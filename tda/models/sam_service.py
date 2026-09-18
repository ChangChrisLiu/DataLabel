"""SAM 2.1 image predictor wrapped as a Qt-free service plus a worker queue.

``SamService`` owns one ``SAM2ImagePredictor`` and caches the image embedding so
repeated clicks on the same viewport crop only pay for the (cheap) mask decoder.
``SamQueue`` runs a service on a single background thread with "latest wins"
semantics, so a fast-clicking user never queues up stale work.

Conventions (see ``tda.core.model``):
  * images are ``HxWx3`` RGB ``uint8`` arrays in *crop* coordinates,
  * masks are boolean ``HxW`` arrays in the same crop coordinates,
  * RLE encoding happens further up the stack, not here.

Torch and sam2 are imported lazily so that importing ``tda.models`` stays cheap
and works on machines without a GPU.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

#: Model config name as shipped inside the ``sam2`` package.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
#: Checkpoint file name expected inside ``paths.yaml: weights_dir``.
CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
#: Radius (crop px) around a new point inside which a refinement is accepted.
REFINE_RADIUS_PX = 48.0
#: Side length of SAM's low-resolution mask input (1024 / 4).
LOW_RES = 256
#: Logit magnitude used when converting a boolean prior mask to mask_input.
MASK_LOGIT_SCALE = 8.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PATHS_YAML = _REPO_ROOT / "configs" / "paths.yaml"


def default_checkpoint() -> Optional[Path]:
    """Return the SAM 2.1 checkpoint path from ``configs/paths.yaml``.

    Falls back to ``<repo>/models/weights/<name>`` when the config file is absent
    or does not define ``weights_dir``. The file is not required to exist.

    Returns ``None`` instead of raising when the config cannot be used at all
    (unparsable YAML, top-level not a mapping, ``weights_dir`` not a path).
    :meth:`SamService.available` and the test suite's collection-time skip guard
    both depend on this never raising: a config typo must degrade to a clean
    "unavailable", not to a collection error.
    """
    try:
        try:
            with open(_PATHS_YAML, "r", encoding="utf-8") as fh:
                cfg: Any = yaml.safe_load(fh)
        except OSError:
            cfg = {}  # no config shipped: use the in-repo fallback below
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise TypeError(f"{_PATHS_YAML} is not a mapping but {type(cfg).__name__}")
        weights_dir: Any = cfg.get("weights_dir") or _REPO_ROOT / "models" / "weights"
        if not isinstance(weights_dir, (str, Path)):
            raise TypeError(f"weights_dir must be a path, got {type(weights_dir).__name__}")
        return Path(weights_dir) / CHECKPOINT_NAME
    except Exception:
        log.warning("cannot resolve the SAM 2.1 checkpoint via %s", _PATHS_YAML, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Request / result payloads
# ---------------------------------------------------------------------------
@dataclass
class SamRequest:
    """One prompt for the image predictor, entirely in crop coordinates.

    Attributes:
        image_crop: ``HxWx3`` RGB uint8 viewport crop (<=1024 long side; the
            caller does the scaling so that mask pixels map back 1:1).
        points: ``(x, y, label)`` triples; ``label`` 1 = foreground, 0 = background.
        box: ``(x0, y0, x1, y1)`` box prompt.
        mask_input: prior boolean ``HxW`` mask; enables *local refinement* — the
            new prediction is only accepted within :data:`REFINE_RADIUS_PX` of a
            point in ``points``, the prior mask is kept everywhere else. With
            ``mask_input`` but no ``points`` there is no anchor for a local edit,
            so the prediction is returned unblended (mask-guided box refinement).
        multimask: ask SAM for three candidates and keep the highest scoring one.
    """

    image_crop: np.ndarray
    points: list[tuple[float, float, int]] = field(default_factory=list)
    box: Optional[tuple[float, float, float, float]] = None
    mask_input: Optional[np.ndarray] = None
    multimask: bool = False


@dataclass
class SamResult:
    """A predicted mask plus the alternatives the annotator may cycle through.

    Attributes:
        mask: boolean ``HxW`` array in crop coordinates -- the best candidate.
        score: SAM's predicted IoU for the *proposal* it produced. After a local
            refinement it still describes that proposal, not the blended mask
            actually returned, so do not read it as a quality score for ``mask``.
        ms: wall-clock duration of :meth:`SamService.predict`, including the
            embedding computation when the crop was not cached yet.
        candidates: every proposal, sorted by score descending, so
            ``candidates[0] is mask``. ``multimask=True`` fills this with SAM's
            three masks; everything else leaves a single entry. After a local
            refinement it holds the *blended* mask only -- the raw proposals
            would undo the blend if they were applied (spec 4.6).
        scores: ``candidates``' scores, same order and length.
    """

    mask: np.ndarray
    score: float
    ms: float
    candidates: list[np.ndarray] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Default the candidate list to the single mask (backward compatible)."""
        if not self.candidates:
            self.candidates = [self.mask]
        if not self.scores:
            self.scores = [self.score] * len(self.candidates)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Validate an image crop and return a C-contiguous ``HxWx3`` uint8 view."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"image_crop must be HxWx3 RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"image_crop must be uint8, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr)


def mask_to_low_res_logits(mask: np.ndarray, scale: float = MASK_LOGIT_SCALE) -> np.ndarray:
    """Convert a boolean crop mask to SAM's ``(1, 256, 256)`` float logit input.

    SAM 2 resizes every image to a fixed 1024x1024 square (no aspect-ratio
    padding), so a plain bilinear resize of the mask to 256x256 lands in exactly
    the frame the mask decoder expects. Values are mapped to ``+/-scale`` logits
    with a soft boundary, since SAM thresholds ``mask_input`` at 0.
    """
    m = np.asarray(mask).astype(np.float32)
    if m.ndim != 2:
        raise ValueError(f"mask_input must be HxW, got shape {m.shape}")
    small = cv2.resize(m, (LOW_RES, LOW_RES), interpolation=cv2.INTER_LINEAR)
    logits = (small - 0.5) * (2.0 * float(scale))
    return logits[None, :, :].astype(np.float32)


def points_within_radius(
    shape: tuple[int, int],
    points: list[tuple[float, float, int]],
    radius: float = REFINE_RADIUS_PX,
) -> np.ndarray:
    """Boolean ``HxW`` map of pixels within ``radius`` px of any given point."""
    h, w = shape
    near = np.zeros((h, w), dtype=bool)
    r2 = float(radius) * float(radius)
    for point in points:
        x, y = float(point[0]), float(point[1])
        x0 = max(0, int(np.floor(x - radius)))
        x1 = min(w, int(np.ceil(x + radius)) + 1)
        y0 = max(0, int(np.floor(y - radius)))
        y1 = min(h, int(np.ceil(y + radius)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        dx = np.arange(x0, x1, dtype=np.float32) - x
        dy = np.arange(y0, y1, dtype=np.float32) - y
        d2 = dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :]
        near[y0:y1, x0:x1] |= d2 <= r2
    return near


def blend_local(
    prior: np.ndarray,
    proposal: np.ndarray,
    points: list[tuple[float, float, int]],
    radius: float = REFINE_RADIUS_PX,
) -> np.ndarray:
    """Keep ``proposal`` near the new points and ``prior`` everywhere else."""
    if not points:
        return prior.copy()
    near = points_within_radius(prior.shape, points, radius)
    return np.where(near, proposal, prior)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class SamService:
    """SAM 2.1 image predictor with an embedding cache. Thread-safe, Qt-free."""

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        config: str = DEFAULT_CONFIG,
        device: str = "cuda",
    ) -> None:
        """Build the model. ``checkpoint=None`` reads ``paths.yaml: weights_dir``."""
        # Resolve the checkpoint before importing torch/sam2: a config or path
        # mistake should fail immediately, not after a multi-second import.
        ckpt = Path(checkpoint) if checkpoint else default_checkpoint()
        if ckpt is None:
            raise FileNotFoundError(
                f"cannot resolve the SAM 2.1 checkpoint; check weights_dir in {_PATHS_YAML}"
            )
        if not ckpt.is_file():
            raise FileNotFoundError(f"SAM 2.1 checkpoint not found: {ckpt}")

        import torch  # local import: keeps ``import tda.models`` cheap
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch.cuda is unavailable")

        self.checkpoint = str(ckpt)
        self.config = config
        self.device = device
        self._torch = torch

        t0 = time.perf_counter()
        self._model = build_sam2(config, str(ckpt), device=device)
        self._predictor = SAM2ImagePredictor(self._model)
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        log.info("SAM 2.1 loaded from %s on %s in %.0f ms", ckpt.name, device, self.load_ms)

        # Re-entrant: predict() holds the lock and calls set_image().
        self._lock = threading.RLock()
        self._image_id: Optional[str] = None
        self._image_src: Optional[np.ndarray] = None  # identity fast path
        self._image_hw: Optional[tuple[int, int]] = None
        self.last_set_image_ms = 0.0

    # -- introspection ------------------------------------------------------
    @staticmethod
    def available(checkpoint: Optional[str] = None) -> bool:
        """True when torch+CUDA and sam2 are importable and the checkpoint exists.

        Never raises. Tests gate on this at import time and the app probes it at
        startup, so every failure mode — missing torch, broken driver, uninstalled
        sam2, unreadable or malformed ``paths.yaml`` — must degrade to ``False``.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            import importlib.util

            if importlib.util.find_spec("sam2.build_sam") is None:
                return False
            ckpt = Path(checkpoint) if checkpoint else default_checkpoint()
            return ckpt is not None and ckpt.is_file()
        except Exception:
            log.debug("SAM 2.1 is unavailable", exc_info=True)
            return False

    @property
    def image_id(self) -> Optional[str]:
        """Id of the crop whose embedding is currently cached, if any."""
        return self._image_id

    # -- inference ----------------------------------------------------------
    def set_image(self, image_crop: np.ndarray) -> str:
        """Compute and cache the image embedding; return the crop's id.

        The embedding is only recomputed when the crop differs from the cached
        one. Identical arrays are detected by object identity first (free) and
        by shape + content hash otherwise, so an equal copy also hits the cache.

        Because of that identity fast path, callers must never mutate a crop
        array in place after handing it over — the same object with new pixels
        looks unchanged and would silently reuse the stale embedding. Pass a new
        array instead.
        """
        img = _as_rgb_uint8(image_crop)
        with self._lock:
            image_id = self._compute_id(image_crop, img)
            if image_id == self._image_id:
                return image_id
            # Drop the cache first: a failed set_image leaves the predictor in an
            # undefined state, which must never be reused by a later predict().
            self._image_id = None
            self._image_src = None
            self._image_hw = None
            t0 = time.perf_counter()
            with self._torch.inference_mode(), self._autocast():
                self._predictor.set_image(img)
            self.last_set_image_ms = (time.perf_counter() - t0) * 1000.0
            self._image_id = image_id
            self._image_src = image_crop
            self._image_hw = (img.shape[0], img.shape[1])
            return image_id

    def predict(self, req: SamRequest) -> SamResult:
        """Run one prompt; performs ``set_image`` first when the crop changed.

        With ``req.mask_input`` set *and* at least one point, the raw prediction
        is only accepted within :data:`REFINE_RADIUS_PX` px of a point and the
        prior mask is kept elsewhere (see :func:`blend_local`).
        """
        t0 = time.perf_counter()
        with self._lock:
            self.set_image(req.image_crop)
            assert self._image_hw is not None
            h, w = self._image_hw

            point_coords: Optional[np.ndarray] = None
            point_labels: Optional[np.ndarray] = None
            if req.points:
                point_coords = np.asarray(
                    [(float(p[0]), float(p[1])) for p in req.points], dtype=np.float32
                )
                point_labels = np.asarray([int(p[2]) for p in req.points], dtype=np.int32)

            box = None
            if req.box is not None:
                box = np.asarray(req.box, dtype=np.float32).reshape(1, 4)

            prior: Optional[np.ndarray] = None
            low_res: Optional[np.ndarray] = None
            if req.mask_input is not None:
                prior = np.asarray(req.mask_input).astype(bool)
                if prior.shape != (h, w):
                    raise ValueError(
                        f"mask_input shape {prior.shape} != crop shape {(h, w)}"
                    )
                low_res = mask_to_low_res_logits(prior)

            if point_coords is None and box is None and low_res is None:
                raise ValueError("SamRequest carries no prompt (points/box/mask_input)")

            with self._torch.inference_mode(), self._autocast():
                masks, scores, _low = self._predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    mask_input=low_res,
                    multimask_output=bool(req.multimask),
                    normalize_coords=True,
                )

            flat = np.asarray(scores).reshape(-1)
            # Stable sort so that equal scores keep SAM's own order and the
            # candidate the annotator sees first never depends on tie breaking.
            order = np.argsort(-flat, kind="stable")
            candidates = [
                np.ascontiguousarray(np.asarray(masks[i]) > 0.5, dtype=bool)
                for i in order
            ]
            ranked = [float(flat[i]) for i in order]
            if prior is not None and req.points:
                # Only the blended mask is a valid edit; offering the raw
                # proposals as alternatives would silently drop the prior.
                candidates = [
                    np.ascontiguousarray(
                        blend_local(prior, candidates[0], req.points, REFINE_RADIUS_PX),
                        dtype=bool,
                    )
                ]
                ranked = ranked[:1]

        return SamResult(
            mask=candidates[0],
            score=ranked[0],
            ms=(time.perf_counter() - t0) * 1000.0,
            candidates=candidates,
            scores=ranked,
        )

    # -- internals ----------------------------------------------------------
    def _autocast(self):
        if self.device.startswith("cuda"):
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def _compute_id(self, original: np.ndarray, contiguous: np.ndarray) -> str:
        """Identity fast path, then shape + content hash.

        SHA-256 over the raw buffer is hardware accelerated here (~0.5 ms for a
        900x900 crop, no intermediate copy), which is cheap next to the ~30 ms
        image encoder it saves whenever the same crop is re-submitted.
        """
        if self._image_id is not None and original is self._image_src:
            return self._image_id
        digest = hashlib.sha256(contiguous).hexdigest()[:32]
        return f"{contiguous.shape[0]}x{contiguous.shape[1]}-{digest}"


# ---------------------------------------------------------------------------
# Background queue
# ---------------------------------------------------------------------------
_STOP = object()


class SamQueue:
    """Runs a :class:`SamService` on one background thread, latest request wins.

    At most one request is in flight. A request that is still waiting when a new
    one arrives is dropped without being run, so a user dragging or clicking
    quickly always gets the newest result and never a backlog. Callbacks are
    invoked **on the worker thread** — Qt code must marshal them itself
    (that is the job of the Qt wrapper, not of this class).
    """

    def __init__(self, service: SamService) -> None:
        self._service = service
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._stopped = False
        self.dropped = 0  # requests replaced before they ever ran
        self.last_error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name="sam-queue", daemon=True)
        self._thread.start()

    def submit(self, req: SamRequest, cb: Callable[[SamResult], None]) -> None:
        """Queue ``req``, dropping any request that has not started running yet."""
        with self._lock:
            if self._stopped:
                raise RuntimeError("SamQueue has been stopped")
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                self.dropped += 1
            self._queue.put((req, cb))

    def stop(self, timeout: float = 30.0) -> None:
        """Drop pending work, let the in-flight request finish, join the thread."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                self._queue.task_done()
                self.dropped += 1
            self._queue.put(_STOP)
        self._thread.join(timeout)
        if self._thread.is_alive():  # pragma: no cover - only on a wedged GPU call
            log.warning("SamQueue worker did not stop within %.1f s", timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                req, cb = item
                try:
                    result = self._service.predict(req)
                except Exception as exc:  # keep the worker alive on bad prompts
                    self.last_error = exc
                    log.exception("SAM predict failed: %s", exc)
                    continue
                try:
                    cb(result)
                except Exception as exc:  # a broken callback must not kill the queue
                    self.last_error = exc
                    log.exception("SAM callback failed: %s", exc)
            finally:
                self._queue.task_done()
