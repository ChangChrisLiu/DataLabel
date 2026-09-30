"""The small-part detector: its configuration, its tiles and its model (task U3).

L2 (``experiments/l2_detector/REPORT.md``) trained a YOLO26n on the old Label
Studio drafts that finds *motherboard screws* on the scanner and on OAK camera
1; this module is that detector as the application runs it, Qt-free:

* :func:`load_detector_config` reads ``configs/detector.yaml`` the way
  :func:`tda.ui.prompt_gate.load_prompt_gate` reads its file -- it never
  raises, and a file that cannot be used is a WARNING and a detector that is
  off;
* :func:`work_crop`, :func:`make_tiles`, :func:`to_native` and :func:`merge`
  are L2's ``tiles.py`` / ``infer.py``, unchanged in what they compute, so the
  model sees at inference exactly the tiles it was trained on;
* :class:`YoloTileDetector` is the model itself.  It is only ever built on the
  detector's worker thread (:mod:`tda.ui.app_detect`): importing ultralytics and
  torch takes seconds, and the window must not wait for them;
* :func:`box_change` is L1's native-resolution dE inside a box, the score the
  candidates are ranked by.

Importing ultralytics patches the whole process, and the window shares that
process; :func:`keep_process_state` takes a snapshot before the import and puts
**every** one of these back the moment the import returns, and again once the
model is built and warmed up (task U3, round 3).  For the import's own second
or two the patched functions are live process-wide -- that cannot be helped
short of a second process:

* ``cv2.setNumThreads(0)`` -- one OpenCV thread for every difference map and
  every decode in the window;
* ``cv2.imread`` / ``cv2.imwrite`` / ``cv2.imshow`` replaced (Windows): the
  replacement *raises* ``cv2.error`` on a 0-byte file where OpenCV returns
  ``None``, and the window's image cache, the difference map's worker, the ROI
  worker and the truth inputs all rely on ``None``;
* ``PIL.Image.open`` and ``torch.save`` replaced;
* numpy's and torch's print options;
* the environment variables in :data:`ULTRALYTICS_ENV` -- *only* those: SAM's
  loader thread runs at the same time and sets its own (torch's inductor sets
  ``TORCHINDUCTOR_CACHE_DIR``), and they stay (round 4);
* the warning filters ultralytics' own code adds -- *only* those, told apart by
  the module that calls ``warnings.filterwarnings`` / ``simplefilter``: sympy,
  imported on SAM's thread meanwhile, adds one of its own, and it stays.

It also writes a settings file under ``%APPDATA%`` unless ``YOLO_CONFIG_DIR``
says where; :func:`prepare_environment` points it at D: and turns downloads
off (those two variables are set on purpose, before the snapshot).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import logging
import math
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import cv2
import numpy as np

__all__ = [
    "CONTAIN", "DEFAULT_FILE", "DEFAULT_YOLO_CONFIG_DIR", "EDGE_PX", "ENV_VAR",
    "MAX_DET", "NMS_IOU", "PAD_VALUE", "PREDICT_CONF", "STORE_CONF",
    "ULTRALYTICS_ENV", "Det",
    "DetectorConfig", "DetectorConfigError", "DetectorUnavailable", "Tile",
    "YoloTileDetector", "box_change", "file_sha1", "keep_process_state",
    "load_detector_config",
    "make_tiles", "merge", "prepare_environment", "to_native", "work_crop",
]

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO / "configs" / "detector.yaml"
#: Another detector file for this process, or ``off``.
ENV_VAR = "TDA_DETECTOR"
#: Where ultralytics keeps its settings when the file does not say.
DEFAULT_YOLO_CONFIG_DIR = "D:/DataSet/.cache/ultralytics_cfg"

# L2's inference constants (``experiments/l2_detector/tiles.py``, ``infer.py``).
PAD_VALUE = 114
#: A detection this close to an interior tile edge may be a cut-off part.
EDGE_PX = 2.0
#: ... and is dropped when this much of it lies in a whole one from another tile.
CONTAIN = 0.6
NMS_IOU = 0.5
#: The model is asked for everything above this; thresholds are filters after.
PREDICT_CONF = 0.001
MAX_DET = 300
#: What the cache keeps of it: far below any operating point, far above noise.
STORE_CONF = 0.01

Box = tuple[float, float, float, float]


class DetectorConfigError(ValueError):
    """Why the detector file cannot be used; the sentence the WARNING says."""


class DetectorUnavailable(RuntimeError):
    """The model cannot run here: no ultralytics, no CUDA, no weights."""


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectorConfig:
    """``configs/detector.yaml``, checked."""

    source: Path
    model: Path
    classes: tuple[str, ...]
    views: tuple[str, ...]
    conf: float
    roi_crop: bool
    work_scale: tuple[tuple[str, float], ...]
    tile: int
    stride: int
    device: Any = 0
    half: bool = True
    yield_ms: float = 20.0
    cache_root: Path = Path("D:/DataSet/cache/det")
    yolo_config_dir: Path = Path(DEFAULT_YOLO_CONFIG_DIR)
    enabled: bool = True

    def applies(self, view: Optional[str]) -> bool:
        """Is the detector on for ``view``?"""
        return str(view) in self.views

    def scale(self, view: str) -> float:
        """The view's work scale (1.0 = native pixels)."""
        return dict(self.work_scale).get(str(view), 1.0)

    @property
    def store_conf(self) -> float:
        """The lowest confidence the cache keeps."""
        return min(STORE_CONF, self.conf)

    def recipe(self) -> dict:
        """Everything that decides what a frame's detections are, bar the model.

        A cache written under another recipe is not an answer for this one.
        """
        return {"roi_crop": bool(self.roi_crop),
                "work_scale": {v: float(s) for v, s in sorted(self.work_scale)},
                "tile": int(self.tile), "stride": int(self.stride),
                "pad": PAD_VALUE, "edge_px": EDGE_PX, "contain": CONTAIN,
                "nms_iou": NMS_IOU, "predict_conf": PREDICT_CONF,
                "max_det": MAX_DET, "store_conf": float(self.store_conf)}


def _read_yaml(path: Path) -> dict:
    import yaml

    if not path.is_file():
        raise DetectorConfigError(f"the detector file {path} does not exist")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DetectorConfigError(f"the detector file {path} cannot be read: "
                                  f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict):
        raise DetectorConfigError(f"the detector file {path} is not a mapping of keys")
    return data


def _number(raw: Any, what: str, low: float, high: float, *,
            low_open: bool = False) -> float:
    if isinstance(raw, bool):
        raise DetectorConfigError(f"{what} = {raw!r} is not a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise DetectorConfigError(f"{what} = {raw!r} is not a number") from None
    if not math.isfinite(value):
        raise DetectorConfigError(f"{what} = {raw!r} is not a finite number")
    if value < low or value > high or (low_open and value <= low):
        raise DetectorConfigError(f"{what} = {raw!r} is outside "
                                  f"{'(' if low_open else '['}{low:g}, {high:g}]")
    return value


def _names(raw: Any, what: str) -> tuple[str, ...]:
    if (not isinstance(raw, list) or not raw
            or not all(isinstance(v, str) and v.strip() for v in raw)):
        raise DetectorConfigError(f"{what} must be a non-empty list of names "
                                  f"({what}: {raw!r})")
    return tuple(dict.fromkeys(v.strip() for v in raw))


def _path(raw: Any, what: str, base: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise DetectorConfigError(f"{what} must be a path ({what}: {raw!r})")
    path = Path(raw.strip())
    return path if path.is_absolute() else (base / path)


def _parse(data: dict, source: Path) -> DetectorConfig:
    """A :class:`DetectorConfig`, or :class:`DetectorConfigError` saying why not."""
    from tda.core.model import VIEWS

    enabled = data.get("enabled", True)
    if not isinstance(enabled, bool):
        raise DetectorConfigError(f"enabled = {enabled!r} is not true or false")
    base = source.parent
    model = _path(data.get("model"), "model", base)
    classes = _names(data.get("classes"), "classes")
    views = _names(data.get("views"), "views")
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise DetectorConfigError(f"views {unknown} are not views ({', '.join(VIEWS)})")
    conf = _number(data.get("conf"), "conf", 0.0, 1.0, low_open=True)
    tiling = data.get("tiling")
    if not isinstance(tiling, dict):
        raise DetectorConfigError(f"tiling must be a mapping (tiling: {tiling!r})")
    roi_crop = tiling.get("roi_crop", True)
    if not isinstance(roi_crop, bool):
        raise DetectorConfigError(f"tiling.roi_crop = {roi_crop!r} is not true or false")
    scales = tiling.get("work_scale")
    if not isinstance(scales, dict):
        raise DetectorConfigError(f"tiling.work_scale must map views to scales "
                                  f"({scales!r})")
    missing = [v for v in views if v not in scales]
    if missing:
        raise DetectorConfigError(f"tiling.work_scale has no scale for {missing}")
    work_scale = tuple(sorted(
        (str(v), _number(s, f"tiling.work_scale.{v}", 0.0, 8.0, low_open=True))
        for v, s in scales.items()))
    tile = int(_number(tiling.get("tile"), "tiling.tile", 32, 4096))
    stride = int(_number(tiling.get("stride"), "tiling.stride", 1, tile))
    if float(tiling.get("tile")) != tile or float(tiling.get("stride")) != stride:
        raise DetectorConfigError("tiling.tile and tiling.stride must be whole pixels")
    device = data.get("device", 0)
    if isinstance(device, bool) or not isinstance(device, (int, str)):
        raise DetectorConfigError(f"device = {device!r} is not a GPU index or name")
    half = data.get("half", True)
    if not isinstance(half, bool):
        raise DetectorConfigError(f"half = {half!r} is not true or false")
    yield_ms = _number(data.get("yield_ms", 20.0), "yield_ms", 0.0, 10_000.0)
    cache_root = _path(data.get("cache_root", "D:/DataSet/cache/det"), "cache_root", base)
    yolo_dir = _path(data.get("yolo_config_dir", DEFAULT_YOLO_CONFIG_DIR),
                     "yolo_config_dir", base)
    return DetectorConfig(
        source=source, model=model, classes=classes, views=views, conf=conf,
        roi_crop=roi_crop, work_scale=work_scale, tile=tile, stride=stride,
        device=device, half=half, yield_ms=yield_ms, cache_root=cache_root,
        yolo_config_dir=yolo_dir, enabled=enabled)


def _load(target: Path) -> DetectorConfig:
    config = _parse(_read_yaml(target), target)
    if not config.enabled:
        return config
    if not config.model.is_file():
        raise DetectorConfigError(f"the model file {config.model} does not exist")
    if importlib.util.find_spec("ultralytics") is None:
        raise DetectorConfigError("ultralytics is not installed in this environment")
    return config


def load_detector_config(path: Optional[str] = None) -> Optional[DetectorConfig]:
    """The detector of ``configs/detector.yaml`` (or ``path``), or ``None``; never raises.

    Without ``path`` the environment's :data:`ENV_VAR` is read first: ``off``
    is a detector that is off, anything else a file to read instead.

    **Never silently.**  ``enabled: false`` and ``TDA_DETECTOR=off`` are one
    INFO line each; a file that cannot be used -- missing, malformed, a model
    file that is not there, no ultralytics -- is one WARNING saying why, and
    the detector is off.  The two checks this cannot make without importing
    torch (is there a GPU, does ultralytics import) happen where the model is
    built, on the worker thread, and end the same way.  The INFO line that
    says the detector is **on** comes from there too, once the model has
    loaded: until then it is not on.
    """
    if path is None:
        chosen = os.environ.get(ENV_VAR, "").strip()
        if chosen.lower() == "off":
            log.info("small-part detector off: %s=off", ENV_VAR)
            return None
        path = chosen or None
    target = Path(path) if path else DEFAULT_FILE
    try:
        config = _load(target)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        log.warning("small-part detector OFF, the guess is the difference map's: %s", exc)
        return None
    if not config.enabled:
        log.info("small-part detector off: enabled is false in %s", target)
        return None
    return config


def prepare_environment(config: DetectorConfig) -> None:
    """Point ultralytics at D: for its settings and forbid it every download."""
    target = Path(config.yolo_config_dir)
    target.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(target)
    os.environ["YOLO_OFFLINE"] = "1"


#: The OpenCV functions ultralytics replaces on Windows.
_CV2_PATCHED = ("imread", "imwrite", "imshow")
#: torch's print options, as ``torch.set_printoptions`` takes them.
_TORCH_PRINT = ("precision", "threshold", "edgeitems", "linewidth", "sci_mode")


class _TorchState:
    """torch's own ``save`` and print options, taken once torch is imported."""

    def __init__(self) -> None:
        self.torch: Any = None
        self.save: Any = None
        self.print_opts: Optional[dict] = None

    def watch(self, torch: Any) -> None:
        """Remember torch's state now -- import torch, then call this, then ultralytics."""
        if self.torch is None:
            self.torch, self.save = torch, torch.save
            self.print_opts = {k: getattr(torch._tensor_str.PRINT_OPTS, k)
                               for k in _TORCH_PRINT}

    def restore(self) -> None:
        if self.torch is not None:
            self.torch.save = self.save
            self.torch.set_printoptions(**self.print_opts)


#: Every environment variable ultralytics (8.4.155) writes or removes, from its
#: source (``os.environ[...] =`` / ``.pop``): on import ``OMP_NUM_THREADS``
#: (``ultralytics/__init__``) and ``NUMEXPR_MAX_THREADS``,
#: ``TF_CPP_MIN_LOG_LEVEL``, ``TORCH_CPP_LOG_LEVEL``, ``KINETO_LOG_LEVEL``
#: (``utils/__init__``); ``CUBLAS_WORKSPACE_CONFIG`` and ``PYTHONHASHSEED``
#: (``utils/torch_utils.init_seeds``, set or popped); and on paths the app never
#: takes -- training, callbacks, exporters -- ``NO_ALBUMENTATIONS_UPDATE``,
#: ``TORCH_NCCL_BLOCKING_WAIT``, ``KMP_DUPLICATE_LIB_OK``,
#: ``COMET_START_ONLINE``, ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION``.  Not
#: ``PATH``: only the Axelera exporter touches it, and puts it back itself.
ULTRALYTICS_ENV = (
    "OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS", "TF_CPP_MIN_LOG_LEVEL",
    "TORCH_CPP_LOG_LEVEL", "KINETO_LOG_LEVEL", "CUBLAS_WORKSPACE_CONFIG",
    "PYTHONHASHSEED", "NO_ALBUMENTATIONS_UPDATE", "TORCH_NCCL_BLOCKING_WAIT",
    "KMP_DUPLICATE_LIB_OK", "COMET_START_ONLINE",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
)


class _FilterWatch:
    """Records the warning filters ultralytics' own code adds, and nobody else's.

    While at least one :func:`keep_process_state` is open,
    ``warnings.filterwarnings`` and ``warnings.simplefilter`` are wrapped: a
    call made from a module of the ``ultralytics`` package has the entry it
    added noted in every open span's list; any other caller -- another thread,
    sympy, torch -- goes straight through, unrecorded.  The entry is the new
    tuple in ``warnings.filters`` whose fields are the call's arguments, so a
    filter another thread adds at the same moment is not mistaken for it; a
    filter that was there before (``filterwarnings`` only moves it to the
    front) is not new and is never taken away.
    """

    _lock = threading.Lock()
    _spans: list = []
    _originals: dict = {}

    @classmethod
    def open(cls) -> list:
        added: list = []
        with cls._lock:
            if not cls._spans:
                for name in ("filterwarnings", "simplefilter"):
                    original = getattr(warnings, name)
                    cls._originals[name] = original
                    setattr(warnings, name, cls._wrap(original))
            cls._spans.append(added)
        return added

    @classmethod
    def close(cls, added: list) -> None:
        with cls._lock:
            cls._spans = [s for s in cls._spans if s is not added]
            if not cls._spans:
                for name, original in cls._originals.items():
                    if getattr(getattr(warnings, name), "__wrapped__", None) is original:
                        setattr(warnings, name, original)
                cls._originals = {}

    @classmethod
    def _wrap(cls, original):
        signature = inspect.signature(original)

        def watched(*args, **kwargs):
            caller = sys._getframe(1).f_globals.get("__name__") or ""
            if caller.partition(".")[0] != "ultralytics":
                return original(*args, **kwargs)
            before = list(warnings.filters)
            result = original(*args, **kwargs)
            call = signature.bind(*args, **kwargs)
            call.apply_defaults()
            new = [f for f in warnings.filters
                   if f not in before and _is_filter(f, call.arguments)]
            with cls._lock:
                for span in cls._spans:
                    span.extend(new)
            return result

        watched.__wrapped__ = original
        return watched


def _is_filter(entry: tuple, arguments: dict) -> bool:
    """Is this ``warnings.filters`` entry the one a call with these arguments adds?"""
    action, message, category, module, lineno = entry
    text = message.pattern if message is not None else ""
    where = module.pattern if module is not None else ""
    return (action == arguments["action"] and category is arguments["category"]
            and lineno == arguments["lineno"] and text == arguments.get("message", "")
            and where == arguments.get("module", ""))


@contextlib.contextmanager
def keep_process_state(torch: Any = None) -> Iterator[_TorchState]:
    """Put back everything importing (and running) ultralytics changes process-wide.

    A snapshot on entry, put back on exit whatever happened in between, of
    what only ultralytics changes: the OpenCV thread count and its ``imread``
    / ``imwrite`` / ``imshow``, ``PIL.Image.open``, numpy's print options, and
    ``torch.save`` and torch's print options once torch is known (pass it, or
    import it inside and call ``watch(torch)`` on what this yields before
    ultralytics is imported).

    What other threads change too is handled one entry at a time (round 4):

    * the environment -- only the variables in :data:`ULTRALYTICS_ENV`: one
      that was absent and is now set is removed, one that changed (or went)
      gets its old value back; every other variable is left as it is, so what
      SAM's loader sets meanwhile stays;
    * the warning filters -- only the entries ultralytics' own calls added in
      between (:class:`_FilterWatch`) are removed; everyone else's stay.
    """
    import PIL.Image

    env = {key: os.environ.get(key) for key in ULTRALYTICS_ENV}
    threads = cv2.getNumThreads()
    cv2_fns = {name: getattr(cv2, name) for name in _CV2_PATCHED}
    pil_open = PIL.Image.open
    np_print = np.get_printoptions()
    torch_state = _TorchState()
    if torch is not None:
        torch_state.watch(torch)
    added = _FilterWatch.open()
    try:
        yield torch_state
    finally:
        _FilterWatch.close(added)
        cv2.setNumThreads(threads)
        for name, fn in cv2_fns.items():
            setattr(cv2, name, fn)
        PIL.Image.open = pil_open
        np.set_printoptions(**np_print)
        torch_state.restore()
        for entry in added:
            # ``warnings`` keeps no two equal entries, so the equal one that
            # ``remove`` takes (in one step, under the GIL) is this very entry
            if any(current is entry for current in warnings.filters):
                with contextlib.suppress(ValueError):
                    warnings.filters.remove(entry)
        if added and hasattr(warnings, "_filters_mutated"):
            warnings._filters_mutated()
        for key, value in env.items():
            if os.environ.get(key) == value:
                continue
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def file_sha1(path: Path) -> str:
    """SHA-1 of a file's bytes: the model's identity in the cache."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# detections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Det:
    """One detection in native frame pixels."""

    box: Box
    cls: str
    conf: float

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))

    @property
    def int_box(self) -> tuple[int, int, int, int]:
        """The box on whole pixels: what dE is measured in and SAM is given (L2)."""
        return tuple(int(round(v)) for v in self.box)  # type: ignore[return-value]

    def row(self) -> list:
        """``[x0, y0, x1, y1, cls, conf]`` as L2's ``detections/*.json`` wrote it."""
        return [round(float(v), 1) for v in self.box] + [str(self.cls),
                                                          round(float(self.conf), 4)]

    @classmethod
    def from_row(cls, row: Sequence) -> "Det":
        return cls(box=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                   cls=str(row[4]), conf=float(row[5]))


# --------------------------------------------------------------------------- #
# tiles (L2's tiles.py)
# --------------------------------------------------------------------------- #
@dataclass
class Tile:
    """One tile: its pixels and where it sits in the work-scale crop."""

    rgb: np.ndarray
    ox: int
    oy: int
    w: int
    h: int


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    s = list(range(0, length - tile + 1, stride))
    if s[-1] != length - tile:
        s.append(length - tile)
    return s


def work_crop(img: np.ndarray, crop: Sequence[int], scale: float) -> np.ndarray:
    """The frame cut to ``crop`` (x0, y0, x1, y1) and brought to ``scale``."""
    x0, y0, x1, y1 = (int(v) for v in crop)
    out = img[y0:y1, x0:x1]
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def make_tiles(crop: np.ndarray, tile: int = 640, stride: int = 512) -> list[Tile]:
    """``tile`` x ``tile`` pieces at ``stride``, the last flush with the edge, grey-padded."""
    height, width = crop.shape[:2]
    out = []
    for oy in _starts(height, tile, stride):
        for ox in _starts(width, tile, stride):
            t = crop[oy:oy + tile, ox:ox + tile]
            h, w = t.shape[:2]
            if h < tile or w < tile:
                pad = np.full((tile, tile, 3), PAD_VALUE, np.uint8)
                pad[:h, :w] = t
                t = pad
            out.append(Tile(rgb=np.ascontiguousarray(t), ox=ox, oy=oy, w=w, h=h))
    return out


def to_native(box: Sequence[float], crop: Sequence[int], scale: float) -> Box:
    """A work-scale crop box in native frame coordinates."""
    return (box[0] / scale + crop[0], box[1] / scale + crop[1],
            box[2] / scale + crop[0], box[3] / scale + crop[1])


def _iou_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix0 = np.maximum(a[0], b[:, 0])
    iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2])
    iy1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-6)


def merge(boxes: np.ndarray, confs: np.ndarray, clss: np.ndarray,
          edge: np.ndarray) -> np.ndarray:
    """Indices kept after L2's edge/containment rule and class-wise NMS.

    A detection touching an *interior* tile edge is dropped when a same-class
    detection from another tile that is not itself cut off contains >= 60 %
    of it (the part is whole there: the tiles overlap by 128 px); then
    class-wise NMS at IoU 0.5 over all tiles.  Highest confidence first.
    """
    n = len(boxes)
    if n == 0:
        return np.zeros(0, int)
    alive = np.ones(n, bool)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in np.nonzero(edge)[0]:
        same = np.nonzero((clss == clss[i]) & (np.arange(n) != i))[0]
        if same.size == 0:
            continue
        b = boxes[same]
        ix0 = np.maximum(boxes[i, 0], b[:, 0])
        iy0 = np.maximum(boxes[i, 1], b[:, 1])
        ix1 = np.minimum(boxes[i, 2], b[:, 2])
        iy1 = np.minimum(boxes[i, 3], b[:, 3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        frac = inter / max(areas[i], 1e-6)
        if np.any((frac >= CONTAIN) & ~edge[same]):
            alive[i] = False
    keep = []
    for c in np.unique(clss):
        idx = np.nonzero(alive & (clss == c))[0]
        idx = idx[np.argsort(-confs[idx])]
        while idx.size:
            i = idx[0]
            keep.append(i)
            if idx.size == 1:
                break
            ious = _iou_rows(boxes[i], boxes[idx[1:]])
            idx = idx[1:][ious < NMS_IOU]
    return np.asarray(sorted(keep, key=lambda i: -confs[i]), int)


def tiles_to_dets(tiles: Sequence[Tile], results: Sequence, crop: Sequence[int],
                  scale: float, crop_hw: tuple[int, int], names: dict) -> list[Det]:
    """Per-tile ``(xyxy, conf, cls)`` arrays (or ``None``) -> merged native detections."""
    height, width = crop_hw
    allb, allc, allk, alle = [], [], [], []
    for t, r in zip(tiles, results):
        if r is None or len(r[0]) == 0:
            continue
        xyxy, cf, cl = (np.asarray(r[0], np.float64).copy(), np.asarray(r[1], np.float64),
                        np.asarray(r[2], int))
        # the padded area is not part of the image
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, t.w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, t.h)
        ok = (xyxy[:, 2] - xyxy[:, 0] >= 1) & (xyxy[:, 3] - xyxy[:, 1] >= 1)
        xyxy, cf, cl = xyxy[ok], cf[ok], cl[ok]
        e = np.zeros(len(xyxy), bool)
        if t.ox > 0:
            e |= xyxy[:, 0] <= EDGE_PX
        if t.oy > 0:
            e |= xyxy[:, 1] <= EDGE_PX
        if t.ox + t.w < width:
            e |= xyxy[:, 2] >= t.w - EDGE_PX
        if t.oy + t.h < height:
            e |= xyxy[:, 3] >= t.h - EDGE_PX
        xyxy[:, [0, 2]] += t.ox
        xyxy[:, [1, 3]] += t.oy
        allb.append(xyxy)
        allc.append(cf)
        allk.append(cl)
        alle.append(e)
    out: list[Det] = []
    if allb:
        b = np.concatenate(allb)
        c = np.concatenate(allc)
        k = np.concatenate(allk)
        e = np.concatenate(alle)
        for i in merge(b, c, k, e):
            nb = to_native(b[i], crop, scale)
            out.append(Det(box=tuple(float(v) for v in nb),  # type: ignore[arg-type]
                           cls=str(names.get(int(k[i]), int(k[i]))), conf=float(c[i])))
    return out


# --------------------------------------------------------------------------- #
# the score (L1's box_change)
# --------------------------------------------------------------------------- #
def box_change(img_j: np.ndarray, img_k: np.ndarray, box: Sequence[int]) -> float:
    """Mean native dE inside ``box`` between frame ``j`` and its neighbour ``k``.

    L1's ``methods.box_change`` with the neighbour unregistered: the box grown
    by a quarter of its side (at least 3 px) is cut from both frames, compared
    with :func:`tda.core.diffmap.diff_delta_e` (``blur=3``, 1 px shift
    tolerance, ``j`` first), and the map is averaged inside the box.  A part
    that left between the two frames is a large number; one still in place is
    near zero.
    """
    from tda.core.diffmap import diff_delta_e

    bx0, by0, bx1, by1 = (int(v) for v in box)
    bw, bh = bx1 - bx0, by1 - by0
    pad = max(3, int(round(0.25 * max(bw, bh))))
    x0 = max(bx0 - pad, 0)
    y0 = max(by0 - pad, 0)
    x1 = min(bx1 + pad, img_k.shape[1], img_j.shape[1])
    y1 = min(by1 + pad, img_k.shape[0], img_j.shape[0])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    cj = np.ascontiguousarray(img_j[y0:y1, x0:x1])
    ck = np.ascontiguousarray(img_k[y0:y1, x0:x1])
    d = diff_delta_e(cj, ck, blur=3, shift_px=1)
    ix0, iy0 = max(0, bx0 - x0), max(0, by0 - y0)
    ix1, iy1 = min(x1 - x0, bx1 - x0), min(y1 - y0, by1 - y0)
    inner = d[iy0:iy1, ix0:ix1]
    return float(inner.mean()) if inner.size else 0.0


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class YoloTileDetector:
    """The configured YOLO, run tile-wise on a frame's crop; built off the GUI thread.

    Everything that can fail does so here, in the constructor: CUDA missing,
    ultralytics not importing, weights that do not load.  The caller turns
    that into a WARNING and a detector that is off.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        started = time.perf_counter()
        if not config.model.is_file():
            raise DetectorUnavailable(f"the model file {config.model} does not exist")
        self.sha1 = file_sha1(config.model)
        #: What the cache is keyed by: another model is another directory.
        self.identity = self.sha1
        prepare_environment(config)
        # torch first, outside any snapshot: what torch itself sets (SAM's
        # loader imports it too) is torch's and stays.
        try:
            import torch
        except Exception as exc:  # noqa: BLE001 - the reason is the message
            raise DetectorUnavailable(f"torch does not import: "
                                      f"{type(exc).__name__}: {exc}") from exc
        if not torch.cuda.is_available():
            raise DetectorUnavailable("CUDA is not available")
        # Two snapshots.  The first ends the moment the import returns: the
        # window's own OpenCV calls see ultralytics' imread only for as long
        # as the import takes.  The second covers building the model and the
        # warm-up (YOLO() sets CUBLAS_WORKSPACE_CONFIG, a first prediction may
        # import more of ultralytics).
        with keep_process_state(torch):
            try:
                from ultralytics import YOLO
            except Exception as exc:  # noqa: BLE001
                raise DetectorUnavailable(f"ultralytics does not import: "
                                          f"{type(exc).__name__}: {exc}") from exc
        with keep_process_state(torch):
            self._torch = torch
            self.model = YOLO(str(config.model))
            self.names = {int(k): str(v) for k, v in dict(self.model.names).items()}
            missing = [c for c in config.classes if c not in self.names.values()]
            if missing:
                raise DetectorUnavailable(f"the model has no class {missing} "
                                          f"(it knows {sorted(self.names.values())})")
            # cudnn autotune and lazy CUDA init, before the first real frame
            dummy = [np.full((config.tile, config.tile, 3), PAD_VALUE, np.uint8)] * 2
            for _ in range(2):
                self._predict(dummy)
        self.load_s = time.perf_counter() - started

    def release(self) -> None:
        """Give the CUDA cache back between passes (the worker thread calls it idle)."""
        try:
            self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - a nicety, never a failure
            pass

    def _predict(self, batch: list[np.ndarray]):
        extra = {"quantize": 16} if self.config.half else {}
        with self._torch.inference_mode():
            return self.model.predict(batch, imgsz=self.config.tile, conf=PREDICT_CONF,
                                      verbose=False, device=self.config.device,
                                      max_det=MAX_DET, **extra)

    def describe(self) -> str:
        """What the INFO line says about the model."""
        device = (f"cuda:{self.config.device}" if isinstance(self.config.device, int)
                  else str(self.config.device))
        return (f"model {self.config.model} (sha1 {self.sha1[:12]}, classes "
                f"{[self.names[k] for k in sorted(self.names)]}), {device} "
                f"{'fp16' if self.config.half else 'fp32'}, loaded in {self.load_s:.1f} s")

    def detect(self, img_rgb: np.ndarray, crop: Sequence[int], view: str,
               step: Optional[int] = None) -> list[Det]:
        """Every detection in ``crop`` of the frame, native pixels, best first."""
        scale = self.config.scale(view)
        work = work_crop(img_rgb, crop, scale)
        tiles = make_tiles(work, self.config.tile, self.config.stride)
        batch = [cv2.cvtColor(t.rgb, cv2.COLOR_RGB2BGR) for t in tiles]
        results = []
        for r in self._predict(batch):
            if r.boxes is None or len(r.boxes) == 0:
                results.append(None)
                continue
            results.append((r.boxes.xyxy.cpu().numpy().astype(np.float64),
                            r.boxes.conf.cpu().numpy().astype(np.float64),
                            r.boxes.cls.cpu().numpy().astype(int)))
        return tiles_to_dets(tiles, results, crop, scale, work.shape[:2], self.names)


def frame_crop(config: DetectorConfig, roi: Optional[Sequence[int]]) -> Optional[tuple]:
    """What a frame is detected on: its ROI, or ``None`` for the whole frame."""
    if not config.roi_crop:
        return None
    return None if roi is None else tuple(int(v) for v in roi)


def whole(img: np.ndarray, crop: Optional[Iterable[int]]) -> tuple[int, int, int, int]:
    """``crop`` clipped to the image, or the whole image for ``None``."""
    h, w = img.shape[:2]
    if crop is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = (int(v) for v in crop)
    return (max(0, x0), max(0, y0), min(w, x1), min(h, y1))


@dataclass(frozen=True)
class FrameDets:
    """One frame's detections and the candidates' change, as the window gets them."""

    step: int
    #: What was detected on: the ROI, or ``None`` for the whole frame.
    crop: Optional[tuple]
    dets: tuple[Det, ...]
    #: The frame dE was measured against, or ``None`` (no neighbour: no dE).
    neighbour: Optional[int]
    #: ``{index into dets: dE}`` for the candidates (configured class, conf).
    change: dict = field(default_factory=dict)
    #: The model ran for this answer (not the disk cache).
    detected: bool = False
    #: dE was measured for this answer (not read from the disk cache).
    measured: bool = False
    seconds: float = 0.0
