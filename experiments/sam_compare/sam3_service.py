"""SAM 3 wrapper shaped like ``tda.models.sam_service.SamService``.

One loaded checkpoint serves both halves of the experiment:

* :meth:`Sam3Service.predict_interactive` -- the PVS / SAM-1-task image
  predictor (click and box to mask), which is what would replace SAM 2.1 as the
  annotation tool's click-to-mask engine;
* :meth:`Sam3Service.detect_text` -- open-vocabulary concept prompts, the thing
  SAM 2.1 cannot do at all.

Both go through ``sam3.model.sam3_image_processor.Sam3Processor``. That matters
for the interactive half: ``SAM3InteractiveImagePredictor.set_image`` cannot be
called directly here, because ``build_sam3_image_model`` builds its tracker
*without* a backbone (``build_tracker(with_backbone=False)``). The interactive
head is designed to run off the detector's shared backbone, which the processor
prepares -- including the ``conv_s0``/``conv_s1`` projection of the two
high-resolution FPN levels -- and ``Sam3Image.predict_inst`` then consumes.

The image embedding is cached on the crop exactly like ``SamService`` does, so
the reported per-prompt times are comparable between the two models.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from experiments.sam_compare import _env

log = logging.getLogger(__name__)

#: Native input resolution of the SAM 3 image backbone.
SAM3_RESOLUTION = 1008


def sam3_checkpoint() -> Path:
    """Path of ``sam3.pt`` inside the D: HuggingFace cache (already downloaded)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=_env.SAM3_REPO, filename="sam3.pt"))


@dataclass
class Sam3Result:
    """One predicted mask, mirroring ``tda.models.sam_service.SamResult``."""

    mask: np.ndarray
    score: float
    ms: float


class Sam3Service:
    """SAM 3 image model with a crop embedding cache. Qt-free, thread-unsafe."""

    def __init__(self, device: str = "cuda", autocast: bool = True) -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self._torch = torch
        self.device = device
        self._use_autocast = autocast

        t0 = time.perf_counter()
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=str(sam3_checkpoint()),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        if model.inst_interactive_predictor is None:  # pragma: no cover
            raise RuntimeError("SAM 3 built without inst_interactive_predictor")
        self._model = model
        self._proc = Sam3Processor(model, resolution=SAM3_RESOLUTION, device=device)
        self.load_ms = (time.perf_counter() - t0) * 1000.0

        self._image_id: Optional[str] = None
        self._state: Optional[dict] = None
        self.last_set_image_ms = 0.0
        log.info("SAM 3 loaded in %.1f s", self.load_ms / 1000.0)

    # -- inference ----------------------------------------------------------
    def set_image(self, image_crop: np.ndarray) -> str:
        """Compute and cache the backbone features for one crop; return its id."""
        img = np.ascontiguousarray(image_crop)
        image_id = f"{img.shape[0]}x{img.shape[1]}-{hashlib.sha256(img).hexdigest()[:32]}"
        if image_id == self._image_id:
            return image_id
        # a failed set_image must not leave a stale state reachable
        self._image_id, self._state = None, None
        t0 = time.perf_counter()
        with self._autocast():
            state = self._proc.set_image(self._to_chw(img))
        self.last_set_image_ms = (time.perf_counter() - t0) * 1000.0
        self._image_id, self._state = image_id, state
        return image_id

    def predict_interactive(
        self,
        image_crop: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask: bool = False,
    ) -> Sam3Result:
        """One point/box prompt on ``image_crop``; best of the returned masks."""
        t0 = time.perf_counter()
        self.set_image(image_crop)
        assert self._state is not None
        with self._autocast():
            masks, scores, _low = self._model.predict_inst(
                self._state,
                point_coords=points,
                point_labels=labels,
                box=box,
                multimask_output=bool(multimask),
                normalize_coords=True,
            )
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        masks = np.asarray(masks)
        if masks.ndim == 2:
            masks = masks[None]
        best = int(np.argmax(scores))
        return Sam3Result(
            mask=np.ascontiguousarray(masks[best] > 0.5, dtype=bool),
            score=float(scores[best]),
            ms=(time.perf_counter() - t0) * 1000.0,
        )

    def detect_text(self, image: np.ndarray, text: str, threshold: float = 0.5) -> dict:
        """Open-vocabulary concept prompt; returns ``{masks, scores, boxes, ms}``.

        ``masks`` is ``Nx H x W`` boolean at the input image's resolution and N
        can be zero. The prompt is re-run from the cached image features, so
        sweeping several concepts over one frame pays the backbone only once.
        """
        t0 = time.perf_counter()
        self.set_image(image)
        assert self._state is not None
        self._proc.reset_all_prompts(self._state)
        self._proc.confidence_threshold = float(threshold)
        with self._autocast():
            state = self._proc.set_text_prompt(text, self._state)
        masks = state.get("masks")
        scores = state.get("scores")
        boxes = state.get("boxes")
        hw = image.shape[:2]
        return {
            "masks": (masks.squeeze(1).cpu().numpy().astype(bool)
                      if masks is not None and masks.numel()
                      else np.zeros((0, *hw), dtype=bool)),
            "scores": (scores.float().cpu().numpy()
                       if scores is not None and scores.numel()
                       else np.zeros((0,), dtype=np.float32)),
            "boxes": (boxes.float().cpu().numpy()
                      if boxes is not None and boxes.numel()
                      else np.zeros((0, 4), dtype=np.float32)),
            "ms": (time.perf_counter() - t0) * 1000.0,
        }

    # -- internals ----------------------------------------------------------
    def _to_chw(self, img: np.ndarray):
        """HxWx3 uint8 RGB -> the CxHxW uint8 tensor the processor expects.

        ``Sam3Processor.set_image`` reads the size off ``shape[-2:]`` for array
        inputs, so a plain HxWx3 array would be read as ``(W, 3)``.
        """
        return self._torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def _autocast(self):
        if self._use_autocast and self.device.startswith("cuda"):
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()
