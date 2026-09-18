"""Read back what :mod:`experiments.sam_compare.prepare_sample` wrote."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from experiments.sam_compare import _env
from experiments.sam_compare.common import unpack_mask


@dataclass
class Ref:
    """One human reference mask plus the geometry the protocol needs."""

    index: int
    desktop: int
    step: int
    label: str
    bbox: tuple[int, int, int, int]
    area: int
    point: tuple[int, int]
    frame_hw: tuple[int, int]
    _packed: np.ndarray

    @property
    def mask(self) -> np.ndarray:
        """Rehydrated full-frame boolean mask."""
        return unpack_mask(self._packed, self.bbox, self.frame_hw)

    @property
    def frame(self) -> tuple[int, int]:
        return self.desktop, self.step


def load_refs(indices: Optional[list[int]] = None) -> list[Ref]:
    """Load reference masks; ``indices=None`` loads all of them."""
    path = _env.OUT_ROOT / "refs.npz"
    with np.load(path, allow_pickle=True) as data:
        meta = json.loads(str(data["meta"]))
        want = range(len(meta)) if indices is None else indices
        out = []
        for i in want:
            m = meta[i]
            out.append(
                Ref(
                    index=i,
                    desktop=m["desktop"],
                    step=m["step"],
                    label=m["label"],
                    bbox=tuple(m["bbox"]),
                    area=m["area"],
                    point=tuple(m["point"]),
                    frame_hw=tuple(m["frame_hw"]),
                    _packed=data[f"m{i}"],
                )
            )
    return out


def load_sample() -> list[int]:
    """The stratified evaluation sample's reference indices."""
    path = _env.OUT_ROOT / "sample.json"
    return list(json.loads(path.read_text(encoding="utf-8"))["indices"])
