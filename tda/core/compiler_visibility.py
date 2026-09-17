"""Derived fields of the layer compiler: the ``visibility`` label and the input hash.

Split out of :mod:`tda.core.compiler` to keep both files small; both public
names are re-exported there, which is where callers should import them from.

Thresholds come straight from spec 3.3 step 7 / 6.2:

* occlusion ratio ``< 0.3`` -> ``visible``, ``0.3 .. 0.95`` ->
  ``occluded_partial``, ``>= 0.95`` -> ``occluded_full``;
* the shorter side of the *visible* bounding box ``< 6 px`` -> ``too_small``,
  ``< 12 px`` -> ``visible_tiny``.

``occluded_full`` wins over both size rules (a handful of surviving pixels of
a 95 %-occluded part is still a fully occluded part), and the size rules win
over ``visible`` / ``occluded_partial``, because a five-pixel sliver cannot be
annotated whatever its occlusion ratio says.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np

from tda.core import masks
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    Similarity,
    Visibility,
    ZOrderRec,
)

__all__ = [
    "OCCLUDED_FULL_RATIO",
    "OCCLUDED_PARTIAL_RATIO",
    "TOO_SMALL_PX",
    "VISIBLE_TINY_PX",
    "derive_visibility",
    "input_hash",
]

OCCLUDED_PARTIAL_RATIO = 0.3
OCCLUDED_FULL_RATIO = 0.95
TOO_SMALL_PX = 6
VISIBLE_TINY_PX = 12

#: how many decimals of the transform enter the hash
HASH_DECIMALS = 6


# --------------------------------------------------------------------------- #
# visibility
# --------------------------------------------------------------------------- #
def derive_visibility(
    visible: Optional[np.ndarray],
    amodal: Optional[np.ndarray],
    occlusion_ratio: float,
) -> str:
    """The automatic ``visibility`` label for one compiled instance.

    ``visible`` is the visible mask (``None`` when the instance has no
    geometry in this frame at all, which is reported as ``out_of_view``),
    ``amodal`` its full shape -- kept in the signature because the truth table
    passes both, even though the ratio already carries their relation -- and
    ``occlusion_ratio`` is ``1 - |V| / |A|``.

    A ``FrameOverride.visibility`` is *not* considered here: the caller applies
    it on top, because a manual label always wins (spec 3.3 step 7).
    """
    if visible is None:
        return Visibility.OUT_OF_VIEW.value
    if float(occlusion_ratio) >= OCCLUDED_FULL_RATIO:
        return Visibility.OCCLUDED_FULL.value
    side = masks.min_side(visible)
    if side < TOO_SMALL_PX:
        return Visibility.TOO_SMALL.value
    if side < VISIBLE_TINY_PX:
        return Visibility.VISIBLE_TINY.value
    if float(occlusion_ratio) >= OCCLUDED_PARTIAL_RATIO:
        return Visibility.OCCLUDED_PARTIAL.value
    return Visibility.VISIBLE.value


# --------------------------------------------------------------------------- #
# input hash
# --------------------------------------------------------------------------- #
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _counts(rle: Optional[dict]) -> Optional[str]:
    """The ``counts`` string of an RLE, decoded when it arrives as bytes."""
    if not rle:
        return None
    counts = rle.get("counts")
    if isinstance(counts, bytes):
        return counts.decode("ascii")
    return None if counts is None else str(counts)


def _rle_digest(rle: Optional[dict]) -> Optional[str]:
    """sha1 of an RLE's counts string -- short, stable and JSON-safe."""
    counts = _counts(rle)
    return None if counts is None else _sha1(counts)


def _keyframe_token(kf: ShapeKeyframe) -> list:
    """What identifies a keyframe in the hash: ``[id, version, content]``.

    The content digest covers the parts (name, sha1 of the RLE counts, box) and
    the fields that decide where the shape applies, so an in-place edit that
    forgets to bump ``version`` -- or a draft that has no ``id`` at all -- still
    changes the hash. ``id`` and ``version`` are kept alongside it because they
    are what the rest of the system versions a shape by.
    """
    parts = [
        [part.name, _rle_digest(part.rle), None if part.box is None else list(part.box)]
        for part in kf.parts
    ]
    content = _sha1(
        json.dumps(
            [
                kf.anchor_step,
                kf.geom_type,
                kf.placement,
                kf.pose_segment,
                sorted(parts, key=str),
            ],
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return [None if kf.id is None else int(kf.id), int(kf.version), content]


def _round(value: float) -> float:
    """Round for the hash, normalising ``-0.0`` so the JSON text is stable."""
    out = round(float(value), HASH_DECIMALS)
    return 0.0 if out == 0 else out


def input_hash(
    *,
    key: FrameKey,
    hw: tuple[int, int],
    needs: dict[str, str],
    placements: dict[str, str],
    selected: dict[str, Optional[ShapeKeyframe]],
    zorder: ZOrderRec,
    layer_order: dict[str, list[tuple[str, str]]],
    overrides: list[PairOverride],
    occluders: list[OccluderMask],
    frame_overrides: dict[str, FrameOverride],
    transform: Similarity,
    pose_segment: Optional[int],
    compiler_version: str,
) -> str:
    """sha1 over a canonical JSON of every input the compilation depends on.

    Contents, all of them sorted so dict and list order cannot leak in:

    * the frame key and canvas size, the ``pose_segment`` and the
      ``compiler_version``;
    * ``needs`` and the resolved ``placements`` as sorted items;
    * per instance the *selected* keyframe's ``[id, version, content digest]``
      (see :func:`_keyframe_token`) -- an edit elsewhere in the chain cannot
      disturb this frame, but an in-place edit of *this* shape always shows;
    * the z-order's ``version`` **and** the layer order actually painted, per
      group, so reordering the rows without bumping the version still shows;
    * the ``(above, below)`` pairs of the overrides, deduplicated;
    * for every occluder of *this* frame its type and the sha1 of its RLE
      counts; occluders of other frames are not inputs at all;
    * per frame override the instance, its forced ``visibility`` and the sha1 of
      its ``visible_rle`` counts;
    * the four :class:`~tda.core.model.Similarity` fields rounded to 6 decimals.

    Same inputs -> same hash; any change to one of them -> a different hash.
    """
    payload = {
        "compiler_version": str(compiler_version),
        "key": [int(key.desktop), int(key.step), str(key.view)],
        "hw": [int(hw[0]), int(hw[1])],
        "pose_segment": None if pose_segment is None else int(pose_segment),
        "needs": sorted((str(k), str(v)) for k, v in needs.items()),
        "placements": sorted((str(k), str(v)) for k, v in placements.items()),
        "keyframes": sorted(
            [inst, None if kf is None else _keyframe_token(kf)]
            for inst, kf in selected.items()
        ),
        "zorder_version": int(zorder.version),
        "layer_order": sorted(
            [group, [[layer[0], layer[1]] for layer in layers]]
            for group, layers in layer_order.items()
        ),
        "overrides": sorted({(str(o.above), str(o.below)) for o in overrides}),
        "occluders": sorted(
            [str(o.occluder_type), _rle_digest(o.rle) or ""] for o in occluders
        ),
        "frame_overrides": sorted(
            [
                str(inst),
                None if fo.visibility is None else str(fo.visibility),
                _rle_digest(fo.visible_rle),
            ]
            for inst, fo in frame_overrides.items()
        ),
        "transform": [
            _round(transform.scale),
            _round(transform.theta),
            _round(transform.tx),
            _round(transform.ty),
        ],
    }
    return _sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
