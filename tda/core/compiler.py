"""The layer compiler -- steps 3-7 of spec 3.3, as one pure function.

:func:`compile_frame` turns the annotated inputs of one frame (shape keyframes,
a z-order with pairwise overrides, frame-level occluders and per-frame
overrides, plus the frame's registration transform) into every instance's
visible mask, occlusion ratio and ``visibility`` label, together with an
``input_hash`` that changes whenever any of those inputs changes.

The module is pure: no I/O, no database, no Qt, no globals, and it never
mutates its arguments -- the same inputs always produce the same output.

How the layering works
----------------------
Each ``(instance, part)`` pair is one *layer*. Layers are grouped by
placement: a part lying on the bench never occludes a part inside the chassis
and vice versa (spec 3.3 step 5), so each group is composited on its own.
Within a group the layers are sorted bottom-to-top and then painted **top-down
into a single claim canvas**: every pixel belongs to the first (topmost) layer
that covers it, which yields all visible masks in one pass over the pixels
instead of a pairwise subtraction per pair of layers. An instance's visible
mask is the union of its parts' claims, its amodal mask the union of its parts.

The order itself is :mod:`tda.core.compiler_layers`' job -- the global z-order
with the pairwise overrides as hard constraints -- and the ``visibility``
label and the ``input_hash`` are :mod:`tda.core.compiler_visibility`'s. Both
modules exist to keep this one readable; :func:`above` and
:func:`derive_visibility` are re-exported here.

Windows: every pixel touched is one a shape could reach
-------------------------------------------------------
The masks this produces are full-frame, in native image coordinates (spec 2.4),
and so is every mask it reads. What it does **not** do is compute them
full-frame. A part covers its own bounding box and nothing else, and that box
is read off the stored RLE's run lengths (:func:`tda.core.masks.rle_bbox_xywh`)
without decoding anything, so the compositing, the occluder subtraction, the
areas, the bounding boxes and the visibility ladder all run on a slice of the
canvas rather than on the canvas.

On a 4032x3040 OAK frame with forty instances that is the difference between
2.9 s and a tenth of it: the old pass did about five full-frame boolean
operations per layer -- ``mask & ~claimed`` alone allocated two 12 MB
temporaries -- plus a 12 MB copy of every decoded part.

Three things follow and are not negotiable, since the ``input_hash`` and the
stored run lengths must not move by one byte
(``tests/test_compiler_golden.py``):

* a window is rounded **outwards** and may be larger than the shape; it may
  never be smaller. A warped part is padded by :data:`WARP_SLACK` because
  nearest-neighbour resampling can land a pixel just outside the transformed
  box, and a part rasterised from a bare rectangle is padded because
  ``fillPoly`` rounds its corners.
* an **empty** part is still a layer. It takes part in the z-order and
  therefore in the hash, exactly as it did when it was a canvas of zeros.
* the canvases are allocated in **Fortran order**, which is what
  :func:`tda.core.masks.encode_rle` needs to store them without transposing,
  and they are ``np.zeros``, which costs nothing until a page is written.

Box-only geometry (``geom_type == "box"``, typically a part lying on the bench)
carries no mask: it is not a layer, it never occludes anything and is never
occluded, and it is reported with ``visible=None``, ``amodal=None``,
``occlusion_ratio=0.0`` and ``visibility="visible"``.

Problems reported in :attr:`CompiledFrame.problems`
---------------------------------------------------
* ``missing_shape:<instance>`` -- an instance inside the chassis needs geometry
  but has no keyframe for this step; this blocks confirming the frame.
* ``bench_missing:<instance>`` -- the same for an instance on the bench, which
  only makes the frame's ``bench_annotated`` false and must **not** block.
* ``zorder_missing:<instance>/<part>`` -- that layer is absent from the z-order;
  it is treated as being above everything else in its group.
* ``zorder_cycle:<a>,<b>`` -- contradictory pairwise overrides.
* ``empty_visible:<instance>`` -- a mask instance whose visible mask came out
  empty and that has no frame override to explain it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tda.core import masks
from tda.core.compiler_layers import LayerKey, above, group_order
from tda.core.compiler_visibility import derive_visibility, input_hash, visibility_for
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    Placement,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    Visibility,
    ZOrderRec,
)

__all__ = [
    "CompiledFrame",
    "CompiledInstance",
    "above",
    "compile_frame",
    "derive_visibility",
    "placements_for",
    "select_keyframe",
]

GEOM_MASK = "mask"
GEOM_BOX = "box"
IN_CHASSIS = Placement.IN_CHASSIS.value
ON_BENCH = Placement.ON_BENCH.value
OUT_OF_VIEW = Visibility.OUT_OF_VIEW.value

_MISSING = "missing"

#: ``(x0, y0, x1, y1)``, upper bounds exclusive, already clipped to the canvas.
Window = tuple[int, int, int, int]

#: Pixels a window is grown by when the shape inside it was resampled rather
#: than read off run lengths: a nearest-neighbour warp, or a rectangle
#: rasterised by ``fillPoly``, can set a pixel just outside the exact box.
WARP_SLACK = 2

#: The transform that leaves a mask where it is -- what an occluder RLE and a
#: frame override are already in, so their windows are their own run lengths'.
IDENTITY = Similarity()


# --------------------------------------------------------------------------- #
# output types
# --------------------------------------------------------------------------- #
@dataclass
class CompiledInstance:
    """One instance's geometry in one frame (a row of the truth table, 3.4).

    ``visible`` and ``amodal`` are ``(H, W)`` bool masks in frame coordinates.
    ``visible`` is ``None`` when the instance has no mask geometry in this
    frame: box-only geometry, a missing shape, or a frame override that put it
    out of view. ``amodal`` is ``None`` for anything but mask geometry.

    ``box`` is ``(x0, y0, x1, y1)`` with the upper bounds exclusive: the
    integer bounding box of the *visible* mask for mask geometry, and the
    transformed part box (floats) for box-only geometry.
    """

    instance: str
    visible: Optional[np.ndarray]
    box: Optional[tuple]
    amodal: Optional[np.ndarray]
    occlusion_ratio: float
    visibility: str
    placement: str
    keyframe_id: Optional[int]
    #: ``(x0, y0, x1, y1)`` outside which ``visible`` is known to be empty, or
    #: ``None`` when nothing is known. It is not geometry -- ``box`` is the
    #: tight bounding box and this may be looser -- it is what lets a caller
    #: encode or repaint this instance without walking the whole canvas
    #: (:func:`tda.core.masks.encode_rle`'s ``window``).
    window: Optional[Window] = None


@dataclass
class CompiledFrame:
    """Everything the compiler derives for one frame.

    ``painted`` is the order the layers were actually composited in, as
    ``placement -> instance keys, bottom-up``: the global z-order *after* the
    pairwise overrides have been applied, with each placement group on its own
    (spec 3.3 step 5). It is what the instance list and the canvas overlay have
    to follow, since anything else would show an order the pixels contradict.
    Instances with no mask layer -- a bench box, a missing shape -- appear in no
    group.
    """

    key: FrameKey
    instances: dict[str, CompiledInstance]
    problems: list[str]
    input_hash: str
    painted: dict[str, list[str]] = field(default_factory=dict)
    #: The same order as the ``(instance, part)`` layer keys it was computed
    #: from -- ``painted`` collapses a multi-part instance to one entry, and
    #: that collapse cannot be undone. It is one of the ingredients of
    #: :attr:`input_hash`, and the only one a reader cannot otherwise recover,
    #: so a caller with a gathered input set can ask whether this compilation
    #: is still the one they make (:func:`tda.core.truth_fresh.hash_of_inputs`)
    #: instead of compiling to find out.
    layers: dict[str, list[tuple[str, str]]] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# keyframe selection
# --------------------------------------------------------------------------- #
def select_keyframe(kfs: list[ShapeKeyframe], step: int) -> Optional[ShapeKeyframe]:
    """The keyframe that applies at ``step``, or ``None`` when there is none.

    ``anchor_step`` is the *latest* logical step a shape applies to, so keyframe
    ``j`` covers ``(previous anchor, anchor_j]`` and the one to use is the
    smallest ``anchor_step >= step`` (spec 3.3 step 3).

    ``kfs`` must already be narrowed to one ``(instance, view, pose_segment)``
    -- and to one placement chain, since the ``in_chassis`` and ``on_bench``
    chains of an instance are independent. :func:`compile_frame` does that
    filtering. Among keyframes sharing an ``anchor_step`` the newest wins
    (highest ``version``, then highest ``id``), so a re-traced shape replaces
    the one it was drawn over.
    """
    best: Optional[ShapeKeyframe] = None
    best_rank: Optional[tuple] = None
    for kf in kfs:
        if kf.anchor_step < step:
            continue
        rank = (kf.anchor_step, -int(kf.version), -(kf.id if kf.id is not None else -1))
        if best_rank is None or rank < best_rank:
            best, best_rank = kf, rank
    return best


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _wrong_size(rle: Optional[dict], hw: tuple[int, int]) -> bool:
    """Is this RLE's own ``size`` something other than the frame's ``hw``?

    Every mask the compiler is handed -- a part, an occluder, an override --
    lives in the coordinates of the frame it belongs to, so a different size is
    an error in the inputs rather than something to pad or scale into place:
    there is no way to know *where* on the frame the smaller mask was meant to
    sit. The caller reports it and drops that mask. Read from the RLE dict, so
    nothing has to be decoded to find out.
    """
    if not rle:
        return False
    size = rle.get("size")
    if size is None:
        return False
    return (int(size[0]), int(size[1])) != hw


def _box_corners(box, transform: Similarity) -> list[float]:
    """The four corners of ``box`` after ``transform``, as a flat polygon."""
    x0, y0, x1, y1 = (float(v) for v in box)
    cos_t = float(transform.scale) * float(np.cos(transform.theta))
    sin_t = float(transform.scale) * float(np.sin(transform.theta))
    out: list[float] = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        out.append(cos_t * x - sin_t * y + float(transform.tx))
        out.append(sin_t * x + cos_t * y + float(transform.ty))
    return out


def _transform_box(box, transform: Similarity) -> tuple[float, float, float, float]:
    """``box`` after ``transform``, as the axis-aligned envelope of its corners."""
    if transform.is_identity():
        return tuple(float(v) for v in box)  # type: ignore[return-value]
    flat = _box_corners(box, transform)
    xs, ys = flat[0::2], flat[1::2]
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #
def _clip_window(box, hw: tuple[int, int]) -> Window:
    """``box`` rounded **outwards** to whole pixels and clipped to the canvas."""
    height, width = int(hw[0]), int(hw[1])
    x0 = int(min(max(np.floor(box[0]), 0), width))
    y0 = int(min(max(np.floor(box[1]), 0), height))
    x1 = int(min(max(np.ceil(box[2]), x0), width))
    y1 = int(min(max(np.ceil(box[3]), y0), height))
    return (x0, y0, x1, y1)


def _empty_window(window: Optional[Window]) -> bool:
    return window is None or window[2] <= window[0] or window[3] <= window[1]


def _union_window(a: Optional[Window], b: Optional[Window]) -> Optional[Window]:
    if _empty_window(a):
        return None if _empty_window(b) else b
    if _empty_window(b):
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _intersect_window(a: Window, b: Window) -> Optional[Window]:
    found = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    return None if _empty_window(found) else found


def _view(canvas: np.ndarray, window: Window) -> np.ndarray:
    """The part of a full-canvas array ``window`` covers (no copy)."""
    return canvas[window[1]:window[3], window[0]:window[2]]


def _merge_layer(
    mask_a: np.ndarray, window_a: Window, mask_b: np.ndarray, window_b: Window
) -> tuple[np.ndarray, Window]:
    """Two parts sharing one name, unioned into one window (spec 3.3 step 5).

    A keyframe may list the same part name twice; that has always been *one*
    layer covering both, and the z-order lists it once.
    """
    if _empty_window(window_b):
        return mask_a, window_a
    if _empty_window(window_a):
        return mask_b, window_b
    window = _union_window(window_a, window_b)
    out = np.zeros((window[3] - window[1], window[2] - window[0]),
                   dtype=bool, order="F")
    for mask, box in ((mask_a, window_a), (mask_b, window_b)):
        out[box[1] - window[1]:box[3] - window[1],
            box[0] - window[0]:box[2] - window[0]] |= mask
    return out, window


def _rle_window(rle: dict, transform: Similarity, hw: tuple[int, int]) -> Window:
    """Where this RLE's pixels are, straight off the run lengths.

    ``toBbox`` reads the encoding, so finding out costs no decode at all -- and
    on a 12 MP canvas measuring it afterwards with :func:`tda.core.masks.bbox`
    would cost more than the compositing the window exists to shrink.
    """
    x, y, width, height = masks.rle_bbox_xywh(rle)
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    box = (x, y, x + width, y + height)
    if transform.is_identity():
        return _clip_window(box, hw)
    warped = _transform_box(box, transform)
    return _clip_window(
        (warped[0] - WARP_SLACK, warped[1] - WARP_SLACK,
         warped[2] + WARP_SLACK, warped[3] + WARP_SLACK), hw
    )


def _part_window(
    part: ShapePart, transform: Similarity, hw: tuple[int, int]
) -> Optional[Window]:
    """A window guaranteed to contain this part's mask, or ``None`` for no mask.

    Always a superset: a window that is too large only costs time, one that is
    too small silently truncates a stored annotation.
    """
    if part.rle is not None:
        return _rle_window(part.rle, transform, hw)
    if part.box is not None:
        box = _transform_box(part.box, transform)
        return _clip_window(
            (box[0] - WARP_SLACK, box[1] - WARP_SLACK,
             box[2] + WARP_SLACK, box[3] + WARP_SLACK), hw
        )
    return None


def _part_mask(
    part: ShapePart, transform: Similarity, hw: tuple[int, int],
    window: Optional[Window] = None,
) -> Optional[np.ndarray]:
    """One part's amodal mask, ``None`` when it has none.

    With ``window`` the array returned covers **only that window** -- which is
    what :func:`compile_frame` composites with -- and without it the whole
    frame, which is what a caller outside this module means by "the part's
    mask". Both are Fortran-ordered, so storing one costs no transpose.

    An RLE is decoded and warped (identity is a fast path: no warp at all); a
    part that carries only a box is rasterised from its transformed corners,
    translated into the window, which is exact because the translation is by
    whole pixels. An RLE whose own size is not ``hw`` yields ``None`` -- see
    :func:`_wrong_size`; the mask loop reports that case before asking.
    """
    if part.rle is not None:
        if _wrong_size(part.rle, hw):
            return None
        if transform.is_identity():
            # shared and read-only: the same forty keyframes are decoded on
            # every commit, every Space and every step, and at 12 MP that was
            # 0.4 s of each of those gestures
            cropped, box = masks.decode_rle_shared(part.rle)
            if window is not None and tuple(window) == tuple(box):
                return cropped
            mask = np.zeros(hw, dtype=bool, order="F")
            if not _empty_window(box):
                _view(mask, box)[...] = cropped
        else:
            mask = np.asfortranarray(
                masks.warp_mask(masks.decode_rle(part.rle), transform, hw)
            )
        if window is None:
            return mask
        return np.asfortranarray(_view(mask, window))
    if part.box is not None:
        flat = _box_corners(part.box, transform)
        if window is None:
            return np.asfortranarray(masks.polygons_to_mask([flat], hw))
        x0, y0, x1, y1 = window
        if _empty_window(window):
            # the rectangle is entirely off the canvas: an empty layer, which
            # is still a layer (it takes part in the z-order and the hash)
            return np.zeros((y1 - y0, x1 - x0), dtype=bool, order="F")
        moved = [v - (x0 if index % 2 == 0 else y0) for index, v in enumerate(flat)]
        return np.asfortranarray(
            masks.polygons_to_mask([moved], (y1 - y0, x1 - x0))
        )
    return None


def _keyframe_box(
    kf: ShapeKeyframe, transform: Similarity, hw: tuple[int, int]
) -> Optional[tuple]:
    """The box of a box-only keyframe: the envelope of its parts' boxes."""
    boxes: list[tuple[float, float, float, float]] = []
    for part in kf.parts:
        if part.box is not None:
            boxes.append(_transform_box(part.box, transform))
            continue
        window = _part_window(part, transform, hw)
        mask = None if window is None else _part_mask(part, transform, hw, window)
        if mask is None:
            continue
        found = masks.bbox(mask)
        if found is not None:
            boxes.append((float(found[0] + window[0]), float(found[1] + window[1]),
                          float(found[2] + window[0]), float(found[3] + window[1])))
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _occlusion_ratio(visible_area: int, amodal_area: int) -> float:
    """``1 - |V| / |A|``, clamped to ``[0, 1]`` (0 when the amodal mask is empty).

    Takes the two pixel counts rather than the two masks: both are measured
    inside the instance's own window, which on a 12 MP canvas is the difference
    between two full-frame scans per instance and two small ones.
    """
    if amodal_area == 0:
        return 0.0
    ratio = 1.0 - visible_area / amodal_area
    return float(min(1.0, max(0.0, ratio)))


def _window_bbox(canvas: np.ndarray, window: Optional[Window]) -> Optional[tuple]:
    """:func:`tda.core.masks.bbox` of what a window holds, in frame coordinates."""
    if _empty_window(window):
        return None
    found = masks.bbox(_view(canvas, window))
    if found is None:
        return None
    return (found[0] + window[0], found[1] + window[1],
            found[2] + window[0], found[3] + window[1])


def _window_area(canvas: np.ndarray, window: Optional[Window]) -> int:
    if _empty_window(window):
        return 0
    return masks.area(_view(canvas, window))


def _label_for(box: Optional[tuple], ratio: float, *, present: bool = True) -> str:
    """:func:`derive_visibility`, asked of a box instead of of a mask.

    The ladder only ever looked at the shorter side of the visible mask's
    bounding box and at the occlusion ratio (spec 3.3 step 7), and the box has
    already been measured by then. ``present=False`` is the "no geometry at
    all" case, which is ``out_of_view``.
    """
    if not present:
        return visibility_for(None, ratio)
    side = 0 if box is None else int(min(box[2] - box[0], box[3] - box[1]))
    return visibility_for(side, ratio)



def _instance_order(layers: list[LayerKey]) -> list[str]:
    """The instance keys of a painted layer list, bottom-up, each one once.

    A multi-part instance contributes several layers; it takes the position of
    its bottom-most one, which is where the instance as a whole starts covering
    anything.
    """
    out: list[str] = []
    for instance, _part in layers:
        if instance not in out:
            out.append(instance)
    return out


# --------------------------------------------------------------------------- #
# the compiler
# --------------------------------------------------------------------------- #
def _placement_of(
    instance: str, needs: dict[str, str], placements: Optional[dict[str, str]]
) -> str:
    """This instance's placement in this frame.

    ``placements`` is authoritative when it covers the instance. Without it the
    instance is treated as ``in_chassis``, except when ``needs`` asks for
    ``"box"`` geometry, which only ever comes from a bench instance (see
    :func:`tda.core.states.needs_geom`).
    """
    if placements is not None and instance in placements:
        return placements[instance]
    return ON_BENCH if needs.get(instance) == GEOM_BOX else IN_CHASSIS


def placements_for(
    needs: dict[str, str], placements: Optional[dict[str, str]]
) -> dict[str, str]:
    """Where each instance **this frame needs geometry for** is (spec 3.3 step 2).

    Narrowed to ``needs``, and that narrowing is the point of the function
    existing at all: :func:`tda.core.truth_inputs.gather` deliberately hands
    over a superset -- an instance that needs nothing here still has a
    placement, and the compiler's problem list and the bench flag both want to
    know it -- while the frame's :func:`input_hash` is taken over *this* dict.
    A second copy of the narrowing drifted from this one immediately: it read
    the superset, so the hash it computed differed on every frame holding an
    instance with nothing to draw, which on a real sheet is most of them.
    """
    return {inst: _placement_of(inst, needs, placements) for inst in sorted(needs)}


def compile_frame(
    key: FrameKey,
    hw: tuple[int, int],
    needs: dict[str, str],
    keyframes: dict[str, list[ShapeKeyframe]],
    zorder: ZOrderRec,
    overrides: list[PairOverride],
    occluders: list[OccluderMask],
    frame_overrides: dict[str, FrameOverride],
    transform: Similarity,
    compiler_version: str = "1",
    *,
    placements: Optional[dict[str, str]] = None,
    pose_segment: Optional[int] = None,
    bench_roi: Optional[list] = None,
) -> CompiledFrame:
    """Compile one frame: steps 3-7 of spec 3.3.

    Parameters
    ----------
    key:
        The frame being compiled; ``key.step`` selects the keyframes and
        ``key.view`` filters them.
    hw:
        ``(H, W)`` of the frame, i.e. of every mask produced.
    needs:
        ``instance -> "mask" | "box"`` from :func:`tda.core.states.needs_geom`.
        Its **keys** are exactly the instances compiled; its values only
        provide the default placement, because the geometry actually produced
        is the one the selected keyframe describes (``geom_type``).
    keyframes:
        ``instance -> keyframes``, which may hold other views, both placement
        chains and several pose segments; they are filtered by ``key.desktop``,
        ``key.view``, the instance's placement and ``pose_segment`` before
        :func:`select_keyframe` runs.
    zorder, overrides:
        The global order and the pairwise overrides of this
        ``(view, pose_segment)``.
    occluders:
        Frame-level occluder masks; those of other frames are ignored. Their
        RLEs are already in frame coordinates, so ``transform`` is not applied
        to them.
    frame_overrides:
        ``instance -> FrameOverride`` for this frame. ``visible_rle`` replaces
        the compiled visible mask before the ``visibility`` label is derived,
        and ``visibility`` always wins over the derived label.
    transform:
        The registration transform from the pose segment's reference frame to
        this frame. Identity is a fast path.
    compiler_version:
        Bumped when the compiler's own semantics change; enters the hash, so
        every row recompiles.
    placements:
        ``instance -> placement`` at this step, which selects the keyframe
        chain and the occlusion group. ``None`` (the default) means
        ``in_chassis`` for every instance except the ones ``needs`` marks as
        ``"box"``.
    bench_roi:
        The staging area this view can see, or ``None``. A part lying on the
        bench is only reported as ``bench_missing`` when there is one: spec 4.2
        asks for a bench box 若该视角有堆放区 ROI, and the scanner never sees one.
    pose_segment:
        The pose segment this frame belongs to. Shapes of two segments are
        drawn in different reference frames and are not comparable, so when
        this is left ``None`` and an instance's chain spans more than one
        segment, that is reported as ``pose_segment_ambiguous:<instance>`` and
        the highest segment is used.

    Notes
    -----
    Bench and chassis instances are composited separately, so they never
    occlude each other however the z-order lists them -- but two bench
    instances do occlude each other, and the frame's occluders are subtracted
    from every instance's visible mask, bench included (spec 3.3 step 6).
    A ``visibility`` of ``out_of_view`` -- from an override, or from a shape
    that is missing altogether -- leaves neither a visible mask nor a box
    behind.
    """
    canvas = (int(hw[0]), int(hw[1]))
    problems: list[str] = []
    instances = sorted(needs)
    placement_of = placements_for(needs, placements)

    # --- step 3: pick the keyframe of each instance ------------------------ #
    selected: dict[str, Optional[ShapeKeyframe]] = {}
    for inst in instances:
        chain = [
            kf
            for kf in keyframes.get(inst, ())
            if kf.view == key.view
            and kf.desktop == key.desktop
            and kf.placement == placement_of[inst]
        ]
        if pose_segment is not None:
            chain = [kf for kf in chain if kf.pose_segment == pose_segment]
        elif len({kf.pose_segment for kf in chain}) > 1:
            # shapes of two pose segments are not comparable: the caller must
            # say which one this frame is in. Newest segment, and a problem.
            problems.append(f"pose_segment_ambiguous:{inst}")
            newest = max(kf.pose_segment for kf in chain)
            chain = [kf for kf in chain if kf.pose_segment == newest]
        selected[inst] = select_keyframe(chain, key.step)

    # --- step 4: transform the shapes into frame coordinates --------------- #
    kinds: dict[str, str] = {}
    layer_masks: dict[LayerKey, np.ndarray] = {}
    layer_windows: dict[LayerKey, Window] = {}
    groups: dict[str, list[LayerKey]] = {}
    amodals: dict[str, np.ndarray] = {}
    #: Where each instance's pixels can be. Everything measured about the
    #: instance afterwards -- its visible mask, its areas, its box -- is
    #: measured inside this and nowhere else.
    windows: dict[str, Optional[Window]] = {}
    boxes: dict[str, Optional[tuple]] = {}
    for inst in instances:
        kf = selected[inst]
        if kf is None:
            kinds[inst] = _MISSING
            if placement_of[inst] != ON_BENCH:
                problems.append(f"missing_shape:{inst}")
            elif bench_roi is not None:
                # a part on the bench is only annotated -- and only missing --
                # where the view has a staging area to see it in (spec 4.2)
                problems.append(f"bench_missing:{inst}")
            continue
        if kf.geom_type == GEOM_BOX:
            kinds[inst] = GEOM_BOX
            boxes[inst] = _keyframe_box(kf, transform, canvas)
            continue
        kinds[inst] = GEOM_MASK
        amodal = np.zeros(canvas, dtype=bool, order="F")
        window: Optional[Window] = None
        for part in kf.parts:
            if _wrong_size(part.rle, canvas):
                problems.append(f"shape_size_mismatch:{inst}/{part.name}")
                continue
            here = _part_window(part, transform, canvas)
            if here is None:
                continue
            mask = _part_mask(part, transform, canvas, here)
            if mask is None:
                continue
            layer = (inst, part.name)
            if layer in layer_masks:  # a repeated part name is one layer
                layer_masks[layer], layer_windows[layer] = _merge_layer(
                    layer_masks[layer], layer_windows[layer], mask, here
                )
            else:
                layer_masks[layer] = mask
                layer_windows[layer] = here
                groups.setdefault(placement_of[inst], []).append(layer)
            _view(amodal, here)[...] |= mask
            window = _union_window(window, here)
        amodals[inst] = amodal
        windows[inst] = window

    # --- step 5: paint each group top-down --------------------------------- #
    visibles = {inst: np.zeros(canvas, dtype=bool, order="F") for inst in amodals}
    painted: dict[str, list[LayerKey]] = {}
    for group in sorted(groups):
        bottom_up, group_problems = group_order(groups[group], zorder, overrides)
        problems.extend(group_problems)
        painted[group] = bottom_up
        claimed = np.zeros(canvas, dtype=bool, order="F")
        for layer in reversed(bottom_up):  # topmost layer claims first
            here = layer_windows[layer]
            if _empty_window(here):
                continue
            mask = layer_masks[layer]
            taken = _view(claimed, here)
            _view(visibles[layer[0]], here)[...] |= mask & ~taken
            taken |= mask

    # --- step 6: occluders ------------------------------------------------- #
    frame_occluders = [occ for occ in occluders if occ.frame == key]
    occluded = None
    occluded_window: Optional[Window] = None
    for occ in frame_occluders:
        if _wrong_size(occ.rle, canvas):
            problems.append(f"shape_size_mismatch:occluder/{occ.occluder_type}")
            continue
        # occluder RLEs are already in frame coordinates (see the docstring),
        # so the window is the run lengths' own box
        cropped, here = masks.decode_rle_shared(occ.rle)
        if occluded is None:
            occluded = np.zeros(canvas, dtype=bool, order="F")
        if not _empty_window(here):
            _view(occluded, here)[...] |= cropped
        occluded_window = _union_window(occluded_window, here)
    if occluded is not None and occluded_window is not None:
        for inst, visible in visibles.items():
            overlap = (None if _empty_window(windows.get(inst))
                       else _intersect_window(windows[inst], occluded_window))
            if overlap is None:
                continue
            _view(visible, overlap)[...] &= ~_view(occluded, overlap)

    # --- step 7: overrides, ratios, labels --------------------------------- #
    compiled: dict[str, CompiledInstance] = {}
    for inst in instances:
        kf = selected[inst]
        kf_id = None if kf is None else kf.id
        placement = placement_of[inst]
        override = frame_overrides.get(inst)
        forced = override.visibility if override is not None else None
        kind = kinds[inst]

        # a "just this frame" visible mask, shared by all three branches
        patch = None
        patch_window: Optional[Window] = None
        if override is not None and override.visible_rle is not None:
            if _wrong_size(override.visible_rle, canvas):
                problems.append(f"shape_size_mismatch:{inst}/override")
            else:
                patch = masks.decode_rle(override.visible_rle)
                patch_window = _rle_window(override.visible_rle, IDENTITY, canvas)

        if kind == _MISSING:
            # no shape to composite, but an override may still say what is
            # visible here; the missing keyframe stays reported either way
            patch_box = (None if patch is None
                         else _window_bbox(patch, patch_window))
            label = forced or _label_for(patch_box, 0.0, present=patch is not None)
            visible = None if label == OUT_OF_VIEW else patch
            compiled[inst] = CompiledInstance(
                instance=inst,
                visible=visible,
                box=None if visible is None else patch_box,
                amodal=None,
                occlusion_ratio=0.0,
                visibility=label,
                placement=placement,
                keyframe_id=kf_id,
                window=None if visible is None else patch_window,
            )
            continue

        if kind == GEOM_BOX:
            visible = patch
            patch_box = (None if patch is None
                         else _window_bbox(patch, patch_window))
            box = boxes.get(inst) if patch is None else patch_box
            label = forced or (
                Visibility.VISIBLE.value
                if patch is None
                else _label_for(patch_box, 0.0)
            )
            if label == OUT_OF_VIEW:
                visible, box = None, None
            compiled[inst] = CompiledInstance(
                instance=inst,
                visible=visible,
                box=box,
                amodal=None,
                occlusion_ratio=0.0,
                visibility=label,
                placement=placement,
                keyframe_id=kf_id,
                window=None if visible is None else patch_window,
            )
            continue

        amodal = amodals[inst]
        window = windows[inst]
        if patch is None:
            visible, visible_window = visibles[inst], window
        else:
            visible, visible_window = patch, patch_window
        visible_area = _window_area(visible, visible_window)
        ratio = _occlusion_ratio(visible_area, _window_area(amodal, window))
        box = _window_bbox(visible, visible_window)
        label = forced or _label_for(box, ratio)
        if override is None and visible_area == 0:
            problems.append(f"empty_visible:{inst}")
        if label == OUT_OF_VIEW:
            visible, box, ratio = None, None, 1.0
        compiled[inst] = CompiledInstance(
            instance=inst,
            visible=visible,
            box=box,
            amodal=amodal,
            occlusion_ratio=ratio,
            visibility=label,
            placement=placement,
            keyframe_id=kf_id,
            window=None if visible is None else visible_window,
        )

    return CompiledFrame(
        key=key,
        instances=compiled,
        painted={group: _instance_order(layers) for group, layers in painted.items()},
        layers={group: list(layers) for group, layers in painted.items()},
        problems=problems,
        input_hash=input_hash(
            key=key,
            hw=canvas,
            needs=needs,
            placements=placement_of,
            selected=selected,
            zorder=zorder,
            layer_order=painted,
            overrides=overrides,
            occluders=frame_occluders,
            frame_overrides=frame_overrides,
            transform=transform,
            pose_segment=pose_segment,
            compiler_version=compiler_version,
        ),
    )
