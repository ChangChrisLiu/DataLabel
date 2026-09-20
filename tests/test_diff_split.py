"""Tests for the change splitter that arms a SAM prompt (spec 4.2, Task B2).

Synthetic pairs only: a "part" rectangle that disappears, a revealed "hole" next
to it, and a lighting shadow.  The point of every case is the same one the real
measurement found -- ``diff_blobs`` merges the part with what its removal
reveals, so the merged region's *centroid* is not on the part even though the
region touches it.  :func:`tda.core.diff_split.propose_parts` has to hand back a
box around the part alone and a point that is provably inside it.

Coordinates follow the rest of the codebase: boxes are ``(x0, y0, x1, y1)`` with
exclusive ends, in **full-frame** coordinates whatever ROI was passed.
"""
from __future__ import annotations

import numpy as np
import pytest

from tda.core.diff_split import PartProposal, propose_parts
from tda.core.diffmap import diff_blobs, diff_delta_e

HW = (300, 400)
ROI = (20, 20, 380, 280)
#: The part that is present in the frame being annotated and gone in the next.
#: Small, like the screws and connectors that dominate the real events.
PART = (100, 100, 122, 122)
#: What its removal reveals -- a socket and a patch of board twenty times the
#: part's size, close enough that ``diff_blobs`` merges the two into one region
#: whose centroid is on neither.
HOLE = (130, 80, 222, 170)


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def _board(seed: int = 7) -> np.ndarray:
    """A textured green board, the background of both frames."""
    rng = _rng(seed)
    img = np.zeros((*HW, 3), dtype=np.uint8)
    img[..., 1] = 90
    img[..., 0] = 30
    img[..., 2] = 40
    noise = rng.integers(-6, 7, size=(*HW, 3))
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _fill(img: np.ndarray, box, colour) -> None:
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = colour


def _textured(img: np.ndarray, box, base, seed: int = 3) -> None:
    """A high-frequency patch -- a socket with pins, or a part's own detail."""
    x0, y0, x1, y1 = box
    rng = _rng(seed)
    patch = rng.integers(0, 2, size=(y1 - y0, x1 - x0, 1)) * 110
    img[y0:y1, x0:x1] = np.clip(np.asarray(base, dtype=np.int16) + patch,
                                0, 255).astype(np.uint8)


def _pair() -> tuple[np.ndarray, np.ndarray]:
    """``(prev, cur)``: prev is the frame being annotated, cur the next step.

    In ``prev`` the part is a flat bright plate and the area beside it is plain
    board.  In ``cur`` the part is gone (bare dark board) and a textured socket
    has been revealed next to it.
    """
    prev = _board()
    cur = _board()
    _fill(prev, PART, (200, 200, 205))
    _fill(cur, PART, (34, 70, 40))
    _fill(prev, HOLE, (28, 88, 38))
    _textured(cur, HOLE, (60, 60, 60))
    return prev, cur


def _inside(box, point) -> bool:
    return bool(box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3])


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


# ---------------------------------------------------------------- degenerate
def test_identical_frames_propose_nothing() -> None:
    """Nothing changed, so there is nothing to point at."""
    frame = _board()
    assert propose_parts(frame, frame.copy(), ROI) == []


def test_whole_roi_change_proposes_nothing() -> None:
    """A lighting jump is not a part: never hand back a full-ROI proposal."""
    prev = _board()
    cur = np.clip(prev.astype(np.int16) + 70, 0, 255).astype(np.uint8)
    for proposal in propose_parts(prev, cur, ROI):
        area = ((proposal.box[2] - proposal.box[0])
                * (proposal.box[3] - proposal.box[1]))
        roi_area = (ROI[2] - ROI[0]) * (ROI[3] - ROI[1])
        assert area <= 0.6 * roi_area, "a full-ROI box is not a prompt"


def test_roi_none_is_allowed() -> None:
    """``roi=None`` compares the whole frame instead of raising."""
    prev, cur = _pair()
    proposals = propose_parts(prev, cur, None)
    assert proposals
    assert all(isinstance(p, PartProposal) for p in proposals)


def test_empty_frames_do_not_raise() -> None:
    flat = np.zeros((*HW, 3), dtype=np.uint8)
    assert propose_parts(flat, flat.copy(), ROI) == []


# ------------------------------------------------------------------ contract
def test_point_is_inside_its_own_box_and_mask() -> None:
    """The point is a distance-transform peak, so it is inside the region."""
    prev, cur = _pair()
    proposals = propose_parts(prev, cur, ROI, max_proposals=3)
    assert proposals
    for p in proposals:
        assert _inside(p.box, p.point)
        assert p.mask is not None
        x, y = p.point
        assert p.mask[y - p.box[1], x - p.box[0]]


def test_boxes_are_in_full_frame_coordinates() -> None:
    prev, cur = _pair()
    for p in propose_parts(prev, cur, ROI):
        assert ROI[0] <= p.box[0] < p.box[2] <= ROI[2]
        assert ROI[1] <= p.box[1] < p.box[3] <= ROI[3]


def test_deterministic() -> None:
    prev, cur = _pair()
    first = propose_parts(prev, cur, ROI)
    second = propose_parts(prev, cur, ROI)
    assert [(p.box, p.point) for p in first] == [(p.box, p.point) for p in second]


def test_max_proposals_is_respected() -> None:
    prev, cur = _pair()
    assert len(propose_parts(prev, cur, ROI, max_proposals=1)) <= 1
    assert len(propose_parts(prev, cur, ROI, max_proposals=2)) <= 2


# ------------------------------------------------------------------ splitting
def test_the_merged_blob_is_split_into_part_and_hole() -> None:
    """``diff_blobs`` sees one region; the splitter has to see two.

    This is the real failure: the part and what its removal reveals are merged,
    and the merged centroid sits between them, on neither.
    """
    prev, cur = _pair()
    delta = diff_delta_e(prev, cur, roi=ROI)
    blobs = diff_blobs(delta)
    assert blobs, "the synthetic pair must produce a change at all"
    merged = blobs[0].box
    assert _iou(merged, PART) < 0.6, "the fixture must merge part and hole"

    proposals = propose_parts(prev, cur, ROI, max_proposals=3)
    assert len(proposals) >= 2
    best_part = max(_iou(p.box, PART) for p in proposals)
    best_hole = max(_iou(p.box, HOLE) for p in proposals)
    assert best_part > _iou(merged, PART)
    assert best_hole > _iou(merged, HOLE)


def test_the_area_prior_picks_the_part_out_of_the_pair() -> None:
    """Given the class's own area band, the part outranks what it revealed."""
    prev, cur = _pair()
    roi_area = float((ROI[2] - ROI[0]) * (ROI[3] - ROI[1]))
    part_area = float((PART[2] - PART[0]) * (PART[3] - PART[1]))
    band = (0.8 * part_area / roi_area, 1.25 * part_area / roi_area)
    top = propose_parts(prev, cur, ROI, expect_area=band, max_proposals=3)[0]
    assert _inside(PART, top.point)
    assert _iou(top.box, PART) > 0.5


def test_the_hole_wins_when_the_prior_asks_for_its_size() -> None:
    """The prior really is what moved the ranking, not the fixture's luck."""
    prev, cur = _pair()
    roi_area = float((ROI[2] - ROI[0]) * (ROI[3] - ROI[1]))
    hole_area = float((HOLE[2] - HOLE[0]) * (HOLE[3] - HOLE[1]))
    band = (0.8 * hole_area / roi_area, 1.25 * hole_area / roi_area)
    top = propose_parts(prev, cur, ROI, expect_area=band, max_proposals=3)[0]
    assert _iou(top.box, HOLE) > _iou(top.box, PART)


def test_known_masks_suppress_what_is_already_drawn() -> None:
    """A shape the annotator has already committed is not proposed again."""
    prev, cur = _pair()
    known = np.zeros(HW, dtype=bool)
    known[HOLE[1]:HOLE[3], HOLE[0]:HOLE[2]] = True
    proposals = propose_parts(prev, cur, ROI, known_masks=[known],
                              max_proposals=3)
    assert proposals
    assert all(_iou(p.box, HOLE) < 0.5 for p in proposals)


def test_blob_index_points_back_at_the_parent_blob() -> None:
    """Every proposal names the ``diff_blobs`` blob it came out of."""
    prev, cur = _pair()
    delta = diff_delta_e(prev, cur, roi=ROI)
    blobs = diff_blobs(delta)
    for p in propose_parts(prev, cur, ROI):
        assert 0 <= p.blob_index < len(blobs)


def test_a_delta_map_can_be_passed_in() -> None:
    """The caller that already has the dE map does not pay for it twice."""
    prev, cur = _pair()
    delta = diff_delta_e(prev, cur, roi=ROI, max_side=None)
    with_map = propose_parts(prev, cur, ROI, delta_e=delta)
    without = propose_parts(prev, cur, ROI)
    assert [p.box for p in with_map] == [p.box for p in without]


def test_a_shadow_outside_the_roi_is_ignored() -> None:
    """The ROI is the window; a change outside it is not a candidate."""
    prev, cur = _pair()
    cur[0:15, 0:15] = 255
    for p in propose_parts(prev, cur, ROI):
        assert p.box[0] >= ROI[0] and p.box[1] >= ROI[1]


def test_bad_input_is_refused() -> None:
    prev, cur = _pair()
    with pytest.raises(ValueError):
        propose_parts(prev[..., :2], cur[..., :2], ROI)
