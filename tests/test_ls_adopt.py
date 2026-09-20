"""Offering an old Label Studio polygon as a candidate (plan B4, step 1).

The database holds 11,191 draft keyframes on provisional ``ls:<Label>#<n>``
keys.  The ordinal in that key is a per-frame left-to-right index, not an
identity, so nothing here ever tries to pick *the* draft of
``ram_module.02``: :func:`tda.core.ls_adopt.drafts_for` offers every draft of
the right **class** near the step and lets geometry rank them.

What the tests below pin down is what may never happen to a draft: it is never
resized to fit a frame it was not traced on, never taken from another view,
never written to, and never handed out as an array the caller can change under
the database's feet.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from tda.core import masks
from tda.core.db import Db
from tda.core.ls_adopt import DraftCandidate, draft_label, drafts_for
from tda.core.model import InstanceRec, ShapeKeyframe, ShapePart
from tda.core.taxonomy import load_taxonomy

DESKTOP = 13
VIEW = "scan"
HW = (64, 64)
LS_SOURCE = "labelstudio"

#: Labels of ``configs/ls_label_map.yaml`` and the classes they map onto.
BOARD = "Motherboard"            # -> motherboard
SCREW = "Motherboard Screw"      # -> screw
RAM = "RAM Module"               # -> ram_module


def rect(x0: int, y0: int, x1: int, y1: int, hw=HW) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def db(tmp_db_path: str):
    d = Db(tmp_db_path)
    yield d
    d.close()


@pytest.fixture
def tax():
    return load_taxonomy()


def add_draft(db: Db, label: str, ordinal: int, step: int, box, *,
              cls: str, view: str = VIEW, hw=HW, desktop: int = DESKTOP) -> str:
    """One importer-shaped draft: a provisional instance plus its keyframe."""
    key = f"ls:{label}#{ordinal}"
    db.upsert_instance(InstanceRec(key=key, desktop=desktop, cls=cls,
                                   raw_names=[label]))
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=key, desktop=desktop, view=view, pose_segment=0,
        anchor_step=step, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box, hw=hw)))],
        amodal_complete=False, source=LS_SOURCE, draft_id=7,
    ))
    return key


def keys(candidates) -> list[str]:
    return [c.key for c in candidates]


# --------------------------------------------------------------------------- #
# label -> class
# --------------------------------------------------------------------------- #
def test_the_label_map_decides_which_class_a_draft_is_offered_for(db, tax):
    """``ls:Motherboard Screw#3`` is a ``screw``, whatever its ordinal says."""
    screw = add_draft(db, SCREW, 3, 20, (2, 2, 6, 6), cls="screw")
    board = add_draft(db, BOARD, 1, 20, (8, 8, 40, 40), cls="motherboard")

    assert keys(drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW)) == [screw]
    assert keys(drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW)) == [board]
    assert drafts_for(db, tax, DESKTOP, VIEW, 20, "psu", hw=HW) == []


def test_draft_label_reads_the_label_back_out_of_the_key():
    assert draft_label("ls:Motherboard Screw#12") == "Motherboard Screw"
    assert draft_label("ls:RAM Module Retention Clip (open)#2") == \
        "RAM Module Retention Clip (open)"
    assert draft_label("screw.psu.01") is None


def test_a_real_instance_is_never_a_draft_candidate(db, tax):
    """Only ``ls:*`` keys are draft material (spec 3.2)."""
    db.upsert_instance(InstanceRec(key="screw.psu.01", desktop=DESKTOP, cls="screw"))
    db.add_keyframe(ShapeKeyframe(
        id=None, instance="screw.psu.01", desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=20, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(2, 2, 6, 6)))],
    ))
    assert drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW) == []


# --------------------------------------------------------------------------- #
# the step window
# --------------------------------------------------------------------------- #
def test_the_step_window_is_the_step_itself_plus_near_steps(db, tax):
    """Drafts are often traced one step away; three steps away is another shape."""
    here = add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    near = add_draft(db, SCREW, 2, 22, (10, 2, 14, 6), cls="screw")
    far = add_draft(db, SCREW, 3, 23, (20, 2, 24, 6), cls="screw")

    found = keys(drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW))
    assert found == [here, near]        # nearest step first, `far` outside ±2
    assert far not in found
    assert keys(drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW,
                           near_steps=0)) == [here]


def test_without_a_reference_mask_the_nearest_step_wins(db, tax):
    before = add_draft(db, SCREW, 1, 19, (2, 2, 6, 6), cls="screw")
    here = add_draft(db, SCREW, 2, 20, (10, 2, 14, 6), cls="screw")
    after = add_draft(db, SCREW, 3, 22, (20, 2, 24, 6), cls="screw")

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW)
    assert keys(found) == [here, before, after]
    assert [c.step for c in found] == [20, 19, 22]
    assert all(c.iou_with_editing == 0.0 for c in found)


# --------------------------------------------------------------------------- #
# geometry decides
# --------------------------------------------------------------------------- #
def test_a_reference_mask_ranks_by_overlap_before_step_distance(db, tax):
    """The armed proposal (or the layer) is what says *which* screw this is."""
    wrong = add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    right = add_draft(db, SCREW, 2, 22, (30, 30, 40, 40), cls="screw")

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW,
                       editing=rect(31, 31, 41, 41))
    assert keys(found) == [right, wrong]
    assert found[0].iou_with_editing > found[1].iou_with_editing
    assert found[1].iou_with_editing == 0.0


def test_an_empty_reference_mask_is_no_reference_at_all(db, tax):
    here = add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    near = add_draft(db, SCREW, 2, 21, (30, 30, 40, 40), cls="screw")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW,
                       editing=np.zeros(HW, dtype=bool))
    assert keys(found) == [here, near]


def test_the_candidate_carries_the_draft_key_step_label_and_mask(db, tax):
    add_draft(db, RAM, 2, 31, (10, 10, 20, 20), cls="ram_module")
    found = drafts_for(db, tax, DESKTOP, VIEW, 30, "ram_module", hw=HW)
    assert len(found) == 1
    candidate = found[0]
    assert isinstance(candidate, DraftCandidate)
    assert candidate.key == "ls:RAM Module#2"
    assert candidate.label == RAM
    assert candidate.step == 31
    assert candidate.mask.dtype == bool and candidate.mask.shape == HW
    assert np.array_equal(candidate.mask, rect(10, 10, 20, 20))


def test_more_candidates_than_the_cap_are_cut_after_ranking(db, tax):
    for i in range(6):
        add_draft(db, SCREW, i + 1, 20, (2 * i, 2, 2 * i + 2, 4), cls="screw")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, max_candidates=3)
    assert len(found) == 3


# --------------------------------------------------------------------------- #
# what is refused: wrong size, wrong view
# --------------------------------------------------------------------------- #
def test_a_wrong_size_draft_is_skipped_and_counted_never_resized(db, tax):
    """A draft traced at another resolution is not this frame's geometry.

    Stretching it would invent pixels nobody drew, so it is dropped and the
    count is handed back for the status bar to say so.
    """
    small = add_draft(db, BOARD, 1, 20, (2, 2, 10, 10), cls="motherboard",
                      hw=(32, 32))
    good = add_draft(db, BOARD, 2, 20, (4, 4, 20, 20), cls="motherboard")
    stats: dict = {}

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW, stats=stats)
    assert keys(found) == [good]
    assert small not in keys(found)
    assert stats["wrong_size"] == 1
    assert all(c.mask.shape == HW for c in found)


def test_every_draft_of_the_wrong_size_leaves_no_candidates_at_all(db, tax):
    add_draft(db, BOARD, 1, 20, (2, 2, 10, 10), cls="motherboard", hw=(32, 32))
    stats: dict = {}
    assert drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW,
                      stats=stats) == []
    assert stats["wrong_size"] == 1 and stats["offered"] == 0


def test_another_view_is_never_offered(db, tax):
    mine = add_draft(db, BOARD, 1, 20, (4, 4, 20, 20), cls="motherboard")
    add_draft(db, BOARD, 1, 20, (4, 4, 20, 20), cls="motherboard", view="oak1")
    assert keys(drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW)) == [mine]


def test_another_desktop_is_never_offered(db, tax):
    add_draft(db, BOARD, 1, 20, (4, 4, 20, 20), cls="motherboard", desktop=24)
    assert drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW) == []


# --------------------------------------------------------------------------- #
# a draft is reference material
# --------------------------------------------------------------------------- #
def snapshot(db: Db) -> list[tuple]:
    rows = db.conn.execute(
        "SELECT k.id, k.instance, k.view, k.anchor_step, k.source, k.version, "
        "p.rle_json FROM shape_keyframe k JOIN shape_part p ON p.keyframe_id=k.id "
        "ORDER BY k.id, p.idx"
    ).fetchall()
    return [tuple(r) for r in rows]


def test_offering_a_draft_writes_nothing_and_hands_out_a_copy(db, tax):
    """Drafts are never modified: reading them may not change one byte."""
    add_draft(db, BOARD, 1, 20, (4, 4, 20, 20), cls="motherboard")
    before = snapshot(db)

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW,
                       editing=rect(4, 4, 20, 20))
    found[0].mask[:] = False          # the caller owns the array it was given

    assert snapshot(db) == before
    again = drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW)
    assert np.array_equal(again[0].mask, rect(4, 4, 20, 20))


def test_nothing_to_offer_is_an_empty_list_and_zero_counts(db, tax):
    stats: dict = {}
    assert drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, stats=stats) == []
    assert stats == {"considered": 0, "wrong_size": 0, "empty": 0, "offered": 0}


# --------------------------------------------------------------------------- #
# round 1 C1: the cursor decides, and nothing is out of reach
# --------------------------------------------------------------------------- #
def test_the_draft_under_the_cursor_comes_first(db, tax):
    """Twenty screws of one class: only where the annotator points tells them apart."""
    boxes = [(2 + 6 * i, 2, 6 + 6 * i, 6) for i in range(9)]
    keys = [add_draft(db, SCREW, i + 1, 20, box, cls="screw")
            for i, box in enumerate(boxes)]

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, cursor=(45, 4))
    assert found[0].key == keys[7]            # x 44..48 is the eighth
    assert found[0].distance == 0.0
    assert len(found) == 9                    # every one of them stays reachable


def test_a_cursor_beside_the_drafts_orders_them_by_distance(db, tax):
    near = add_draft(db, SCREW, 1, 20, (40, 40, 46, 46), cls="screw")
    far = add_draft(db, SCREW, 2, 20, (2, 2, 6, 6), cls="screw")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, cursor=(50, 50))
    assert keys(found) == [near, far]
    assert found[0].distance > 0 and found[0].distance < found[1].distance


def test_the_cursor_beats_the_step_and_the_step_beats_the_key(db, tax):
    here = add_draft(db, SCREW, 9, 20, (2, 2, 6, 6), cls="screw")
    under = add_draft(db, SCREW, 1, 22, (40, 40, 46, 46), cls="screw")
    also = add_draft(db, SCREW, 2, 21, (40, 40, 46, 46), cls="screw")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, cursor=(42, 42))
    # both under the cursor: the nearer step wins; `here` is far away
    assert keys(found) == [also, under, here]


def test_a_cursor_inside_the_mask_beats_one_only_inside_the_bounding_box(db, tax):
    """An L-shaped draft's box covers the neighbour it does not touch."""
    big = f"ls:{BOARD}#1"
    db.upsert_instance(InstanceRec(key=big, desktop=DESKTOP, cls="motherboard",
                                   raw_names=[BOARD]))
    hollow = np.zeros(HW, dtype=bool)
    hollow[10:40, 10:14] = True               # a bar down the left of a 30x30 box
    hollow[36:40, 10:40] = True               # and along the bottom
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=big, desktop=DESKTOP, view=VIEW, pose_segment=0,
        anchor_step=20, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(hollow))],
        amodal_complete=False, source=LS_SOURCE))
    inside = add_draft(db, BOARD, 2, 20, (20, 20, 30, 30), cls="motherboard")

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "motherboard", hw=HW,
                       cursor=(25, 25))
    assert keys(found) == [inside, big]       # (25, 25) is in the hole of `big`
    assert found[0].distance == 0.0 and found[1].distance == 0.0


def test_drafts_of_another_class_under_the_cursor_are_offered_last(db, tax):
    """Label Studio's labels are noisy, so what is *under* the cursor counts."""
    mine = add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    other = add_draft(db, BOARD, 1, 20, (20, 20, 40, 40), cls="motherboard")
    add_draft(db, RAM, 1, 20, (50, 50, 60, 60), cls="ram_module")   # not pointed at

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, cursor=(25, 25))
    assert keys(found) == [mine, other]
    assert [c.same_class for c in found] == [True, False]
    assert [c.cls for c in found] == ["screw", "motherboard"]
    assert found[1].label == BOARD


def test_without_a_cursor_nothing_of_another_class_is_offered(db, tax):
    add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    add_draft(db, BOARD, 1, 20, (20, 20, 40, 40), cls="motherboard")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW)
    assert all(c.same_class for c in found)


def test_the_mask_is_decoded_only_when_it_is_asked_for(db, tax):
    add_draft(db, SCREW, 1, 20, (2, 2, 6, 6), cls="screw")
    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW, cursor=(3, 3))
    candidate = found[0]
    assert candidate.decoded is False
    assert np.array_equal(candidate.mask, rect(2, 2, 6, 6))
    assert candidate.decoded is True
    candidate.release()
    assert candidate.decoded is False
    assert candidate.rle["size"] == [64, 64]
    assert candidate.keyframe_id is not None


# --------------------------------------------------------------------------- #
# round 1 I1: a draft of another pose segment is another pose
# --------------------------------------------------------------------------- #
def test_a_draft_from_another_pose_segment_is_never_offered(db, tax):
    """Drafts sit in segment 0 and are not re-keyed, so the step range decides."""
    db.set_pose_segment(DESKTOP, VIEW, 1, 1, 20, 20, None, None)
    db.set_pose_segment(DESKTOP, VIEW, 2, 21, 42, 42, None, None)
    mine = add_draft(db, SCREW, 1, 19, (2, 2, 6, 6), cls="screw")
    over = add_draft(db, SCREW, 2, 21, (10, 2, 14, 6), cls="screw")

    found = drafts_for(db, tax, DESKTOP, VIEW, 20, "screw", hw=HW)
    assert keys(found) == [mine]
    assert over not in keys(found)
    # and from the other side of the break the answer is the other way round
    assert keys(drafts_for(db, tax, DESKTOP, VIEW, 21, "screw", hw=HW)) == [over]
