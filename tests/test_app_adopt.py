"""Adopting a Label Studio draft from the window (plan B4, step 2).

``Shift+A`` offers the old polygons of the class being drawn as a **ghost**:
display-only pixels in their own colour, over the frame, under the annotator's
own layer.  While one is showing it owns ``Enter`` and ``Esc`` exactly as the
ROI rectangle does -- ``Enter`` copies it into the editing layer as one
undoable stroke, ``Esc`` takes the ghost away and leaves the layer alone.

The tests below are about the two things that make it safe: the ghost never
writes anything by itself, and everything that changes what the annotator is
looking at (a frame, an instance, a view, the ROI rectangle) takes it away
again -- the same hygiene the SAM prompts get.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    COOLER,
    DESKTOP,
    SCREWS,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
)
from tda.core import masks
from tda.core.db import Db
from tda.core.model import FrameKey, InstanceRec, ShapeKeyframe, ShapePart
from tda.ui import app_actions as A
from tda.ui import app_adopt as adopt
from tda.ui.app import MainWindow

#: Labels of ``configs/ls_label_map.yaml``; the classes they map onto are the
#: ones the truncated D13 scene actually has.
SCREW_LABEL = "CPU Cooling Fan Screw"       # -> screw
COOLER_LABEL = "CPU Cooling Fan"            # -> cpu_cooler
HW = (64, 64)
#: The step the tests work on: the cooler and its screws are in the chassis.
WORK_STEP = 12


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester",
                     sam_queue=StubSamQueue())
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    if win.roi_editing:            # the first frame of a machine offers an ROI
        win.act_commit()
    # A frame where the cooler and its four screws are still in the chassis and
    # on the task card, so an adopted draft is one an annotator could commit.
    win.timeline_goto(WORK_STEP)
    yield win
    close_window(win)


# --------------------------------------------------------------------------- #
# scene helpers
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int, hw=HW) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def add_draft(db: Db, label: str, ordinal: int, step: int, box, *, cls: str,
              view: str = VIEW, hw=HW) -> str:
    """One importer-shaped draft on the open machine."""
    key = f"ls:{label}#{ordinal}"
    db.upsert_instance(InstanceRec(key=key, desktop=DESKTOP, cls=cls,
                                   raw_names=[label]))
    db.add_keyframe(ShapeKeyframe(
        id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=0,
        anchor_step=step, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box, hw=hw)))],
        amodal_complete=False, source="labelstudio", draft_id=7,
    ))
    return key


def seed_screw_drafts(win, boxes=((2, 2, 5, 5), (30, 30, 33, 33))) -> list[str]:
    """Two ``screw`` drafts on the open step, left to right.

    Nine pixels each: a screw's area prior is 0.004-0.55 % of the ROI, and a
    draft outside that band would raise the area warning at commit time, which
    is a different conversation (it has its own tests).
    """
    step = win.session.current().step
    return [add_draft(win.db, SCREW_LABEL, i + 1, step, box, cls="screw")
            for i, box in enumerate(boxes)]


def seed_cooler_draft(win, box=(20, 20, 30, 30)) -> str:
    """One ``cpu_cooler`` draft: 100 px, which its area prior calls plausible.

    The commit tests need a draft the area warning has nothing to say about, and
    a screw cannot be one here: on a 46x46 ROI its band tops out at 10 px while
    the absolute floor for any mask is 12.
    """
    step = win.session.current().step
    return add_draft(win.db, COOLER_LABEL, 1, step, box, cls="cpu_cooler")


def edit_a_screw(win) -> str:
    """Begin an ordinary edit on a real instance of class ``screw``."""
    win.on_request_edit(SCREWS[0])
    assert win.session.editing_instance == SCREWS[0]
    return SCREWS[0]


def edit_the_cooler(win) -> str:
    win.on_request_edit(COOLER)
    assert win.session.editing_instance == COOLER
    return COOLER


def ghost(win):
    return None if win.overlay is None else win.overlay.ghost


def db_snapshot(db: Db) -> list[tuple]:
    rows = db.conn.execute(
        "SELECT k.id, k.instance, k.anchor_step, k.source, k.version, p.rle_json "
        "FROM shape_keyframe k JOIN shape_part p ON p.keyframe_id=k.id "
        "WHERE k.instance LIKE 'ls:%' ORDER BY k.id, p.idx").fetchall()
    return [tuple(r) for r in rows]


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #
def test_the_adopt_binding_is_in_the_one_key_map_and_takes_a_free_key():
    action = next(a for a in A.ACTIONS if a.name == "adopt_draft")
    assert action.slot == "act_adopt_draft"
    assert action.modes == (A.MODE_ANNOTATE,)
    combos = [c for a in A.ACTIONS if a is not action for c in A.combos_of(a)]
    assert not set(A.combos_of(action)) & set(combos)


def test_plain_a_still_toggles_the_overlays():
    from PySide6.QtCore import Qt

    assert A.action_for(Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier).name == \
        "toggle_overlays"
    assert A.action_for(Qt.Key.Key_A, Qt.KeyboardModifier.ShiftModifier).name == \
        "adopt_draft"


# --------------------------------------------------------------------------- #
# offering
# --------------------------------------------------------------------------- #
def test_adopting_needs_an_instance_being_edited(window):
    seed_screw_drafts(window)
    window.act_adopt_draft()
    assert not window.showing_draft_ghost()
    assert adopt.ADOPT_NEEDS_INSTANCE in window.status_message()


def test_no_candidate_says_so_and_shows_nothing(window):
    edit_a_screw(window)
    window.act_adopt_draft()
    assert not window.showing_draft_ghost()
    assert "no Label Studio draft" in window.status_message()


def test_the_ghost_is_display_only_until_enter(window):
    drafts = seed_screw_drafts(window)
    edit_a_screw(window)
    before = window.session.editing_mask().copy()
    ops = len(window.session.undo_stack)

    window.act_adopt_draft()

    assert window.showing_draft_ghost()
    assert window.overlay.has_ghost
    assert np.array_equal(ghost(window), rect(2, 2, 5, 5))
    # nothing of the annotator's has moved
    assert np.array_equal(window.session.editing_mask(), before)
    assert not window.overlay.editing.any()
    assert len(window.session.undo_stack) == ops
    assert window._sidecar_pending is None
    assert drafts[0] in window.status_message()


def test_shift_a_again_walks_the_candidates(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_adopt_draft()
    window.act_adopt_draft()
    assert np.array_equal(ghost(window), rect(30, 30, 33, 33))
    assert "2/2" in window.status_message()
    window.act_adopt_draft()                      # wraps round
    assert np.array_equal(ghost(window), rect(2, 2, 5, 5))


def test_a_wrong_size_draft_is_counted_and_never_shown(window):
    step = window.session.current().step
    add_draft(window.db, SCREW_LABEL, 1, step, (2, 2, 5, 5), cls="screw",
              hw=(32, 32))
    edit_a_screw(window)
    window.act_adopt_draft()
    assert not window.showing_draft_ghost()
    assert "1" in window.status_message() and "尺寸" in window.status_message()


def test_adoption_is_inert_while_another_frame_is_flashed_over_the_canvas(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_flash_compare(True)
    assert window.is_flashing()
    window.act_adopt_draft()
    assert not window.showing_draft_ghost()
    window.act_flash_compare(False)


# --------------------------------------------------------------------------- #
# Enter: one undoable stroke on the same history
# --------------------------------------------------------------------------- #
def test_enter_on_the_ghost_replaces_an_empty_layer_in_one_undoable_op(window):
    drafts = seed_screw_drafts(window)
    instance = edit_a_screw(window)
    ops = len(window.session.undo_stack)
    keyframes = len(window.db.keyframes(DESKTOP, VIEW, instance))

    window.act_adopt_draft()
    window.act_commit()                      # Enter belongs to the ghost

    assert np.array_equal(window.session.editing_mask(), rect(2, 2, 5, 5))
    assert np.array_equal(window.overlay.editing, rect(2, 2, 5, 5))
    assert len(window.session.undo_stack) == ops + 1
    assert not window.showing_draft_ghost() and not window.overlay.has_ghost
    # the ghost's Enter is not the edit's Enter: nothing was written
    assert len(window.db.keyframes(DESKTOP, VIEW, instance)) == keyframes
    assert window.session.editing_instance == instance
    assert drafts[0] in window.status_message()
    assert "replaced" in window.status_message()


def test_enter_unions_a_layer_that_already_holds_pixels(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.set_editing_mask(rect(50, 50, 60, 60), undoable=True)

    window.act_adopt_draft()
    window.act_commit()

    assert np.array_equal(window.session.editing_mask(),
                          rect(50, 50, 60, 60) | rect(2, 2, 5, 5))
    assert "unioned" in window.status_message()


def test_the_adopted_stroke_reaches_the_crash_sidecar(window):
    seed_screw_drafts(window)
    instance = edit_a_screw(window)
    window.act_adopt_draft()
    window.act_commit()

    assert window._sidecar_pending is not None
    window.flush_sidecar()
    assert window.sidecar_writes == 1
    pending = window.sidecar.pending_for(window.session.current(), instance, HW)
    assert pending is not None
    assert np.array_equal(pending["mask"], rect(2, 2, 5, 5))


def test_one_ctrl_z_takes_the_whole_adopted_draft_back(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    before = window.session.editing_mask().copy()
    window.act_adopt_draft()
    window.act_commit()

    window.act_undo()

    assert np.array_equal(window.session.editing_mask(), before)
    assert not window.overlay.editing.any()


# --------------------------------------------------------------------------- #
# Esc: the ghost only
# --------------------------------------------------------------------------- #
def test_esc_dismisses_the_ghost_and_keeps_the_edit(window):
    seed_screw_drafts(window)
    instance = edit_a_screw(window)
    window.set_editing_mask(rect(50, 50, 60, 60), undoable=True)
    window.act_adopt_draft()

    window.act_clear_edit()

    assert not window.showing_draft_ghost()
    assert window.session.editing_instance == instance
    assert np.array_equal(window.session.editing_mask(), rect(50, 50, 60, 60))
    # and the second Esc is the ordinary one
    window.act_clear_edit()
    assert window.session.editing_instance is None


# --------------------------------------------------------------------------- #
# what the commit records
# --------------------------------------------------------------------------- #
def last_op(win, kind: str = "commit_keyframe") -> dict:
    rows = win.db.ops(DESKTOP, VIEW, limit=20)
    for row in rows:
        if row["kind"] == kind:
            payload = row["payload"]
            return json.loads(payload) if isinstance(payload, str) else payload
    raise AssertionError(f"no {kind} in the op log")


def test_the_commit_records_which_draft_was_adopted(window):
    draft = seed_cooler_draft(window)
    step = window.session.current().step
    edit_the_cooler(window)
    window.act_adopt_draft()
    window.act_commit()             # the ghost
    window.act_commit()             # the edit

    payload = last_op(window)
    assert payload["adopted_from"] == draft
    assert payload["adopted_step"] == step
    assert payload["instance"] == COOLER
    assert "area_warning_overridden" not in payload


def test_an_area_override_and_an_adoption_ride_on_the_same_commit(window):
    # 2,116 px for a cooler on a 46x46 ROI: over its band, so the commit asks
    draft = seed_cooler_draft(window, box=(6, 6, 52, 52))
    edit_the_cooler(window)
    window.act_adopt_draft()
    window.act_commit()             # the ghost
    window.act_commit()             # raises the area warning
    assert window._pending_warning is not None
    window.act_commit()             # the second Enter goes ahead

    payload = last_op(window)
    assert payload["adopted_from"] == draft
    assert payload["area_warning_overridden"] is True


def test_a_commit_after_a_discarded_ghost_claims_nothing(window):
    seed_cooler_draft(window)
    edit_the_cooler(window)
    window.act_adopt_draft()
    window.act_clear_edit()                       # the ghost goes
    window.set_editing_mask(rect(40, 40, 50, 50), undoable=True)
    window.act_commit()

    assert "adopted_from" not in last_op(window)


def test_adopting_and_committing_never_touches_the_draft(window):
    seed_cooler_draft(window)
    edit_the_cooler(window)
    before = db_snapshot(window.db)
    window.act_adopt_draft()
    window.act_commit()
    window.act_commit()
    assert db_snapshot(window.db) == before


# --------------------------------------------------------------------------- #
# hygiene: what takes the ghost away
# --------------------------------------------------------------------------- #
def test_a_frame_change_dismisses_the_ghost_and_clears_the_candidates(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_adopt_draft()
    assert window.showing_draft_ghost()

    window.act_step(-1)

    assert not window.showing_draft_ghost()
    assert not window.overlay.has_ghost
    assert window.draft_candidates() == []


def test_editing_another_instance_dismisses_the_ghost(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_adopt_draft()
    window.on_request_edit(COOLER)
    assert not window.showing_draft_ghost()


def test_a_view_change_dismisses_the_ghost(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_adopt_draft()
    window.act_set_view("oak1")
    assert not window.showing_draft_ghost()


def test_the_roi_rectangle_takes_the_ghost_away_with_enter_and_esc(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_adopt_draft()
    window.act_edit_roi()
    assert window.roi_editing
    assert not window.showing_draft_ghost()


def test_the_ghost_is_refused_while_a_bar_owns_enter_and_esc(window):
    seed_screw_drafts(window)
    edit_a_screw(window)
    window.act_edit_roi()
    window.act_adopt_draft()
    assert not window.showing_draft_ghost()
    assert adopt.ADOPT_BUSY.split("/")[0].strip() in window.status_message()
