"""``Ctrl+Shift+B``, the audit's proposal bar and the timeline's marks (task B1).

The window is the D13 scanner scene of ``app_scene.py``.  The split dialog is
answered through :meth:`tda.ui.app_pose.PoseMixin.ask_pose_split`, which is a
method precisely so a test can answer it without a modal event loop; the dialog
itself is built once, headless, to prove it assembles.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_pose import NO_SPLIT_HERE, PoseSplitDialog, proposal_text

CUT = 8


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path: Path):
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    seed_shapes(session, LAST_STEP)
    session.goto(LAST_STEP, force=True)
    win.render_frame()
    yield win
    close_window(win)


def answer(win: MainWindow, carry) -> list:
    """Make the split dialog answer ``carry`` (or cancel with ``None``)."""
    asked: list = []

    def ask(key, straddles, default, note=""):
        asked.append({"step": key.step, "straddles": straddles, "default": default,
                      "note": note})
        return carry

    win.ask_pose_split = ask
    return asked


def segments(win: MainWindow) -> list[tuple]:
    return [(r["seg"], r["start_step"], r["end_step"])
            for r in win.db.pose_segments(DESKTOP, VIEW)]


# --------------------------------------------------------------------------- #
# the action
# --------------------------------------------------------------------------- #
def test_the_action_is_in_the_one_key_map():
    action = next(a for a in A.ACTIONS if a.name == "split_pose")
    assert action.keys == ("Ctrl+Shift+B",)
    assert action.slot == "act_split_pose"
    assert A.action_for(*_combo("Ctrl+Shift+B")) is action


def _combo(spec: str):
    from PySide6.QtGui import QKeySequence

    combination = QKeySequence.fromString(spec)[0]
    return combination.key(), combination.keyboardModifiers()


def test_ctrl_shift_b_cuts_the_view_at_this_frame(window: MainWindow):
    window.session.goto(CUT, force=True)
    asked = answer(window, carry=False)

    window.act_split_pose()

    assert asked and asked[0]["step"] == CUT
    assert segments(window) == [(1, 1, CUT - 1), (2, CUT, LAST_STEP)]
    assert window.db.pose_break(DESKTOP, VIEW, CUT)["status"] == "accepted"
    assert window.db.pose_break(DESKTOP, VIEW, CUT)["source"] == "manual:tester"
    # the window is still standing on the same frame it cut at
    assert window.session.current().step == CUT


def test_cancelling_the_dialog_writes_nothing(window: MainWindow):
    window.session.goto(CUT, force=True)
    answer(window, carry=None)

    window.act_split_pose()

    assert segments(window) == [(1, 1, LAST_STEP)]
    assert window.db.pose_break(DESKTOP, VIEW, CUT) is None


def test_the_dialog_is_told_how_many_shapes_the_boundary_cuts(window: MainWindow):
    window.session.goto(CUT, force=True)
    expected = len(window.db.straddling_keyframes(DESKTOP, VIEW, CUT))
    asked = answer(window, carry=True)

    window.act_split_pose()

    assert expected > 0 and asked[0]["straddles"] == expected
    carried = [k for k in window.db.keyframes(DESKTOP, VIEW) if k.source == "carried"]
    assert len(carried) == expected
    assert all(k.anchor_step == CUT - 1 and k.pose_segment == 1 for k in carried)


def test_a_split_at_the_first_step_is_refused_with_a_reason(window: MainWindow):
    window.session.goto(min(window.session.steps()), force=True)
    asked = answer(window, carry=False)

    window.act_split_pose()

    assert asked == []                       # the dialog never opened
    assert segments(window) == [(1, 1, LAST_STEP)]
    assert "first step" in window.status_message()


def test_the_action_is_refused_by_the_one_gate_while_the_layer_is_dirty(window: MainWindow):
    """No second gate here: `leave_frame` answers, and its hint is the hint."""
    window.session.goto(CUT, force=True)
    instance = next(str(r["key"]) for r in window.session.instance_rows()
                    if r.get("placement") == "in_chassis")
    window.on_request_edit(instance)
    mask = window.session.editing_mask().copy()
    mask[2:6, 2:6] = True
    window.set_editing_mask(mask)
    assert window.has_uncommitted_edit()
    asked = answer(window, carry=False)

    window.act_split_pose()

    assert asked == []
    assert segments(window) == [(1, 1, LAST_STEP)]
    assert "Enter" in window.status_message()
    assert window.session.editing_instance == instance


# --------------------------------------------------------------------------- #
# the proposal bar
# --------------------------------------------------------------------------- #
def test_a_proposed_break_shows_the_bar_on_its_own_frame(window: MainWindow):
    window.db.add_pose_break(DESKTOP, VIEW, CUT, status="proposed", kind="camera",
                             magnitude_px=190.45, source="audit:events.csv")

    window.session.goto(CUT, force=True)
    window.render_frame()

    assert window.pose_bar.isVisibleTo(window)
    assert "190 px" in window.pose_bar.label.text()
    assert "Tab" in window.pose_bar.label.text()

    window.session.goto(CUT - 1, force=True)
    window.render_frame()
    assert not window.pose_bar.isVisibleTo(window)


def test_accepting_from_the_bar_opens_the_same_dialog_and_cuts(window: MainWindow):
    window.db.add_pose_break(DESKTOP, VIEW, CUT, status="proposed", kind="camera",
                             magnitude_px=8.6, source="audit:events.csv",
                             note="tape rectangle displaced")
    window.session.goto(CUT, force=True)
    window.render_frame()
    asked = answer(window, carry=True)

    window.accept_pose_proposal()

    assert asked[0]["default"] is True       # 8.6 px is under the carry threshold
    assert asked[0]["note"] == "tape rectangle displaced"
    assert segments(window) == [(1, 1, CUT - 1), (2, CUT, LAST_STEP)]
    row = window.db.pose_break(DESKTOP, VIEW, CUT)
    assert row["status"] == "accepted" and row["kind"] == "camera"
    assert row["source"] == "audit:events.csv"   # the provenance is not overwritten
    assert not window.pose_bar.isVisibleTo(window)


def test_a_large_move_does_not_carry_the_shapes_by_default(window: MainWindow):
    window.db.add_pose_break(DESKTOP, VIEW, CUT, status="proposed", kind="camera",
                             magnitude_px=190.45, source="audit:events.csv")
    window.session.goto(CUT, force=True)
    window.render_frame()
    asked = answer(window, carry=False)

    window.accept_pose_proposal()

    assert asked[0]["default"] is False


def test_rejecting_keeps_the_row_and_cuts_nothing(window: MainWindow):
    window.db.add_pose_break(DESKTOP, VIEW, CUT, status="proposed", kind="camera",
                             magnitude_px=8.6, source="audit:events.csv")
    window.session.goto(CUT, force=True)
    window.render_frame()

    window.reject_pose_proposal()

    assert window.db.pose_break(DESKTOP, VIEW, CUT)["status"] == "rejected"
    assert segments(window) == [(1, 1, LAST_STEP)]
    assert not window.pose_bar.isVisibleTo(window)


def test_rejecting_an_accepted_break_merges_the_segments_back(window: MainWindow):
    window.session.goto(CUT, force=True)
    answer(window, carry=True)
    window.act_split_pose()
    assert segments(window) == [(1, 1, CUT - 1), (2, CUT, LAST_STEP)]
    window.db.set_pose_break_status(DESKTOP, VIEW, CUT, "proposed")
    window.render_frame()

    window.reject_pose_proposal()

    assert segments(window) == [(1, 1, LAST_STEP)]
    assert [k for k in window.db.keyframes(DESKTOP, VIEW) if k.source == "carried"] == []


# --------------------------------------------------------------------------- #
# the timeline mark
# --------------------------------------------------------------------------- #
def test_the_timeline_marks_every_accepted_break_of_this_view(window: MainWindow):
    window.session.goto(CUT, force=True)
    answer(window, carry=False)

    window.act_split_pose()

    assert window.timeline.break_steps() == [CUT]
    # a proposal is not a boundary and is not marked
    window.db.add_pose_break(DESKTOP, VIEW, 11, status="proposed", kind="camera",
                             source="audit:events.csv")
    window.render_frame()
    assert window.timeline.break_steps() == [CUT]


def test_another_views_breaks_are_not_marked(window: MainWindow):
    window.db.add_pose_break(DESKTOP, "oak1", 5, status="accepted", kind="camera",
                             source="audit:events.csv")

    window.render_frame()

    assert window.timeline.break_steps() == []


# --------------------------------------------------------------------------- #
# the dialog itself
# --------------------------------------------------------------------------- #
def test_the_dialog_assembles_with_and_without_images(qapp):
    rgb = np.zeros((16, 24, 3), dtype=np.uint8)
    dialog = PoseSplitDialog(None, 19, rgb, rgb, 3, True, note="tape displaced")
    assert dialog.carry() is True
    dialog.deleteLater()

    empty = PoseSplitDialog(None, 19, None, None, 0, False)
    assert empty.carry() is False
    empty.deleteLater()


def test_the_bar_text_names_the_step_and_the_size():
    text = proposal_text({"step": 19, "kind": "camera", "magnitude_px": 190.45})
    assert "19" in text and "190 px" in text
    plain = proposal_text({"step": 19, "kind": None, "magnitude_px": None})
    assert "19" in plain and "px" not in plain


def test_the_refusal_line_names_the_step():
    assert "7" in NO_SPLIT_HERE.format(step=7)
