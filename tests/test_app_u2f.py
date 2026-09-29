"""U2f: the two review minors that came with the fast comparison.

* **a** -- arriving at a frame, the session reuses the compilation it holds only
  when it is still what the frame's inputs make (``usable``), not because its
  own edit epoch has not moved: another writer moves the inputs without
  touching it.
* **b** -- the task card's "已经确认 ✓" follows the frame on screen when a
  re-check demotes it, from the queue signal the window already gets, without
  rebuilding the card.

The comparison itself is ``tests/test_disagreement_window.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import DESKTOP, LAST_STEP, VIEW, cell, close_window
from test_app_u2e import (
    KEY5,
    STEP,
    answer_roi,
    confirm_row_frame,
    confirmed_five,
    no_space_promise,
    open_window,
    surfaces,
)
from tda.core import masks
from tda.core.model import ShapeKeyframe, ShapePart, ZOrderRec
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.panels.taskcard import CONFIRMED_TAIL, CONFIRMED_TEXT


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def compiles_of(monkeypatch, key) -> list:
    """Every compilation of ``key`` made on this thread from now on."""
    import tda.core.truth as truth_mod

    made: list = []
    real = truth_mod.compile_frame
    here = threading.current_thread()

    def counted(*a, **k):
        if threading.current_thread() is here and a[0] == key:
            made.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(truth_mod, "compile_frame", counted)
    return made


# --------------------------------------------------------------------------- #
# a: a held compilation is reused only while it is still the answer
# --------------------------------------------------------------------------- #
def test_arrival_reuses_the_held_compilation_only_while_the_inputs_make_it(
        qapp, tmp_path, monkeypatch):
    session = confirmed_five(tmp_path)
    try:
        # behind but agreeing: the layer order moved, no pixel did
        session.set_zorder_move("connector.01", "connector.02")
        session.truth.run_pending_rechecks(DESKTOP, VIEW)
        session.goto(STEP, force=True)
        held = session.compiled()
        made = compiles_of(monkeypatch, KEY5)

        # steady state: arriving again compiles nothing and keeps what it holds
        session.goto(STEP + 1, force=True)
        session.goto(STEP, force=True)
        assert made == [] and session.compiled() is held

        # another writer -- a second window, `cli check`, a repair script --
        # moves the layer order again and brings the rows up to date itself:
        # the digest is current, and the session's epoch never moved
        zorder = session.db.zorder(DESKTOP, VIEW, 1)
        order = list(zorder.order)
        order[0], order[-1] = order[-1], order[0]      # distinct cells: no pixel moves
        session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, order,
                                        version=zorder.version + 1))
        assert session.truth.refresh(KEY5)["standing"] == 0
        assert session.truth._digest_is_current(KEY5, session.truth.inputs_digest(KEY5))

        session.goto(STEP + 1, force=True)
        made.clear()
        session.goto(STEP, force=True)
        assert made == [KEY5], "the held compilation is not these inputs' any more"
        fresh = session.truth.compile(KEY5)
        assert fresh.input_hash != held.input_hash
        assert session.compiled() is not held, "a stale compilation was reused"
        assert session.compiled().input_hash == fresh.input_hash
    finally:
        session.close(force=True)


# --------------------------------------------------------------------------- #
# b: "已经确认 ✓" follows the frame on screen
# --------------------------------------------------------------------------- #
@pytest.fixture
def confirmed_on_screen(qapp, tmp_path):
    session = confirm_row_frame(tmp_path)
    assert session.confirm_frame() is True
    win = open_window(tmp_path, session)
    answer_roi(win)
    session.goto(STEP, force=True)
    # the arrival's difference map lands on its own; let it, before counting
    assert win.assist.wait(10.0), "the difference map did not finish"
    QApplication.processEvents()
    seen = surfaces(win)
    assert seen["header"] == f"第 {STEP} 帧：{CONFIRMED_TEXT}"
    assert seen["plan"].phase == G.PHASE_CONFIRMED
    yield session, win
    close_window(win)


def _count(monkeypatch, obj, name) -> list:
    calls: list = []
    real = getattr(obj, name)
    monkeypatch.setattr(obj, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _recheck_queues_a_conflict(session) -> None:
    """What the sweeper does to the frame on screen: the chassis moved elsewhere.

    A new version of its shape reaches step 5 (written the way an edit on
    another frame, or another writer, writes it), and the re-check of step 5
    queues the disagreement -- no frame change, no announce.
    """
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance="chassis", desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask", version=9,
        parts=[ShapePart("main", masks.encode_rle(cell(60)))]))
    assert session.truth.refresh(KEY5)["conflicts"] == 1


@pytest.mark.parametrize("verdict", ["demoted", "conflict"])
def test_the_card_stops_saying_confirmed_when_a_recheck_moves_the_frame_on_screen(
        confirmed_on_screen, monkeypatch, verdict):
    """The sweeper's verdict lands while the annotator stands on the frame."""
    session, win = confirmed_on_screen
    rebuilt = _count(monkeypatch, session, "task_card")
    if verdict == "demoted":
        session.truth.demote_frame(KEY5, "the verified frame gained connector.99")
        expected = api.STATUS_NEEDS_REVIEW
    else:
        _recheck_queues_a_conflict(session)
        expected = api.STATUS_CONFLICT
    session._on_queues_changed()        # what the sweeper's queue signal runs
    QApplication.processEvents()

    assert rebuilt == [], "the card was rebuilt to learn one status"
    assert session.frame_status(STEP) == expected
    seen = surfaces(win)
    assert CONFIRMED_TEXT not in seen["header"], seen["header"]
    assert not any(CONFIRMED_TAIL in row for row in seen["rows"]), seen["rows"]
    assert seen["plan"].phase != G.PHASE_CONFIRMED
    assert expected in win.frame_label.text()
    if verdict == "conflict":
        # Space refuses now, and everything on screen says why instead of "Space"
        assert [c for c in seen["pane"] if c.startswith("open_conflict:")], seen["pane"]
        no_space_promise(seen)
        assert not session.confirm_frame()


def test_a_second_conflict_on_a_frame_already_in_conflict_reaches_the_pane(
        confirmed_on_screen, monkeypatch):
    """U2h: the status stays ``conflict``; the number of open conflicts moves.

    The pane used to learn of the second one only at the next announce.
    """
    from app_scene import chassis_instances

    session, win = confirmed_on_screen
    _recheck_queues_a_conflict(session)
    session._on_queues_changed()
    QApplication.processEvents()
    assert session.frame_status(STEP) == api.STATUS_CONFLICT
    first = [c for c in surfaces(win)["pane"] if c.startswith("open_conflict:")]
    assert len(first) == 1, first

    rebuilt = _count(monkeypatch, session, "task_card")
    follows = _count(monkeypatch, win.task_card, "follow_status")
    other = next(k for k in chassis_instances(session, STEP) if k != "chassis")
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=other, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask", version=9,
        parts=[ShapePart("main", masks.encode_rle(cell(61)))]))
    assert session.truth.refresh(KEY5)["conflicts"] == 1      # the second one
    queries = _count(monkeypatch, session.db, "conflicts")
    session._on_queues_changed()
    QApplication.processEvents()

    assert session.frame_status(STEP) == api.STATUS_CONFLICT   # unchanged
    seen = surfaces(win)
    now = [c for c in seen["pane"] if c.startswith("open_conflict:")]
    assert len(now) == 2, now
    no_space_promise(seen)
    assert follows == [1], "the second conflict was not followed, or followed twice"
    assert rebuilt == [], "the card was rebuilt to learn one number"
    # one read of the view's conflicts for the review panel's queues and one
    # for the memo the timeline's statuses and this count share -- nothing
    # per frame, nothing for the count itself
    assert len(queries) <= 2, len(queries)

    # the same queue signal again: nothing moved, nothing is repainted
    follows.clear()
    session._on_queues_changed()
    QApplication.processEvents()
    assert follows == []


def test_a_queue_change_that_leaves_the_status_alone_repaints_nothing(
        confirmed_on_screen, monkeypatch):
    session, win = confirmed_on_screen
    statuses = _count(monkeypatch, win, "update_status")
    follows = _count(monkeypatch, win.task_card, "follow_status")
    session._on_queues_changed()
    QApplication.processEvents()
    assert statuses == [] and follows == []
    assert surfaces(win)["header"] == f"第 {STEP} 帧：{CONFIRMED_TEXT}"


def test_the_card_says_confirmed_again_when_the_frame_is(confirmed_on_screen):
    """``follow_status`` goes both ways: the same rule as a rebuild."""
    session, win = confirmed_on_screen
    card = win.task_card
    before = (card.header_text(), card.row_texts())
    card.follow_status(api.STATUS_NEEDS_REVIEW)
    assert CONFIRMED_TEXT not in card.header_text()
    card.follow_status(api.STATUS_RECHECK)             # confirmed, re-check pending
    assert (card.header_text(), card.row_texts()) == before
    card.refresh()
    assert (card.header_text(), card.row_texts()) == before
