"""U2e: every reason Space would refuse is on screen before Space is pressed.

The walk that made it likely: the annotator fixes a part with Enter on frame
j, the shape reaches a frame already confirmed, its re-check opens a
conflict -- and that frame then said "按 Space" while Space refused.  The four
reasons of ``verify_frame`` come from one function now
(:func:`tda.core.truth_refusals.blocking_reasons`); these tests walk each of
them through the window: the card's pane, its header and ✔ row, the guide,
and the refusal Space then gives.

Also here: the scope bar's Alt+Enter button, the ROI "box ready" wording and
the ✔ row's tooltip.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    cell,
    close_window,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core import masks
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, StepType
from tda.core.truth_refusals import (
    CONFLICT,
    FROZEN,
    RACE,
    VerifyRefused,
    blocking_reasons,
)
from tda.ui import app_actions as A
from tda.ui import guide as G
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.panels.taskcard import CONFLICT_SENTENCE, RACE_SENTENCE, VIEW_ROLE

#: Typed ``dupli``, step 5's card is one ✔ row and its header "这一帧不用画".
STEP = 5


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def open_window(tmp_path: Path, session) -> MainWindow:
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    win.resize(1400, 900)
    return win


def answer_roi(win: MainWindow) -> None:
    if win.roi_editing:
        win.wait_for_roi_proposal()
        win.act_commit()
    if win.roi_editing:
        win.act_clear_edit()


def draw(session, instance: str, index: int, scope: str = api.SCOPE_KEYFRAME) -> None:
    session.begin_edit(instance)
    session.set_editing_mask(cell(index))
    session.commit_edit(scope)
    session.clear_edit()


def confirm_row_frame(tmp_path: Path, skip=()):
    """Step 5 typed ``dupli``, every part drawn but ``skip``."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False           # the re-checks run where a test says
    db = session.db
    steps = db.steps(DESKTOP)
    for rec in steps:
        if rec.step == STEP:
            rec.step_type = StepType.DUPLI.value
    db.replace_steps(DESKTOP, steps, db.actions(DESKTOP))
    session.open(DESKTOP, VIEW)
    session.sweeper_enabled = False
    seed_shapes(session, STEP, skip=skip)
    session.goto(STEP + 1, force=True)
    session.goto(STEP, force=True)
    assert session.current_problems() == [f"missing_shape:{p}" for p in skip]
    return session


def confirmed_then_changed(tmp_path: Path):
    """The realistic walk: confirm step 5, then fix the chassis with Enter on
    step 4 -- its keyframe reaches step 5 -- and let the re-check run."""
    session = confirm_row_frame(tmp_path)
    assert session.confirm_frame() is True            # step 5 confirmed; now on 4
    draw(session, "chassis", 50)                      # Enter on step 4
    assert STEP in session.db.rechecks(DESKTOP, VIEW)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    return session


def surfaces(win: MainWindow) -> dict:
    """What the frame on screen says about Space, everywhere it says it."""
    QApplication.processEvents()
    lw = win.task_card.list_widget()
    views = [lw.item(i).data(VIEW_ROLE) for i in range(lw.count())]
    return {"header": win.task_card.header_text(),
            "rows": [v["sentence"] for v in views],
            "tips": [lw.item(i).toolTip() for i in range(lw.count())],
            "plan": win.guide_plan(),
            "pane": [r["code"] for r in win.task_card.problem_rows()],
            "count": win.task_card.problem_count()}


def promises_space(text: str) -> bool:
    return "直接 Space" in text or "按 Space" in text


def no_space_promise(seen: dict) -> None:
    n = seen["count"]
    assert n > 0
    blocked = f"但下面还有 {n} 个问题挡住 Space（见下方）"
    assert seen["header"].endswith(blocked), seen["header"]
    assert seen["rows"] == [f"这一帧不用画，{blocked}"], seen["rows"]
    plan = seen["plan"]
    assert plan.phase == G.PHASE_BLOCKED and plan.action == ""
    for text in [seen["header"], plan.now, *seen["rows"], *seen["tips"],
                 *(t for _s, t in plan.steps)]:
        assert not promises_space(text), text
        assert "confirm the frame" not in text, text


# --------------------------------------------------------------------------- #
# the one function
# --------------------------------------------------------------------------- #
def test_a_frozen_disagreement_is_a_reason_until_its_conflict_is_queued(qapp, tmp_path):
    session = confirm_row_frame(tmp_path)
    assert session.confirm_frame() is True
    draw(session, "chassis", 50)                      # step 5's chassis moved
    key = FrameKey(DESKTOP, STEP, VIEW)
    compiled = session.truth.compile(key)
    reasons = blocking_reasons(session.db, key, compiled)
    assert [(b.kind, b.code) for b in reasons] == [(FROZEN, "frozen_disagreement:chassis")]

    # verify_frame reads the same list, refuses for it, and queues the conflict
    with pytest.raises(VerifyRefused) as refused:
        session.truth.verify_frame(key, "tester", compiled)
    assert [b.kind for b in refused.value.blockers] == [FROZEN]
    after = blocking_reasons(session.db, key, compiled)
    assert [b.kind for b in after] == [CONFLICT], "named once: as the conflict"
    assert after[0].code.startswith("open_conflict:") and after[0].instance == "chassis"
    with pytest.raises(VerifyRefused) as again:
        session.truth.verify_frame(key, "tester", compiled)
    assert again.value.blockers == after


# --------------------------------------------------------------------------- #
# an open conflict: shown on arrival, clicked to Review, settled there
# --------------------------------------------------------------------------- #
def test_an_open_conflict_is_on_screen_before_space(qapp, tmp_path):
    session = confirmed_then_changed(tmp_path)
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        session.goto(STEP, force=True)                # arrive at the confirmed frame
        seen = surfaces(win)
        cid, _inst = session.db.open_conflicts_at(FrameKey(DESKTOP, STEP, VIEW))[0]
        assert seen["pane"] == [f"open_conflict:{cid}/chassis"]
        line = win.task_card.problem_rows()[0]["text"]
        assert line.startswith(CONFLICT_SENTENCE) and f"#{cid}" in line
        no_space_promise(seen)
        assert "单击一条去处理" in seen["plan"].now          # the line is clickable
        assert "1 个问题要先处理 — 见任务卡" in win.status_message()

        assert win.act_confirm() is False                   # ... and Space refuses
        assert f"conflict(s) {cid} are still open" in win.task_card.problems()[0]
        assert f"open_conflict:{cid}/chassis" in [r["code"] for r in
                                                   win.task_card.problem_rows()]
        no_space_promise(surfaces(win))
    finally:
        close_window(win)


def test_a_click_on_the_conflict_opens_review_on_it_and_the_verdict_gives_space_back(
        qapp, tmp_path):
    session = confirmed_then_changed(tmp_path)
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        session.goto(STEP, force=True)
        QApplication.processEvents()
        cid, _inst = session.db.open_conflicts_at(FrameKey(DESKTOP, STEP, VIEW))[0]
        lw = win.task_card._problems_list
        win.task_card._problems_list.itemClicked.emit(lw.item(0))
        QApplication.processEvents()
        assert win.mode == A.MODE_REVIEW
        assert win.review.selected_conflict() == cid
        assert f"冲突 #{cid} 已选中" in win.status_message()

        win.dispatch(A.action_named("review_accept_new"))    # N: take the new shape
        QApplication.processEvents()
        assert session.db.open_conflicts_at(FrameKey(DESKTOP, STEP, VIEW)) == []
        win.set_mode(A.MODE_ANNOTATE)
        seen = surfaces(win)
        assert seen["pane"] == [] and seen["count"] == 0
        assert seen["header"].endswith("直接 Space"), seen["header"]
        assert seen["rows"] == ["这一帧不用画，直接 Space"]
        # taking the new shape re-froze the frame: the guide calls it confirmed,
        # and Space (which would refuse a blocked frame) goes through
        assert seen["plan"].phase == G.PHASE_CONFIRMED, seen["plan"]
        assert "挡住" not in seen["plan"].now
        assert win.act_confirm() is True
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# a frozen disagreement not queued yet
# --------------------------------------------------------------------------- #
def test_a_frozen_disagreement_is_on_screen_and_space_turns_it_into_a_conflict(
        qapp, tmp_path):
    """Standing on the confirmed frame while its inputs move under it -- here a
    write the session did not make -- the next announce shows the part."""
    session = confirm_row_frame(tmp_path)
    assert session.confirm_frame() is True
    session.goto(STEP, force=True)                    # standing on the confirmed frame
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        # another writer re-traces the chassis: no refresh, no re-check
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance="chassis", desktop=DESKTOP, view=VIEW, pose_segment=1,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask", version=9,
            parts=[ShapePart("main", masks.encode_rle(cell(50)))]))
        session._invalidate()
        session._announce()                           # what any edit ends with
        seen = surfaces(win)
        assert seen["pane"] == ["frozen_disagreement:chassis"]
        assert "按 F5" in win.task_card.problem_rows()[0]["text"]
        no_space_promise(seen)
        assert "单击一条去处理" in seen["plan"].now

        assert win.act_confirm() is False             # Space refuses and queues it
        codes = [r["code"] for r in win.task_card.problem_rows()]
        assert any(c.startswith("open_conflict:") for c in codes), codes
        assert "frozen_disagreement:chassis" not in codes, "named once, as the conflict"
        no_space_promise(surfaces(win))
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# blocking compiler codes (the U2d reason, through the same function)
# --------------------------------------------------------------------------- #
def test_a_blocking_code_is_on_screen_before_space(qapp, tmp_path):
    session = confirm_row_frame(tmp_path, skip=("connector.01",))   # not a ✔-frame row
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        seen = surfaces(win)
        assert seen["pane"] == ["missing_shape:connector.01"]
        no_space_promise(seen)
        assert win.act_confirm() is False
        assert "missing_shape:connector.01" in win.task_card.problems()[0]
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# the input race: only a refusal knows it, and it says to press Space again
# --------------------------------------------------------------------------- #
def test_an_input_race_says_press_space_again(qapp, tmp_path, monkeypatch):
    session = confirm_row_frame(tmp_path)
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        assert surfaces(win)["plan"].phase == G.PHASE_CONFIRM   # nothing to see coming
        real = session.truth.inputs_digest
        monkeypatch.setattr(session.truth, "inputs_digest",
                            lambda key, cache=None: "moved while confirming")
        assert win.act_confirm() is False
        rows = win.task_card.problem_rows()
        assert [r["text"] for r in rows if r["code"] == "inputs_changed:"] == [RACE_SENTENCE]
        seen = surfaces(win)
        assert seen["count"] == 1
        assert seen["plan"].now == f"现在：{RACE_SENTENCE}"
        assert "单击一条去处理" not in seen["plan"].now
        assert seen["header"].endswith("但下面还有 1 个问题挡住 Space（见下方）")

        monkeypatch.setattr(session.truth, "inputs_digest", real)
        assert win.act_confirm() is True               # ... and it was transient
    finally:
        close_window(win)


def test_the_race_blocker_is_the_refusal_verify_raises(qapp, tmp_path, monkeypatch):
    session = confirm_row_frame(tmp_path)
    monkeypatch.setattr(session.truth, "inputs_digest", lambda key, cache=None: "x")
    with pytest.raises(VerifyRefused) as refused:
        session.truth.verify_frame(session.current(), "tester", session.prepared())
    assert [b.kind for b in refused.value.blockers] == [RACE]
    assert "press Space again" in str(refused.value)
