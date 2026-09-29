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

import numpy as np
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
from tda.core.model import FrameKey, FrameOverride, ShapeKeyframe, ShapePart, StepType
from tda.core.truth_refusals import (
    CONFLICT,
    FROZEN,
    PROBLEM,
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


def thorough(session, key, compiled) -> list:
    """What ``verify_frame`` refuses for (its mode of the one function)."""
    return blocking_reasons(session.db, key, lambda: (compiled, None, None), thorough=True)


def shown(session, key, compiled) -> list:
    """What the window lists (the display's mode, digest-gated)."""
    return blocking_reasons(
        session.db, key, lambda: (compiled, None, None), thorough=False,
        ruled_out=lambda made: session.truth.disagreement_ruled_out(key, made))


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
    # the ✔ row's tooltip: Chinese first, and the row's own sentence (item 4)
    for tip, row in zip(seen["tips"], seen["rows"]):
        assert tip.startswith(row), (tip, row)
        assert f"{n} problem(s) below stop Space" in tip


# --------------------------------------------------------------------------- #
# the one function
# --------------------------------------------------------------------------- #
def test_a_frozen_disagreement_is_a_reason_until_its_conflict_is_queued(qapp, tmp_path):
    session = confirm_row_frame(tmp_path)
    assert session.confirm_frame() is True
    draw(session, "chassis", 50)                      # step 5's chassis moved
    key = FrameKey(DESKTOP, STEP, VIEW)
    compiled = session.truth.compile(key)
    reasons = thorough(session, key, compiled)
    assert [(b.kind, b.code) for b in reasons] == [(FROZEN, "frozen_disagreement:chassis")]
    assert shown(session, key, compiled) == reasons    # the display agrees

    # verify_frame reads the same list, refuses for it, and queues the conflict
    with pytest.raises(VerifyRefused) as refused:
        session.truth.verify_frame(key, "tester", compiled)
    assert refused.value.blockers == reasons
    after = thorough(session, key, compiled)
    assert [b.kind for b in after] == [CONFLICT]
    assert after[0].code.startswith("open_conflict:") and after[0].instance == "chassis"
    assert shown(session, key, compiled) == after, "named once: as the conflict"
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
        assert seen["tips"][0].startswith("这一帧不用画，直接 Space")
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
        # the refusal's own reasons, carried rather than looked for again
        assert "frozen_disagreement:chassis" in codes, codes
        assert "now in the review queue" in win.task_card.problems()[0]
        no_space_promise(surfaces(win))
        session._announce()                           # the next arrival or edit
        codes = [r["code"] for r in win.task_card.problem_rows()]
        assert [c for c in codes if c.startswith("open_conflict:")] and \
            "frozen_disagreement:chassis" not in codes, "named once, as the conflict"
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


# --------------------------------------------------------------------------- #
# item 2: the scope bar's 仅本帧 is greyed like the palette's, and never errors
# --------------------------------------------------------------------------- #
def scope_bar_over(win: MainWindow, instance: str) -> None:
    """Pixels painted on ``instance`` and a layering suggestion on the bar."""
    win.on_request_edit(instance)
    win.session.set_editing_mask(cell(40))
    win._sync_editing_layer()
    win._note_edit_facts()
    win._offer_scope("zorder:above:chassis")
    QApplication.processEvents()
    assert win.scope_bar.isVisibleTo(win) and win.layer_facts()[2]


def test_the_scope_bars_alt_enter_is_greyed_with_the_reason(qapp, tmp_path):
    session = confirm_row_frame(tmp_path, skip=("connector.01",))  # no shape here
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        scope_bar_over(win, "connector.01")
        button = win.scope_override_button
        assert not button.isEnabled()
        assert button.toolTip() == api.OVERRIDE_NEEDS_SHAPE
        assert button.toolTip() == win.palette.button("commit_override").reason()

        errors = win.last_error_message()
        ops = len(session.db.ops(DESKTOP, VIEW, limit=10_000))
        win._scope_bar_override()                 # what a click runs, were it live
        assert api.OVERRIDE_NEEDS_SHAPE in win.status_message()
        assert win.last_error_message() == errors, "said as an error"
        assert "refused" not in win.status_message()
        assert len(session.db.ops(DESKTOP, VIEW, limit=10_000)) == ops
        assert session.editing_instance == "connector.01"
    finally:
        close_window(win)


def test_the_scope_bars_alt_enter_is_live_on_a_part_with_a_shape(qapp, tmp_path):
    session = confirm_row_frame(tmp_path)
    win = open_window(tmp_path, session)
    try:
        answer_roi(win)
        scope_bar_over(win, "connector.01")
        assert win.scope_override_button.isEnabled()
        assert win.scope_override_button.toolTip() == ""
    finally:
        close_window(win)


# --------------------------------------------------------------------------- #
# item 3: "框好了" names Esc the way the bar does
# --------------------------------------------------------------------------- #
def test_box_ready_says_esc_as_the_bar_does(qapp, tmp_path):
    from tda.ui.app_roi import (
        ROI_BAR_KEYS,
        ROI_BAR_KEYS_STORED,
        ROI_BOX_READY,
        ROI_BOX_READY_STORED,
    )

    win = open_window(tmp_path, make_session(tmp_path))
    try:
        win.wait_for_roi_proposal()
        assert win.roi_editing and win.roi() is None
        win.on_roi_box((8.0, 8.0, 56.0, 56.0))
        assert win.status_message() == ROI_BOX_READY
        assert "Esc 先跳过" in ROI_BOX_READY and "Esc 先跳过" in ROI_BAR_KEYS
        win.act_commit()                          # stored
        win.act_edit_roi()                        # Shift+R: over the stored one
        win.on_roi_box((6.0, 6.0, 58.0, 58.0))
        assert win.status_message() == ROI_BOX_READY_STORED
        assert "Esc 不改" in ROI_BOX_READY_STORED and "Esc 不改" in ROI_BAR_KEYS_STORED
    finally:
        close_window(win)


# =========================================================================== #
# round 2: the display's mode and verify's mode of the one function agree
# =========================================================================== #
MOVED, DEPARTED, LABELLED, BROKEN = "chassis", "connector.06", "connector.03", "connector.04"
KEY5 = FrameKey(DESKTOP, STEP, VIEW)


def confirmed_five(tmp_path: Path, confirm: bool = True, skip=()):
    """Step 5, every part drawn (but ``skip``), confirmed; the session on step 4."""
    session = make_session(tmp_path)
    session.sweeper_enabled = False           # re-checks run where a scenario says
    seed_shapes(session, STEP, skip=skip)
    session.goto(STEP + 1, force=True)
    session.goto(STEP, force=True)
    if confirm:
        assert session.confirm_frame() is True
        assert session.current().step == STEP - 1
    return session


def retrace(session, instance: str, index: int) -> None:
    """Enter on step 4: the keyframe in force reaches step 5."""
    draw(session, instance, index)


def with_conflict(session) -> int:
    retrace(session, MOVED, 50)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    (cid, _inst), = session.db.open_conflicts_at(KEY5)
    return cid


def wrong_size(session, instance: str) -> None:
    """A keyframe of the wrong canvas size: ``shape_size_mismatch`` (blocking)."""
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=instance, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask", version=9,
        parts=[ShapePart("main", masks.encode_rle(np.ones((32, 32), dtype=bool)))]))
    session._invalidate()


def s_clean(tmp_path):
    return confirmed_five(tmp_path, confirm=False)


def s_code(tmp_path):
    return confirmed_five(tmp_path, confirm=False, skip=(DEPARTED,))


def s_behind_agreeing(tmp_path):
    session = confirmed_five(tmp_path)
    # the layer order moves (a new version), the pixels do not: the re-check
    # agrees and leaves step 5's rows with their old input_hash on purpose
    session.set_zorder_move("connector.01", "connector.02")
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    return session


def s_conflict(tmp_path):
    session = confirmed_five(tmp_path)
    with_conflict(session)
    return session


def s_conflict_code_frozen_other(tmp_path):
    session = confirmed_five(tmp_path)
    with_conflict(session)
    wrong_size(session, BROKEN)                # a blocking code (its empty shape is
    retrace(session, LABELLED, 52)             # within the re-trace tolerance) ...
    return session                             # ... and another part moved


def s_conflict_frozen_same(tmp_path):
    session = confirmed_five(tmp_path)
    with_conflict(session)
    retrace(session, MOVED, 51)                # moved again, not re-checked
    return session


def s_conflict_frozen_other(tmp_path):
    session = confirmed_five(tmp_path)
    with_conflict(session)
    retrace(session, LABELLED, 52)             # another part moved, not re-checked
    return session


def s_frozen_alone(tmp_path):
    session = confirmed_five(tmp_path)
    retrace(session, MOVED, 50)                # nobody has looked since
    return session


def s_departed_and_moved(tmp_path):
    session = confirmed_five(tmp_path)
    session.db.delete_instance(DESKTOP, DEPARTED)   # the part leaves the frame
    retrace(session, MOVED, 50)
    return session


def s_label_only(tmp_path):
    session = confirmed_five(tmp_path)
    session.db.set_frame_override(FrameOverride(KEY5, LABELLED, None, "occluded_partial"))
    session._invalidate()
    return session


def s_keep_old(tmp_path):
    session = confirmed_five(tmp_path)
    cid = with_conflict(session)
    session.resolve_conflict(cid, api.RESOLVE_KEEP_OLD)
    return session


def s_accept_new(tmp_path):
    session = confirmed_five(tmp_path)
    cid = with_conflict(session)
    session.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW)
    return session


#: What each scenario shows before anybody looks: ``(kind, instance)`` per line.
SCENARIOS = {
    "clean": (s_clean, []),
    "code": (s_code, [(PROBLEM, "")]),
    "behind_agreeing": (s_behind_agreeing, []),
    "conflict": (s_conflict, [(CONFLICT, MOVED)]),
    "conflict_code_frozen_other": (s_conflict_code_frozen_other,
                                   [(CONFLICT, MOVED), (PROBLEM, ""), (FROZEN, LABELLED)]),
    "conflict_frozen_same": (s_conflict_frozen_same, [(CONFLICT, MOVED)]),
    "conflict_frozen_other": (s_conflict_frozen_other,
                              [(CONFLICT, MOVED), (FROZEN, LABELLED)]),
    "frozen_alone": (s_frozen_alone, [(FROZEN, MOVED)]),
    "departed_and_moved": (s_departed_and_moved, [(FROZEN, DEPARTED), (FROZEN, MOVED)]),
    "label_only": (s_label_only, [(FROZEN, LABELLED)]),
    "keep_old": (s_keep_old, []),
    "accept_new": (s_accept_new, []),
}


def modes_agree(session, compiled) -> tuple[list, list]:
    """verify's answer is the display's first category; both are empty together."""
    th = thorough(session, KEY5, compiled)
    sh = shown(session, KEY5, compiled)
    if not th:
        assert sh == [], sh
    else:
        assert [b for b in sh if b.kind == th[0].kind] == th, (th, sh)
    return th, sh


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_display_and_verify_modes_agree(qapp, tmp_path, name):
    build, expected = SCENARIOS[name]
    session = build(tmp_path)
    try:
        # as the inputs stand, before anybody looks
        th, sh = modes_agree(session, session.truth.compile(KEY5))
        assert [(b.kind, b.instance) for b in sh] == expected, sh
        # arriving: the visit refresh runs first, then the frame is announced
        session.goto(STEP, force=True)
        th, sh = modes_agree(session, session.compiled())
        assert {b.code for b in sh if b.kind != PROBLEM} <= set(session.current_problems())
        # Space: verify_frame's own refusal is the thorough answer, word for word
        if th:
            with pytest.raises(VerifyRefused) as refused:
                session.truth.verify_frame(KEY5, "tester", session.prepared())
            assert refused.value.blockers == th
        else:
            session.truth.verify_frame(KEY5, "tester", session.prepared())
    finally:
        session.close(force=True)


def test_departed_parts_are_queued_first_as_verify_always_did(qapp, tmp_path):
    session = s_departed_and_moved(tmp_path)
    try:
        with pytest.raises(VerifyRefused):
            session.truth.verify_frame(KEY5, "tester")
        queued = [inst for _cid, inst in session.db.open_conflicts_at(KEY5)]
        assert queued == [DEPARTED, MOVED]
    finally:
        session.close(force=True)


def test_a_behind_but_agreeing_frame_is_not_decoded_on_arrival(qapp, tmp_path, monkeypatch):
    """The reviewer's case: the digest rules it out, nothing is compared."""
    import tda.core.truth_refusals as refusals

    session = s_behind_agreeing(tmp_path)
    try:
        session.goto(STEP, force=True)                 # the visit refresh stamps it
        assert session.db.frozen_rows_behind(KEY5, session.compiled().input_hash)
        assert session.truth.disagreement_ruled_out(KEY5, session.compiled())
        compared: list = []
        real = refusals.frozen_disagreements
        monkeypatch.setattr(refusals, "frozen_disagreements",
                            lambda *a, **k: compared.append(a[1]) or real(*a, **k))
        session.goto(STEP + 1, force=True)
        session.goto(STEP, force=True)                 # steady state: the second arrival
        assert compared == [], "the display decoded a frame its digest rules out"
        session.truth.verify_frame(KEY5, "tester", session.prepared())
        assert compared, "verify_frame compares, whatever the digest says"
    finally:
        session.close(force=True)


def test_what_a_refresh_just_compared_is_not_compared_again(qapp, tmp_path, monkeypatch):
    """Enter on a confirmed frame: the commit's own refresh compared every
    frozen row and queued the disagreement; the announce that follows shows
    the conflict without decoding a row (at 12 MP that doubled the commit)."""
    import tda.core.truth_refusals as refusals

    session = confirmed_five(tmp_path)
    try:
        session.goto(STEP, force=True)                 # standing on the confirmed frame
        compared: list = []
        real = refusals.frozen_disagreements
        monkeypatch.setattr(refusals, "frozen_disagreements",
                            lambda *a, **k: compared.append(a[1]) or real(*a, **k))
        retrace(session, MOVED, 50)                    # Enter here: a conflict is queued
        codes = session.current_problems()
        assert [c for c in codes if c.startswith("open_conflict:")], codes
        assert session.db.frame_digest(KEY5) is None, "the digest cannot help here"
        assert compared == [], "the announce compared what the refresh just had"
        # anything that moves the inputs outside the session's own writes does
        session._invalidate()
        session._announce()
        assert compared, "a new epoch is compared again"
    finally:
        session.close(force=True)


def test_the_shortcut_rests_on_the_digest_invariant_and_space_still_holds(qapp, tmp_path):
    """Break the invariant on purpose: a writer that changes a confirmed row
    behind the truth service's back, leaving the digest current.  The display
    then misses the disagreement -- that is the one assumption it makes -- and
    Space still refuses, because verify never takes the shortcut."""
    session = s_behind_agreeing(tmp_path)
    try:
        session.goto(STEP, force=True)
        row = session.db.compiled(KEY5)[MOVED]
        other = masks.encode_rle(cell(60))
        session.db.put_compiled(KEY5, MOVED, other, 0.0, row["visibility"],
                                row["placement"], "verified", row["input_hash"],
                                verified_by="somebody else")
        compiled = session.compiled()
        assert session.truth.disagreement_ruled_out(KEY5, compiled), \
            "the digest is current although a disagreement now stands"
        assert shown(session, KEY5, compiled) == []            # the blind spot
        assert [b.kind for b in thorough(session, KEY5, compiled)] == [FROZEN]
        with pytest.raises(VerifyRefused):
            session.truth.verify_frame(KEY5, "tester", session.prepared())
    finally:
        session.close(force=True)
