"""Lazy truth, the background re-check of frozen frames, and the time budgets.

Annotating means one commit every few seconds, so a commit that recompiles the
whole interval of a chassis-sized shape at 1600x1600 -- 13 s measured, ~40 s on
a 120-step desktop -- is not a slow tool, it is an unusable one.  The rules this
module pins down (spec 3.4 read as "the compiled rows of an *unverified* frame
are a cache"):

* a commit compiles the frame the annotator is looking at, and nothing else;
* every other unverified frame in the interval is compiled when it is visited,
  or by a batch sweep before an export;
* every *verified* frame in the interval must still be re-checked, because that
  is where conflicts come from -- but off the GUI thread, and the request
  survives a crash;
* a frame whose re-check is still pending reads as ``recheck``.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import statistics
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from session_scene import (
    ANNOTATOR,
    BOARD_STEP,
    CHASSIS,
    COOLER,
    DESKTOP,
    LAST_STEP,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
    seed_shapes,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def as_shipped(monkeypatch):
    """Measure what the annotator runs, not what the test suite runs.

    ``tests/conftest.py`` turns :data:`tda.core.masks.CHECK_ENCODE_WINDOW` on
    for the whole session, which is right -- it guards every windowed encode in
    the codebase on every run -- and it costs four ``any`` passes over the
    canvas outside each window. On a 42-instance 12 MP commit that is 289 ms of
    checking (274 ms -> 563 ms measured), and a budget is a promise about the
    shipped configuration, where the check is off.
    """
    from tda.core import masks as masks_mod

    monkeypatch.setattr(masks_mod, "CHECK_ENCODE_WINDOW", False)


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    yield made
    made.close(force=True)  # the worker thread must never outlive the object it signals


def compiled_rows(session, step: int) -> dict:
    return session.db.compiled(FrameKey(DESKTOP, step, VIEW))


# --------------------------------------------------------------------------- #
# a commit costs one frame (ruling a)
# --------------------------------------------------------------------------- #
def test_a_commit_compiles_only_the_frame_in_front_of_the_annotator(session):
    session.goto(10)
    result = draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)

    assert result["affected"] == list(range(1, 13))  # the reach is still reported
    assert result["compiled"] == [10]  # ... but only this frame was compiled
    assert COOLER in compiled_rows(session, 10)
    assert compiled_rows(session, 7) == {}


def test_a_commit_undo_and_redo_each_compile_the_frame_once(session, monkeypatch):
    """The refresh already made the frame; nothing may make it a second time."""
    session.sweeper_enabled = False  # the prefetch of k-1 is not GUI-thread work
    session.sweeper.stop()
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)

    import tda.core.truth as truth_mod

    original = truth_mod.compile_frame
    steps: list[int] = []

    def counted(key, *a, **k):
        steps.append(key.step)
        return original(key, *a, **k)

    monkeypatch.setattr(truth_mod, "compile_frame", counted)

    session.begin_edit(COOLER)
    session.set_editing_mask(cell(5))
    session.commit_edit(api.SCOPE_KEYFRAME)
    assert steps == [10], f"commit compiled {steps}"

    steps.clear()
    session.undo()
    assert steps == [10], f"undo compiled {steps}"

    steps.clear()
    session.redo()
    assert steps == [10], f"redo compiled {steps}"


def test_visiting_a_frame_compiles_it(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    assert compiled_rows(session, 7) == {}

    session.goto(7)
    assert COOLER in compiled_rows(session, 7)


def test_a_stale_frame_is_recompiled_on_the_next_visit(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.goto(7)
    session.goto(10)
    draw(session, COOLER, cell(5), api.SCOPE_KEYFRAME)

    session.goto(7)  # step 7's rows are from before the re-trace
    row = compiled_rows(session, 7)[COOLER]
    from tda.core import masks

    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(5))


def test_refresh_all_still_compiles_the_whole_view(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    session.refresh_all()
    assert COOLER in compiled_rows(session, 3)


def test_frame_status_does_not_depend_on_stored_rows(session):
    """An unlabeled frame stays unlabeled, and a drawn one reads auto, uncompiled."""
    assert session.frame_status(7) == api.STATUS_UNLABELED
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)

    assert compiled_rows(session, 7) == {}  # never compiled
    assert session.frame_status(7) == api.STATUS_AUTO  # ... and it still knows


def test_missing_shape_queue_needs_no_compilation(session):
    session.goto(10)
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)
    missing = session.queues()[api.QUEUE_MISSING_SHAPE]
    at_seven = {entry["instance"] for entry in missing if entry["step"] == 7}

    assert compiled_rows(session, 7) == {}
    assert CHASSIS in at_seven and COOLER not in at_seven


# --------------------------------------------------------------------------- #
# verified frames are re-checked in the background (ruling b)
# --------------------------------------------------------------------------- #
def _verify(session, step: int) -> None:
    session.goto(step)
    seed_shapes(session, step)
    assert session.confirm_frame() is True


def test_a_verified_frame_in_the_interval_is_queued_and_then_conflicts(session):
    _verify(session, 12)
    assert session.frame_status(12) == api.STATUS_VERIFIED

    session.goto(10)
    result = draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    assert 12 in result["rechecks"]
    assert session.frame_status(12) == api.STATUS_RECHECK
    assert session.db.rechecks(DESKTOP, VIEW) == [12]

    assert session.drain_sweeper(timeout=20.0) is True
    assert session.db.rechecks(DESKTOP, VIEW) == []
    assert session.frame_status(12) == api.STATUS_CONFLICT
    assert [e["step"] for e in session.queues()[api.QUEUE_CONFLICTS]] == [12]


def test_an_agreeing_recheck_leaves_the_frame_verified(session):
    session.goto(12)
    drawn = seed_shapes(session, 12)
    assert session.confirm_frame() is True

    # re-trace the cooler one pixel row shorter: a change, but inside the
    # re-tracing tolerance of spec 3.4, so the frozen frame still agrees
    nudged = cell(drawn.index(COOLER)).copy()
    nudged[np.nonzero(nudged)[0].max(), :] = False
    session.goto(10)
    result = draw(session, COOLER, nudged, api.SCOPE_KEYFRAME)
    assert 12 in result["rechecks"]

    assert session.drain_sweeper(timeout=20.0) is True
    assert session.queues()[api.QUEUE_CONFLICTS] == []
    assert session.frame_status(12) == api.STATUS_VERIFIED


def test_pending_rechecks_survive_close_and_open(qapp, tmp_path):
    session = make_session(tmp_path)
    session.sweeper_enabled = False  # nothing drains it while we look
    _verify(session, 12)
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    assert session.db.rechecks(DESKTOP, VIEW) == [12]
    session.close()

    again = AnnotationSession(session.db, session.tax, session.truth,
                              session.cache_dir, ANNOTATOR)
    again.open(DESKTOP, VIEW)
    assert again.frame_status(12) == api.STATUS_RECHECK
    assert again.drain_sweeper(timeout=20.0) is True
    assert again.frame_status(12) == api.STATUS_CONFLICT
    again.close()


def test_close_joins_the_sweeper_with_work_queued(session):
    _verify(session, 12)
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    assert session.sweeper.is_running is True

    session.close()  # must not hang and must not raise
    assert session.is_open is False
    # no thread may outlive the session: it holds a second connection to the
    # database and would go on writing behind a window that has closed
    assert session.sweeper.is_running is False


def test_a_prefetch_that_lands_after_an_edit_is_never_displayed(session):
    session.goto(11)  # queues a prefetch of step 10, compiled as it is now
    draw(session, COOLER, cell(0), api.SCOPE_KEYFRAME)  # ... and then it changes
    session.drain_sweeper(timeout=20.0)  # the stale compilation arrives here

    session.goto(10)
    visible = session.compiled().instances[COOLER].visible
    assert visible is not None and np.array_equal(visible, cell(0))


def test_a_bench_part_is_not_this_views_work_without_a_staging_area(session):
    """Spec 3.3 step 2: on a view with no bench ROI a bench part is not an instance."""
    session.goto(14)  # the cooler is out of the machine here
    missing = {e["instance"] for e in session.queues()[api.QUEUE_MISSING_SHAPE]
               if e["step"] == 14}

    assert CHASSIS in missing  # a chassis instance with no shape does block
    assert COOLER not in missing
    found = session.review.drawn()[14]
    assert COOLER not in found.bench_missing and COOLER not in found.missing


def test_the_sweeper_reports_progress(session):
    seen: list = []
    session.sigSweepProgress.connect(lambda done, total: seen.append((done, total)))
    _verify(session, 12)
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    session.drain_sweeper(timeout=20.0)
    QApplication.processEvents()
    assert seen and seen[-1][0] == seen[-1][1]


def test_pending_rechecks_are_visible_to_the_truth_service(session):
    _verify(session, 12)
    session.sweeper_enabled = False
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    assert session.truth.pending_rechecks(DESKTOP, VIEW) == [12]
    assert session.truth.run_pending_rechecks(DESKTOP, VIEW)["conflicts"] >= 1
    assert session.truth.pending_rechecks(DESKTOP, VIEW) == []


# --------------------------------------------------------------------------- #
# one refresh implementation (ruling e)
# --------------------------------------------------------------------------- #
def test_refresh_range_can_key_its_problems_by_step(qapp, tmp_path):
    session = make_session(tmp_path)
    stats = session.truth.refresh_range(DESKTOP, VIEW, [1, 2], per_step=True)
    assert set(stats["problems"]) == {1, 2}
    flat = session.truth.refresh_range(DESKTOP, VIEW, [1, 2])
    assert isinstance(flat["problems"], list)


# --------------------------------------------------------------------------- #
# the budgets (ruling f)
# --------------------------------------------------------------------------- #
#: How many times each budgeted operation is measured.  The budget is asserted
#: on the **best** of them: what is under test is what the code costs, and a run
#: that was descheduled while another suite had the machine measures the machine
#: instead.  A regression makes every one of the three slow, so best-of-3 is no
#: weaker a guard -- it only stops the test failing for somebody else's CPU.
BEST_OF = 3


def best_of(measure, setup=None, times: int = BEST_OF) -> tuple[float, list[float]]:
    """``(best, every run)`` for one budgeted operation."""
    runs: list[float] = []
    for attempt in range(times):
        if setup is not None:
            setup(attempt)
        started = time.perf_counter()
        measure(attempt)
        runs.append(time.perf_counter() - started)
    return min(runs), runs


def _under(budget: float, what: str, runs: list[float]) -> None:
    """Assert the **best** of ``runs`` is inside ``budget``.

    Best, not every run: a run that was descheduled while another suite had
    the machine measures the machine.  The cost of that choice is that one
    lucky run hides a regression of the typical case, which is why the 12 MP
    window gestures are also checked on their median (:func:`_median_under`).
    """
    best = min(runs)
    assert best <= budget, (
        f"{what} took {best:.3f}s at best, over the {budget:.2f}s best-of-"
        f"{len(runs)} budget; all runs: "
        + ", ".join(f"{r:.3f}s" for r in runs)
    )


#: How far over its budget the **median** may sit. The budget itself is what the
#: code costs when the machine is this test's (:func:`_under`, best of N); the
#: median is measured while other work shares the machine -- a review agent
#: exporting 66 desktops put the 12 MP timeline jump at 0.549 s against 0.50 s
#: with nothing wrong in the code (0.41 s idle). 30 % keeps the guard's point --
#: a regression of the typical case still fails it -- without making the whole
#: suite flaky under load.
MEDIAN_SLACK = 1.3


def _median_under(budget: float, what: str, runs: list[float]) -> None:
    """Assert the **median** of ``runs`` is inside ``budget`` x :data:`MEDIAN_SLACK`.

    The guard the best-of cannot give: a regression that leaves one run fast
    and the rest slow passes :func:`_under` and fails here.
    """
    middle = statistics.median(runs)
    limit = budget * MEDIAN_SLACK
    assert middle <= limit, (
        f"{what} took {middle:.3f}s at the median of {len(runs)}, over "
        f"{limit:.2f}s ({budget:.2f}s budget x {MEDIAN_SLACK}); all runs: "
        + ", ".join(f"{r:.3f}s" for r in runs)
    )


@pytest.mark.slow
def test_gui_thread_budgets_at_full_scanner_resolution(qapp, tmp_path, as_shipped):
    """A chassis-sized commit at 1600x1600 over 40 steps, plus browsing.

    Asserted on the **best of** :data:`BEST_OF` runs (see :func:`_under`).
    """
    session = make_session(tmp_path, last_step=40, hw=(1600, 1600))
    session.goto(2)  # almost everything is still in the chassis here
    drawn = seed_shapes(session, 2, grid=8, anchor=40)
    assert len(drawn) >= 30

    reached: list[int] = []

    def commit(attempt: int) -> None:
        # a different mask each time: re-committing the same pixels is a no-op
        result = session.commit_edit(api.SCOPE_KEYFRAME)
        reached.append(len(result["affected"]))

    def before_commit(attempt: int) -> None:
        session.goto(20)
        session.begin_edit(CHASSIS)
        session.set_editing_mask(
            rect(100, 100, 1500 - attempt, 1500 - attempt, (1600, 1600))
        )

    _, commit_runs = best_of(commit, before_commit)
    assert min(reached) >= 20  # it really does reach that far

    def warm(attempt: int) -> None:
        session.goto(18)
        session.compiled()
        session.image()

    def before_warm(attempt: int) -> None:
        session.goto(19)  # warms the prefetch of 18
        session.drain_prefetch(timeout=20.0)

    _, warm_runs = best_of(warm, before_warm)

    # measure the cold arrival on its own: with the prefetch of k-1 running the
    # two compete for the same cores, and what is being asked here is how long
    # the GUI thread takes, not how the machine schedules two of them
    session.drain_prefetch(timeout=20.0)
    session.sweeper_enabled = False
    session.sweeper.stop()

    def cold(attempt: int) -> None:
        session.goto(10)
        session.compiled()
        session.image()

    def before_cold(attempt: int) -> None:
        session.goto(30)          # so that arriving at 10 is a real move
        session.images.clear()    # ... with neither its image nor its rows in hand
        session._invalidate()

    _, cold_runs = best_of(cold, before_cold)

    session.close()
    _under(0.8, "commit", commit_runs)
    _under(0.15, "warm goto", warm_runs)
    _under(0.8, "cold goto", cold_runs)


@pytest.mark.slow
@pytest.mark.parametrize("step", [2, 12])
def test_confirming_a_frame_is_under_budget_at_full_scanner_resolution(
    qapp, tmp_path, monkeypatch, as_shipped, step: int
):
    """Space, in the configuration the annotator actually runs.

    Asserted on the **best of** :data:`BEST_OF` runs (see :func:`_under`).

    Two costs had to go. ``verify_frame`` compares every frozen row against a
    fresh compilation before it will confirm anything (the I1 gate), and
    comparing masks means decoding them -- so a row whose stored ``input_hash``
    is this compilation's is skipped undecoded, the reasoning ``refresh``
    already uses. And the compilation itself is one the session already has:
    arriving at a frame compiles it, or the sweeper prefetched it, and Space
    hands that frame to the truth service rather than paying for it twice.

    **The sweeper stays on**, and the frame confirmed is the prefetched
    ``k-1``: in reverse-order annotation every frame arrives that way, and a
    budget met only with the background worker switched off is a budget for a
    tool nobody runs. The compile count is asserted alongside the clock,
    because a wall time is also met by a fast machine.

    Both an early frame and a late one: by **step 12** several instances have
    left the machine and need no geometry, so the frame's placements are a
    strict subset of the desktop's. Reading the wrong one of those two dicts
    made the handover fail on exactly the later frames -- which is where the
    annotation happens -- and this test, standing only on step 2, said nothing.
    """
    session = make_session(tmp_path, last_step=40, hw=(1600, 1600))
    session.goto(step)
    # seeded for the frame being confirmed: its needs are a superset of the
    # later ones', so k+1 is drawn too and neither frame has a missing shape
    drawn = seed_shapes(session, step, grid=8, anchor=40)
    assert len(drawn) >= 30
    assert session.sweeper_enabled is True

    compiles = _count_compiles_in(monkeypatch)

    def confirm(attempt: int) -> None:
        assert session.confirm_frame() is True

    def before_confirm(attempt: int) -> None:
        session.goto(step + 1)                # k, so the sweeper prefetches k-1
        session.drain_prefetch(timeout=20.0)
        session.goto(step)                    # ... and arrive on the prefetched one
        session.drain_prefetch(timeout=20.0)  # which warms k-2 in its turn
        session.db.set_frame_flags(FrameKey(DESKTOP, step, VIEW), review_status=None)
        compiles.clear()

    _, confirm_runs = best_of(confirm, before_confirm)
    # The frame Space confirmed was compiled by nobody on this thread: it came
    # from the prefetch and the truth service proved it still current. Stepping
    # back to k-2 afterwards does compile that frame -- arriving anywhere does,
    # and the goto budgets above are what covers it.
    assert [k.step for k in compiles] == [step - 1], \
        f"confirm_frame compiled {compiles}"
    assert len(session.db.compiled(FrameKey(DESKTOP, step, VIEW))) >= 30

    session.close()
    _under(0.6, f"confirm_frame at step {step}", confirm_runs)


# --------------------------------------------------------------------------- #
# the 12 MP budgets (plan B task B3 step 5)
# --------------------------------------------------------------------------- #
#: What both OAK cameras shoot (spec 2.4). 12.26 MP, 7.5x a scanner frame.
OAK_HW = (3040, 4032)
#: The frame the budgets are measured on. Annotation runs in reverse, so the
#: *last* frame of a session is the *first* step of the teardown -- the one
#: where the whole machine is still in the chassis and every instance has a row
#: in the truth table. That is the expensive frame, and measuring the cheap one
#: is the F1 lesson this test exists to avoid repeating.
OAK_STEP = 2
#: Spec 4.2 read through plan B: what an annotator may wait for one gesture.
BUDGET_COMMIT = 0.45
BUDGET_CONFIRM = 0.6
BUDGET_FRAME_CHANGE = 0.15
BUDGET_TIMELINE_JUMP = 0.5
#: Samples per gesture **in this test only**, against :data:`BEST_OF` for the
#: scanner gates above, which are unchanged and must stay that way.
#:
#: This machine is shared, and a 12 MP gesture is long enough that being
#: descheduled once doubles it: a full-suite run with another worker on the box
#: measured 0.56, 0.58 and 0.59 s for a commit that is 0.30-0.35 s on its own,
#: and the commit's median sits at about 85 % of its budget. Five samples is
#: the same guard as three -- a regression makes every one of them slow -- with
#: a better chance of catching a moment when the machine is this test's.
BEST_OF_12MP = 5


@pytest.mark.slow
def test_gui_thread_budgets_on_a_12mp_frame_with_forty_instances(qapp, tmp_path, as_shipped):
    """Commit, Space, frame change and timeline jump on a 4032x3040 frame.

    Asserted on the **best of** :data:`BEST_OF_12MP` runs (see :func:`_under`).

    **In the shipped configuration**: the sweeper is on, so the prefetch of
    ``k-1`` competes for the same cores and the same database; the frame is the
    one with every instance still in the machine, so the truth table has forty
    rows to derive and write; and the mask committed is chassis-sized, so the
    reach of the edit is the whole interval.

    What made this impossible before (measured, same scene, same machine):
    commit 4.1 s, Space 2.5 s, timeline jump 2.5 s. The compiler composited
    every layer across the whole 12 MP canvas and the truth table transposed
    every visible mask into a Fortran buffer before encoding it.
    """
    session = make_session(tmp_path, last_step=BOARD_STEP, hw=OAK_HW)
    session.goto(OAK_STEP)
    drawn = seed_shapes(session, OAK_STEP, grid=7, anchor=BOARD_STEP, refresh=False)
    assert len(drawn) >= 40, f"the scene has only {len(drawn)} instances"
    assert session.sweeper_enabled is True
    session.compiled()  # the frame the gestures start from

    def before_commit(attempt: int) -> None:
        session.goto(OAK_STEP)
        session.begin_edit(CHASSIS)
        # a different mask each time: re-committing the same pixels is a no-op
        session.set_editing_mask(
            rect(100, 100, 3900 - attempt, 2900 - attempt, OAK_HW)
        )

    def commit(attempt: int) -> None:
        session.commit_edit(api.SCOPE_KEYFRAME)

    _, commit_runs = best_of(commit, before_commit, times=BEST_OF_12MP)
    assert len(session.db.compiled(FrameKey(DESKTOP, OAK_STEP, VIEW))) >= 40

    def before_confirm(attempt: int) -> None:
        session.goto(OAK_STEP + 1)
        session.drain_prefetch(timeout=120.0)
        session.goto(OAK_STEP)                # arrive on the prefetched frame
        session.drain_prefetch(timeout=120.0)
        session.db.set_frame_flags(FrameKey(DESKTOP, OAK_STEP, VIEW),
                                   review_status=None)

    def confirm(attempt: int) -> None:
        assert session.confirm_frame() is True

    _, confirm_runs = best_of(confirm, before_confirm, times=BEST_OF_12MP)

    def before_change(attempt: int) -> None:
        session.goto(OAK_STEP + 3)
        session.drain_prefetch(timeout=120.0)  # warms k-1, which is where we go

    def change(attempt: int) -> None:
        session.goto(OAK_STEP + 2)
        session.compiled()
        session.image()

    _, change_runs = best_of(change, before_change, times=BEST_OF_12MP)

    def before_jump(attempt: int) -> None:
        session.goto(BOARD_STEP - 2)
        session.images.clear()
        session._invalidate()

    def jump(attempt: int) -> None:
        session.goto(OAK_STEP + 6)
        session.compiled()
        session.image()

    _, jump_runs = best_of(jump, before_jump, times=BEST_OF_12MP)

    session.close(force=True)
    _under(BUDGET_COMMIT, "commit", commit_runs)
    _under(BUDGET_CONFIRM, "confirm_frame (Space)", confirm_runs)
    _under(BUDGET_FRAME_CHANGE, "frame change", change_runs)
    _under(BUDGET_TIMELINE_JUMP, "timeline jump", jump_runs)


# --------------------------------------------------------------------------- #
# the window's half of the same three gestures (plan B task B7)
# --------------------------------------------------------------------------- #
#: Spec 4.2 read through plan B, for the whole gesture -- key press to a
#: repainted canvas and panels that agree with it, not just the session call
#: underneath.  The measurements above stop at the session; these do not.
BUDGET_WINDOW_COMMIT = 0.6
BUDGET_WINDOW_FRAME_CHANGE = 0.12
BUDGET_WINDOW_TIMELINE_JUMP = 0.5
BUDGET_WINDOW_REPAINT = 0.05
#: Where the annotator stands on an OAK frame: the ROI zoomed to fill the
#: canvas, which on a 4032x3040 frame in this window is about 59 %.
OAK_ZOOM = 0.59


def _open_window(session, tmp_path: Path):
    """A real ``MainWindow`` on this session, laid out like the annotator's."""
    from PySide6.QtCore import Qt

    from tda.ui.app import MainWindow

    window = MainWindow(session, {
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(tmp_path / "tda.sqlite"),
        "backup_dir": str(tmp_path / "backups"),
        "app_dir": str(tmp_path / "state"),
    }, ANNOTATOR)
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.resize(1920, 1200)
    window.show()
    QApplication.processEvents()
    return window


def _settle(window) -> None:
    """Let Qt deliver everything the gesture posted, and really repaint.

    ``processEvents`` alone leaves the canvas with a scheduled update; the
    budget is about what the annotator waits for, which ends when the pixels
    are on screen.
    """
    QApplication.processEvents()
    window.canvas.viewport().repaint()


@pytest.mark.slow
def test_window_gesture_budgets_on_a_12mp_frame_best_and_median_of_five(
    qapp, tmp_path, as_shipped
):
    """Enter, a frame change, a timeline jump and a repaint, through the window.

    Asserted twice: on the **best** of five, which is what the code costs when
    the machine is this test's, and on the **median** of five, so that a
    regression of the typical case cannot hide behind one lucky run.  (The
    session budgets above are best-of only; this is the newer, stricter shape
    and the one to copy.)

    The session budgets above measure ``commit_edit`` and ``goto``.  What the
    annotator waits for is longer than either: ``Enter`` also asks what the
    edit *meant*, repaints the overlay, re-syncs the editing layer and
    refreshes the panels, and a frame change repaints a 12 MP canvas.
    Measured on main before this task, the whole ``Enter`` was 1.8-1.9 s
    against a 0.5 s ``commit_edit``.

    **In the shipped configuration**: the sweeper is on, the frame is the one
    with every instance still in the machine, the canvas stands where the
    annotator stands -- zoomed into the ROI, not fitted to the whole frame --
    and the cache carries no offline timeline thumbnails, which is the state
    ``oak1`` and ``oak2`` are in today, so the timeline rows are read from the
    12 MP frames themselves.
    """
    session = make_session(tmp_path, last_step=BOARD_STEP, hw=OAK_HW)
    session.goto(OAK_STEP)
    drawn = seed_shapes(session, OAK_STEP, grid=7, anchor=BOARD_STEP, refresh=False)
    assert len(drawn) >= 40, f"the scene has only {len(drawn)} instances"
    session.compiled()
    window = _open_window(session, tmp_path)
    try:
        assert session.sweeper_enabled is True

        def stand() -> None:
            window.canvas.set_zoom(OAK_ZOOM)
            window.canvas.center_on((OAK_HW[1] / 2, OAK_HW[0] / 2))

        stand()
        _settle(window)
        shown = window.canvas.viewport_image_rect()
        assert (shown[2] - shown[0]) < OAK_HW[1], \
            "the canvas is showing the whole frame; that is not where an annotator stands"

        # -- the whole Enter gesture ---------------------------------------
        def before_commit(attempt: int) -> None:
            window.act_clear_edit()
            session.clear_edit()
            session.goto(OAK_STEP, force=True)
            _settle(window)
            window.on_request_edit(CHASSIS)
            mask = session.editing_mask()
            painted = (np.zeros(OAK_HW, dtype=bool) if mask is None
                       else mask.copy())
            painted[1200:1400, 1500 + attempt:1700 + attempt] ^= True
            window.set_editing_mask(painted, undoable=True)
            _settle(window)

        def commit(attempt: int) -> None:
            window.act_commit()
            for _ in range(2):
                if not (window.warn_bar.isVisible() or window.scope_bar.isVisible()):
                    break
                window.act_commit()
            _settle(window)

        _, commit_runs = best_of(commit, before_commit, times=BEST_OF_12MP)
        assert session.editing_instance is None, "the commit was refused"
        assert len(session.db.compiled(FrameKey(DESKTOP, OAK_STEP, VIEW))) >= 40

        # -- a frame change -------------------------------------------------
        window.act_clear_edit()
        session.clear_edit()
        _settle(window)

        def before_change(attempt: int) -> None:
            session.goto(OAK_STEP + 1 + (attempt % 2) * 2, force=True)
            session.drain_prefetch(timeout=120.0)
            _settle(window)

        def change(attempt: int) -> None:
            window.act_step(-1)
            _settle(window)

        _, change_runs = best_of(change, before_change, times=BEST_OF_12MP)

        # -- a timeline click ------------------------------------------------
        steps = sorted(session.steps())

        def before_jump(attempt: int) -> None:
            session.goto(steps[-2], force=True)
            session.images.clear()
            session._invalidate()
            _settle(window)

        def jump(attempt: int) -> None:
            window.timeline_goto(OAK_STEP + 6 + attempt)
            _settle(window)

        _, jump_runs = best_of(jump, before_jump, times=BEST_OF_12MP)
        assert session.current().step == OAK_STEP + 6 + BEST_OF_12MP - 1, \
            "a jump was refused; that is not a jump time"

        # -- panning, which composites the strip it reveals -------------------
        session.goto(OAK_STEP, force=True)
        stand()
        _settle(window)
        bar = window.canvas.horizontalScrollBar()
        pan_runs = []
        for _attempt in range(4 * BEST_OF_12MP):
            started = time.perf_counter()
            bar.setValue(bar.value() + 40)
            _settle(window)
            pan_runs.append(time.perf_counter() - started)

        zoom_runs = []
        for attempt in range(2 * BEST_OF_12MP):
            started = time.perf_counter()
            window.canvas.set_zoom(
                window.canvas.zoom_factor() * (1.25 if attempt % 2 else 0.8)
            )
            _settle(window)
            zoom_runs.append(time.perf_counter() - started)
    finally:
        window.shutdown()
        window.hide()
        QApplication.processEvents()

    for check in (_under, _median_under):
        check(BUDGET_WINDOW_COMMIT, "the whole Enter gesture", commit_runs)
        check(BUDGET_WINDOW_FRAME_CHANGE, "frame change (window)", change_runs)
        check(BUDGET_WINDOW_TIMELINE_JUMP, "timeline jump (window)", jump_runs)
    # A repaint is a fiftieth of the smallest of those, and there are twenty of
    # them: the best-of is the measurement, the median only says the machine
    # was not stolen mid-test.
    _under(BUDGET_WINDOW_REPAINT, "pan repaint", pan_runs)
    _under(BUDGET_WINDOW_REPAINT, "zoom repaint", zoom_runs)
    _median_under(BUDGET_WINDOW_REPAINT, "pan repaint", pan_runs)
    _median_under(BUDGET_WINDOW_REPAINT, "zoom repaint", zoom_runs)


def _count_compiles_in(monkeypatch) -> list:
    """Record every pixel compilation made on **this** thread from now on.

    The sweeper compiles on its own thread and is not what is being measured;
    what is, is whether the GUI thread compiles a frame it was handed.
    """
    import threading

    import tda.core.truth as truth_mod

    calls: list = []
    real = truth_mod.compile_frame
    here = threading.current_thread()

    def counted(*a, **k):
        if threading.current_thread() is here:
            calls.append(a[0])
        return real(*a, **k)

    monkeypatch.setattr(truth_mod, "compile_frame", counted)
    return calls
