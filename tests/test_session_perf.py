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
    best = min(runs)
    assert best <= budget, (
        f"{what} took {best:.3f}s, over the {budget:.2f}s budget; "
        f"all {len(runs)} runs: " + ", ".join(f"{r:.3f}s" for r in runs)
    )


@pytest.mark.slow
def test_gui_thread_budgets_at_full_scanner_resolution(qapp, tmp_path):
    """A chassis-sized commit at 1600x1600 over 40 steps, plus browsing."""
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
