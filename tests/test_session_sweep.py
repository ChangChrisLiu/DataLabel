"""What the truth sweeper does when the work fails, and when it races an edit.

The sweeper is the only place in the tool where a conflict can go missing: it
runs off the GUI thread, against its own connection, on a queue that has to
survive a crash.  Two things therefore have to be true whatever happens, and
both are tested here rather than reasoned about:

* **a frame is never dropped.**  A re-check that raises is logged, reported and
  retried; the persisted request stays until the frame has actually been
  compared; and a request that arrives while the sweeper is working on that very
  frame is not swallowed by the clear that follows.
* **a re-check never writes what the inputs no longer say.**  The frame's
  ``input_hash`` is taken before the pixel work and checked again inside the
  write transaction, so an edit landing in between rolls the write back and
  re-queues the step instead of freezing a conflict nobody caused.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core.model import FrameKey
from tda.core.truth import TruthService
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from tda.ui.session_sweep import TruthSweeper
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
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
    made.close(force=True)  # the worker must never outlive the object it signals


def exploding(message: str, fail_times: int = 1_000_000, then=None):
    """A ``_recheck`` replacement that raises the first ``fail_times`` calls.

    A plain function, because the worker looks the attribute up on the class:
    anything that is not a descriptor would not be handed ``self``.
    """
    state = {"calls": 0}

    def _recheck(self, db, truth, step, gen=None):
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise RuntimeError(message)
        if then is not None:
            then(self, db, truth, step, gen)

    _recheck.state = state
    return _recheck


def verified_edit(session, verified_step: int = 12, edit_step: int = 10) -> None:
    """Freeze a frame, then edit a shape that reaches it."""
    session.goto(verified_step)
    seed_shapes(session, verified_step)
    assert session.confirm_frame() is True
    session.goto(edit_step)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)


# --------------------------------------------------------------------------- #
# a failing re-check (finding 1)
# --------------------------------------------------------------------------- #
def test_a_failing_recheck_is_reported_and_the_frame_stays_pending(session, monkeypatch):
    session.sweeper_enabled = False
    verified_edit(session)
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)
    errors: list[tuple] = []
    session.sigSweepError.connect(lambda step, text: errors.append((step, text)))
    progress: list[tuple] = []
    session.sigSweepProgress.connect(lambda *a: progress.append(a))

    boom = exploding("the truth service fell over")
    monkeypatch.setattr(TruthSweeper, "_recheck", boom)
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    session.drain_sweeper(timeout=20.0)

    assert boom.state["calls"] >= 1
    assert errors and errors[0][0] == 12 and "fell over" in errors[0][1]
    assert any("fell over" in text for group in problems for text in group)
    # nothing may look finished while a step failed, and nothing may be lost
    assert all(done < total for done, total, *_ in progress)
    assert session.db.rechecks(DESKTOP, VIEW) == [12]
    assert session.frame_status(12) == api.STATUS_RECHECK


def test_a_failing_recheck_is_retried_with_a_backoff(session, monkeypatch):
    session.sweeper_enabled = False
    verified_edit(session)

    flaky = exploding("transient", fail_times=1, then=TruthSweeper._recheck)
    monkeypatch.setattr(TruthSweeper, "_recheck", flaky)
    monkeypatch.setattr("tda.ui.session_sweep.RETRY_DELAYS", (0.05, 0.1))
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    assert session.drain_sweeper(timeout=20.0) is True

    assert flaky.state["calls"] == 2  # it failed once and then went through
    assert session.db.rechecks(DESKTOP, VIEW) == []


def test_a_parked_step_does_not_spin(session, monkeypatch):
    session.sweeper_enabled = False
    verified_edit(session)
    boom = exploding("always")
    monkeypatch.setattr(TruthSweeper, "_recheck", boom)
    monkeypatch.setattr("tda.ui.session_sweep.RETRY_DELAYS", (0.01, 0.02))
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    session.drain_sweeper(timeout=20.0)

    assert boom.state["calls"] == 3  # the first attempt plus the two retries
    parked = boom.state["calls"]
    session.drain_sweeper(timeout=2.0)
    assert boom.state["calls"] == parked  # ... and it stays parked


def test_a_worker_that_cannot_open_its_database_says_so(qapp, tmp_path, monkeypatch):
    session = make_session(tmp_path)
    errors: list[tuple] = []
    session.sigSweepError.connect(lambda step, text: errors.append((step, text)))

    monkeypatch.setattr("tda.ui.session_sweep.Db",
                        lambda path: (_ for _ in ()).throw(RuntimeError("locked out")))
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([3])
    session.drain_sweeper(timeout=20.0)

    assert errors and "locked out" in errors[-1][1]
    assert session.sweeper.is_running is False
    session.close()


def test_a_worker_that_cannot_start_is_not_running_the_moment_it_says_so(
    qapp, tmp_path, monkeypatch
):
    """The real race behind the intermittent failure, three hundred times over.

    The failure path set ``_idle`` and *then* returned, so ``wait_idle`` -- which
    keys on ``_idle`` -- could hand control back while the thread was still
    unwinding. ``is_running`` then answered ``True`` for a worker that had
    already given up, about once in two hundred runs, and a caller that
    reasonably stops using the sweeper after that was racing a live thread on
    the same database.
    """
    session = make_session(tmp_path)
    monkeypatch.setattr("tda.ui.session_sweep.Db",
                        lambda path: (_ for _ in ()).throw(RuntimeError("locked out")))
    try:
        for attempt in range(300):
            session.sweeper.open(DESKTOP, VIEW)
            session.sweeper.enqueue([3])
            session.drain_sweeper(timeout=20.0)
            assert session.sweeper.is_running is False, f"still alive on attempt {attempt}"
    finally:
        session.close()


def test_the_emit_guard_only_swallows_a_receiver_that_is_gone(qapp, tmp_path):
    """Everything else is a bug in the slot, and a bug has to be reportable."""
    sweeper = TruthSweeper(str(tmp_path / "x.sqlite"), None, str(tmp_path))

    class Gone:
        def emit(self, *a):
            raise RuntimeError("Signal source has been deleted")

    class Broken:
        def emit(self, *a):
            raise RuntimeError("the slot raised")

    sweeper._emit(Gone())  # noqa: SLF001 - the guard is what is under test

    with pytest.raises(RuntimeError, match="the slot raised"):
        sweeper._emit(Broken())


def test_stop_keeps_is_running_truthful_when_the_join_times_out(session):
    session.sweeper.open(DESKTOP, VIEW)
    holding = threading.Event()

    def never_finish(self, *a, **k):
        holding.set()
        threading.Event().wait(30.0)

    original = TruthSweeper._recheck
    TruthSweeper._recheck = never_finish
    try:
        session.sweeper.enqueue([5])
        assert holding.wait(10.0) is True
        session.sweeper.stop(timeout=0.2)
        assert session.sweeper.is_running is True  # it did not actually stop
    finally:
        TruthSweeper._recheck = original


# --------------------------------------------------------------------------- #
# an edit landing mid-sweep (finding 10)
# --------------------------------------------------------------------------- #
def test_an_edit_during_a_recheck_rolls_the_write_back_and_requeues(session, monkeypatch):
    """The probe: the inputs move between gathering the frame and writing it.

    Simulated on the digest rather than by editing from the worker thread --
    the session's connection belongs to the GUI thread and may not be touched
    from here, which is the whole reason the sweeper has one of its own.
    """
    session.sweeper_enabled = False
    verified_edit(session)
    conflicts_before = len(session.queues()[api.QUEUE_CONFLICTS])

    original = TruthService.inputs_digest
    seen = {"n": 0}

    def moving(self, key, cache=None):
        seen["n"] += 1
        digest = original(self, key, cache)
        # call 1 is the guard, call 2 the check inside the write transaction
        return digest + ":moved" if seen["n"] == 2 else digest

    monkeypatch.setattr(TruthService, "inputs_digest", moving)
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    assert session.drain_sweeper(timeout=30.0) is True

    assert seen["n"] >= 4  # it was tried again after the abandoned write
    # the abandoned pass wrote nothing; the second one, against the inputs that
    # are actually there, is the one whose verdict stands
    assert len(session.queues()[api.QUEUE_CONFLICTS]) == conflicts_before + 1
    assert session.db.rechecks(DESKTOP, VIEW) == []


def test_a_request_arriving_mid_sweep_is_not_cleared_away(session):
    """The generation stamp: a newer request outlives the older one's clear."""
    db = session.db
    db.add_rechecks(DESKTOP, VIEW, [9])
    first = db.recheck_generation(DESKTOP, VIEW, 9)

    db.add_rechecks(DESKTOP, VIEW, [9])  # the annotator edits again mid-sweep
    assert db.recheck_generation(DESKTOP, VIEW, 9) == first + 1

    db.clear_recheck(DESKTOP, VIEW, 9, gen=first)  # the sweeper finishes the old one
    assert db.rechecks(DESKTOP, VIEW) == [9]  # ... and the newer request survives

    db.clear_recheck(DESKTOP, VIEW, 9, gen=first + 1)
    assert db.rechecks(DESKTOP, VIEW) == []


def test_a_request_that_arrives_mid_check_survives_the_clear(session, monkeypatch):
    """End to end: the sweeper must retire only the request it picked up.

    The stamp is read before the comparison, so a request made while the frame
    is being compared carries a newer one and the clear does not match it -- the
    frame stays queued and is looked at again, rather than being retired on the
    strength of a check that predates the edit.
    """
    session.sweeper_enabled = False
    verified_edit(session)
    original = TruthService.refresh
    bumped = {"done": False}

    def refresh_then_request_again(self, key, cache=None, guard=None):
        result = original(self, key, cache, guard)
        if not bumped["done"]:
            bumped["done"] = True  # the annotator edits this frame again, now
            self.db.add_rechecks(key.desktop, key.view, [key.step])
        return result

    monkeypatch.setattr(TruthService, "refresh", refresh_then_request_again)
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    session.drain_sweeper(timeout=20.0)

    assert bumped["done"]
    assert session.db.rechecks(DESKTOP, VIEW) == [12]


# --------------------------------------------------------------------------- #
# queue notifications (minor)
# --------------------------------------------------------------------------- #
def test_queue_changes_are_coalesced(session):
    session.sweeper_enabled = False
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    for step in (11, 10, 9):
        session.goto(step)
        seed_shapes(session, step)
        session.confirm_frame()

    changes: list[int] = []
    session.sigQueuesChanged.connect(lambda: changes.append(1))
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([9, 10, 11, 12])
    session.drain_sweeper(timeout=30.0)
    QApplication.processEvents()

    assert changes  # the panels are told
    assert len(changes) < 4  # ... but not once per frame
