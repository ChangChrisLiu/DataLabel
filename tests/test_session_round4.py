"""The round-4 findings: stamping, the bench gate, and the escape hatch.

Three of these are about a database that already exists.  The digest table is
created empty by the migration, so **every** pre-existing view starts with rows
that are already up to date and no digest to say so; unless that path stamps,
the short-circuit never engages and nothing is ever cheaper.  The bench gate is
the same story from the other side: an ``on_bench`` instance on a view that
cannot see the staging area is not annotated, not missing, and must not count
towards ``bench_annotated`` either (spec 3.3 step 2).
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    LAST_STEP,
    SCREWS,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
    seed_shapes,
)

BENCH_ROI = (0, 40, 64, 64)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    made.sweeper_enabled = False
    made.sweeper.stop()
    yield made
    made.close()


@pytest.fixture
def counter(monkeypatch):
    import tda.core.truth as truth_mod

    original = truth_mod.compile_frame
    steps: list[int] = []
    monkeypatch.setattr(truth_mod, "compile_frame",
                        lambda key, *a, **k: (steps.append(key.step),
                                              original(key, *a, **k))[1])
    return steps


def with_bench_roi(session) -> None:
    session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, LAST_STEP, LAST_STEP, None, None)
    session.db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, BENCH_ROI)
    session.review.invalidate()
    session._invalidate()


# --------------------------------------------------------------------------- #
# 1. a database that predates the digest table
# --------------------------------------------------------------------------- #
def test_a_view_with_no_digests_is_stamped_on_the_first_pass(session, counter):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    session.truth.ensure_fresh(DESKTOP, VIEW)
    # exactly what the migration leaves behind: rows that are already correct
    # and not one digest to say so
    session.db.conn.execute("DELETE FROM frame_digest")
    session.db.conn.commit()

    counter.clear()
    session.truth.ensure_fresh(DESKTOP, VIEW)
    first = list(counter)
    assert first, "the rows carry no digest, so the first pass has to look"
    assert session.db.frame_digests(DESKTOP, VIEW), "... and it has to stamp them"

    counter.clear()
    session.truth.ensure_fresh(DESKTOP, VIEW)
    assert counter == []


def test_a_frame_the_refresh_had_nothing_to_do_on_is_stamped(session):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    key = FrameKey(DESKTOP, 6, VIEW)
    session.truth.refresh(key)
    session.db.clear_frame_digest(key)

    out = session.truth.refresh(key)  # every row already carries the hash
    assert out["skipped"] and out["updated"] == 0
    assert session.db.frame_digest(key) is not None


def test_an_edited_resolution_leaves_the_frame_judged_again(session):
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]
    key = FrameKey(DESKTOP, 12, VIEW)
    # a stamp the frame is carrying while the disagreement is open: `edited`
    # writes nothing at all, so it is the one resolution that could leave a
    # digest claiming rows nobody judged
    session.db.set_frame_digest(key, session.truth.inputs_digest(key),
                                session.truth.compiler_version,
                                len(session.db.compiled(key)))

    session.truth.resolve_conflict(cid, "edited", "tester")
    assert session.db.frame_digest(key) is None


def test_confirming_a_frame_with_an_open_conflict_says_why(session):
    """Space on a demoted frame: the refusal is a problem line, not silence.

    The compilation of such a frame is perfectly fine, so the old code emitted
    an empty problem list and the frame simply did not confirm.
    """
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]

    said: list[list[str]] = []
    session.sigProblems.connect(said.append)
    session.goto(12)

    assert session.confirm_frame() is False
    assert said and str(cid) in said[-1][0]
    assert "conflict" in said[-1][0]
    assert session.db.conflicts(DESKTOP, VIEW, open_only=True)


# --------------------------------------------------------------------------- #
# 2. refresh_all must not die of a missing logger
# --------------------------------------------------------------------------- #
def test_refresh_all_refuses_when_the_sweeper_will_not_stop(session):
    class Stubborn:
        is_running = True

        def stop(self, timeout: float = 30.0) -> bool:
            return False

    session.sweeper = Stubborn()
    with pytest.raises(RuntimeError, match="sweeper"):
        session.refresh_all()


# --------------------------------------------------------------------------- #
# 3. the parked backlog can be asked for again
# --------------------------------------------------------------------------- #
def test_retry_rechecks_re_arms_what_the_sweeper_parked(qapp, tmp_path, monkeypatch):
    from tda.ui.session_sweep import TruthSweeper

    session = make_session(tmp_path)
    session.sweeper_enabled = False
    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)

    state = {"fail": True}
    original = TruthSweeper._recheck

    def flaky(self, db, truth, step, gen=None):
        if state["fail"]:
            raise RuntimeError("not yet")
        original(self, db, truth, step, gen)

    monkeypatch.setattr(TruthSweeper, "_recheck", flaky)
    monkeypatch.setattr("tda.ui.session_sweep.RETRY_DELAYS", (0.01, 0.02))
    session.sweeper_enabled = True
    session.sweeper.open(DESKTOP, VIEW)
    session.sweeper.enqueue([12])
    session.drain_sweeper(timeout=20.0)
    assert session.sweeper.parked() == [12]
    assert session.db.rechecks(DESKTOP, VIEW) == [12]

    state["fail"] = False
    assert session.retry_rechecks() == 1
    assert session.drain_sweeper(timeout=20.0) is True
    assert session.db.rechecks(DESKTOP, VIEW) == []
    session.close()


def test_retry_rechecks_on_a_quiet_view_is_zero(session):
    assert session.retry_rechecks() == 0
    session.close()
    assert session.retry_rechecks() == 0


def test_a_refresh_reads_the_inputs_once(session, monkeypatch):
    """The digest needs them and so does the compilation; one read serves both.

    Gathering twice re-reads the whole keyframe table per frame, which is most
    of what a batch pass over a view spends its time on.
    """
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    import tda.core.truth as truth_mod

    original = truth_mod.gather
    calls: list = []
    monkeypatch.setattr(truth_mod, "gather",
                        lambda *a, **k: (calls.append(1), original(*a, **k))[1])

    session.truth.refresh(FrameKey(DESKTOP, 6, VIEW))  # a frame that must compile
    assert calls == [1]


def test_retry_rechecks_re_arms_a_step_only_the_sweeper_remembers(session):
    """A synchronous drain can clear the row while the sweeper still parks it."""
    class Parked:
        is_running = True
        asked: list = []

        def parked(self) -> list[int]:
            return [7]

        def enqueue(self, steps) -> None:
            self.asked = list(steps)

        def stop(self, timeout: float = 30.0) -> bool:
            return True

    stub = Parked()
    session.sweeper = stub
    session.sweeper_enabled = True  # the stub stands in for a running worker
    assert session.db.rechecks(DESKTOP, VIEW) == []  # nothing persisted at all

    assert session.retry_rechecks() == 1
    assert stub.asked == [7]


# --------------------------------------------------------------------------- #
# 4. a view with no staging area has no bench instances at all
# --------------------------------------------------------------------------- #
def test_a_bench_part_is_not_compiled_on_a_view_that_cannot_see_it(session):
    session.goto(LAST_STEP)
    compiled = session.compiled()

    assert COOLER not in compiled.instances
    for screw in SCREWS:
        assert screw not in compiled.instances
    assert CHASSIS in compiled.instances


def test_bench_annotated_is_left_alone_without_a_staging_area(session):
    session.goto(LAST_STEP)
    seed_shapes(session, LAST_STEP)
    session.truth.refresh(FrameKey(DESKTOP, LAST_STEP, VIEW))
    assert not session.db.get_frame(FrameKey(DESKTOP, LAST_STEP, VIEW))["bench_annotated"]

    assert session.confirm_frame() is True
    frame = session.db.get_frame(FrameKey(DESKTOP, LAST_STEP, VIEW))
    assert not frame["bench_annotated"]  # there is no bench: the flag means nothing


def test_with_a_staging_area_the_bench_parts_come_back(session):
    with_bench_roi(session)
    session.goto(LAST_STEP)
    compiled = session.compiled()

    assert COOLER in compiled.instances
    assert any(p.startswith("bench_missing:") for p in compiled.problems)

    session.commit_box(COOLER, (2.0, 44.0, 12.0, 54.0))
    session.truth.refresh(FrameKey(DESKTOP, LAST_STEP, VIEW))
    assert COOLER in session.db.compiled(FrameKey(DESKTOP, LAST_STEP, VIEW))


def test_the_queues_agree_with_the_compiler_about_the_bench(session):
    session.goto(LAST_STEP)
    found = session.review.drawn()[LAST_STEP]
    assert COOLER not in found.bench_missing
    assert COOLER not in found.missing

    with_bench_roi(session)
    found = session.review.drawn()[LAST_STEP]
    assert COOLER in found.bench_missing
    assert COOLER not in found.missing  # a bench box never blocks a frame


# --------------------------------------------------------------------------- #
# 5. the escape hatch really is one
# --------------------------------------------------------------------------- #
def test_refresh_all_forces_by_default_and_can_be_cheap(session, counter):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    session.refresh_all()

    counter.clear()
    session.refresh_all()  # the escape hatch does not trust the digests
    assert counter

    counter.clear()
    session.refresh_all(force=False)  # ... but it can be asked to
    assert counter == []
