"""Not compiling a frame whose inputs have not moved (spec 3.4).

``refresh`` compared the stored ``input_hash`` *after* compiling, which means
every batch pass over a view paid the pixel work whether or not anything had
changed: the second export of an untouched desktop cost exactly as much as the
first.  The comparison now happens before any of it, against a persisted
:func:`~tda.core.truth.TruthService.inputs_digest` -- the same pixel-free
fingerprint the sweeper guards a re-check with -- written in the same
transaction as the rows it describes.

The frozen-truth invariant is untouched by this: a matching digest means the
inputs are identical, so there is nothing a recompilation could disagree with.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core.export.coco import export_coco
from tda.core.model import FrameKey
from tda.core.truth import TruthService
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    VIEW,
    cell,
    draw,
    make_session,
    rect,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    made.sweeper_enabled = False  # nothing may compile behind the counter
    made.sweeper.stop()
    yield made
    made.close(force=True)


@pytest.fixture
def counter(monkeypatch):
    """Counts every ``compile_frame`` the truth service performs, by step."""
    import tda.core.truth as truth_mod

    original = truth_mod.compile_frame
    steps: list[int] = []

    def counted(key, *a, **k):
        steps.append(key.step)
        return original(key, *a, **k)

    monkeypatch.setattr(truth_mod, "compile_frame", counted)
    return steps


def _drawn(session) -> None:
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(2), api.SCOPE_KEYFRAME)


# --------------------------------------------------------------------------- #
# the digest short-circuit
# --------------------------------------------------------------------------- #
def test_a_second_export_compiles_nothing(session, tmp_path, counter):
    _drawn(session)
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "a.json"),
                only_verified=False)
    assert counter, "the first export has to do the work"

    counter.clear()
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "b.json"),
                only_verified=False)
    assert counter == []


def test_only_the_frames_whose_inputs_moved_are_recompiled(session, tmp_path, counter):
    _drawn(session)
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "a.json"),
                only_verified=False)

    session.goto(10)
    draw(session, COOLER, cell(6), api.SCOPE_KEYFRAME)  # reaches steps 1-12
    counter.clear()
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "b.json"),
                only_verified=False)

    # the cooler's chain ends at 12, and step 10 was compiled by the commit
    assert set(counter) == set(range(1, 13)) - {10}


def test_a_frame_whose_rows_went_missing_is_compiled_again(session, tmp_path, counter):
    _drawn(session)
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "a.json"),
                only_verified=False)
    key = FrameKey(DESKTOP, 6, VIEW)
    for instance in list(session.db.compiled(key)):
        session.db.delete_compiled(key, instance)

    counter.clear()
    export_coco(session.db, session.tax, [DESKTOP], VIEW, str(tmp_path / "b.json"),
                only_verified=False)
    assert 6 in counter  # the digest matches but the rows it describes are gone


def test_a_visit_still_gets_its_compiled_frame(session, counter):
    _drawn(session)
    session.goto(6)
    assert session.compiled().instances  # the arrays the canvas draws
    counter.clear()

    session.goto(7)
    frame = session.compiled()
    assert frame.instances
    assert counter == [7]  # once, and only for the frame arrived at


def test_the_digest_is_written_with_the_rows_it_describes(session):
    _drawn(session)
    session.truth.refresh_range(DESKTOP, VIEW, [6])
    stored = session.db.frame_digest(FrameKey(DESKTOP, 6, VIEW))

    assert stored is not None
    assert stored["digest"] == session.truth.inputs_digest(FrameKey(DESKTOP, 6, VIEW))
    assert stored["n_rows"] == len(session.db.compiled(FrameKey(DESKTOP, 6, VIEW)))


def test_verifying_a_frame_records_its_digest_too(session):
    from session_scene import seed_shapes

    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    stored = session.db.frame_digest(FrameKey(DESKTOP, 12, VIEW))
    assert stored["digest"] == session.truth.inputs_digest(FrameKey(DESKTOP, 12, VIEW))


# --------------------------------------------------------------------------- #
# the compiler version has to match (a mass re-queue otherwise)
# --------------------------------------------------------------------------- #
def test_a_foreign_compiler_version_is_refused(session, tmp_path):
    _drawn(session)
    session.truth.ensure_fresh(DESKTOP, VIEW)

    other = TruthService(session.db, session.tax, compiler_version="99")
    with pytest.raises(RuntimeError, match="compiler version"):
        other.ensure_fresh(DESKTOP, VIEW)
    assert other.ensure_fresh(DESKTOP, VIEW, force=True)["updated"] >= 0


def test_a_resolved_conflict_leaves_the_frame_to_be_judged_again(session):
    """Resolving writes rows (or nothing at all), so no digest may claim it is done."""
    from session_scene import seed_shapes

    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    session.truth.run_pending_rechecks(DESKTOP, VIEW)
    cid = session.queues()[api.QUEUE_CONFLICTS][0]["id"]

    # a frame can carry a stamp while a conflict of it is open -- a pass that
    # found nothing to do wrote one, a repair script wrote one -- and resolving
    # changes the rows under it, so the stamp has to go or the next pass would
    # skip a frame it has never judged
    key = FrameKey(DESKTOP, 12, VIEW)
    session.db.set_frame_digest(key, session.truth.inputs_digest(key),
                                session.truth.compiler_version,
                                len(session.db.compiled(key)))
    assert session.db.frame_digest(key) is not None

    session.truth.resolve_conflict(cid, api.RESOLVE_ACCEPT_NEW, "tester")
    assert session.db.frame_digest(key) is None


def test_coverage_never_decodes_an_image(session, monkeypatch):
    """It runs on the timeline's repaint path, so it may not touch the disk."""
    _drawn(session)
    import cv2

    monkeypatch.setattr(cv2, "imread", _refuse)
    found = session.review.drawn()
    assert found and any(f.needed for f in found.values())


def _refuse(*a, **k):
    raise AssertionError("coverage decoded an image")


def test_ensure_fresh_reports_both_halves(session, tmp_path):
    from session_scene import seed_shapes

    session.goto(12)
    seed_shapes(session, 12)
    assert session.confirm_frame() is True
    session.goto(10)
    # big enough to be a real disagreement: a 6x6 cell fits inside its own
    # two-pixel re-tracing tolerance band, so moving one is not a conflict
    draw(session, CHASSIS, rect(2, 34, 34, 62), api.SCOPE_KEYFRAME)
    assert session.truth.pending_rechecks(DESKTOP, VIEW) == [12]

    totals = session.truth.ensure_fresh(DESKTOP, VIEW)
    assert totals["rechecked"]["conflicts"] >= 1  # what the drain found
    assert totals["refreshed"]["updated"] >= 1  # ... and what the sweep did
    # and the flat totals are the two added together, not one of them
    assert totals["conflicts"] == (totals["rechecked"]["conflicts"]
                                   + totals["refreshed"]["conflicts"])
    assert totals["updated"] == (totals["rechecked"]["updated"]
                                 + totals["refreshed"]["updated"])


# --------------------------------------------------------------------------- #
# the hash a reader computes is the hash the compiler wrote
# --------------------------------------------------------------------------- #
def _hash_matches(session, step: int) -> tuple[bool, str]:
    """``(does it match, why not)`` for one frame of the open view."""
    from tda.core.truth_fresh import hash_of_inputs
    from tda.core.truth_inputs import gather

    key = FrameKey(DESKTOP, step, VIEW)
    inputs = gather(session.db, session.tax, key, cache_dir=session.truth.cache_dir)
    compiled = session.truth.compile(key)
    mine = hash_of_inputs(inputs, compiled.layers, session.truth.compiler_version)
    return mine == compiled.input_hash, (
        f"step {step}: needs={len(inputs.needs)} placements={len(inputs.placements)}"
    )


def test_every_frame_of_the_scene_hashes_the_way_the_compiler_hashed_it(session):
    """The property the whole handover rests on, on every frame of a real sheet.

    ``gather`` hands out the placement of every instance the desktop has; the
    compiler hashes the ones this frame needs geometry for. Reading the
    superset made the hashes differ on every frame with an instance that needs
    nothing here -- 7 of the 14 steps of D13, and the *later* ones, which is
    where the annotation actually happens.
    """
    from session_scene import seed_shapes

    session.goto(12)
    seed_shapes(session, 12)  # real shapes, so the frames have layers to order
    results = {step: _hash_matches(session, step) for step in session.steps()}
    bad = {step: why for step, (ok, why) in results.items() if not ok}

    assert not bad, f"{len(bad)}/{len(results)} frames hash differently: {bad}"


def test_the_handover_is_accepted_where_the_frame_needs_less_than_the_desktop(session):
    """The regression: a frame whose placements are a superset of its needs."""
    from session_scene import seed_shapes
    from tda.core.truth_inputs import gather
    from tda.core.truth_verify import usable

    session.goto(12)
    seed_shapes(session, 12)
    key = FrameKey(DESKTOP, 12, VIEW)
    inputs = gather(session.db, session.tax, key, cache_dir=session.truth.cache_dir)
    assert set(inputs.placements) != set(inputs.needs), "the scene lost its point"

    assert usable(session.truth.compile(key), key, inputs,
                  session.truth.compiler_version)
