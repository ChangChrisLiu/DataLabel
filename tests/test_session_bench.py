"""Bench items appear only where there is a staging area to see (spec 4.2).

"若该视角有堆放区 ROI" -- *if this view has a staging-area ROI*.  The scanner
looks straight down at the board and cannot see the bench at all, so asking its
annotator to box thirty removed parts is asking for something that is not in the
picture.  A bench item is therefore emitted only when the frame's pose segment
has a bench ROI recorded, and it carries its own kind, ``add_bench_box``, so the
window arms the box tool rather than the brush -- the old card said "draw" and
the mask commit that followed was refused.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from tda.core.db import Db
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
    make_session,
)

BENCH_ROI = [0, 40, 64, 64]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> AnnotationSession:
    made = make_session(tmp_path)
    yield made
    made.close(force=True)


def with_bench_roi(session) -> AnnotationSession:
    """Record a staging area for the open view's pose segment."""
    session.db.set_pose_segment(DESKTOP, VIEW, 1, 1, LAST_STEP, LAST_STEP, None, None)
    session.db.set_pose_segment_bench_roi(DESKTOP, VIEW, 1, BENCH_ROI)
    session.review.invalidate()
    return session


def kinds(session, instance: str) -> list[str]:
    return [i["kind"] for i in session.task_card() if i["instance"] == instance]


# --------------------------------------------------------------------------- #
# no staging area: no bench items anywhere
# --------------------------------------------------------------------------- #
def test_the_scanner_start_card_asks_for_no_bench_boxes(session):
    listed = session.task_card()
    assert all(item["kind"] != api.KIND_ADD_BENCH_BOX for item in listed)
    # the parts that are out of the machine are simply not this view's business
    assert COOLER not in {item["instance"] for item in listed}
    for screw in SCREWS:
        assert screw not in {item["instance"] for item in listed}
    assert CHASSIS in {item["instance"] for item in listed}


def test_without_a_staging_area_a_bench_part_is_no_problem(session):
    session.goto(LAST_STEP)
    problems = session.compiled().problems
    assert not [p for p in problems if p.startswith("bench_missing:")]


def test_forward_browsing_asks_for_nothing_bench_shaped(session):
    session.goto(13)
    session.browse_forward()
    assert kinds(session, COOLER) == []


# --------------------------------------------------------------------------- #
# with a staging area: bench items, and they are box work
# --------------------------------------------------------------------------- #
def test_the_start_card_asks_for_a_box_when_the_view_has_a_staging_area(session):
    with_bench_roi(session)
    items = {i["instance"]: i for i in session.task_card()}

    assert items[COOLER]["kind"] == api.KIND_ADD_BENCH_BOX
    assert "box" in items[COOLER]["text"]
    assert "Draw the staging-area box" in items[COOLER]["text"]
    assert CHASSIS in items and items[CHASSIS]["kind"] == api.KIND_ADD_SHAPE


def test_forward_browsing_asks_for_the_box_of_the_part_just_removed(session):
    with_bench_roi(session)
    session.goto(13)
    session.browse_forward()
    items = {i["instance"]: i for i in session.task_card()}

    assert items[COOLER]["kind"] == api.KIND_ADD_BENCH_BOX
    assert "staging-area box" in items[COOLER]["text"]
    assert items[COOLER]["done"] is False

    # ... and the kind is honest: the box tool writes it without a refusal
    session.commit_box(COOLER, (2.0, 44.0, 12.0, 54.0))
    items = {i["instance"]: i for i in session.task_card()}
    assert items[COOLER]["done"] is True


def test_a_bench_part_with_no_box_is_a_problem_once_there_is_a_staging_area(session):
    with_bench_roi(session)
    session.goto(LAST_STEP)
    session._invalidate()
    problems = session.compiled().problems
    assert any(p.startswith("bench_missing:") for p in problems)


def test_the_bench_box_still_ends_where_the_part_goes_back_in(session):
    with_bench_roi(session)
    session.commit_box(COOLER, (2.0, 44.0, 12.0, 54.0))
    session.goto(12)
    assert api.KIND_REMOVE_BENCH_BOX in kinds(session, COOLER)


# --------------------------------------------------------------------------- #
# the accessor itself
# --------------------------------------------------------------------------- #
def test_the_bench_roi_round_trips(tmp_path):
    db = Db(str(tmp_path / "roi.sqlite"))
    db.set_pose_segment(7, "oak2", 1, 1, 9, 9, None, None)
    assert db.bench_roi(7, "oak2", 1) is None

    db.set_pose_segment_bench_roi(7, "oak2", 1, (0, 40, 64, 64))
    assert db.bench_roi(7, "oak2", 1) == [0, 40, 64, 64]
    assert db.pose_segments(7, "oak2")[0]["bench_roi"] == [0, 40, 64, 64]

    db.set_pose_segment_bench_roi(7, "oak2", 1, None)
    assert db.bench_roi(7, "oak2", 1) is None
    db.close()


def test_a_malformed_bench_roi_is_refused(tmp_path):
    db = Db(str(tmp_path / "roi.sqlite"))
    db.set_pose_segment(7, "oak2", 1, 1, 9, 9, None, None)
    for bad in ((1, 2, 3), (10, 0, 2, 5), "nope"):
        with pytest.raises(ValueError):
            db.set_pose_segment_bench_roi(7, "oak2", 1, bad)
    db.close()


def test_the_bench_roi_is_clamped_and_logged_like_the_chassis_one(tmp_path):
    """One validator owns both rectangles: same clamping, same audit trail."""
    db = Db(str(tmp_path / "roi.sqlite"))
    db.set_pose_segment(7, "oak2", 1, 1, 9, 9, None, None)

    stored = db.set_pose_segment_bench_roi(7, "oak2", 1, (-5, 39.6, 10_000, 64),
                                           hw=(64, 64))

    assert stored == [0, 40, 64, 64]
    assert db.bench_roi(7, "oak2", 1) == stored
    logged = [op for op in db.ops(7, "oak2") if op["kind"] == "set_bench_roi"]
    assert logged and logged[0]["payload"]["roi"] == stored
    assert logged[0]["inverse"]["roi"] is None
    assert db.set_pose_segment_bench_roi(7, "oak2", 1, None) is None
    db.close()
