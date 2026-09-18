"""The task card describes the frame on screen (spec 4.2, controller ruling).

Reverse-order annotation arrives at frame ``j`` from ``j + 1``, which is already
done.  The work that belongs to ``j`` is therefore the difference between the
two *evaluated at j*: a part that is ``removed`` at ``j + 1`` and installed at
``j`` is a part the annotator can see, and has to draw, on the image in front of
them.

The card used to be one frame out -- it listed, at ``k``, the work of the
transition ``k -> k - 1``.  That asked for a shape on a frame where the part is
removed (correctly refused, since bench geometry is a box) and, after ``Space``
stepped back, the item was gone from the card.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tda.core import masks
from tda.core.model import FrameKey
from tda.ui import session_api as api
from tda.ui.session import AnnotationSession
from session_scene import (
    CHASSIS,
    COOLER,
    DESKTOP,
    FAN_CONNECTOR,
    LAST_STEP,
    SCREWS,
    VIEW,
    cell,
    draw,
    make_session,
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
    made.close()


def card(session) -> dict[str, dict]:
    return {item["instance"]: item for item in session.task_card()}


# --------------------------------------------------------------------------- #
# the card belongs to the frame on screen
# --------------------------------------------------------------------------- #
def test_the_cooler_is_asked_for_on_the_frame_where_it_is_back(session):
    """Action 13 removes the cooler, so frame 12 is the one that still has it."""
    session.goto(12)
    items = card(session)

    assert items[COOLER]["kind"] == api.KIND_ADD_SHAPE
    for screw in SCREWS:
        assert items[screw]["kind"] == api.KIND_ADD_SHAPE
    assert session.task_card()[0]["instance"] == COOLER  # the parent leads


def test_the_frame_where_the_cooler_is_gone_does_not_ask_for_it(session):
    session.goto(13)
    assert COOLER not in card(session)


def test_the_card_item_can_actually_be_drawn_and_goes_done(session):
    session.goto(12)
    assert card(session)[COOLER]["done"] is False

    session.begin_edit(COOLER)
    session.set_editing_mask(cell(3))
    result = session.commit_edit(api.SCOPE_KEYFRAME)  # no refusal: it is in the chassis

    assert result["changed"] is True
    assert card(session)[COOLER]["done"] is True
    row = session.db.compiled(FrameKey(DESKTOP, 12, VIEW))[COOLER]
    assert np.array_equal(masks.decode_rle(row["visible_rle"]), cell(3))


def test_a_latch_and_a_connector_ask_for_a_split_on_their_own_frame(session):
    session.goto(6)  # step 7 opens the SSD drive latch, so frame 6 still has it shut
    assert card(session)["drive_latch.01"]["kind"] == api.KIND_SPLIT_KEYFRAME

    session.goto(11)  # step 12 unplugs the CPU-fan connector
    assert card(session)[FAN_CONNECTOR]["kind"] == api.KIND_SPLIT_KEYFRAME


def test_a_screw_that_was_only_loosened_is_state_only(session):
    session.goto(2)  # step 3 unscrews the first cooler screw
    assert card(session)[SCREWS[0]]["kind"] == api.KIND_STATE_ONLY


def test_a_bench_box_ends_on_the_frame_the_part_is_back_in(session):
    session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))  # drawn on the start frame
    session.goto(12)
    kinds = [item["kind"] for item in session.task_card() if item["instance"] == COOLER]
    assert api.KIND_REMOVE_BENCH_BOX in kinds


# --------------------------------------------------------------------------- #
# the start frame
# --------------------------------------------------------------------------- #
def test_the_start_frame_lists_everything_that_still_needs_a_shape(session):
    assert session.current().step == LAST_STEP
    items = session.task_card()
    listed = [item["instance"] for item in items]

    assert CHASSIS in listed and "motherboard.01" in listed
    assert all(item["kind"] == api.KIND_ADD_SHAPE for item in items)
    assert all(item["done"] is False for item in items)
    # the chassis is drawn first: everything else is layered on top of it
    assert listed[0] == CHASSIS


def test_a_drawn_instance_leaves_the_start_frame_card(session):
    session.begin_edit(CHASSIS)
    session.set_editing_mask(cell(1))
    session.commit_edit(api.SCOPE_KEYFRAME)
    assert CHASSIS not in card(session)


def test_the_start_frame_asks_for_a_confirmation_once_it_is_drawn(qapp, tmp_path):
    session = make_session(tmp_path)
    seed_shapes(session, LAST_STEP)
    session.commit_box(COOLER, (2.0, 2.0, 12.0, 12.0))
    for screw in SCREWS:
        session.commit_box(screw, (14.0, 2.0, 22.0, 12.0))
    for key in ("storage_drive.ssd.01", "drive_cage.01"):
        session.commit_box(key, (24.0, 2.0, 34.0, 12.0))

    kinds = {item["kind"] for item in session.task_card()}
    assert kinds == {api.KIND_CONFIRM}
    session.close()


# --------------------------------------------------------------------------- #
# which frame the card is diffed against
# --------------------------------------------------------------------------- #
def test_task_neighbour_is_the_frame_we_came_from(session):
    assert session.task_neighbour() is None  # the start frame has nobody after it
    session.goto(10)
    assert session.task_neighbour() == 11


def test_task_neighbour_skips_a_frame_with_no_image(qapp, tmp_path):
    session = make_session(tmp_path, missing=(11,))
    session.goto(10)
    assert session.task_neighbour() == 12
    session.close()


def test_task_neighbour_follows_the_browsing_direction(session):
    session.goto(10)
    session.browse_forward()
    assert session.task_neighbour() == 9
    session.browse_reverse()
    assert session.task_neighbour() == 11


def test_flash_compare_shows_the_frame_the_card_is_about(session):
    session.goto(10)
    assert np.array_equal(session.flash_compare(), session.image_at(11))
    assert np.array_equal(session.flash_compare(other=True), session.image_at(9))


def test_browsing_forward_asks_for_the_bench_box_of_the_part_just_removed(session):
    session.goto(13)  # step 13 removed the cooler; going forward we arrive here
    session.browse_forward()
    items = card(session)
    assert items[COOLER]["kind"] == api.KIND_ADD_SHAPE
    assert items[COOLER]["done"] is False


# --------------------------------------------------------------------------- #
# confirming names the item (controller ruling 4)
# --------------------------------------------------------------------------- #
def test_a_refused_confirmation_names_the_shape_to_draw(session):
    session.goto(12)
    seed_shapes(session, 12, skip=(CHASSIS,))
    problems: list[list[str]] = []
    session.sigProblems.connect(problems.append)

    assert session.confirm_frame() is False
    assert any(f"draw {CHASSIS} on this frame" in text for text in problems[-1])
    assert f"missing_shape:{CHASSIS}" in problems[-1]  # the raw problem is still there


# --------------------------------------------------------------------------- #
# preview of a layering scope (controller ruling 5)
# --------------------------------------------------------------------------- #
def test_preview_of_a_zorder_scope_reports_the_pairs_reach(session):
    session.goto(10)
    draw(session, CHASSIS, cell(1), api.SCOPE_KEYFRAME)
    draw(session, COOLER, cell(2), api.SCOPE_KEYFRAME)
    session.begin_edit(COOLER)

    pair = session.preview(f"zorder:above:{CHASSIS}")
    # both carry geometry while the cooler is still in the chassis: steps 1-12
    assert pair["steps"] == list(range(1, 13))
    assert pair["steps"] != session.preview(api.SCOPE_FRAME_OVERRIDE)["steps"]


def test_a_refusal_is_a_session_refusal(session):
    session.goto(14)
    with pytest.raises(api.SessionRefusal):
        session.begin_edit(COOLER)
        session.set_editing_mask(cell(0))
        session.commit_edit(api.SCOPE_KEYFRAME)
