"""Model assist wiring: difference map, SAM prompts and candidates (task 13b).

The difference map runs off the GUI thread, so the payload carries the thread it
was computed on and the tests assert on it.  SAM answers come from the stub
queue of :mod:`tests.app_scene`, which only replies when a test tells it to --
that is what makes "a result for the previous frame is dropped" testable.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app_scene import (
    DESKTOP,
    LAST_STEP,
    VIEW,
    StubSamQueue,
    make_paths,
    make_session,
    seed_shapes,
)
from tda.core.model import FrameKey
from tda.ui import app_actions as A
from tda.ui.app import MainWindow
from tda.ui.app_assist import AssistController


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = make_session(tmp_path)
    queue = StubSamQueue()
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=queue)
    win.resize(900, 700)
    win.show()
    QApplication.processEvents()
    win.set_mode(A.MODE_ANNOTATE)
    win.sam_queue = queue
    yield win
    win.shutdown()


def wait_for_assist(win: MainWindow, timeout: float = 5.0) -> dict:
    assert win.assist.wait(timeout), "the difference map did not finish"
    QApplication.processEvents()
    payload = win.assist_result
    assert payload is not None
    return payload


# --------------------------------------------------------------------------- #
# the difference map
# --------------------------------------------------------------------------- #
def test_the_difference_map_runs_off_the_gui_thread(window):
    window.session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    assert payload["key"] == FrameKey(DESKTOP, LAST_STEP - 1, VIEW)
    assert payload["thread"] != threading.current_thread().name
    assert "explained" in payload and "unexplained" in payload


def test_the_difference_map_is_restricted_to_the_roi(window):
    window.act_commit()                     # accept the proposed ROI
    window.session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    x0, y0, x1, y1 = window.roi()
    for blob in payload["explained"] + payload["unexplained"]:
        bx0, by0, bx1, by1 = blob.box
        assert x0 <= bx0 and by0 >= y0 and bx1 <= x1 and by1 <= y1


def test_a_frame_without_a_predecessor_produces_no_blobs(window):
    window.session.goto(min(window.session.steps()))
    assert window.assist.wait(5.0)
    QApplication.processEvents()
    assert window.assist_result is None or not window.assist_result["unexplained"]


def test_assist_controller_runs_synchronously_when_asked(qapp):
    controller = AssistController()
    try:
        a = np.full((64, 64, 3), 30, dtype=np.uint8)
        b = a.copy()
        b[20:40, 20:40] = 220
        payload = controller.compute(FrameKey(1, 2, VIEW), b, a, (0, 0, 64, 64), [])
        assert payload["unexplained"], "a bright square is a change"
        assert payload["unexplained"][0].area > 0
    finally:
        controller.shutdown()


# --------------------------------------------------------------------------- #
# the prompt box
# --------------------------------------------------------------------------- #
def test_the_best_unexplained_blob_becomes_the_sam_prompt_box(window):
    window.session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    if not payload["unexplained"]:
        pytest.skip("the synthetic frames produced no unexplained change")
    window.act_tool("sam_point")
    window.begin_add_shape(payload["unexplained"][0])
    assert window.sam_point.prompt_box is not None
    assert tuple(int(v) for v in window.sam_point.prompt_box) == \
        tuple(int(v) for v in payload["unexplained"][0].box)


def test_the_prompt_box_is_cleared_on_a_frame_change(window):
    window.act_tool("sam_point")
    window.sam_point.set_prompt_box((1.0, 2.0, 30.0, 40.0))
    window.session.goto(LAST_STEP - 2)
    assert window.sam_point.prompt_box is None


# --------------------------------------------------------------------------- #
# SAM results
# --------------------------------------------------------------------------- #
def start_edit(win: MainWindow) -> str:
    card = [row for row in win.session.task_card() if row.get("instance")]
    instance = str(card[0]["instance"])
    win.task_card.sigRequestEdit.emit(instance)
    return instance


def test_a_sam_click_lands_in_the_editing_layer(window):
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    assert window.sam_queue.pending() == 1
    window.sam_queue.flush()
    QApplication.processEvents()
    assert window.overlay.editing.any()
    assert np.array_equal(window.session.editing_mask(), window.overlay.editing)


def test_c_cycles_the_candidates_and_the_status_bar_counts_them(window):
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush(multimask=True)
    QApplication.processEvents()
    assert window.sam_point.candidate_count == 3
    assert "1/3" in window.sam_label.text()
    first = window.overlay.editing.copy()
    window.act_cycle_candidate()
    assert not np.array_equal(window.overlay.editing, first)
    assert "2/3" in window.sam_label.text()


def test_a_result_for_the_previous_frame_is_dropped(window):
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.session.goto(LAST_STEP - 1)       # the answer is now stale
    untouched = window.overlay.editing.copy()
    window.sam_queue.flush()
    QApplication.processEvents()
    assert np.array_equal(window.overlay.editing, untouched)
    assert "dropped" in window.last_error_message().lower()


def test_sam_hints_and_errors_reach_the_status_bar(window):
    window.sam_point.sigHint.emit("a hint for the annotator")
    assert "a hint for the annotator" in window.status_message()
    window.sam_point.sigError.emit("something was dropped")
    assert "something was dropped" in window.status_message()


def test_an_unavailable_sam_disables_the_sam_tools_with_a_reason(qapp, tmp_path):
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=None)
    try:
        win.set_sam_unavailable("no checkpoint in D:/DataSet/models/weights")
        assert win.sam_available is False
        assert "no checkpoint" in win.sam_label.text()
        win.act_tool("sam_point")
        assert win.active_tool is not win.sam_point   # the tool refuses to arm
    finally:
        win.shutdown()


# --------------------------------------------------------------------------- #
# the unexplained queue and the heat map
# --------------------------------------------------------------------------- #
def test_unexplained_blobs_are_handed_to_the_session_on_confirm(window, monkeypatch):
    handed: list[tuple] = []
    monkeypatch.setattr(window.session, "set_unexplained",
                        lambda step, boxes: handed.append((step, list(boxes))),
                        raising=False)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP)
    wait_for_assist(window)
    window.act_confirm()
    assert handed and handed[0][0] == LAST_STEP


def test_the_diff_heat_toggle_paints_and_clears_an_overlay_item(window):
    window.session.goto(LAST_STEP - 1)
    wait_for_assist(window)
    window.act_toggle_heat()
    assert window.heat_visible is True
    assert window.heat_item.pixmap().isNull() is False
    window.act_toggle_heat()
    assert window.heat_visible is False
