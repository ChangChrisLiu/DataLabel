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
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from app_scene import (
    close_window,
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
from tda.ui import session_api as api
from tda.ui.app import MainWindow
from tda.ui.app_assist import AssistController
from tda.ui.app_diff import _pixels_of


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
    # The diff map only proposes prompt boxes once the segment has an ROI, so
    # accept the one the window offers -- which is what an annotator does on
    # the first frame of a machine.
    if win.roi_editing:
        win.wait_for_roi_proposal()   # measured off-thread; Enter needs the box
        win.act_commit()
    yield win
    close_window(win)


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
    window.wait_for_roi_proposal()          # the box is measured off-thread
    window.act_commit()                     # accept the proposed ROI
    window.session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    x0, y0, x1, y1 = window.roi()
    for blob in payload["explained"] + payload["unexplained"]:
        bx0, by0, bx1, by1 = blob.box
        assert x0 <= bx0 and by0 >= y0 and bx1 <= x1 and by1 <= y1


def test_a_frame_without_a_neighbour_produces_no_blobs(window):
    """The card is written against ``j + 1``; the newest frame has no such side."""
    window.session.goto(max(window.session.steps()))
    assert window.assist.wait(5.0)
    QApplication.processEvents()
    assert window.assist_result is None


def test_the_comparison_is_against_the_task_card_neighbour(window):
    """The blob must mark what re-appears in the frame on screen, not what left."""
    from tda.ui import app_compat as compat

    window.session.goto(LAST_STEP - 2)
    payload = wait_for_assist(window)
    neighbour = compat.task_neighbour(window.session)
    assert neighbour == LAST_STEP - 1          # the frame the annotator came from
    assert payload["key"].step == LAST_STEP - 2


def test_a_commit_does_not_compare_the_same_two_frames_again(window):
    """Drawing changes what is *explained*, never what the difference is.

    A commit makes the session re-announce the frame, which used to throw the
    comparison away and start it over -- a 12 MP difference map, on two frames
    that had not changed, whose only effect was to make the ``Space`` that
    followed wait 130 ms for it.  The blobs are re-split instead.
    """
    from app_scene import cell

    window.session.goto(LAST_STEP - 2)
    before = wait_for_assist(window)
    asked: list = []
    real = window.assist.request
    window.assist.request = lambda *a, **k: (asked.append(a[0]), real(*a, **k))[1]

    instance = str(window.session.task_card()[0]["instance"])
    window.on_request_edit(instance)
    window.set_editing_mask(cell(3), undoable=True)
    window.act_commit()
    if window.scope_bar.isVisible() or window.warn_bar.isVisible():
        window.act_commit()
    QApplication.processEvents()

    assert window.session.editing_instance is None, "the commit was refused"
    assert asked == [], f"the commit asked for {len(asked)} more comparison(s)"
    assert window.assist_result is not None, "the commit threw the comparison away"
    assert window.assist_result["key"] == before["key"]
    assert window.assist_result["blobs"] == before["blobs"]

    # ... and stepping to another frame does start a new one
    window.act_clear_edit()
    window.session.clear_edit()
    window.session.goto(LAST_STEP - 3)
    assert len(asked) == 1, "a real frame change did not start a comparison"


def _blob(box, area: int = 40, score: float = 10.0):
    """One hand-made difference blob at a known box."""
    from tda.core.diffmap import DiffBlob

    return DiffBlob(box=tuple(int(v) for v in box), area=int(area), score=float(score))


def _card_wants_a_shape(window) -> bool:
    """Is the task card's current item one that a prompt box is armed for?"""
    rows = window.session.task_card()
    index = window.task_card.current_index()
    return (0 <= index < len(rows)
            and rows[index].get("kind") == api.KIND_ADD_SHAPE)


def _stand_on_a_card_item(window) -> None:
    """Put the window on a frame whose card asks for a shape to be drawn."""
    for step in range(LAST_STEP - 1, 1, -1):
        window.session.goto(step)
        wait_for_assist(window)
        if _card_wants_a_shape(window):
            return
    pytest.skip("no frame of the scene asks for a shape to be drawn")


def _re_announce(window) -> None:
    """What a commit does to the window: the session re-announces the frame."""
    window.on_frame_changed_assist(window.session.current())


def test_a_commit_re_arms_the_prompt_box_of_the_blob_it_did_not_explain(window):
    """The next SAM click after a commit must still carry the difference box.

    A re-announced frame clears the prompt box, both SAM tools' copies and the
    rubber band before it decides whether to compare again -- so the branch
    that decides *not* to has to put them back, or every click after a commit
    goes out point-only, which is the configuration that returned the whole
    chassis on 7 of 13 real frames.
    """
    _stand_on_a_card_item(window)
    roi = window.roi()
    assert roi is not None, "the fixture did not accept the proposed ROI"
    x0, y0, x1, y1 = roi
    drawn = _blob((x0 + 2, y0 + 2, x0 + 8, y0 + 8))
    wanted = _blob((x1 - 12, y1 - 12, x1 - 4, y1 - 4), area=30, score=9.0)

    window.assist_result = {"key": window.session.current(),
                            "blobs": [drawn, wanted], "delta": None,
                            "roi": roi, "expected": [],
                            "explained": [], "unexplained": [drawn, wanted]}
    window._assist_asked = window._assist_subject()
    # what the commit just added to the frame explains the first blob
    window.expected_now = lambda: [drawn.box]

    _re_announce(window)

    assert [b.box for b in window.assist_result["unexplained"]] == [wanted.box]
    box = tuple(float(v) for v in wanted.box)
    assert window._prompt_box == box
    assert window.sam_point.prompt_box == box
    assert window.sam_box.prompt_box == box
    assert window.canvas._rubber_band == box


def test_a_commit_that_explains_every_blob_arms_nothing(window):
    _stand_on_a_card_item(window)
    roi = window.roi()
    drawn = _blob((roi[0] + 2, roi[1] + 2, roi[0] + 8, roi[1] + 8))
    window.assist_result = {"key": window.session.current(), "blobs": [drawn],
                            "delta": None, "roi": roi, "expected": [],
                            "explained": [], "unexplained": [drawn]}
    window._assist_asked = window._assist_subject()
    window.expected_now = lambda: [drawn.box]

    _re_announce(window)

    assert window.assist_result["unexplained"] == []
    assert window._prompt_box is None
    assert window.sam_point.prompt_box is None
    assert window.canvas._rubber_band is None


def test_the_heat_map_survives_a_commit_while_d_is_on(window):
    """``D`` is a display choice about the frame, not about the last edit.

    Main repainted it when the comparison it started came back; the branch that
    starts none has to repaint it itself, or the map on screen is whatever was
    drawn before and nothing says so.
    """
    _stand_on_a_card_item(window)
    window.act_toggle_heat()
    assert window.heat_visible is True
    assert window.heat_item.isVisible() is True
    window._assist_asked = window._assist_subject()
    painted: list = []
    real = window._paint_heat
    window._paint_heat = lambda: (painted.append(1), real())[1]

    _re_announce(window)

    assert painted, "the heat map was not repainted"
    assert window.heat_item.isVisible() is True


def test_a_new_roi_starts_a_new_comparison(window):
    """The ROI is what the difference map looks *inside*: a new one is a new answer."""
    window.session.goto(LAST_STEP - 2)
    wait_for_assist(window)
    asked: list = []
    real = window.assist.request
    window.assist.request = lambda *a, **k: (asked.append(a[3]), real(*a, **k))[1]

    window.start_roi_edit()
    window.on_roi_box(4.0, 4.0, 40.0, 40.0)
    window.accept_roi()
    QApplication.processEvents()
    assert asked and asked[-1] is not None
    assert tuple(int(v) for v in asked[-1]) == tuple(int(v) for v in window.roi())


def test_the_neighbour_is_read_by_the_worker_when_the_gui_has_not_got_it(window):
    """A timeline click must not decode 12 MP to hand a worker an array.

    Stepping back, ``k+1`` is the frame just left and is already in the image
    cache, so the pixels go straight over.  A jump to a frame nobody has
    visited has no such luck, and decoding its neighbour on the GUI thread was
    46 ms of a 12 MP timeline click spent on work the worker could do itself.
    """
    session = window.session
    target = LAST_STEP - 4
    neighbour = target + 1

    session.goto(neighbour)     # ... and then step back, the way annotating goes
    session.goto(target)
    assert session.peek_image_at(neighbour) is not None   # walked here: in hand
    assert isinstance(window._neighbour_pixels(neighbour), np.ndarray)

    session.images.clear()
    handed = window._neighbour_pixels(neighbour)
    assert isinstance(handed, str), f"the GUI thread decoded {type(handed)}"
    assert session.peek_image_at(neighbour) is None, \
        "asking for the path decoded the image anyway"

    # ... and the answer is the same one the pixels would have produced
    controller = AssistController()
    try:
        image = session.image_at(target)
        by_path = controller.compute(
            FrameKey(DESKTOP, target, VIEW),
            image, _pixels_of(handed), None, [])
        by_array = controller.compute(
            FrameKey(DESKTOP, target, VIEW),
            image, session.image_at(neighbour), None, [])
    finally:
        controller.shutdown()
    assert np.array_equal(by_path["delta"], by_array["delta"])
    assert [b.box for b in by_path["unexplained"]] == \
        [b.box for b in by_array["unexplained"]]


def test_an_unreadable_neighbour_path_is_no_comparison_rather_than_a_failure(qapp):
    """The same answer a frame with no neighbour gives: ``None``, not an error."""
    controller = AssistController()
    results: list = []
    controller.sigBlobs.connect(results.append)
    failures: list = []
    controller.sigFailed.connect(failures.append)
    try:
        image = np.full((32, 32, 3), 30, dtype=np.uint8)
        controller.request(FrameKey(1, 2, VIEW), image, "does/not/exist.png", None)
        assert controller.wait(5.0)
        QApplication.processEvents()
    finally:
        controller.shutdown()
    assert results == [None], f"an unreadable neighbour gave {results!r}"
    assert failures == []


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


def test_the_prompt_box_survives_a_detour_through_the_brush(window):
    """``detach()`` clears the tool's box, so re-arming has to put it back."""
    window.session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    if not payload["unexplained"]:
        pytest.skip("the synthetic frames produced no unexplained change")
    window.act_tool("sam_point")
    window.begin_add_shape(payload["unexplained"][0])
    box = window.sam_point.prompt_box
    window.act_tool("brush")
    assert window.sam_point.prompt_box is None      # the tool was detached
    window.act_tool("sam_point")
    assert window.sam_point.prompt_box == box


def test_a_frame_without_an_image_unsets_the_frame_token(qapp, tmp_path):
    """No token means the tools refuse to prompt, which is what we want there."""
    session = make_session(tmp_path, missing=(LAST_STEP - 1,))
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=StubSamQueue())
    try:
        win.session.goto(LAST_STEP - 1)
        assert win.sam_point.frame_token is None
        win.session.goto(LAST_STEP)
        assert win.sam_point.frame_token == FrameKey(DESKTOP, LAST_STEP, VIEW)
    finally:
        close_window(win)


def test_the_sam_tools_always_know_which_instance_they_write(window):
    instance = start_edit(window)
    assert window.sam_point.instance == instance
    assert window.sam_box.instance == instance
    window.act_clear_edit()
    assert window.sam_point.instance is None


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


def test_a_failed_inference_reaches_the_status_bar(qapp, tmp_path):
    """The queue's on_error hook existed but nobody passed it."""
    from tda.models.sam_service import SamQueue

    class _Boom:
        def predict(self, req):
            raise RuntimeError("CUDA out of memory")

    session = make_session(tmp_path)
    queue = SamQueue(_Boom())
    win = MainWindow(session, make_paths(tmp_path), "tester", sam_queue=queue)
    try:
        win.resize(900, 700)
        win.show()
        QApplication.processEvents()
        win.set_mode(A.MODE_ANNOTATE)
        if win.roi_editing:
            win.wait_for_roi_proposal()
            win.act_commit()
        start_edit(win)
        win.act_tool("sam_point")
        win.sam_point.on_press(32.0, 32.0, None)
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            QApplication.processEvents()
            if "out of memory" in win.last_error_message():
                break
            time.sleep(0.005)
        assert "out of memory" in win.last_error_message()
    finally:
        queue.stop()
        close_window(win)


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
        close_window(win)


# --------------------------------------------------------------------------- #
# the unexplained queue and the heat map
# --------------------------------------------------------------------------- #
def test_unexplained_blobs_are_handed_to_the_session_on_confirm(window, monkeypatch):
    handed: list[tuple] = []
    monkeypatch.setattr(window.session, "set_unexplained",
                        lambda step, boxes: handed.append((step, list(boxes))),
                        raising=False)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP - 1)
    wait_for_assist(window)
    step = window.session.current().step
    window.act_confirm()
    assert handed and handed[0][0] == step
    assert step not in window.unanalysed


def test_a_frame_whose_comparison_never_finished_is_recorded_as_not_analysed(
        window, monkeypatch):
    """An empty list would claim a check that never happened."""
    handed: list[tuple] = []
    monkeypatch.setattr(window.session, "set_unexplained",
                        lambda step, boxes: handed.append((step, list(boxes))),
                        raising=False)
    seed_shapes(window.session, LAST_STEP)
    window.session.goto(LAST_STEP - 1)
    window.assist_result = None
    monkeypatch.setattr(window.assist, "wait", lambda timeout=1.0: False)
    step = window.session.current().step
    window.act_confirm()
    assert handed == []
    assert step in window.unanalysed
    assert "not analysed" in window.last_error_message()


def test_expected_boxes_come_from_the_masks_not_from_the_box_column(qapp):
    """The old version read ``compiled.box``, which only bench rows ever fill."""
    from tda.ui.app_assist import expected_boxes, expected_payload

    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 30:44] = True
    payload = expected_payload({"part.01": mask}, [(1, 2, 3, 4)])
    assert sorted(expected_boxes(payload)) == [(1, 2, 3, 4), (30, 10, 44, 20)]
    assert expected_boxes(()) == []


def test_a_drawn_part_stops_being_an_unexplained_difference(window, monkeypatch):
    """Before the shape exists the blob is unexplained; after it, explained.

    The instance is named directly because the task card's own rule is being
    changed in the session at the same time (it will list the work of the frame
    on screen); what is under test here is the window's mask -> box -> explain
    path, which is the same either way.
    """
    from tda.core import masks as _masks
    from tda.core.model import ShapeKeyframe, ShapePart, ZOrderRec

    session = window.session
    session.goto(LAST_STEP - 1)
    payload = wait_for_assist(window)
    blobs = list(payload["blobs"])
    assert blobs, "the synthetic frames must differ somewhere"
    assert payload["unexplained"], "nothing is drawn yet, so nothing is explained"

    target = "chassis"
    x0, y0, x1, y1 = blobs[0].box
    mask = np.zeros((64, 64), dtype=bool)
    mask[y0:y1, x0:x1] = True
    session.db.add_keyframe(ShapeKeyframe(
        id=None, instance=target, desktop=DESKTOP, view=VIEW, pose_segment=1,
        anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
        parts=[ShapePart("main", _masks.encode_rle(mask))],
    ))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(target, "main")]))
    session.refresh_all()
    monkeypatch.setattr(window, "_card_instances", lambda: {target})

    assert expected_covers(window, blobs[0].box)
    after = window.re_explain()
    assert any(b.box == blobs[0].box for b in after["explained"])
    assert all(b.box != blobs[0].box for b in after["unexplained"])


def expected_covers(win: MainWindow, box) -> bool:
    from tda.ui.app_assist import expected_boxes

    return any(tuple(b) == tuple(box) for b in expected_boxes(win.expected_now()))


def test_the_diff_heat_toggle_paints_and_clears_an_overlay_item(window):
    window.session.goto(LAST_STEP - 1)
    wait_for_assist(window)
    window.act_toggle_heat()
    assert window.heat_visible is True
    assert window.heat_item.pixmap().isNull() is False
    window.act_toggle_heat()
    assert window.heat_visible is False


# --------------------------------------------------------------------------- #
# the points belong to one prompt (final review, item 1)
# --------------------------------------------------------------------------- #
def points_of(win: MainWindow) -> list:
    return list(win.sam_point.points)


def last_request(win: MainWindow):
    return win.sam_queue.requests[-1]


def test_the_points_do_not_follow_the_annotator_to_the_next_instance(window):
    """S, click A, Enter, activate B, click B sent both points as one prompt.

    Every mask after the first commit was a union over everything clicked in
    the session, and ``multimask=False`` meant ``C`` offered nothing to fix it.
    """
    card = [str(r["instance"]) for r in window.session.task_card() if r.get("instance")]
    window.task_card.sigRequestEdit.emit(card[0])
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    assert len(last_request(window).points) == 1
    window.act_commit()                      # Enter: part A is written

    window.task_card.sigRequestEdit.emit(card[1])
    window.sam_point.on_press(20.0, 20.0, None)

    assert len(last_request(window).points) == 1, points_of(window)
    assert last_request(window).multimask is True


def test_the_points_do_not_follow_the_annotator_to_the_next_frame(window):
    """Two of the three points were in the previous frame's coordinates."""
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()

    window.session.goto(LAST_STEP - 1, force=True)   # S is still armed
    QApplication.processEvents()
    window.sam_point.on_press(20.0, 20.0, None)

    assert len(last_request(window).points) == 1, points_of(window)


def test_a_commit_forgets_the_points(window):
    """The mask is written; the clicks that made it are not a prompt any more."""
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    window.act_commit()
    assert points_of(window) == []


def test_esc_forgets_the_points(window):
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    window.act_clear_edit()
    assert points_of(window) == []


def test_an_undo_that_replaces_the_layer_forgets_the_points(window):
    """The layer the points were refining is gone, so they refine nothing."""
    start_edit(window)
    window.act_tool("sam_point")
    window.sam_point.on_press(32.0, 32.0, None)
    window.sam_queue.flush()
    QApplication.processEvents()
    assert points_of(window)

    window.act_undo()

    assert points_of(window) == []


def test_holding_pgdn_does_not_queue_a_comparison_per_frame(window):
    """Ten repeats, one pending comparison: the mailbox holds the newest only.

    Stepping on auto-repeat is only usable if the work it triggers coalesces --
    a diff thread per skipped frame was 0.8 GB at scanner resolution.
    """
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent as _QEvent

    window.set_mode(A.MODE_ANNOTATE)
    steps = sorted(window.session.steps())
    window.session.goto(steps[-1], force=True)
    QApplication.processEvents()

    def repeat() -> QKeyEvent:
        return QKeyEvent(_QEvent.Type.KeyPress, int(Qt.Key.Key_PageDown),
                         Qt.KeyboardModifier.NoModifier, 0, 0, 0, "", True)

    moved = 0
    for _ in range(10):
        before = window.session.current().step
        window.handle_key(repeat())
        moved += int(window.session.current().step != before)

    assert moved >= 5, "the repeats did not step the frame"
    assert window.assist.queued() <= 1
    assert window.sam_queue.pending() == 0


def test_a_blob_covering_most_of_the_roi_is_not_a_prompt_box(window):
    """"Everything changed" is not a prompt; it is the absence of one.

    A box over 60 % of the ROI tells SAM nothing it did not already know, and
    on the rehearsal that is exactly when the single candidate came back as the
    whole chassis.
    """
    from tda.core.diffmap import DiffBlob

    roi = window.roi()
    assert roi is not None
    x0, y0, x1, y1 = roi
    window.act_tool("sam_point")

    big = DiffBlob(box=(float(x0), float(y0), float(x1), float(y1)),
                   area=(x1 - x0) * (y1 - y0), score=9.0)
    window.begin_add_shape(big)
    assert window.sam_point.prompt_box is None
    assert "整块" in window.status_message() or "whole" in window.status_message()

    w, h = (x1 - x0) // 4, (y1 - y0) // 4
    small = DiffBlob(box=(float(x0), float(y0), float(x0 + w), float(y0 + h)),
                     area=w * h, score=9.0)
    window.begin_add_shape(small)
    assert window.sam_point.prompt_box is not None


# --------------------------------------------------------------------------- #
# the checkpoint comes from the paths the app was started with (item 19)
# --------------------------------------------------------------------------- #
def test_the_window_passes_its_own_weights_dir_to_sam(window, monkeypatch, tmp_path):
    """``default_checkpoint()`` read ``<repo>/configs/paths.yaml`` and ignored --paths.

    Started with another paths file -- a second data disk, a colleague's copy --
    the app looked for the repo's checkpoint instead of the configured one.
    """
    import tda.models.sam_service as sam_service

    seen: list[dict] = []

    class FakeService:
        def __init__(self, checkpoint=None, **kwargs):
            seen.append({"checkpoint": checkpoint, **kwargs})

    monkeypatch.setattr(sam_service, "SamService", FakeService)
    monkeypatch.setattr(sam_service, "SamQueue", lambda service: service)
    window.sam_queue = None
    window._sam_loading = False
    window.paths = dict(window.paths, weights_dir=str(tmp_path / "w"))

    window.start_sam()
    window._sam_loader_thread.join(10.0)
    QApplication.processEvents()

    assert seen, "SAM was never constructed"
    assert str(tmp_path / "w") in str(seen[0]["checkpoint"])


def test_default_checkpoint_still_works_for_library_use():
    from tda.models.sam_service import CHECKPOINT_NAME, checkpoint_in, default_checkpoint

    fallback = default_checkpoint()
    assert fallback is None or str(fallback).endswith(CHECKPOINT_NAME)
    picked = checkpoint_in({"weights_dir": "D:/somewhere"})
    assert str(picked).endswith(CHECKPOINT_NAME) and "somewhere" in str(picked)
