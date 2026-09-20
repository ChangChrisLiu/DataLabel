"""Offscreen tests for the asynchronous SAM prompt tools (spec 4.6).

Split out of ``tests/test_canvas.py`` together with ``tda/ui/canvas/sam_tools.py``.
No GPU is involved: every test drives the tools through a stub queue, which is
also the only way to control *when* a result comes back -- most of what is
tested here is what happens when it comes back too late.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from tda.core import masks as M
from tda.core.model import FrameKey
from tda.models.sam_service import SamRequest, SamResult
from tda.ui.canvas.overlay import LabelOverlay
from tda.ui.canvas.sam_tools import (
    ERR_FRAME_CHANGED,
    ERR_NO_FRAME_TOKEN,
    ERR_OUT_OF_BOUNDS,
    HINT_EDITED,
    SamBoxTool,
    SamPointTool,
)
from tda.ui.canvas.tools import BrushTool
from tda.ui.canvas.view import ImageCanvas
from tda.ui.commands import UndoStack, edit_editing_mask_op

MAIN_THREAD = threading.get_ident()


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _rgb(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = np.arange(w, dtype=np.uint8)[None, :]
    img[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    img[..., 2] = 90
    return img


def _shown(canvas: ImageCanvas, w: int = 400, h: int = 400) -> ImageCanvas:
    canvas.resize(w, h)
    canvas.show()
    QApplication.processEvents()
    return canvas


def _press(x: float, y: float, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    pos = QPointF(x, y)
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        pos,
        pos,
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )


def _spin(predicate, timeout: float = 5.0) -> bool:
    """Run the Qt event loop until ``predicate()`` or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _drain(rounds: int = 6) -> None:
    """Give every queued signal a chance to run, for "nothing happened" checks."""
    for _ in range(rounds):
        QApplication.processEvents()
        time.sleep(0.005)


#: The frame every tool in this file is told it is prompting on.
FRAME_K = FrameKey(13, 12, "scan")


@pytest.fixture(params=["preset-instance", "fresh-overlay"])
def rig(request, qapp):
    """Canvas + overlay, with and without an editing instance already chosen.

    ``fresh-overlay`` is the production state before the first mask of a frame:
    ``editing_instance`` is still ``None`` and the tool's own fallback is what
    names the instance. Every core test runs under both, because a tool that
    only works once somebody else has called ``set_editing`` is a tool that
    never works on the first part of a frame.
    """
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(60, 80))
    ov = LabelOverlay((60, 80))
    if request.param == "preset-instance":
        ov.set_editing("inst-x", np.zeros((60, 80), dtype=bool))
    canvas.set_overlay(ov)
    return canvas, ov


@pytest.fixture
def zoomed(rig):
    """The rig with the whole 60x80 image visible: crop coords == image coords."""
    canvas, ov = rig
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    return canvas, ov


def _point_tool(canvas, ov, queue=None, *, token=FRAME_K, cls=None, **kwargs):
    """A point tool that has been told which frame it is on (see fix round 2)."""
    tool = (cls or SamPointTool)(canvas, ov, queue, **kwargs)
    if token is not None:
        tool.set_frame_token(token)
    return tool


def _box_tool(canvas, ov, queue=None, *, token=FRAME_K, **kwargs) -> SamBoxTool:
    tool = SamBoxTool(canvas, ov, queue, **kwargs)
    if token is not None:
        tool.set_frame_token(token)
    return tool


class StubQueue:
    """Stands in for ``SamQueue``: records requests, replies synchronously."""

    def __init__(self, maker=None) -> None:
        self.requests: list[SamRequest] = []
        self._maker = maker

    def submit(self, req: SamRequest, cb) -> None:
        self.requests.append(req)
        if self._maker is not None:
            cb(self._maker(req))

    @property
    def last(self) -> SamRequest:
        return self.requests[-1]


class ThreadedQueue(StubQueue):
    """Replies from a *worker* thread, like the real ``SamQueue`` does."""

    def __init__(self, maker) -> None:
        super().__init__(None)
        self._maker = maker
        self.threads: list[threading.Thread] = []

    def submit(self, req: SamRequest, cb) -> None:
        self.requests.append(req)
        maker = self._maker
        thread = threading.Thread(target=lambda: cb(maker(req)), daemon=True)
        self.threads.append(thread)
        thread.start()

    def join(self) -> None:
        for thread in self.threads:
            thread.join(5.0)
            assert not thread.is_alive()


class HoldingQueue(StubQueue):
    """Never answers until :meth:`flush`, which can also answer out of order."""

    def __init__(self, maker) -> None:
        super().__init__(None)
        self._maker = maker
        self.pending: list[tuple[SamRequest, object]] = []

    def submit(self, req: SamRequest, cb) -> None:
        self.requests.append(req)
        self.pending.append((req, cb))

    def flush(self, newest_first: bool = False) -> None:
        pending = list(reversed(self.pending)) if newest_first else list(self.pending)
        self.pending = []
        for req, cb in pending:
            cb(self._maker(req))


def _blob_result(req: SamRequest) -> SamResult:
    h, w = req.image_crop.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 4 : h // 2, w // 4 : w // 2] = True
    return SamResult(mask=mask, score=0.9, ms=1.0)


def _tagged_result(req: SamRequest) -> SamResult:
    """A 3x3 blob at the newest point, so the applied result is identifiable."""
    h, w = req.image_crop.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    px, py, _ = req.points[-1]
    mask[int(py) - 1 : int(py) + 2, int(px) - 1 : int(px) + 2] = True
    return SamResult(mask=mask, score=0.9, ms=1.0)


def _multi_result(req: SamRequest) -> SamResult:
    """Three nested candidates (part -> sub-assembly -> assembly), best first."""
    h, w = req.image_crop.shape[:2]
    candidates = []
    for divisor in (4, 3, 2):
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4 : h // 4 + h // divisor, w // 4 : w // 4 + w // divisor] = True
        candidates.append(mask)
    return SamResult(
        mask=candidates[0],
        score=0.9,
        ms=1.0,
        candidates=candidates,
        scores=[0.9, 0.8, 0.7],
    )


def _wire_undo(tool, overlay) -> UndoStack:
    """A real UndoStack fed by ``sigStroke`` + ``stroke_before``, as the app will."""
    stack = UndoStack()

    def set_mask(payload: dict) -> None:
        overlay.set_editing(payload["instance"], M.decode_rle(payload["rle_after"]))

    stack.register("edit_editing_mask", set_mask, set_mask)

    def on_stroke(_rect) -> None:
        assert tool.stroke_before is not None, "no before-snapshot to build an op from"
        stack.push(
            edit_editing_mask_op(
                overlay.editing_instance, tool.stroke_before, overlay.editing.copy()
            ),
            apply=False,
        )

    tool.sigStroke.connect(on_stroke)
    return stack


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------
def test_sam_point_tool_submits_a_viewport_crop_with_crop_coords(zoomed):
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)

    tool.on_press(20.0, 30.0, _press(0, 0, Qt.MouseButton.LeftButton))
    req = queue.last
    assert req.image_crop.shape == (60, 80, 3)
    assert req.image_crop.dtype == np.uint8
    assert req.points == [(20.0, 30.0, 1)]
    assert req.mask_input is None
    assert req.multimask is True  # a single point asks for candidates


def test_sam_point_tool_right_click_is_a_negative_point(zoomed):
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, _press(0, 0, Qt.MouseButton.LeftButton))
    tool.on_press(40.0, 30.0, _press(0, 0, Qt.MouseButton.RightButton))
    assert queue.last.points == [(20.0, 30.0, 1), (40.0, 30.0, 0)]
    # candidates are offered for every prompt without a prior mask (item 15)
    assert queue.last.multimask is True


def test_sam_point_tool_downscales_a_large_crop_and_scales_points(qapp):
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(2000, 1500))
    ov = LabelOverlay((2000, 1500))
    canvas.set_overlay(ov)
    canvas.set_zoom(0.05)  # everything visible => crop is the whole image
    QApplication.processEvents()
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)

    tool.on_press(1000.0, 1000.0, None)
    req = queue.last
    h, w = req.image_crop.shape[:2]
    assert max(h, w) == 1024
    scale = 1024 / 2000
    px, py, label = req.points[0]
    assert px == pytest.approx(1000.0 * scale, abs=1.0)
    assert py == pytest.approx(1000.0 * scale, abs=1.0)
    assert label == 1


def test_sam_point_tool_refine_mode_passes_the_cropped_editing_mask(zoomed):
    canvas, ov = zoomed
    prior = np.zeros((60, 80), dtype=bool)
    prior[10:20, 10:20] = True
    ov.set_editing("inst-x", prior)
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue, refine=True)

    tool.on_press(15.0, 15.0, None)
    req = queue.last
    assert req.mask_input is not None
    assert req.mask_input.shape == req.image_crop.shape[:2]
    assert req.mask_input[15, 15]
    assert not req.mask_input[50, 50]


def test_sam_point_tool_refine_does_not_ask_for_candidates(zoomed):
    canvas, ov = zoomed
    prior = np.zeros((60, 80), dtype=bool)
    prior[10:20, 10:20] = True
    ov.set_editing("inst-x", prior)
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue, refine=True)

    tool.on_press(15.0, 15.0, None)
    assert queue.last.mask_input is not None
    assert queue.last.multimask is False


def test_sam_point_tool_sends_point_plus_box_when_a_prompt_box_is_set(zoomed):
    """The diff map (or the box tool) supplies the box; the click supplies the point."""
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)

    tool.set_prompt_box((10.0, 12.0, 50.0, 52.0))
    tool.on_press(20.0, 30.0, None)
    req = queue.last
    assert req.points == [(20.0, 30.0, 1)]
    assert req.box == (10.0, 12.0, 50.0, 52.0)
    assert req.multimask is True, "a box narrows the question but does not answer it"

    tool.set_prompt_box(None)
    tool.clear_points()
    tool.on_press(20.0, 30.0, None)
    assert queue.last.box is None
    assert queue.last.multimask is True


def test_a_prompt_box_outside_the_viewport_is_dropped(qapp):
    """A box the crop clips to nothing would make SAM return an empty mask."""
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(200, 200))
    ov = LabelOverlay((200, 200))
    canvas.set_overlay(ov)
    canvas.zoom_to((0, 0, 60, 60))
    QApplication.processEvents()
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)

    tool.set_prompt_box((150.0, 150.0, 190.0, 190.0))  # far outside the viewport
    tool.on_press(20.0, 20.0, None)
    assert queue.last.box is None
    assert queue.last.multimask is True, "without a usable box the point is ambiguous"


def test_sam_point_tool_clear_points(zoomed):
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _point_tool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, None)
    tool.clear_points()
    assert tool.points == []
    tool.on_press(40.0, 40.0, None)
    assert queue.last.points == [(40.0, 40.0, 1)]


def test_sam_box_tool_submits_the_dragged_box(zoomed):
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _box_tool(canvas, ov, queue)

    tool.on_press(40.0, 45.0, None)
    tool.on_move(10.0, 15.0, None)  # dragged backwards: must be normalised
    assert tool.box == (10.0, 15.0, 40.0, 45.0)
    tool.on_release(10.0, 15.0, None)

    req = queue.last
    assert req.box == (10.0, 15.0, 40.0, 45.0)
    assert req.points == []
    assert req.image_crop.shape == (60, 80, 3)


def test_sam_box_tool_ignores_a_degenerate_drag(rig):
    canvas, ov = rig
    queue = StubQueue()
    tool = _box_tool(canvas, ov, queue)
    tool.on_press(20.0, 20.0, None)
    tool.on_release(20.0, 20.0, None)
    assert queue.requests == []


def test_sam_tools_without_a_queue_do_not_crash(rig):
    canvas, ov = rig
    tool = _point_tool(canvas, ov, None)
    tool.on_press(10.0, 10.0, None)  # no queue: collects the point, submits nothing
    assert tool.points == [(10.0, 10.0, 1)]


# ---------------------------------------------------------------------------
# applying a result
# ---------------------------------------------------------------------------
def test_sam_point_tool_result_is_added_to_the_editing_layer(zoomed):
    """A prompt SAM saw no prior mask for may only *add* (task U1, report 1).

    This used to assert the opposite -- "non-refine results replace the layer"
    -- which is the defect the annotator hit on their first day: the pixels
    already in the layer are theirs and SAM was never shown them.
    """
    canvas, ov = zoomed
    ov.editing[0:5, 0:5] = True
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    # _blob_result fills rows h/4..h/2, cols w/4..w/2 of the 60x80 crop
    assert ov.editing[20, 25]
    assert ov.editing[0, 0], "the pixels the annotator owned were thrown away"


def test_sam_point_tool_refine_keeps_the_mask_outside_the_crop(qapp):
    canvas = _shown(ImageCanvas(), 200, 200)
    canvas.set_image(_rgb(200, 200))
    ov = LabelOverlay((200, 200))
    prior = np.zeros((200, 200), dtype=bool)
    prior[180:200, 180:200] = True  # far from the zoomed viewport
    ov.set_editing("inst-x", prior)
    canvas.set_overlay(ov)
    canvas.zoom_to((0, 0, 60, 60))
    QApplication.processEvents()

    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, refine=True)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)
    tool.on_press(20.0, 20.0, None)
    assert _spin(lambda: bool(strokes))
    assert ov.editing[190, 190], "refine must keep pixels outside the crop"


class ThreadRecordingSamTool(SamPointTool):
    """Records which thread actually touches the overlay."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.apply_threads: list[int] = []

    def _on_result(self, payload) -> None:
        self.apply_threads.append(threading.get_ident())
        super()._on_result(payload)


def test_sam_result_is_applied_on_the_gui_thread(zoomed):
    canvas, ov = zoomed
    queue = ThreadedQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, cls=ThreadRecordingSamTool)

    tool.on_press(20.0, 30.0, None)
    queue.join()
    # The worker callback has returned, yet nothing may have been applied yet:
    # anything else would mean the overlay was written from the worker thread.
    assert not ov.editing.any(), "the mask was applied off the GUI thread"
    assert tool.apply_threads == []

    assert _spin(lambda: bool(tool.apply_threads)), "no SAM result arrived"
    assert tool.apply_threads == [MAIN_THREAD]
    assert ov.editing[20, 25]


# ---------------------------------------------------------------------------
# stale results (request token + frame identity)
# ---------------------------------------------------------------------------
def test_only_the_newest_request_is_applied(zoomed):
    """Two clicks in flight: the first answer must not reach the overlay."""
    canvas, ov = zoomed
    queue = HoldingQueue(_tagged_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    tool.clear_points()
    tool.on_press(50.0, 40.0, None)
    assert len(queue.pending) == 2

    queue.flush()  # answers in submission order
    assert _spin(lambda: bool(strokes))
    _drain()
    assert len(strokes) == 1, "the superseded result was applied too"
    assert ov.editing[40, 50], "the newest result must win"
    assert not ov.editing[30, 20], "the stale result leaked into the layer"


def test_a_stale_result_delivered_last_is_still_dropped(zoomed):
    """A non-FIFO executor may answer the old prompt after the new one."""
    canvas, ov = zoomed
    queue = HoldingQueue(_tagged_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    tool.clear_points()
    tool.on_press(50.0, 40.0, None)

    queue.flush(newest_first=True)  # the stale answer lands last
    assert _spin(lambda: bool(strokes))
    _drain()
    assert len(strokes) == 1
    assert ov.editing[40, 50]
    assert not ov.editing[30, 20], "the late stale result overwrote the newest one"


def test_a_result_for_the_previous_frame_is_dropped(zoomed):
    """Silent annotation corruption: frame k's mask written into frame k-1."""
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    tool.set_frame_token(FrameKey(13, 12, "scan"))
    tool.set_prompt_box((5.0, 5.0, 50.0, 50.0))
    errors: list[str] = []
    strokes: list[object] = []
    tool.sigError.connect(errors.append)
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    tool.set_frame_token(FrameKey(13, 11, "scan"))  # annotator stepped back
    assert tool.prompt_box is None, "the box belonged to the other frame"
    assert tool.candidate_count == 0

    queue.flush()
    assert _spin(lambda: bool(errors))
    assert errors == [ERR_FRAME_CHANGED]
    assert strokes == []
    assert not ov.editing.any()


def test_a_result_for_another_instance_is_dropped(zoomed):
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    errors: list[str] = []
    tool.sigError.connect(errors.append)

    tool.on_press(20.0, 30.0, None)
    ov.set_editing("inst-y", np.zeros((60, 80), dtype=bool))  # another instance
    queue.flush()
    assert _spin(lambda: bool(errors))
    assert errors == [ERR_FRAME_CHANGED]
    assert not ov.editing.any()


def test_a_result_whose_crop_no_longer_fits_is_dropped_not_raised(zoomed):
    """A smaller frame must not blow up inside the Qt slot."""
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    # The instance is pinned so that only the *size* differs after the swap;
    # otherwise the identity check would catch it first and the bounds check
    # would never be exercised.
    tool = _point_tool(canvas, ov, queue, instance="inst-x")
    errors: list[str] = []
    tool.sigError.connect(errors.append)

    tool.on_press(20.0, 30.0, None)
    smaller = LabelOverlay((20, 20))
    smaller.set_editing("inst-x", np.zeros((20, 20), dtype=bool))
    tool.overlay = smaller  # the session moved to a frame of another size

    queue.flush()
    assert _spin(lambda: bool(errors))
    assert errors == [ERR_OUT_OF_BOUNDS]
    assert not smaller.editing.any()


def test_changing_the_frame_token_resets_candidates_and_the_prompt_box(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    tool.set_frame_token(FrameKey(13, 12, "scan"))
    tool.set_prompt_box((5.0, 5.0, 50.0, 50.0))

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: tool.candidate_count == 3)

    tool.set_frame_token(FrameKey(13, 12, "scan"))  # same frame: no reset
    assert tool.candidate_count == 3
    assert tool.prompt_box is not None

    tool.set_frame_token(FrameKey(13, 11, "scan"))
    assert tool.candidate_count == 0
    assert tool.prompt_box is None
    assert tool.cycle_candidate() == 0


def test_a_fresh_overlay_can_click_once_and_press_c(qapp):
    """The tool names the instance itself; that must not read as a drift.

    On a fresh overlay ``editing_instance`` is ``None`` at submit and becomes
    the tool's own fallback once the first mask lands. If the identity is
    stamped before the fallback and compared after it, the tool invalidates its
    own result: candidates vanish and the headline gesture stops working.
    """
    canvas = _shown(ImageCanvas(), 300, 300)
    canvas.set_image(_rgb(60, 80))
    ov = LabelOverlay((60, 80))
    canvas.set_overlay(ov)
    canvas.set_zoom(0.1)
    QApplication.processEvents()
    assert ov.editing_instance is None, "this test is about the fresh state"

    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 3

    assert tool.cycle_candidate() == 1, "click once, press C: nothing happened"
    assert tool.candidate_count == 3, "the tool discarded its own candidates"
    assert ov.editing[32, 25], "candidate 1 was not applied"
    assert tool.cycle_candidate() == 2
    assert ov.editing[40, 25]


def test_the_prompt_box_survives_consecutive_clicks(zoomed):
    """The box describes the frame's changed region, not one click."""
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    box = (5.0, 5.0, 50.0, 50.0)
    tool.set_prompt_box(box)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: ov.editing.any())
    assert tool.prompt_box == box, "the box was cleared by the tool's own result"

    tool.clear_points()
    tool.on_press(25.0, 35.0, None)
    assert len(queue.requests) == 2
    assert queue.requests[0].box == box
    assert queue.requests[1].box == box, "the second click lost the box"


# ---------------------------------------------------------------------------
# the frame token is mandatory
# ---------------------------------------------------------------------------
def test_submitting_without_a_frame_token_is_refused(zoomed):
    """No identity means no way to tell a stale result from a fresh one."""
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    tool = SamPointTool(canvas, ov, queue)  # deliberately not told the frame
    errors: list[str] = []
    strokes: list[object] = []
    tool.sigError.connect(errors.append)
    tool.sigStroke.connect(strokes.append)

    assert tool.frame_token is None
    tool.on_press(20.0, 30.0, None)
    assert queue.requests == [], "a prompt was sent without a frame token"
    assert errors == [ERR_NO_FRAME_TOKEN]
    _drain()
    assert strokes == []
    assert not ov.editing.any()


def test_unsetting_the_frame_token_disables_the_tool(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    errors: list[str] = []
    tool.sigError.connect(errors.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: tool.candidate_count == 3)

    tool.set_frame_token(None)  # e.g. the session closed the frame
    assert tool.frame_token is None
    assert tool.candidate_count == 0
    assert tool.prompt_box is None

    tool.clear_points()
    tool.on_press(30.0, 30.0, None)
    assert len(queue.requests) == 1, "a prompt was sent while disabled"
    assert errors == [ERR_NO_FRAME_TOKEN]


def test_one_overlay_reused_for_two_frames_drops_the_stale_result(zoomed):
    """The session repaints one overlay per frame, so object identity is useless."""
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, token=FrameKey(13, 12, "scan"))
    errors: list[str] = []
    tool.sigError.connect(errors.append)

    tool.on_press(20.0, 30.0, None)
    # ... the annotator steps back: same overlay object, same size, new frame.
    tool.set_frame_token(FrameKey(13, 11, "scan"))

    queue.flush()
    assert _spin(lambda: bool(errors))
    assert errors == [ERR_FRAME_CHANGED]
    assert not ov.editing.any()


# ---------------------------------------------------------------------------
# cancelling
# ---------------------------------------------------------------------------
def test_detach_cancels_the_in_flight_prompt(zoomed):
    """Switching tool must not let the old tool paint a second later."""
    canvas, ov = zoomed
    queue = HoldingQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    tool.set_prompt_box((5.0, 5.0, 50.0, 50.0))
    strokes: list[object] = []
    errors: list[str] = []
    tool.sigStroke.connect(strokes.append)
    tool.sigError.connect(errors.append)

    tool.on_press(20.0, 30.0, None)
    tool.detach()

    queue.flush()
    _drain()
    assert strokes == [], "a late result was applied after the tool was switched away"
    assert errors == [], "cancelling is not an error"
    assert not ov.editing.any()
    assert tool.candidate_count == 0
    assert tool.prompt_box is None
    assert tool.points == [], "re-attaching must start from a clean prompt"


def test_a_detached_box_tool_forgets_its_drag(zoomed):
    canvas, ov = zoomed
    queue = StubQueue()
    tool = _box_tool(canvas, ov, queue)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(40.0, 40.0, None)
    assert tool.box is not None

    tool.detach()
    assert tool.box is None
    tool.on_release(40.0, 40.0, None)
    assert queue.requests == [], "a drag that was cancelled still submitted"


def test_switching_instance_resets_candidates_and_the_prompt_box(zoomed):
    """The overlay changes instance behind the tool's back; it must notice."""
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    tool.set_prompt_box((5.0, 5.0, 50.0, 50.0))

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: tool.candidate_count == 3)

    ov.set_editing("inst-y", np.zeros((60, 80), dtype=bool))
    assert tool.cycle_candidate() == 0
    assert tool.candidate_count == 0
    assert tool.prompt_box is None, "the box described the previous instance"
    assert not ov.editing.any(), "the other instance's layer was overwritten"

    tool.clear_points()
    tool.on_press(20.0, 30.0, None)
    assert queue.last.box is None
    assert queue.last.multimask is True


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def test_sam_point_tool_cycles_through_the_candidates(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 3
    assert tool.candidate_index == 0
    # candidate 0: rows 15..30, cols 20..40
    assert ov.editing[20, 25] and not ov.editing[32, 25]

    assert tool.cycle_candidate() == 1
    # candidate 1: rows 15..35, cols 20..46
    assert ov.editing[32, 25] and not ov.editing[40, 25]

    tool.cycle_candidate()
    assert tool.candidate_index == 2
    assert ov.editing[40, 25], "candidate 2 is the largest"

    tool.cycle_candidate()  # wraps around
    assert tool.candidate_index == 0
    assert not ov.editing[32, 25], "wrapping must restore the first candidate"

    tool.cycle_candidate(-1)  # and cycles backwards
    assert tool.candidate_index == 2
    assert ov.editing[40, 25]
    assert len(strokes) == 5, "every candidate swap is one undoable edit"


def test_sam_point_tool_resets_the_candidates_on_a_new_request(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    tool.cycle_candidate()
    assert tool.candidate_index == 1

    tool.clear_points()
    tool.on_press(30.0, 30.0, None)
    assert tool.candidate_index == 0, "a new request resets the index"
    assert tool.candidate_count == 0, "and forgets the stale candidates"
    assert _spin(lambda: len(strokes) > 2)
    assert tool.candidate_count == 3


def test_cycle_candidate_without_a_result_is_a_no_op(rig):
    canvas, ov = rig
    tool = _point_tool(canvas, ov, StubQueue())
    assert tool.candidate_count == 0
    assert tool.cycle_candidate() == 0
    assert not ov.editing.any()


def test_cycle_candidate_does_nothing_when_sam_offered_only_one_mask(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)  # a single-candidate result
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 1
    assert tool.cycle_candidate() == 0
    assert len(strokes) == 1, "a pointless swap must not create an undo step"


def test_a_manual_edit_cancels_cycling_and_is_never_discarded(zoomed):
    """A brush stroke on top of the proposal outranks the candidate list."""
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)
    hints: list[str] = []
    tool.sigHint.connect(hints.append)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: bool(strokes))
    assert tool.candidate_count == 3

    brush = BrushTool(canvas, ov, radius=2)
    brush.on_press(5.0, 55.0, None)  # far outside every candidate
    brush.on_release(5.0, 55.0, None)
    assert ov.editing[55, 5]

    assert tool.cycle_candidate() == 0
    assert ov.editing[55, 5], "the manual stroke was overwritten"
    assert tool.candidate_count == 0, "candidates must be dropped, not re-applied"
    assert hints == [HINT_EDITED]


def test_cycle_candidate_follows_the_layer_after_undo(zoomed):
    """Undo moves the layer behind the tool's back; the index must follow it."""
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    stack = _wire_undo(tool, ov)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: len(stack) == 1)
    tool.cycle_candidate()
    tool.cycle_candidate()
    assert tool.candidate_index == 2

    stack.undo()  # the layer is candidate 1 again
    assert ov.editing[32, 25] and not ov.editing[40, 25]

    assert tool.cycle_candidate() == 2, "the index did not follow the layer"
    assert ov.editing[40, 25]


def test_candidate_switches_are_undoable_through_a_real_undo_stack(zoomed):
    canvas, ov = zoomed
    pre_sam = np.zeros((60, 80), dtype=bool)
    pre_sam[50:58, 60:70] = True  # something already painted, far from the candidates
    ov.set_editing("inst-x", pre_sam)

    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    stack = _wire_undo(tool, ov)

    tool.on_press(20.0, 30.0, None)
    assert _spin(lambda: len(stack) == 1)
    tool.cycle_candidate()
    tool.cycle_candidate()
    assert len(stack) == 3, "one op per applied mask"
    candidate_2 = ov.editing.copy()
    assert candidate_2[40, 25]

    stack.undo()
    stack.undo()
    stack.undo()
    assert np.array_equal(ov.editing, pre_sam), "undo must reach the pre-SAM mask"

    stack.redo()
    stack.redo()
    stack.redo()
    assert np.array_equal(ov.editing, candidate_2)


# ---------------------------------------------------------------------------
# the points belong to one prompt, not to the session (final review, item 1)
# ---------------------------------------------------------------------------
def test_changing_the_frame_token_forgets_the_points(zoomed):
    """Clicks on the previous frame are in the previous frame's coordinates.

    ``set_frame_token`` dropped the candidates and the box but kept ``points``,
    so the first click on the next frame went out as a two-point prompt -- one
    of them pointing at whatever now occupies that spot.
    """
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, None)
    assert len(queue.last.points) == 1

    tool.set_frame_token(FrameKey(13, 11, "scan"))
    tool.on_press(40.0, 20.0, None)

    assert len(queue.last.points) == 1, f"points leaked: {queue.last.points}"
    assert queue.last.multimask is True, "a lone first point must offer candidates"


def test_switching_instance_forgets_the_points(zoomed):
    """A point that belongs to part A is not a prompt for part B.

    The reviewer's sequence: S, click A, Enter, activate B, click B -- the
    request carried both points and ``multimask=False``, so ``C`` offered
    nothing and the mask was a union of two parts.
    """
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, instance="part.a")
    tool.on_press(20.0, 30.0, None)
    assert len(queue.last.points) == 1

    tool.instance = "part.b"
    tool.on_press(40.0, 20.0, None)

    assert len(queue.last.points) == 1, f"points leaked: {queue.last.points}"
    assert queue.last.multimask is True


def test_reset_prompt_forgets_the_points_and_the_drag(zoomed):
    """One call the window can make whenever the layer stops being the tool's."""
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    point = _point_tool(canvas, ov, queue)
    point.on_press(20.0, 30.0, None)
    box = _box_tool(canvas, ov, queue)
    box.on_press(10.0, 10.0, None)
    box.on_move(30.0, 30.0, None)

    point.reset_prompt()
    box.reset_prompt()

    assert point.points == []
    assert box.box is None and box._dragging is False
    point.on_press(40.0, 20.0, None)
    assert len(queue.last.points) == 1


# ---------------------------------------------------------------------------
# candidates are offered whenever there is no prior mask (addendum, item 15)
# ---------------------------------------------------------------------------
def test_a_box_prompt_asks_for_candidates(zoomed):
    """A single candidate for a box prompt was the whole chassis, twice.

    Measured on the rehearsal: the motherboard (s41) and the PSU (s33) had a
    correct but large diff box and SAM's one answer was 627k / 620k px of
    machine, with no way out but the eraser.
    """
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _box_tool(canvas, ov, queue)
    tool.on_press(10.0, 10.0, None)
    tool.on_move(40.0, 40.0, None)
    tool.on_release(40.0, 40.0, None)

    assert queue.last.multimask is True
    assert _spin(lambda: tool.candidate_count >= 2), tool.candidate_count


def test_a_point_and_box_prompt_asks_for_candidates(zoomed):
    canvas, ov = zoomed
    queue = StubQueue(_multi_result)
    tool = _point_tool(canvas, ov, queue)
    tool.set_prompt_box((8.0, 8.0, 44.0, 44.0))
    tool.on_press(20.0, 30.0, None)

    assert queue.last.multimask is True
    assert _spin(lambda: tool.candidate_count >= 2)


def test_a_refinement_still_asks_for_one(zoomed):
    """With a prior mask the answer is not ambiguous, and blending needs one."""
    canvas, ov = zoomed
    ov.set_editing("inst-x", np.zeros((60, 80), dtype=bool))
    ov.editing[20:30, 20:30] = True
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, refine=True)
    tool.on_press(25.0, 25.0, None)

    assert queue.last.mask_input is not None
    assert queue.last.multimask is False


# ---------------------------------------------------------------------------
# a prompt that was reset is a prompt that was cancelled (F3 round 2, item 1)
# ---------------------------------------------------------------------------
def test_a_reset_prompt_drops_a_result_that_is_still_in_flight(zoomed):
    """Click, Esc, re-activate the SAME instance: the late mask landed anyway.

    ``reset_prompt`` forgot the points and the candidates but left the token
    alone, so the answer to a prompt the annotator had already discarded
    painted 2,464 px into the layer they had just emptied.
    """
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, instance="part.a")
    tool.on_press(20.0, 30.0, None)
    assert queue.pending, "nothing was submitted"

    tool.reset_prompt()          # what Esc does, through the window
    ov.set_editing("part.a", np.zeros((60, 80), dtype=bool))
    queue.flush()                # the answer arrives now
    _drain()

    assert not ov.editing.any(), "a discarded prompt painted the layer"


def test_a_reset_prompt_does_not_stop_the_next_one(zoomed):
    """The tool stays armed: the point after the reset must still work."""
    canvas, ov = zoomed
    queue = HoldingQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue)
    tool.on_press(20.0, 30.0, None)
    tool.reset_prompt()

    tool.on_press(40.0, 20.0, None)
    queue.flush()
    _drain()

    assert ov.editing.any(), "the prompt after the reset was dropped too"


# ---------------------------------------------------------------------------
# a new prompt never destroys the pixels the annotator owns (task U1, report 1)
# ---------------------------------------------------------------------------
# "SAM 给出掩码之后，如果我用 B 来补充的话，同时会删掉所有的 SAM 做的本身的掩码，
# 就是成了替代了，而不是添加" -- the annotator's first trial.  Measured on the
# real D13/scan/step 42 with SAM 2.1: a box prompt produced 595,473 px, two
# brush strokes added 2,372 px, and the next box prompt left 244,356 px with
# **0** of the hand-painted ones.  The rule this section pins down: SAM may only
# take away pixels it was *shown* (``mask_input``); everything else in the layer
# is owned by the annotator and is composed with the result, never replaced.
def _painted(ov, box) -> np.ndarray:
    """Put a block into the editing layer and hand back a copy of it."""
    x0, y0, x1, y1 = box
    layer = ov.editing.copy()
    layer[y0:y1, x0:x1] = True
    ov.set_editing(ov.editing_instance or "inst-x", layer)
    own = np.zeros_like(layer)
    own[y0:y1, x0:x1] = True
    return own


def test_a_box_prompt_keeps_the_pixels_the_annotator_owns(zoomed):
    """The report, in one test: a box prompt is an addition, not a replacement."""
    canvas, ov = zoomed
    own = _painted(ov, (60, 50, 70, 58))  # far from _blob_result's block
    queue = StubQueue(_blob_result)
    tool = _box_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(44.0, 44.0, None)
    tool.on_release(44.0, 44.0, None)
    assert _spin(lambda: bool(strokes))

    assert ov.editing[20, 25], "the SAM result did not land"
    assert (ov.editing & own).sum() == own.sum(), "the annotator's pixels were erased"


def test_a_second_box_prompt_keeps_the_first_result_and_the_brush_strokes(zoomed):
    """X, drag, B, paint, X, drag -- the exact sequence of the trial."""
    canvas, ov = zoomed
    queue = StubQueue(_blob_result)
    tool = _box_tool(canvas, ov, queue)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(44.0, 44.0, None)
    tool.on_release(44.0, 44.0, None)
    assert _spin(lambda: bool(strokes))
    first = ov.editing.copy()

    brush = BrushTool(canvas, ov, radius=2)
    brush.on_press(5.0, 55.0, None)
    brush.on_release(5.0, 55.0, None)
    hand = ov.editing & ~first
    assert hand.any(), "the brush stroke did not land"

    tool.reset_prompt()          # a new box is a new prompt
    tool.on_press(12.0, 12.0, None)
    tool.on_move(46.0, 46.0, None)
    tool.on_release(46.0, 46.0, None)
    assert _spin(lambda: len(strokes) > 1)

    assert (ov.editing & hand).sum() == hand.sum(), "the brush strokes were erased"
    assert (ov.editing & first).sum() == first.sum(), "the first mask was erased"


def test_cycling_swaps_only_the_current_prompts_contribution(zoomed):
    """``C`` walks SAM's three answers; the owned pixels never move."""
    canvas, ov = zoomed
    own = _painted(ov, (60, 50, 70, 58))
    queue = StubQueue(_multi_result)
    tool = _box_tool(canvas, ov, queue)
    tool.on_press(10.0, 10.0, None)
    tool.on_move(44.0, 44.0, None)
    tool.on_release(44.0, 44.0, None)
    assert _spin(lambda: tool.candidate_count == 3)

    seen = set()
    for _ in range(3):
        assert (ov.editing & own).sum() == own.sum(), "cycling ate the owned pixels"
        seen.add(int((ov.editing & ~own).sum()))
        tool.cycle_candidate()
    assert len(seen) == 3, f"the candidates did not differ: {seen}"


def test_one_undo_takes_the_whole_sam_application_back(zoomed):
    canvas, ov = zoomed
    own = _painted(ov, (60, 50, 70, 58))
    queue = StubQueue(_blob_result)
    tool = _box_tool(canvas, ov, queue)
    stack = _wire_undo(tool, ov)

    tool.on_press(10.0, 10.0, None)
    tool.on_move(44.0, 44.0, None)
    tool.on_release(44.0, 44.0, None)
    assert _spin(lambda: len(stack) == 1)

    stack.undo()
    assert np.array_equal(ov.editing, own), "one undo did not reach the owned mask"


def test_a_refinement_may_still_remove_the_pixels_sam_was_shown(zoomed):
    """Negative points stay a way to take pixels off: SAM saw that mask."""
    canvas, ov = zoomed
    prior = np.zeros((60, 80), dtype=bool)
    prior[20:40, 20:40] = True
    ov.set_editing("inst-x", prior)
    queue = StubQueue(_blob_result)
    tool = _point_tool(canvas, ov, queue, refine=True)
    strokes: list[object] = []
    tool.sigStroke.connect(strokes.append)

    tool.on_press(25.0, 25.0, None)
    assert _spin(lambda: bool(strokes))

    assert queue.last.mask_input is not None, "the prior mask was not sent"
    # _blob_result covers rows 15..30, cols 20..40 of the 60x80 crop: the part of
    # the prior mask below row 30 was shown to SAM and comes back removed.
    assert not ov.editing[35, 25], "a refinement must be able to remove pixels"
