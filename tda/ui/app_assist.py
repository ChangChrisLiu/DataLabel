"""Model assist: the frame-difference map, off the GUI thread (spec 4.2, 5.1).

Stepping from k to k-1 the annotator has to find what the action changed.  The
difference map does that for them: the two frames are compared inside the pose
segment's ROI, the changed regions are turned into blobs, and every blob that
the step's own task items already account for is filtered out
(:func:`tda.core.diffmap.explain_blobs`).  What is left is either the part being
annotated -- its box becomes SAM's box prompt, which the model comparison
measured at IoU 0.76 against 0.24 for a lone point -- or something nobody has
explained, which belongs in the review queue.

At 12 MP a comparison costs tens to hundreds of milliseconds, far too long for a
frame change, so :class:`AssistController` runs it on a worker thread and hands
the result back through a queued signal.  Only the newest request may deliver: a
result whose token has been superseded is dropped, exactly like a late SAM
answer.
"""
from __future__ import annotations

import threading
from typing import Any, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem

from tda.core.diffmap import (
    MIN_SCALE_DELTA_E,
    ROBUST_PCT,
    DiffBlob,
    diff_blobs,
    diff_delta_e,
    explain_blobs,
    heat_to_rgba,
)

__all__ = ["AssistController", "AssistMixin", "best_unexplained", "blob_boxes",
           "heat_rgba"]

Box = tuple[int, int, int, int]


def blob_boxes(blobs: Sequence[DiffBlob]) -> list[Box]:
    """Just the boxes, which is what the review queue stores."""
    return [tuple(int(v) for v in blob.box) for blob in blobs]  # type: ignore[misc]


def heat_rgba(delta: np.ndarray, roi: Optional[Box]) -> np.ndarray:
    """RGBA overlay for a dE map, rescaled the way ``diff_heat`` would.

    The rescale is repeated here rather than calling ``diff_heat`` so the pair
    of frames is only compared once: the heat map and the blobs come from the
    same ``delta``.
    """
    h, w = delta.shape
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    heat = np.zeros((h, w), dtype=np.float32)
    if x1 > x0 and y1 > y0:
        inside = delta[y0:y1, x0:x1]
        scale = max(float(np.percentile(inside, ROBUST_PCT)), MIN_SCALE_DELTA_E)
        heat[y0:y1, x0:x1] = np.clip(inside / scale, 0.0, 1.0)
    return heat_to_rgba(heat)


class _Bridge(QObject):
    """Carries a worker-thread payload onto the GUI thread (queued connection)."""

    sigPayload = Signal(object)


class AssistController(QObject):
    """Runs the difference map for one frame at a time, newest request wins.

    Signals:
        sigBlobs: ``{"key", "explained", "unexplained", "delta", "roi",
            "thread"}`` on the GUI thread.
        sigFailed: the comparison raised; the text is for the status bar.
    """

    sigBlobs = Signal(object)
    sigFailed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._token = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stopped = False
        self._bridge = _Bridge(self)
        self._bridge.sigPayload.connect(self._deliver,
                                        Qt.ConnectionType.QueuedConnection)

    # -- computation --------------------------------------------------------
    def compute(self, key: Any, image: np.ndarray, previous: np.ndarray,
                roi: Optional[Box], expected_boxes: Sequence[Box]) -> dict:
        """The whole comparison, synchronously; safe to call from any thread."""
        delta = diff_delta_e(previous, image, roi=roi)
        blobs = diff_blobs(delta)
        explained, unexplained = explain_blobs(blobs, list(expected_boxes))
        return {
            "key": key,
            "explained": explained,
            "unexplained": unexplained,
            "delta": delta,
            "roi": roi,
            "thread": threading.current_thread().name,
        }

    def request(self, key: Any, image: Optional[np.ndarray],
                previous: Optional[np.ndarray], roi: Optional[Box],
                expected_boxes: Sequence[Box]) -> None:
        """Schedule a comparison; a pending one is abandoned, not cancelled.

        The running thread is left to finish -- interrupting a numpy call is not
        possible -- but its result is stamped with a token that no longer
        matches, so it is dropped in :meth:`_deliver` instead of being shown for
        the wrong frame.
        """
        with self._lock:
            if self._stopped:
                return
            self._token += 1
            token = self._token
        if image is None or previous is None:
            self._bridge.sigPayload.emit((token, None, None))
            return
        boxes = [tuple(int(v) for v in box) for box in expected_boxes]
        thread = threading.Thread(
            target=self._run, args=(token, key, image, previous, roi, boxes),
            name=f"tda-diff-{token}", daemon=True,
        )
        self._thread = thread
        thread.start()

    def _run(self, token: int, key, image, previous, roi, boxes) -> None:
        try:
            payload = self.compute(key, image, previous, roi, boxes)
        except Exception as exc:  # noqa: BLE001 - a worker must never crash Qt
            self._bridge.sigPayload.emit((token, None, f"{type(exc).__name__}: {exc}"))
            return
        self._bridge.sigPayload.emit((token, payload, None))

    def _deliver(self, stamped: object) -> None:
        token, payload, error = stamped  # type: ignore[misc]
        if token != self._token:
            return  # a superseded comparison: the frame moved on
        if error:
            self.sigFailed.emit(str(error))
            return
        if payload is not None:
            self.sigBlobs.emit(payload)
        else:
            self.sigBlobs.emit(None)

    # -- lifecycle ----------------------------------------------------------
    def wait(self, timeout: float = 5.0) -> bool:
        """Join the running comparison (tests and the smoke run); ``True`` if done."""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self) -> None:
        """Refuse new requests and wait for the one in flight."""
        with self._lock:
            self._stopped = True
        self.wait(5.0)


def best_unexplained(payload: Optional[dict]) -> Optional[DiffBlob]:
    """The strongest unexplained blob of a payload, or ``None``.

    ``diff_blobs`` already sorts by score, so this is the first entry; it is a
    function rather than an index so the rule has one place to live.
    """
    if not payload:
        return None
    unexplained = payload.get("unexplained") or []
    return unexplained[0] if unexplained else None


class _SamLoader(QObject):
    """Carries the outcome of the background SAM load onto the GUI thread."""

    sigLoaded = Signal(object)


class AssistMixin:
    """The window half of the assist: SAM tools, prompt boxes, the heat map."""

    # ------------------------------------------------------------------ setup
    def _init_assist(self, sam_queue: Any) -> None:
        from tda.ui.canvas.sam_tools import SamBoxTool, SamPointTool

        self.sam_queue = sam_queue
        self._owns_queue = False
        self.sam_available = sam_queue is not None
        self.sam_reason = "" if sam_queue is not None else "not loaded yet"
        self.sam_point = SamPointTool(self.canvas, None, queue=sam_queue, refine=True)
        self.sam_box = SamBoxTool(self.canvas, None, queue=sam_queue)
        for tool in (self.sam_point, self.sam_box):
            tool.sigStroke.connect(self.on_stroke)
            tool.sigHint.connect(self.report)
            tool.sigError.connect(self.report_error)

        self.assist = AssistController(self)
        self.assist.sigBlobs.connect(self._on_blobs)
        self.assist.sigFailed.connect(self.report_error)
        self.assist_result: Optional[dict] = None
        self._unexplained: dict[int, list[Box]] = {}
        #: The diff-map box prompt of the frame, kept by the window because
        #: ``SamToolBase.detach()`` clears the tool's own copy: switching to the
        #: brush and back must not silently downgrade point+box to point-only.
        self._prompt_box: Optional[tuple[float, float, float, float]] = None

        self.heat_visible = False
        self.heat_item = QGraphicsPixmapItem()
        self.heat_item.setZValue(0.5)  # over the frame, under the label overlay
        self.heat_item.setVisible(False)
        self.canvas.scene().addItem(self.heat_item)
        self._sam_loader = _SamLoader(self)
        self._sam_loader.sigLoaded.connect(self._on_sam_loaded,
                                           Qt.ConnectionType.QueuedConnection)

    def _sam_tool(self):
        return self.sam_box if self._tool_name == "sam_box" else self.sam_point

    def set_sam_instance(self, instance: Optional[str]) -> None:
        """Name the instance an applied mask belongs to.

        The tools have a ``"editing"`` fallback for when nothing says; relying
        on it would stamp two different edits with the same identity, so the
        window always answers explicitly.
        """
        for tool in (self.sam_point, self.sam_box):
            tool.instance = instance

    def clear_prompt_box(self) -> None:
        """Forget the box prompt; the next frame's diff map proposes its own."""
        self._prompt_box = None

    def rearm_sam(self) -> None:
        """Give a freshly attached SAM tool its frame token and prompt box back.

        ``detach()`` cancels the in-flight prompt and clears the box, which is
        what makes a tool switch safe; the cost is that re-arming has to be
        explicit, or the next click would go out point-only.
        """
        tool = self._sam_tool()
        if not self.sam_available or self._tool_name not in ("sam_point", "sam_box"):
            return
        from tda.ui import app_compat as compat

        if compat.is_open(self.session) and self.session.image() is not None:
            tool.set_frame_token(self.session.current())
        if self._prompt_box is not None:
            tool.set_prompt_box(self._prompt_box)

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_assist(self, key) -> None:
        """Re-stamp the SAM tools and start the comparison for the new frame.

        The token is **mandatory**: without one the tools refuse to prompt, so a
        frame with no image gets ``None`` (prompting is meaningless there) and
        every other frame gets its :class:`~tda.core.model.FrameKey`.
        """
        token = key if self.session.image() is not None else None
        self.clear_prompt_box()
        for tool in (self.sam_point, self.sam_box):
            tool.overlay = self.overlay
            tool.set_frame_token(token)
            # set_frame_token drops the box only when the token really changes,
            # and re-arming an attached tool may already have set it: say it.
            tool.set_prompt_box(None)
        self.set_sam_instance(getattr(self.session, "editing_instance", None))
        if not self.roi_editing:
            self.canvas.set_rubber_band(None)
        self.assist_result = None
        self.heat_item.setVisible(False)
        self.request_assist()

    def request_assist(self) -> None:
        """Compare the open frame with the previous one, off the GUI thread."""
        from tda.ui import app_compat as compat

        if not compat.is_open(self.session):
            return
        key = self.session.current()
        image = self.session.image()
        previous, previous_key = self._previous_image(key)
        self.assist.request(key, image, previous, self.roi(),
                            self._expected_boxes(previous_key))

    def _previous_image(self, key) -> tuple[Optional[np.ndarray], Any]:
        """The newest earlier step that actually has an image, and its pixels."""
        from tda.core.model import FrameKey

        for step in sorted((s for s in self.session.steps() if s < key.step),
                           reverse=True):
            image = self.session.image_at(step)
            if image is not None:
                return image, FrameKey(key.desktop, step, key.view)
        return None, None

    def _expected_boxes(self, previous_key) -> list[Box]:
        """Boxes of the instances already shaped at k-1 (a plain table read)."""
        if previous_key is None:
            return []
        boxes: list[Box] = []
        for row in self.db.compiled(previous_key).values():
            box = row.get("box")
            if box:
                boxes.append(tuple(int(round(float(v))) for v in box))  # type: ignore[arg-type]
        return boxes

    # --------------------------------------------------------------- results
    def _on_blobs(self, payload: object) -> None:
        """A comparison came back on the GUI thread."""
        from tda.ui import app_compat as compat

        self.assist_result = payload if isinstance(payload, dict) else None
        if self.assist_result is None or not compat.is_open(self.session):
            return
        if self.assist_result.get("key") != self.session.current():
            self.assist_result = None
            return
        if self.heat_visible:
            self._paint_heat()
        blob = best_unexplained(self.assist_result)
        if blob is not None and self._task_is_add_shape():
            self.begin_add_shape(blob)

    def _task_is_add_shape(self) -> bool:
        from tda.ui import session_api as api

        rows = self.session.task_card()
        index = self.task_card.current_index()
        return 0 <= index < len(rows) and rows[index].get("kind") == api.KIND_ADD_SHAPE

    def begin_add_shape(self, blob: DiffBlob) -> None:
        """Feed a changed region's box to SAM as the box half of point+box."""
        box = tuple(float(v) for v in blob.box)
        self._prompt_box = box
        for tool in (self.sam_point, self.sam_box):
            tool.set_prompt_box(box)
        if not self.roi_editing:
            self.canvas.set_rubber_band(box)
        self.report(f"prompt box from the difference map: "
                    f"{tuple(int(v) for v in blob.box)} ({blob.area} px)")

    def unexplained_boxes(self) -> list[Box]:
        """Boxes of the changes nothing on this frame accounts for."""
        return blob_boxes((self.assist_result or {}).get("unexplained") or [])

    def hand_over_unexplained(self, step: int, boxes: list[Box]) -> None:
        """Give the review queue what the frame never explained (spec 4.4)."""
        from tda.ui import app_compat as compat

        if not compat.set_unexplained(self.session, step, boxes):
            self._unexplained[int(step)] = list(boxes)

    # ------------------------------------------------------------ SAM status
    def sam_status_text(self) -> str:
        """The SAM part of the status bar: state plus the candidate counter."""
        if not self.sam_available:
            return f"SAM unavailable: {self.sam_reason}"
        tool = self._sam_tool()
        count = tool.candidate_count
        if count:
            return f"SAM ready · {tool.candidate_index + 1}/{count}"
        return "SAM ready"

    def set_sam_unavailable(self, reason: str) -> None:
        """Disable the SAM tools and say why; everything else keeps working."""
        self.sam_available = False
        self.sam_reason = str(reason)
        for tool in (self.sam_point, self.sam_box):
            tool.queue = None
        if self._tool_name in ("sam_point", "sam_box"):
            self._tool_name = "brush"
            self._attach_tool()
        self.update_status()

    def set_sam_queue(self, queue: Any, owns: bool = True) -> None:
        """Arm the SAM tools once a checkpoint has finished loading."""
        self.sam_queue = queue
        self._owns_queue = bool(owns)
        self.sam_available = queue is not None
        self.sam_reason = ""
        for tool in (self.sam_point, self.sam_box):
            tool.queue = queue
        self.update_status()

    def start_sam(self) -> None:
        """Load SAM on a background thread; the first frame must not wait for it."""
        if self.sam_queue is not None:
            return
        self.sam_reason = "loading"
        self.update_status()
        loader = self._sam_loader

        def load() -> None:
            try:
                from tda.models.sam_service import SamQueue, SamService

                loader.sigLoaded.emit(SamQueue(SamService()))
            except Exception as exc:  # noqa: BLE001 - the app works without SAM
                loader.sigLoaded.emit(f"{type(exc).__name__}: {exc}")

        threading.Thread(target=load, name="tda-sam-load", daemon=True).start()

    def _on_sam_loaded(self, outcome: object) -> None:
        if isinstance(outcome, str):
            self.set_sam_unavailable(outcome)
            return
        self.set_sam_queue(outcome, owns=True)
        self.report("SAM ready")

    # -------------------------------------------------------------- actions
    def act_cycle_candidate(self) -> None:
        """``C``: the next of SAM's proposals for the same click."""
        tool = self._sam_tool()
        if tool.candidate_count < 2:
            self.report("no other SAM candidate")
            return
        tool.cycle_candidate()
        self.update_status()

    def act_toggle_heat(self) -> None:
        """``D``: the frame-difference heat map over the image."""
        self.heat_visible = not self.heat_visible
        if self.heat_visible:
            self._paint_heat()
        else:
            self.heat_item.setVisible(False)
        self.report("difference heat map " + ("on" if self.heat_visible else "off"))

    def _paint_heat(self) -> None:
        payload = self.assist_result
        if not payload or payload.get("delta") is None:
            self.report("no difference map for this frame yet")
            self.heat_item.setVisible(False)
            return
        rgba = np.ascontiguousarray(heat_rgba(payload["delta"], payload.get("roi")))
        height, width = rgba.shape[:2]
        image = QImage(rgba.data, width, height, 4 * width,
                       QImage.Format.Format_RGBA8888)
        self.heat_item.setPixmap(QPixmap.fromImage(image.copy()))
        self.heat_item.setVisible(True)

    # ------------------------------------------------------------- lifecycle
    def shutdown_assist(self) -> None:
        """Stop the comparison thread and, when we started it, the SAM queue."""
        self.assist.shutdown()
        if self._owns_queue and self.sam_queue is not None:
            try:
                self.sam_queue.stop()
            except Exception:  # pragma: no cover - a wedged GPU call
                pass
            self.sam_queue = None
