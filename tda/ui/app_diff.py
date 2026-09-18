"""The frame-difference map and the worker that computes it (spec 4.2, 5.1).

Stepping back one frame, the annotator has to find what the action changed.
This does that for them: the frame on screen is compared with the one the task
card is written against, inside the pose segment's ROI; the changed regions
become blobs, and every blob already accounted for by an instance drawn on this
frame is filtered out (:func:`tda.core.diffmap.explain_blobs`).  What is left is
either the part being annotated -- its box becomes SAM's box prompt, measured at
IoU 0.76 against 0.24 for a lone point -- or a change nobody has explained,
which belongs in the review queue.

The window half of the wiring (the SAM tools, the prompt box, the heat map) is
:class:`tda.ui.app_assist.AssistMixin`; everything here is Qt-thread plumbing
and pure functions, so it can be tested without a window.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional, Sequence

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal

from tda.core import masks as _masks
from tda.core.diffmap import (
    MIN_SCALE_DELTA_E,
    ROBUST_PCT,
    DiffBlob,
    diff_blobs,
    diff_delta_e,
    explain_blobs,
    heat_to_rgba,
)

__all__ = ["MAX_DIFF_SIDE", "AssistController", "Box", "best_unexplained",
           "blob_boxes", "expected_boxes", "expected_payload", "heat_rgba"]

Box = tuple[int, int, int, int]

#: The comparison is downscaled to this long side.  A 12 MP pair costs ~1.4 s
#: and 300 MB at native resolution and ~90 ms at 1600, and the blobs are only
#: ever used as a box prompt and as a "look here" marker, so the precision the
#: downscale costs buys nothing.
MAX_DIFF_SIDE = 1600


def blob_boxes(blobs: Sequence[DiffBlob]) -> list[Box]:
    """Just the boxes, which is what the review queue stores."""
    return [tuple(int(v) for v in blob.box) for blob in blobs]  # type: ignore[misc]


def expected_payload(masks: dict, boxes: Sequence[Box] = ()) -> dict:
    """What the window hands the worker as "already accounted for".

    Masks are passed **by value** (they are the session's arrays and the GUI
    thread keeps editing them) and turned into boxes on the worker.
    """
    return {
        # A real copy, not a view: these are the session's arrays and the GUI
        # thread keeps editing them while the worker is measuring them.
        "masks": {str(k): np.array(v, dtype=bool, copy=True)
                  for k, v in dict(masks).items()},
        "boxes": [tuple(int(round(float(v))) for v in box) for box in boxes],
    }


def expected_boxes(expected: Any) -> list[Box]:
    """Boxes from either a plain sequence or an :func:`expected_payload` dict."""
    if not expected:
        return []
    if isinstance(expected, dict):
        out: list[Box] = list(expected.get("boxes") or [])
        for mask in (expected.get("masks") or {}).values():
            box = _masks.bbox(mask)
            if box is not None:
                out.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
        return out
    return [tuple(int(round(float(v))) for v in box) for box in expected]  # type: ignore[misc]


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
    """Runs the difference map on **one** worker thread, newest request wins.

    A thread per frame change was the obvious first version and the wrong one:
    six quick ``PgDn`` presses left six live threads, each holding a pair of
    frames and a float32 dE map -- about 0.8 GB at 1600x1600 and 3.9 GB at
    12 MP.  This keeps one long-lived worker and a single-slot mailbox, exactly
    like :class:`~tda.models.sam_service.SamQueue`: a request that has not
    started yet is replaced, and a result whose token has been superseded is
    dropped instead of shown for the wrong frame.

    Signals:
        sigBlobs: ``{"key", "explained", "unexplained", "delta", "roi",
            "expected", "thread"}`` on the GUI thread, or ``None``.
        sigFailed: the comparison raised; the text is for the status bar.
    """

    sigBlobs = Signal(object)
    sigFailed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._token = 0
        self._lock = threading.Condition()
        self._pending: Optional[tuple] = None
        self._busy = False
        self._stopped = False
        self._bridge = _Bridge(self)
        self._bridge.sigPayload.connect(self._deliver,
                                        Qt.ConnectionType.QueuedConnection)
        self._thread = threading.Thread(target=self._run, name="tda-diff",
                                        daemon=True)
        self._thread.start()

    # -- computation --------------------------------------------------------
    def compute(self, key: Any, image: np.ndarray, previous: np.ndarray,
                roi: Optional[Box], expected: Any = ()) -> dict:
        """The whole comparison, synchronously; safe to call from any thread.

        ``expected`` is either a list of boxes or the richer payload
        :func:`expected_payload` builds -- masks are turned into boxes *here*,
        on the worker, so the GUI thread never pays for a bounding box over a
        12 MP array.
        """
        boxes = expected_boxes(expected)
        delta = diff_delta_e(previous, image, roi=roi, max_side=MAX_DIFF_SIDE)
        blobs = diff_blobs(delta)
        explained, unexplained = explain_blobs(blobs, boxes)
        return {
            "key": key,
            "blobs": blobs,
            "explained": explained,
            "unexplained": unexplained,
            "delta": delta,
            "roi": roi,
            "expected": boxes,
            "thread": threading.current_thread().name,
        }

    def request(self, key: Any, image: Optional[np.ndarray],
                previous: Optional[np.ndarray], roi: Optional[Box],
                expected: Any = ()) -> None:
        """Queue a comparison, replacing one that has not started yet."""
        with self._lock:
            if self._stopped:
                return
            self._token += 1
            token = self._token
            if image is None or previous is None:
                self._bridge.sigPayload.emit((token, None, None))
                return
            self._pending = (token, key, image, previous, roi, expected)
            self._lock.notify()

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._stopped:
                    self._lock.wait()
                if self._stopped and self._pending is None:
                    return
                job, self._pending = self._pending, None
                self._busy = True
            token, key, image, previous, roi, expected = job
            try:
                payload = self.compute(key, image, previous, roi, expected)
                self._bridge.sigPayload.emit((token, payload, None))
            except Exception as exc:  # noqa: BLE001 - a worker must never crash Qt
                self._bridge.sigPayload.emit(
                    (token, None, f"{type(exc).__name__}: {exc}")
                )
            finally:
                with self._lock:
                    self._busy = False
                    self._lock.notify_all()

    def _deliver(self, stamped: object) -> None:
        token, payload, error = stamped  # type: ignore[misc]
        if token != self._token:
            return  # a superseded comparison: the frame moved on
        if error:
            self.sigFailed.emit(str(error))
            return
        self.sigBlobs.emit(payload)

    # -- lifecycle ----------------------------------------------------------
    def wait(self, timeout: float = 5.0) -> bool:
        """Block until the queue is empty and the worker idle; ``True`` if it is."""
        deadline = time.perf_counter() + float(timeout)
        with self._lock:
            while self._pending is not None or self._busy:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return False
                self._lock.wait(left)
        return True

    def shutdown(self) -> None:
        """Refuse new requests, let the one in flight finish, join the worker."""
        with self._lock:
            self._stopped = True
            self._pending = None
            self._lock.notify_all()
        self._thread.join(10.0)

    @property
    def alive(self) -> bool:
        """Whether the worker thread is still running (tests and the leak check)."""
        return self._thread.is_alive()


def best_unexplained(payload: Optional[dict]) -> Optional[DiffBlob]:
    """The strongest unexplained blob of a payload, or ``None``.

    ``diff_blobs`` already sorts by score, so this is the first entry; it is a
    function rather than an index so the rule has one place to live.
    """
    if not payload:
        return None
    unexplained = payload.get("unexplained") or []
    return unexplained[0] if unexplained else None


