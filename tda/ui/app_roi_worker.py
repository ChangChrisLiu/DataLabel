"""Measuring a pose segment's chassis ROI without stopping the window.

The proposal is no longer one frame's answer but the union of three
(:func:`tda.core.cache.suggest_roi_over`), which means decoding three images
and running the detector on each. Measured on the real data: 165 ms for a
12 MP OAK segment and 465 ms for a 1600x1600 scanner one, both with the images
already in the OS cache and more when they are not -- against a first paint
that has to happen now. So it happens on a worker, exactly the way the
difference map does (:class:`tda.ui.app_diff.AssistController`): one long-lived
thread, a single-slot mailbox so that walking through segments cannot pile up
requests, and a token so that an answer for a segment nobody is looking at any
more is dropped rather than drawn.

The worker reads the images itself, from paths. That is not only where the time
is -- it is also the only place the channel order is not a question:
``cv2.imread`` gives BGR, which is what :mod:`tda.core.cache` measures, so
nothing has to be converted and nothing can be converted wrongly.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterable, Optional

import cv2
from PySide6.QtCore import QObject, Qt, Signal

from tda.core.cache import suggest_roi_over

__all__ = ["RoiProposal", "RoiProposer", "measure_paths"]


class RoiProposal(dict):
    """``{"segment", "box", "paths", "ms", "thread"}`` -- a dict with a name."""


def measure_paths(paths: Iterable[str], view: str) -> Optional[tuple]:
    """The segment's box from the images at ``paths``; ``None`` without any.

    Synchronous and thread-safe, so a test (or a caller with no event loop) can
    ask for the same answer the worker would give.
    """
    images = []
    for path in paths:
        if not path:
            continue
        found = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if found is not None:
            images.append(found)
    if not images:
        return None
    return tuple(int(v) for v in suggest_roi_over(images, view))


class _Bridge(QObject):
    """Carries a worker-thread payload onto the GUI thread (queued connection)."""

    sigPayload = Signal(object)


class RoiProposer(QObject):
    """Measures one pose segment's ROI on a worker thread, newest request wins.

    Signals:
        sigProposed: a :class:`RoiProposal` on the GUI thread. ``box`` is
            ``None`` when the segment has no readable frame at all; it is the
            whole frame when the detector found nothing, which is what
            :func:`tda.core.cache.suggest_roi_over` means by "no proposal".
        sigFailed: the measurement raised; the text is for the status bar.
    """

    sigProposed = Signal(object)
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
        self._thread = threading.Thread(target=self._run, name="tda-roi",
                                        daemon=True)
        self._thread.start()

    # -- requests -----------------------------------------------------------
    def request(self, segment: Any, view: str, paths: Iterable[str]) -> None:
        """Queue a measurement, replacing one that has not started yet."""
        wanted = [str(p) for p in paths if p]
        with self._lock:
            if self._stopped:
                return
            self._token += 1
            token = self._token
            if not wanted:
                self._bridge.sigPayload.emit(
                    (token, RoiProposal(segment=segment, box=None, paths=[],
                                        ms=0.0, thread=""), None)
                )
                return
            self._pending = (token, segment, str(view), wanted)
            self._lock.notify()

    def cancel(self) -> None:
        """Abandon the answer in flight: nobody is waiting for it any more.

        Bumping the token is what does the work -- a measurement already on the
        worker cannot be stopped, and does not need to be, because
        :meth:`_deliver` drops a payload whose token has moved on. Without this
        an answer requested before an ``Enter`` stayed valid: it was still about
        the same segment, nothing had been dragged since, and the next
        ``Shift+R`` armed the tool just in time for it to land and overwrite the
        rectangle the annotator had just stored.
        """
        with self._lock:
            self._token += 1
            self._pending = None
            self._lock.notify_all()

    def queued(self) -> int:
        """How many measurements are waiting: the mailbox holds at most one."""
        with self._lock:
            return int(self._pending is not None)

    def pending(self) -> bool:
        """Is an answer still on its way -- queued, or being measured now?"""
        with self._lock:
            return self._pending is not None or self._busy

    def wait(self, timeout: float = 10.0) -> bool:
        """Block until the mailbox is empty and the worker idle; ``True`` if so."""
        deadline = time.perf_counter() + float(timeout)
        with self._lock:
            while self._pending is not None or self._busy:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return False
                self._lock.wait(left)
        return True

    def shutdown(self, timeout: float = 5.0) -> None:
        """Refuse new requests, let the one in flight finish, join the worker."""
        with self._lock:
            self._stopped = True
            self._pending = None
            self._lock.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the worker ---------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._stopped:
                    self._lock.wait()
                if self._stopped and self._pending is None:
                    return
                job, self._pending = self._pending, None
                self._busy = True
            token, segment, view, paths = job
            try:
                started = time.perf_counter()
                box = measure_paths(paths, view)
                payload = RoiProposal(
                    segment=segment, box=box, paths=list(paths),
                    ms=(time.perf_counter() - started) * 1000,
                    thread=threading.current_thread().name,
                )
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
            return  # a superseded measurement: the annotator moved on
        if error:
            self.sigFailed.emit(str(error))
            return
        self.sigProposed.emit(payload)
