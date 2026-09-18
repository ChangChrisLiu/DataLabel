"""The background re-check of frozen frames, and the background prefetch.

Spec 3.4 requires every ``verified`` frame an edit reaches to be compared
against its frozen rows, or a conflict would silently never be raised.  Spec 4.2
requires the annotator to be able to commit an edit every few seconds.  At
1600x1600 a single frame costs ~300 ms to compile, so the two are only
compatible if the comparison happens somewhere other than the GUI thread.

:class:`TruthSweeper` is that somewhere.  It owns **its own**
:class:`~tda.core.db.Db` connection to the same file -- SQLite connections are
thread-affine, and WAL plus ``busy_timeout`` is what lets two of them write --
and it touches nothing else the session owns.  Everything that crosses back is a
Qt signal, which Qt queues onto the receiving thread.

It does two jobs off one worker thread:

* **re-checks**, taken newest-request-first, because the frame the annotator
  just edited is the one whose verdict matters soonest.  Each is one short
  transaction, and the persisted request (``recheck_queue``) is cleared only
  once the frame has actually been compared, so a crash re-queues it;
* **prefetch** of the frame the annotator is about to reach (``k-1``, spec 4.2),
  compiled and handed over as a whole :class:`~tda.core.compiler.CompiledFrame`.
  Every request carries the session's edit counter and the result is dropped if
  anything changed meanwhile, so a stale compilation can never be displayed.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Iterable, Optional

from PySide6.QtCore import QObject, Signal

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.ui.session_images import ImageCache

__all__ = ["TruthSweeper"]


class TruthSweeper(QObject):
    """A worker thread that re-checks frozen frames and prefetches the next one."""

    #: ``(done, total)`` of the current run of re-checks.
    sigProgress = Signal(int, int)
    #: Something the review queues show has changed.
    sigQueuesChanged = Signal()
    #: ``(step, epoch, CompiledFrame)`` for a frame compiled ahead of time.
    sigPrefetched = Signal(int, int, object)
    #: ``(step, RGB array)`` for a frame decoded ahead of time.
    sigPrefetchedImage = Signal(int, object)

    def __init__(self, db_path: str, tax: Taxonomy, cache_dir: str,
                 compiler_version: str = "1", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._db_path = str(db_path)
        self._cache_dir = str(cache_dir)
        self._tax = tax
        self._compiler_version = compiler_version
        self.desktop: Optional[int] = None
        self.view: str = ""

        self._lock = threading.Condition()
        self._rechecks: deque[int] = deque()  # newest request first
        self._prefetch: Optional[tuple[int, int]] = None  # (step, epoch)
        self._done = 0
        self._total = 0
        self._busy = False
        self._stopping = False
        self._idle = threading.Event()
        self._idle.set()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ----------------------------------------------------------
    def open(self, desktop: int, view: str) -> None:
        """Point the sweeper at one desktop/view and start its thread."""
        self.stop()
        self.desktop, self.view = int(desktop), str(view)
        self._stopping = False
        self._done = self._total = 0
        self._rechecks.clear()
        self._prefetch = None
        self._thread = threading.Thread(target=self._run, name="tda-truth-sweeper",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Ask the worker to finish the frame in hand and join it.

        Called from :meth:`AnnotationSession.close`, which is the one place the
        GUI is allowed to wait for this thread.
        """
        thread = self._thread
        if thread is None:
            return
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        thread.join(timeout)
        self._thread = None

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue is empty; ``True`` when it drained in time."""
        if self._thread is None:
            return not self._rechecks
        return self._idle.wait(timeout)

    # -- requests -----------------------------------------------------------
    def enqueue(self, steps: Iterable[int]) -> None:
        """Queue frozen frames for a re-check, newest request served first."""
        wanted = [int(s) for s in steps]
        if not wanted:
            return
        with self._lock:
            for step in wanted:
                if step in self._rechecks:
                    self._rechecks.remove(step)  # re-requested: it goes to the front
                else:
                    self._total += 1
                self._rechecks.appendleft(step)
            self._idle.clear()
            self._lock.notify_all()

    def prefetch(self, step: int, epoch: int) -> None:
        """Ask for one frame to be compiled ahead of the annotator reaching it."""
        with self._lock:
            self._prefetch = (int(step), int(epoch))
            self._idle.clear()
            self._lock.notify_all()

    def pending(self) -> int:
        with self._lock:
            return len(self._rechecks)

    @property
    def is_running(self) -> bool:
        """Is the worker thread alive? ``close()`` must leave this ``False``."""
        return self._thread is not None and self._thread.is_alive()

    # -- the worker ---------------------------------------------------------
    def _run(self) -> None:
        db: Optional[Db] = None
        try:
            db = Db(self._db_path)
            truth = TruthService(db, self._tax, self._compiler_version)
            images = ImageCache(db, self._cache_dir)
            while True:
                job = self._take()
                if job is None:
                    return
                kind, payload = job
                try:
                    if kind == "recheck":
                        self._recheck(db, truth, int(payload))
                    else:
                        self._compile_ahead(truth, images, payload)
                except Exception:  # a bad frame must not take the thread down
                    pass
                finally:
                    self._finished(kind)
        finally:
            if db is not None:
                db.close()

    def _take(self) -> Optional[tuple[str, object]]:
        """The next job, waiting for one; ``None`` when the sweeper is stopping."""
        with self._lock:
            while True:
                if self._stopping:
                    return None
                if self._rechecks:
                    self._busy = True
                    return ("recheck", self._rechecks.popleft())
                if self._prefetch is not None:
                    self._busy = True
                    job, self._prefetch = self._prefetch, None
                    return ("prefetch", job)
                self._busy = False
                self._idle.set()
                self._lock.wait(0.05)

    def _finished(self, kind: str) -> None:
        with self._lock:
            self._busy = False
            if kind == "recheck":
                self._done += 1
                done, total = self._done, self._total
            else:
                done = total = None
            drained = not self._rechecks and self._prefetch is None
            if drained:
                self._idle.set()
        if done is not None:
            self.sigProgress.emit(done, total)
        if kind == "recheck":
            self.sigQueuesChanged.emit()

    def _recheck(self, db: Db, truth: TruthService, step: int) -> None:
        """Compare one frozen frame against its inputs and retire the request."""
        truth.refresh(FrameKey(self.desktop, step, self.view))
        db.clear_recheck(self.desktop, self.view, step)

    def _compile_ahead(self, truth: TruthService, images: ImageCache, job) -> None:
        """Do for ``k-1`` what visiting it would: refresh its rows and read it.

        Both halves matter.  Compiling alone would still leave the GUI thread to
        write the truth rows the visit brings up to date, and that write is a
        second compilation; decoding the image here is what keeps a 12 MP read
        off the thread that has to stay responsive.
        """
        step, epoch = job
        key = FrameKey(self.desktop, step, self.view)
        stats = truth.refresh(key)
        self.sigPrefetched.emit(step, epoch, stats["compiled"])
        rgb = images.get(key)
        if rgb is not None:
            images.clear()  # the GUI side owns the cache; this one just decodes
            self.sigPrefetchedImage.emit(step, rgb)
