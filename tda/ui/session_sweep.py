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

Nothing may be lost
-------------------
The sweeper is the one place a conflict can go missing, so three things are
deliberate rather than incidental:

* **a failure is loud.**  A re-check that raises is logged, reported on
  :attr:`sigSweepError`, retried on the :data:`RETRY_DELAYS` backoff and then
  parked -- never dropped, never spun on.  ``done`` counts successes only, so
  the progress reported can never read "finished" while a frame failed.  A
  worker that cannot even open its database says so and stops, and the session
  falls back to the synchronous path.
* **a request outlives the work on it.**  The persisted row carries a
  generation stamp; the sweeper clears it only for the generation it read, so a
  request arriving mid-check is not cleared away with the older one.
* **a result describes the inputs it was computed from.**  Every re-check takes
  the frame's cheap input digest first and checks it again inside the write
  transaction; if the annotator edited in between, nothing is written and the
  frame is queued again.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Iterable, Optional

import shiboken6
from PySide6.QtCore import QObject, Signal

from tda.core.db import Db
from tda.core.model import FrameKey
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService
from tda.ui.session_images import ImageCache

__all__ = ["IDLE_WAIT", "QUEUE_SIGNAL_INTERVAL", "RETRY_DELAYS",
           "STALE_QUIET_PERIOD", "TruthSweeper"]

log = logging.getLogger("tda.sweeper")

#: How long to wait before each retry of a frame whose re-check raised. After
#: the last one the frame is parked -- still queued in the database, so the next
#: ``enqueue`` or ``open`` picks it up, but never retried in a hot loop.
RETRY_DELAYS: tuple[float, ...] = (2.0, 10.0)

#: At most one ``sigQueuesChanged`` per this many seconds while a sweep runs
#: (plus one when it drains): a fifty-frame sweep must not make the review panel
#: rebuild fifty times.
QUEUE_SIGNAL_INTERVAL = 0.25

#: How long the worker blocks when it has nothing to do. Every request notifies
#: the condition, so this is only the ceiling on noticing a backoff coming due.
IDLE_WAIT = 0.25

#: How long a frame that was edited *while* being re-checked waits before the
#: next attempt: re-queueing it at the front would spin against an annotator who
#: is still drawing on it.
STALE_QUIET_PERIOD = 1.0

#: The step a worker-level failure (its database, not one frame) is reported as.
NO_STEP = -1


class TruthSweeper(QObject):
    """A worker thread that re-checks frozen frames and prefetches the next one."""

    #: ``(done, total, failed)`` of the current run of re-checks; ``done``
    #: counts successes only, so it reaches ``total`` only when all of them went
    #: through. A slot taking two arguments still works: Qt drops the third.
    sigProgress = Signal(int, int, int)
    #: Something the review queues show has changed (coalesced).
    sigQueuesChanged = Signal()
    #: ``(step, text)`` -- a re-check failed; :data:`NO_STEP` means the worker
    #: itself could not start.
    sigError = Signal(int, str)
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
        self._retry_at: dict[int, float] = {}  # step -> when it may be tried again
        self._tries: dict[int, int] = {}
        self._parked: set[int] = set()
        self._prefetch: Optional[tuple[int, int]] = None  # (step, epoch)
        self._done = 0
        self._total = 0
        self._failed = 0
        self._stopping = False
        self._idle = threading.Event()
        self._idle.set()
        self._thread: Optional[threading.Thread] = None
        self._queues_dirty = False
        self._queues_sent_at = 0.0

    # -- lifecycle ----------------------------------------------------------
    def open(self, desktop: int, view: str) -> None:
        """Point the sweeper at one desktop/view and start its thread."""
        self.stop()
        self.desktop, self.view = int(desktop), str(view)
        with self._lock:
            self._stopping = False
            self._done = self._total = self._failed = 0
            self._rechecks.clear()
            self._retry_at.clear()
            self._tries.clear()
            self._parked.clear()
            self._prefetch = None
        self._thread = threading.Thread(target=self._run, name="tda-truth-sweeper",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> bool:
        """Ask the worker to finish the frame in hand and join it.

        Returns whether it actually stopped.  A thread that did not join is
        **not** forgotten: dropping the handle would make :attr:`is_running` lie
        about a thread still writing to the database.
        """
        thread = self._thread
        if thread is None:
            return True
        with self._lock:
            self._stopping = True
            self._lock.notify_all()
        thread.join(timeout)
        if thread.is_alive():
            log.error("truth sweeper did not stop within %.1fs; it is still running",
                      timeout)
            return False
        self._thread = None
        return True

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until the queue is empty; ``True`` when it drained in time.

        A worker that has **given up** is waited for as well as waited on.
        ``_idle`` is set from inside the thread, so it becomes observable
        before the thread has finished unwinding, and a caller that reasonably
        stops using the sweeper the moment this returns was racing a thread
        still holding a connection to the same database. Whenever the worker is
        stopping, this joins it briefly before answering, so "not running" is
        true by the time anybody can read it.
        """
        deadline = time.monotonic() + timeout
        while True:
            idle = self._idle.wait(min(0.05, max(0.0, deadline - time.monotonic())))
            self._settle(deadline)
            if idle:
                return True
            if not self.is_running:
                return not self._rechecks  # the worker is gone; nothing will drain
            if time.monotonic() >= deadline:
                return False

    def _settle(self, deadline: float) -> None:
        """Let a worker that is on its way out actually get out.

        Only while ``_stopping`` is set -- the worker itself sets it when it
        cannot start, and :meth:`stop` sets it on the way down -- so this never
        waits on a sweeper that is simply busy.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        with self._lock:
            stopping = self._stopping
        if not stopping:
            return
        thread.join(max(0.0, min(0.2, deadline - time.monotonic())))

    @property
    def is_running(self) -> bool:
        """Is the worker thread alive? ``close()`` must leave this ``False``.

        Always the thread's own answer, never a flag that stands in for it: a
        flag is set at some point *inside* the thread and the thread is alive
        for a while afterwards.
        """
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- requests -----------------------------------------------------------
    def enqueue(self, steps: Iterable[int]) -> None:
        """Queue frozen frames for a re-check, newest request served first."""
        wanted = [int(s) for s in steps]
        if not wanted:
            return
        with self._lock:
            for step in wanted:
                self._parked.discard(step)
                self._retry_at.pop(step, None)
                self._tries.pop(step, None)
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

    def parked(self) -> list[int]:
        """Steps that failed their retries and are waiting to be asked again."""
        with self._lock:
            return sorted(self._parked)

    # -- the worker ---------------------------------------------------------
    def _emit(self, signal, *args) -> None:
        """Emit from the worker thread without letting the emit kill it.

        A sweeper can outlive the window that owns it -- a session nobody
        closed, a teardown that ran while a frame was in hand -- and then the
        Qt object behind the signal is already gone and ``emit`` raises
        ``RuntimeError: wrapped C/C++ object has been deleted``. Raised from
        inside the worker's *own* error path, that killed the thread with the
        failure it was trying to report, so nothing was reported at all and the
        real exception surfaced as an unhandled one somewhere else entirely.

        There is nothing to do about a receiver that no longer exists except
        note it: the work itself is in the database either way. Every *other*
        ``RuntimeError`` is a bug in a slot and is re-raised -- swallowing
        those would make this guard the next place a failure goes missing.

        Which of the two it was is asked of shiboken rather than read off the
        exception: what PySide raises for a destroyed object is PySide's
        business, it has been worded differently across versions, and the
        version is not pinned. ``isValid`` is checked *after* the emit as well
        as before, because the window can be torn down between the two.
        """
        if not shiboken6.isValid(self):
            log.debug("truth sweeper has no receiver left for %s", signal)
            return
        try:
            signal.emit(*args)
        except RuntimeError:
            if shiboken6.isValid(self):
                raise  # the object is alive, so this is a slot's own failure
            log.debug("truth sweeper could not deliver %s", signal)

    def _run(self) -> None:
        try:
            db = Db(self._db_path)
        except Exception as exc:  # no connection: say so instead of dying quietly
            log.exception("truth sweeper could not open %s", self._db_path)
            with self._lock:
                self._stopping = True
                self._idle.set()
            self._emit(self.sigError, NO_STEP, f"{type(exc).__name__}: {exc}")
            return
        try:
            # the same local cache the session reads from, so a re-check
            # measures a frame's canvas off the copy rather than off F:
            truth = TruthService(db, self._tax, self._compiler_version,
                                 cache_dir=self._cache_dir)
            images = ImageCache(db, self._cache_dir)
            while True:
                job = self._take()
                if job is None:
                    return
                kind, payload = job
                failed = False
                try:
                    if kind == "recheck":
                        step, gen = payload
                        self._recheck(db, truth, int(step), gen)
                    else:
                        self._compile_ahead(truth, images, payload)
                except Exception as exc:
                    failed = True
                    step = payload[0] if kind == "recheck" else NO_STEP
                    log.exception("truth sweeper failed on %s step %s", kind, step)
                    self._emit(self.sigError, int(step),
                               f"{type(exc).__name__}: {exc}")
                    if kind == "recheck":
                        self._schedule_retry(int(step))
                self._finished(kind, failed)
        finally:
            db.close()

    def _take(self) -> Optional[tuple[str, object]]:
        """The next job, waiting for one; ``None`` when the sweeper is stopping."""
        while True:
            with self._lock:
                if self._stopping:
                    return None
                step = self._next_recheck()
                if step is not None:
                    # the generation stamp is read on the worker's own
                    # connection, inside _recheck_impl
                    return ("recheck", (step, None))
                if self._prefetch is not None:
                    job, self._prefetch = self._prefetch, None
                    return ("prefetch", job)
                if not self._retry_at:
                    self._idle.set()
                    self._lock.wait(IDLE_WAIT)
                else:
                    # something is waiting out its backoff: look again when the
                    # earliest one is due rather than poll
                    due = min(self._retry_at.values()) - time.monotonic()
                    self._lock.wait(max(0.01, min(IDLE_WAIT, due)))
            self._flush_queue_signal(force=False)

    def _next_recheck(self) -> Optional[int]:
        """The next step due; ``None`` while the queue is empty or only waiting."""
        now = time.monotonic()
        for _ in range(len(self._rechecks)):
            step = self._rechecks.popleft()
            if self._retry_at.get(step, 0.0) <= now:
                self._retry_at.pop(step, None)  # it is being tried now
                return step
            self._rechecks.append(step)  # not due yet: look at the others
        return None

    def _requeue_quietly(self, step: int) -> None:
        """Put an overtaken frame back, but not before the annotator has stopped.

        A re-check abandoned because the inputs moved says the frame is being
        worked on right now; trying again immediately would burn a compilation
        per brush stroke.
        """
        with self._lock:
            self._retry_at[step] = time.monotonic() + STALE_QUIET_PERIOD
            if step not in self._rechecks:
                self._rechecks.append(step)
            self._idle.clear()
            self._lock.notify_all()

    def _schedule_retry(self, step: int) -> None:
        """Try a failed frame again later, or park it after the last attempt."""
        with self._lock:
            tries = self._tries.get(step, 0) + 1
            self._tries[step] = tries
            if tries > len(RETRY_DELAYS):
                self._parked.add(step)
                log.error("truth sweeper parked step %s after %d attempts; it stays "
                          "queued in the database for the next open()", step, tries)
                return
            self._retry_at[step] = time.monotonic() + RETRY_DELAYS[tries - 1]
            self._rechecks.append(step)
            self._idle.clear()

    def _finished(self, kind: str, failed: bool) -> None:
        with self._lock:
            if kind == "recheck":
                if failed:
                    self._failed += 1
                else:
                    self._done += 1
                    self._queues_dirty = True
                done, total, bad = self._done, self._total, self._failed
            else:
                done = None
            if not self._rechecks and self._prefetch is None and not self._retry_at:
                self._idle.set()
                drained = True
            else:
                drained = False
        if done is not None:
            self._emit(self.sigProgress, done, total, bad)
        self._flush_queue_signal(force=drained)

    def _flush_queue_signal(self, force: bool) -> None:
        """Emit at most one ``sigQueuesChanged`` per :data:`QUEUE_SIGNAL_INTERVAL`."""
        now = time.monotonic()
        with self._lock:
            if not self._queues_dirty:
                return
            if not force and now - self._queues_sent_at < QUEUE_SIGNAL_INTERVAL:
                return
            self._queues_dirty = False
            self._queues_sent_at = now
        self._emit(self.sigQueuesChanged)

    # -- the work -----------------------------------------------------------
    def _recheck(self, db: Db, truth: TruthService, step: int,
                 gen: Optional[int] = None) -> None:
        """Compare one frozen frame against its inputs and retire the request.

        Split from :meth:`_recheck_impl` so that a test can make the attempt
        fail without having to break the comparison itself.
        """
        self._recheck_impl(db, truth, step, gen)

    def _recheck_impl(self, db: Db, truth: TruthService, step: int,
                      gen: Optional[int]) -> None:
        key = FrameKey(self.desktop, step, self.view)
        stamp = db.recheck_generation(self.desktop, self.view, step)
        guard = truth.inputs_digest(key)
        result = truth.refresh(key, guard=guard)
        if result.get("stale"):
            # the annotator edited this frame while it was being compiled: the
            # comparison describes inputs nobody has any more, so nothing was
            # written and the frame goes back on the queue
            db.add_rechecks(self.desktop, self.view, [step])
            self._requeue_quietly(step)
            return
        if stamp is not None:
            db.clear_recheck(self.desktop, self.view, step, stamp)

    def _compile_ahead(self, truth: TruthService, images: ImageCache, job) -> None:
        """Do for ``k-1`` what visiting it would: refresh its rows and read it.

        Both halves matter.  Compiling alone would still leave the GUI thread to
        write the truth rows the visit brings up to date, and that write is a
        second compilation; decoding the image here is what keeps a 12 MP read
        off the thread that has to stay responsive.
        """
        step, epoch = job
        key = FrameKey(self.desktop, step, self.view)
        stats = truth.refresh(key, want_compiled=True)
        self._emit(self.sigPrefetched, step, epoch, stats["compiled"])
        rgb = images.get(key)
        if rgb is not None:
            images.clear()  # the GUI side owns the cache; this one just decodes
            self._emit(self.sigPrefetchedImage, step, rgb)
