"""The compiled-frame cache, and everything the background worker hands back.

One concern, seen from two sides.  A frame's compilation is expensive at
scanner resolution (~300 ms) and the annotator walks back and forth over the
same two frames all day, so it is kept: per step, and per *edit epoch*, a
counter bumped by every change to the annotation inputs.  That counter is also
what makes a compilation arriving from the sweeper safe to adopt -- a prefetch
that was in flight while the annotator drew something is simply not the frame
they are looking at, and carries the epoch to prove it.

The same object therefore owns the callbacks the worker signals into: a failed
re-check, a finished one, a frame compiled ahead, an image decoded ahead.  None
of them run on the worker's thread; Qt queues them here.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QCoreApplication

from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey

__all__ = ["COMPILED_CACHE_SIZE", "TruthCacheMixin"]

#: Compiled frames kept in memory. Four covers the Tab compare between a frame
#: and its neighbour plus the prefetched next one, at a few MB of masks each.
COMPILED_CACHE_SIZE = 4


class TruthCacheMixin:
    """Compiled frames in memory, and the background worker's callbacks."""

    def compiled(self) -> CompiledFrame:
        """Compiler output for the current frame.

        Kept in memory per step and per edit epoch, so that stepping back and
        forth between two frames -- which is what the ``Tab`` compare does -- is
        free, and so that a frame the sweeper compiled ahead of time is used
        rather than compiled again.
        """
        key = self.current()
        hit = self._compiled.get(key.step)
        if hit is not None and hit[0] == self._epoch:
            return hit[1]
        compiled = self.truth.compile(key)
        self._keep_compiled(key.step, self._epoch, compiled)
        return compiled

    def _compile_on_visit(self) -> None:
        """Bring the frame just opened up to date in the truth table (spec 3.4).

        A commit only compiles the frame in front of the annotator, so the rows
        of every other frame it reached are stale until somebody looks at them.
        This is that moment: one frame, and only when its stored rows do not
        already carry the current inputs' hash.
        """
        key = self.current()
        if key.step not in self._available:
            return
        held = self._compiled.get(key.step)
        if held is not None and held[0] == self._epoch:
            stored = self.db.compiled(key)
            if stored and all(row["input_hash"] == held[1].input_hash
                              for row in stored.values()):
                return  # the sweeper already brought this frame up to date
        # one compilation for the whole visit: the refresh hands back the frame
        # it made its decisions from, which is the one the panels are about to
        # ask for -- compiling it twice is what made arriving cost 0.9 s
        stats = self.truth.refresh(key)
        self.review.problems[key.step] = list(stats["problems"])
        self.review.invalidate()
        self._keep_compiled(key.step, self._epoch, stats["compiled"])

    def _keep_compiled(self, step: int, epoch: int, compiled: CompiledFrame) -> None:
        """Remember one compilation, and what the truth table owes because of it."""
        self._compiled[int(step)] = (int(epoch), compiled)
        self.review.problems[int(step)] = list(compiled.problems)
        while len(self._compiled) > COMPILED_CACHE_SIZE:
            self._compiled.pop(next(iter(self._compiled)))

    def _invalidate(self) -> None:
        """The annotation inputs changed: everything derived from them is stale."""
        self._epoch += 1
        self._compiled.clear()
        self.review.invalidate()

    # -------------------------------------------------- the background worker
    def _on_sweep_error(self, step: int, text: str) -> None:
        """A background re-check failed: say so, and leave the frame pending."""
        where = "the truth sweeper" if step < 0 else f"step {step}"
        self.review.invalidate()
        self.sigSweepError.emit(step, text)
        self.sigProblems.emit([f"{where}: re-check failed: {text}"])

    def _on_queues_changed(self) -> None:
        """A re-check finished: its verdict may have changed a status or a queue."""
        self.review.invalidate()
        self.sigQueuesChanged.emit()

    def _on_prefetched(self, step: int, epoch: int, compiled: object) -> None:
        """Adopt a frame the sweeper compiled ahead, unless it went stale."""
        if epoch == self._epoch and isinstance(compiled, CompiledFrame):
            self._keep_compiled(step, epoch, compiled)

    def _on_prefetched_image(self, step: int, rgb: object) -> None:
        """Adopt a frame the sweeper decoded ahead of the annotator reaching it."""
        key = self._key(step)
        if key is not None and isinstance(rgb, np.ndarray):
            self.images.put(key, rgb)

    def _prefetch_next(self) -> None:
        """Warm the frame the annotator is about to reach: ``k-1`` (spec 4.2).

        Reverse-order annotation always lands there next, and both halves of the
        cost -- bringing its truth rows up to date and decoding its image -- are
        done on the sweeper thread, so arriving is free.
        """
        if not self.sweeper_enabled or self._step is None:
            return
        earlier = [s for s in self._available if s < self._step]
        if not earlier:
            return
        step = earlier[-1]
        if self._compiled.get(step, (None,))[0] != self._epoch:
            self.sweeper.prefetch(step, self._epoch)

    def drain_sweeper(self, timeout: float = 30.0) -> bool:
        """Wait for the background re-checks to finish (tests, exports, quit)."""
        drained = self.sweeper.wait_idle(timeout)
        QCoreApplication.processEvents()
        self.review.invalidate()
        return drained

    #: The GUI never waits for a prefetch; a test that measures one does.
    drain_prefetch = drain_sweeper
