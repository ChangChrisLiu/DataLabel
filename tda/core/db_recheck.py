"""The queue of frozen frames still waiting to be compared against their inputs.

Spec 3.4 says a ``verified`` row is never silently overwritten: when the inputs
change, the frame is recompiled and any disagreement is queued as a conflict.
Doing that for every frame an edit reaches is what made a commit cost seconds,
so the work moved off the GUI thread -- and the moment it is asynchronous, the
*request* has to outlive the process.  A crash between "the shape changed" and
"the frozen frame was checked" would otherwise lose a conflict for good, and a
conflict that is never raised is worse than a slow tool.

The table is therefore the queue itself, not a cache of it: one row per frame
waiting, added with ``INSERT OR IGNORE`` and removed when the check is done, so
two connections (the GUI's and the sweeper's) can work on it without ever
reading a value in order to write it back.
"""
from __future__ import annotations

from typing import Iterable, Optional

from tda.core import dbrows as R

__all__ = ["RecheckMixin"]


class RecheckMixin:
    """Pending truth re-checks, for a repository holding ``self.conn``."""

    def add_rechecks(self, desktop: int, view: str, steps: Iterable[int]) -> list[int]:
        """Queue these frames for a re-check; returns the steps queued.

        Asking again for a frame that is already queued does not add a second
        row -- it bumps that row's ``gen``. That is what makes a request landing
        while the sweeper is working on the very same frame safe: the sweeper
        clears the row only for the generation it read, so the newer request
        outlives the older one's clear (and outlives a crash, since it is a row
        rather than something held in memory).
        """
        wanted = sorted({int(s) for s in steps})
        if not wanted:
            return []
        with self._tx():
            self.conn.execute("INSERT OR IGNORE INTO desktop(id) VALUES(?)", (int(desktop),))
            self.conn.executemany(
                "INSERT INTO recheck_queue(desktop, step, view, requested_at, gen) "
                "VALUES(?, ?, ?, ?, 0) "
                "ON CONFLICT(desktop, step, view) DO UPDATE SET "
                "gen = gen + 1, requested_at = excluded.requested_at",
                [(int(desktop), step, str(view), R.now_iso()) for step in wanted],
            )
        return wanted

    def rechecks(self, desktop: int, view: Optional[str] = None) -> list[int]:
        """Logical steps still waiting for a re-check, ascending."""
        return [step for step, _gen in self.recheck_items(desktop, view)]

    def recheck_items(self, desktop: int, view: Optional[str] = None
                      ) -> list[tuple[int, int]]:
        """``(step, gen)`` for every frame still waiting, ascending by step.

        The generation is what a worker has to carry back to
        :meth:`clear_recheck`; reading the steps without it is only safe for
        display.
        """
        sql = "SELECT step, gen FROM recheck_queue WHERE desktop=?"
        args: list = [int(desktop)]
        if view is not None:
            sql += " AND view=?"
            args.append(str(view))
        return [(int(r["step"]), int(r["gen"]))
                for r in self.conn.execute(sql + " ORDER BY step", args)]

    def recheck_generation(self, desktop: int, view: str, step: int) -> Optional[int]:
        """The generation stamp of one queued frame, or ``None`` when it is not."""
        row = self.conn.execute(
            "SELECT gen FROM recheck_queue WHERE desktop=? AND step=? AND view=?",
            (int(desktop), int(step), str(view)),
        ).fetchone()
        return None if row is None else int(row["gen"])

    def clear_recheck(self, desktop: int, view: str, step: int,
                      gen: Optional[int] = None) -> bool:
        """Retire one re-check request; ``True`` when a row was actually removed.

        With ``gen`` the row goes only if it still carries that stamp, so a
        request that arrived while the frame was being checked survives. Without
        it the row goes whatever it says, which is only right for a caller that
        holds the write lock for the whole drain.
        """
        sql = "DELETE FROM recheck_queue WHERE desktop=? AND step=? AND view=?"
        args: list = [int(desktop), int(step), str(view)]
        if gen is not None:
            sql += " AND gen=?"
            args.append(int(gen))
        with self._tx():
            return self.conn.execute(sql, args).rowcount > 0
