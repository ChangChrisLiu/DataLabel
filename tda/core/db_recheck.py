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

    def add_rechecks(self, desktop: int, view: str, steps: Iterable[int]) -> int:
        """Queue these frames for a re-check; returns how many were new.

        Idempotent: asking twice for the same frame leaves one request, which is
        what lets a caller queue an edit's whole interval without checking.
        """
        rows = [(int(desktop), int(step), str(view), R.now_iso())
                for step in sorted({int(s) for s in steps})]
        if not rows:
            return 0
        with self._tx():
            self.conn.execute("INSERT OR IGNORE INTO desktop(id) VALUES(?)", (int(desktop),))
            before = self._recheck_count(desktop, view)
            self.conn.executemany(
                "INSERT OR IGNORE INTO recheck_queue(desktop, step, view, requested_at) "
                "VALUES(?, ?, ?, ?)",
                rows,
            )
            return self._recheck_count(desktop, view) - before

    def _recheck_count(self, desktop: int, view: Optional[str]) -> int:
        sql = "SELECT COUNT(*) AS n FROM recheck_queue WHERE desktop=?"
        args: list = [int(desktop)]
        if view is not None:
            sql += " AND view=?"
            args.append(str(view))
        return int(self.conn.execute(sql, args).fetchone()["n"])

    def rechecks(self, desktop: int, view: Optional[str] = None) -> list[int]:
        """Logical steps still waiting for a re-check, ascending."""
        sql = "SELECT step FROM recheck_queue WHERE desktop=?"
        args: list = [int(desktop)]
        if view is not None:
            sql += " AND view=?"
            args.append(str(view))
        return [int(r["step"]) for r in self.conn.execute(sql + " ORDER BY step", args)]

    def clear_recheck(self, desktop: int, view: str, step: int) -> None:
        """Mark one frame checked; a frame that is not queued is not an error."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM recheck_queue WHERE desktop=? AND step=? AND view=?",
                (int(desktop), int(step), str(view)),
            )
