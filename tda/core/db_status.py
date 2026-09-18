"""The counting queries behind ``python -m tda.cli status``, mixed into ``Db``.

They answer "how much of each desktop is in the database" and are the only
aggregate reads in the repository, so they sit apart from the row-level
accessors in ``db.py``. The callers name a *counter*, never a table or a
clause: the SQL is written here and the name is looked up in
:data:`DESKTOP_COUNTERS` / :data:`VIEW_COUNTERS`, so nothing from the command
line can reach the query.
"""
from __future__ import annotations

__all__ = ["DESKTOP_COUNTERS", "VIEW_COUNTERS", "StatusMixin"]

#: counter -> the table whose rows are counted per desktop.
DESKTOP_COUNTERS: dict[str, str] = {
    "steps": "step",
    "actions": "action",
    "instances": "instance",
    "events": "state_event",
    "keyframes": "shape_keyframe",
}
#: counter -> the ``FROM``/``WHERE`` clause counted per ``(desktop, view)``.
VIEW_COUNTERS: dict[str, str] = {
    "frames": "FROM frame",
    "missing": "FROM frame WHERE missing=1",
    "keyframes": "FROM shape_keyframe",
    "verified": "FROM frame WHERE review_status='verified'",
    # frozen frames whose inputs moved and that nobody has compared yet: until
    # this is zero the truth table of that view is not one to export (spec 3.4)
    "rechecks": "FROM recheck_queue",
}


class StatusMixin:
    """Aggregate counters over one database."""

    def desktop_ids(self) -> list[int]:
        """Every desktop id the database knows, ascending."""
        return [r["id"] for r in self.conn.execute("SELECT id FROM desktop ORDER BY id")]

    def count_per_desktop(self, counter: str) -> dict[int, int]:
        """``{desktop: n}`` for one of :data:`DESKTOP_COUNTERS`."""
        try:
            table = DESKTOP_COUNTERS[counter]
        except KeyError:
            raise ValueError(f"unknown desktop counter: {counter!r}") from None
        rows = self.conn.execute(
            f"SELECT desktop, COUNT(*) AS n FROM {table} GROUP BY desktop"
        ).fetchall()
        return {r["desktop"]: r["n"] for r in rows}

    def count_per_view(self, counter: str) -> dict[tuple[int, str], int]:
        """``{(desktop, view): n}`` for one of :data:`VIEW_COUNTERS`."""
        try:
            source = VIEW_COUNTERS[counter]
        except KeyError:
            raise ValueError(f"unknown view counter: {counter!r}") from None
        rows = self.conn.execute(
            f"SELECT desktop, view, COUNT(*) AS n {source} GROUP BY desktop, view"
        ).fetchall()
        return {(r["desktop"], r["view"]): r["n"] for r in rows}

    def step_type_counts(self, desktop: int) -> dict[str, int]:
        """``{step_type: n}`` for one desktop, ordered by type."""
        rows = self.conn.execute(
            "SELECT step_type, COUNT(*) AS n FROM step WHERE desktop=? "
            "GROUP BY step_type ORDER BY step_type", (desktop,)
        ).fetchall()
        return {r["step_type"]: r["n"] for r in rows}

    def steps_with_note_prefix(self, desktop: int, prefix: str) -> list[int]:
        """Steps of one desktop carrying a note line that starts with ``prefix``.

        Used before a destructive re-import, to say how much imported annotation
        is about to be rewritten.
        """
        rows = self.conn.execute(
            "SELECT step, notes FROM step WHERE desktop=? ORDER BY step", (desktop,)
        ).fetchall()
        return [
            r["step"] for r in rows
            if any(line.startswith(prefix) for line in (r["notes"] or "").splitlines())
        ]
