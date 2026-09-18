"""Connection-level behaviour of :class:`tda.core.db.Db`: migrations and transactions.

:mod:`tda.core.db` is the repository -- one method per query. The two things
that are about the *connection* rather than about any table live here, mixed
into ``Db``:

* :meth:`ConnectionMixin._add_missing_columns` -- ``schema.sql`` is replayed on
  every open, but its statements are ``CREATE TABLE IF NOT EXISTS``, so a table
  that already exists keeps the shape it was created with. Columns added after
  schema version 1 are therefore listed in :data:`MIGRATIONS` and added with
  ``ALTER TABLE`` when an older file is opened.
* :meth:`ConnectionMixin.transaction` -- every repository method commits on its
  own, which is right for a single edit and wrong for a routine that must land
  whole (confirming a frame, resolving a conflict). Inside this block the
  per-call commits become no-ops and the outermost block commits once, or rolls
  everything back when it raises.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

__all__ = ["MIGRATIONS", "ConnectionMixin"]

#: Columns added after schema_version 1, per table. Every one an existing
#: database is missing is added when it is opened, so upgrading is just
#: re-opening the file.
MIGRATIONS: dict[str, dict[str, str]] = {
    "compiled_mask": {
        "geom_type": "TEXT NOT NULL DEFAULT 'mask'",
        "box_json": "TEXT",
    },
}


class ConnectionMixin:
    """Schema migration and grouped writes for a repository holding ``self.conn``."""

    def _add_missing_columns(self, table: str, columns: dict[str, str]) -> None:
        """``ALTER TABLE ADD COLUMN`` for every column ``table`` does not have yet."""
        present = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if not present:  # the table itself is missing: nothing to migrate
            return
        for name, ddl in columns.items():
            if name not in present:
                self.conn.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" {ddl}')

    @contextmanager
    def transaction(self) -> Iterator["ConnectionMixin"]:
        """Commit every write inside the block at once, or none of them.

        Re-entrant, so a caller may wrap a routine that already uses it: only
        the outermost block commits, and an exception rolls the whole thing
        back.
        """
        if self._tx_depth:
            self._tx_depth += 1
            try:
                yield self
            finally:
                self._tx_depth -= 1
            return
        self._tx_depth = 1
        try:
            with self.conn:  # commits on success, rolls back on an exception
                yield self
        finally:
            self._tx_depth = 0

    def _tx(self):
        """What a single write commits with: the connection, or nothing inside a block."""
        return nullcontext() if self._tx_depth else self.conn
