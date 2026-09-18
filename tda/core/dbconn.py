"""Connection-level behaviour of :class:`tda.core.db.Db`: migrations and transactions.

:mod:`tda.core.db` is the repository -- one method per query. The two things
that are about the *connection* rather than about any table live here, mixed
into ``Db``:

* :meth:`ConnectionMixin.init_schema` -- ``schema.sql`` is replayed on every
  open, but its statements are ``CREATE TABLE IF NOT EXISTS``, so a table that
  already exists keeps the shape it was created with. Columns added after schema
  version 1 are therefore listed in :data:`MIGRATIONS` and added with
  ``ALTER TABLE``; a file from a newer build is refused instead.
* :meth:`ConnectionMixin.transaction` -- every repository method commits on its
  own, which is right for a single edit and wrong for a routine that must land
  whole (confirming a frame, resolving a conflict). Inside this block the
  per-call commits become no-ops and the outermost block commits once, or rolls
  everything back when it raises.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Optional

__all__ = ["MIGRATIONS", "ConnectionMixin"]

#: Columns added after schema_version 1, per table. Every one an existing
#: database is missing is added when it is opened, so upgrading is just
#: re-opening the file.
MIGRATIONS: dict[str, dict[str, str]] = {
    "compiled_mask": {
        "geom_type": "TEXT NOT NULL DEFAULT 'mask'",
        "box_json": "TEXT",
    },
    # `recheck_queue` arrived with schema version 3; the generation stamp was
    # added in the same version, so only a database written by a pre-release
    # build of that branch can be missing it.
    "recheck_queue": {
        "gen": "INTEGER NOT NULL DEFAULT 0",
    },
}


class ConnectionMixin:
    """Schema bootstrap and grouped writes for a repository holding ``self.conn``.

    The host also provides ``self.path`` (for error messages) and a
    ``self._tx_depth`` initialised to 0.
    """

    def init_schema(self, sql_path: Path, version: int) -> None:
        """Replay the DDL, migrate an older file, refuse a newer one, stamp the version.

        A file written by a *newer* build is refused rather than opened: this
        build does not know what its extra columns mean, and stamping the older
        version onto it would hide that from the build that does. Everything
        here is idempotent, so opening a database again is a no-op.
        """
        sql = Path(sql_path).read_text(encoding="utf-8")
        with self.conn:
            self.conn.executescript(sql)
            found = self._stored_version()
            if found is not None and found > version:
                raise RuntimeError(
                    f"{self.path} was written by a newer TDA (schema version {found}); "
                    f"this build knows version {version} -- upgrade tda to open it"
                )
            for table, columns in MIGRATIONS.items():
                self._add_missing_columns(table, columns)
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(version),),
            )

    def _stored_version(self) -> Optional[int]:
        """The file's own schema version, or ``None`` when it carries no readable one."""
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            return None
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return None

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
