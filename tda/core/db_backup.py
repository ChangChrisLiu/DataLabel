"""The safety copy and the single-user lock, mixed into :class:`tda.core.db.Db`.

Both are about the *file* rather than about any table, which is why they sit
apart from the repository in ``db.py``.

A backup is only worth taking if it is worth restoring, so nothing here trusts
the copy it just made. ``sqlite3``'s backup API can fail part-way through -- a
full volume, a dropped network share -- and what it leaves behind is a small,
**valid** SQLite file under the real ``tda_<timestamp>.sqlite`` name: it is not
empty, ``PRAGMA quick_check`` says ``ok``, and the one person who ever reads
these names will pick it as "the latest backup". So a copy that raises is
deleted again, and a copy that returns is checked against a fingerprint of the
source taken in the same read transaction (:data:`FINGERPRINT_TABLES`): a stub
cannot match it.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from tda.core import dbrows as R

__all__ = [
    "FINGERPRINT_TABLES",
    "LOCK_SUFFIX",
    "LOCK_TTL",
    "BackupLockMixin",
    "acquire_lock_file",
    "lock_path_for",
    "read_lock_file",
    "release_lock_file",
]

LOCK_TTL = timedelta(hours=12)
#: Suffix of the single-user lock file, next to the database (spec 3.5).
LOCK_SUFFIX = ".lock"

#: The tables whose row counts fingerprint a database. Five ``COUNT(*)`` over a
#: 18 MB file cost milliseconds, and between them they cover every stage of the
#: pipeline, so a copy that agrees on all five (plus both schema versions) is
#: the same database -- while the truncated stub a half-finished copy leaves has
#: none of these tables at all.
FINGERPRINT_TABLES = ("frame", "instance", "shape_keyframe", "compiled_mask", "op_log")


def fingerprint(conn: sqlite3.Connection) -> tuple:
    """Cheap identity of a database: its schema version plus five row counts.

    ``PRAGMA schema_version`` is deliberately **not** in it: it is SQLite's own
    DDL cookie and the backup API is free to write a different one into the
    copy, which it does. The stored ``meta.schema_version`` is TDA's version of
    the schema, is an ordinary row, and is copied verbatim.

    Raises ``sqlite3.Error`` when a table is missing, which is itself the
    answer: the file is not a copy of this database.
    """
    version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    out: list = [
        None if version is None else str(version[0]),
        int(conn.execute("PRAGMA user_version").fetchone()[0]),
    ]
    for table in FINGERPRINT_TABLES:
        out.append(int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]))
    return tuple(out)


def _discard(out: Path) -> None:
    """Remove a backup that must not be left where a restore would find it."""
    try:
        out.unlink()
    except OSError:  # leaving it is bad, but the error the caller raises is the point
        pass


class BackupLockMixin:
    """``backup()`` and the single-user lock for a repository holding ``self.conn``."""

    # ----------------------------------------------------------------- backup

    def backup(self, dest_dir: str) -> str:
        """Copy the live database with the SQLite backup API; returns the new path.

        The file name carries **local** time (``tda_YYYYmmdd_HHMMSS.sqlite``),
        because the one person who reads these names is looking for "the copy
        from before lunch" on their own clock; two copies inside the same second
        are told apart by a ``_1``, ``_2`` suffix. The consequence is that a
        daylight-saving step back can make one name sort before an older one --
        the file's own mtime is the authority, not the name.

        What is written is **verified**, in three steps, because a backup nobody
        checked is what makes the ``--force`` run that rewrites the step table
        look safe:

        1. a copy that raises part-way is deleted and the error re-raised --
           otherwise an 8 KB stub stays behind under the real backup name;
        2. the copy is opened and must be non-empty and pass ``PRAGMA
           quick_check``;
        3. its :func:`fingerprint` must equal the source's, read inside the same
           transaction the copy was taken in.

        Anything else removes the file and raises ``OSError``; ``sqlite3.Error``
        from the copy itself propagates unchanged. Every caller of the safety
        copy turns both into one printed line and exit 1
        (:func:`tda.cli._safety_backup`).
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        out = self._backup_name(dest)
        target = sqlite3.connect(str(out))
        try:
            with self._read_snapshot():
                source = fingerprint(self.conn)
                self.conn.backup(target)
        except BaseException:
            target.close()
            _discard(out)  # a partial copy under a real name is worse than none
            raise
        else:
            target.close()
        self._verify_backup(out, source)
        return str(out)

    @staticmethod
    def _backup_name(dest: Path) -> Path:
        """``tda_<local timestamp>.sqlite``, with a serial for the same second."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # local time, see backup()
        out = dest / f"tda_{stamp}.sqlite"
        serial = 1
        while out.exists():
            out = dest / f"tda_{stamp}_{serial}.sqlite"
            serial += 1
        return out

    def _read_snapshot(self):
        """A read transaction around the fingerprint and the copy, when possible.

        It pins the source so the two cannot describe different moments. Inside
        an open :meth:`~tda.core.dbconn.ConnectionMixin.transaction` there is
        already one, and a driver that will not start a bare ``BEGIN`` here is
        not worth failing the backup over: holding the single-user lock, nobody
        else is writing anyway.
        """
        from contextlib import contextmanager

        @contextmanager
        def _snapshot():
            started = False
            if not getattr(self, "_tx_depth", 0):
                try:
                    self.conn.execute("BEGIN")
                    started = True
                except sqlite3.Error:
                    started = False
            try:
                yield
            finally:
                if started:
                    try:
                        self.conn.rollback()  # read-only: nothing to commit
                    except sqlite3.Error:
                        pass

        return _snapshot()

    @staticmethod
    def _verify_backup(out: Path, source: tuple) -> None:
        """Raise (and remove the file) unless ``out`` really is that database."""
        try:
            size = out.stat().st_size
        except OSError:
            raise OSError(f"the backup was not written: {out}") from None
        if size == 0:
            _discard(out)
            raise OSError(f"the backup is empty and was removed again: {out}")

        copy = sqlite3.connect(str(out))
        try:
            integrity = copy.execute("PRAGMA quick_check").fetchone()[0]
            if str(integrity).lower() != "ok":
                raise OSError(
                    f"the backup is corrupt ({integrity}) and was removed again: {out}"
                )
            try:
                made = fingerprint(copy)
            except sqlite3.Error as exc:
                raise OSError(
                    f"the backup does not match the database ({exc}) and was "
                    f"removed again: {out}"
                ) from None
            if made != source:
                raise OSError(
                    f"the backup does not match the database it was made from "
                    f"(got {made}, expected {source}) and was removed again: {out}"
                )
        except BaseException:
            copy.close()
            _discard(out)
            raise
        copy.close()

    # ------------------------------------------------------------------- lock

    def _read_lock(self) -> Optional[dict]:
        """Parse the lock file, or None when it is missing, unreadable or not an object.

        A lock whose content is not a JSON object carries no annotator, so it is
        treated like a stale one and the next ``acquire_lock`` takes it over.
        """
        try:
            held = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return held if isinstance(held, dict) else None

    def acquire_lock(self, annotator: str) -> None:
        """Take the single-user lock; raises if another annotator holds a fresh one.

        :func:`acquire_lock_file` does the same thing *without* a ``Db``, which
        is what the application uses: opening the database replays the schema
        and the migrations, so the lock has to be taken before that, not after.
        """
        held = self._read_lock()
        if held and held.get("annotator") != annotator:
            _refuse_if_fresh(held)
        self._lock_path.write_text(
            json.dumps({"annotator": annotator, "ts": R.now_iso()}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._lock_annotator = annotator

    def release_lock(self) -> None:
        """Remove the lock file if present; safe to call more than once."""
        self._lock_path.unlink(missing_ok=True)
        self._lock_annotator = None


# --------------------------------------------------------------------------- #
# the single-user lock, without a Db
# --------------------------------------------------------------------------- #
def _refuse_if_fresh(held: dict) -> None:
    """Raise when somebody else's lock is younger than :data:`LOCK_TTL`.

    A lock whose timestamp cannot be read counts as abandoned: an unreadable
    stamp is a bug in whoever wrote it, not a reason to lock a annotator out of
    their own database forever.
    """
    try:
        ts = datetime.fromisoformat(str(held.get("ts")))
    except ValueError:
        return
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    if datetime.now(timezone.utc) - ts < LOCK_TTL:
        raise RuntimeError(
            f"database locked by {held.get('annotator')!r} since {held.get('ts')}"
        )


def lock_path_for(db_path: str) -> Path:
    """The lock file that belongs to a database path."""
    return Path(str(db_path) + LOCK_SUFFIX)


def read_lock_file(db_path: str) -> Optional[dict]:
    """The holder recorded in a lock file, or ``None`` when there is none."""
    try:
        held = json.loads(lock_path_for(db_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return held if isinstance(held, dict) else None


def acquire_lock_file(db_path: str, annotator: str) -> None:
    """Take the single-user lock **without opening the database**.

    :meth:`tda.core.db.Db.__init__` creates the file when it is missing and
    replays the schema and the migrations on an existing one, so by the time a
    ``Db`` exists the database has already been written to.  The application
    therefore has to take the lock first: a refused launch must leave the file
    exactly as it was, down to its modification time.

    Raises ``RuntimeError`` when a *fresh* lock is held by somebody else; a lock
    older than :data:`LOCK_TTL`, or one whose timestamp cannot be read, counts as
    abandoned and is taken over -- the same rule as
    :meth:`BackupLockMixin.acquire_lock`.
    """
    held = read_lock_file(db_path)
    if held and held.get("annotator") != annotator:
        _refuse_if_fresh(held)
    path = lock_path_for(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"annotator": annotator, "ts": R.now_iso()}, ensure_ascii=False),
        encoding="utf-8",
    )


def release_lock_file(db_path: str) -> None:
    """Drop the lock file; safe to call more than once."""
    lock_path_for(db_path).unlink(missing_ok=True)
