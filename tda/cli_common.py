"""What every command-line module shares: the exit codes, the lock, the backup.

:mod:`tda.cli` builds the parser and owns the four import commands,
:mod:`tda.cli_app` the app/check/export ones, :mod:`tda.cli_relations`
``infer-relations`` and :mod:`tda.cli_graph` ``constraints``.  All four need the
same four things -- the exit codes, ``--desktops``, a session that holds the
single-user lock, and the safety copy a destructive command takes first -- so
they live here rather than in whichever module happened to define them.

That is not tidiness.  ``python -m tda.cli`` runs ``cli.py`` as ``__main__``, so
a ``from tda.cli import ...`` inside another module **imports the file a second
time** and builds a second :class:`Locked` class.  ``main()`` then caught its
own ``__main__.Locked`` while the export raised ``tda.cli.Locked``, and five
subcommands answered a held lock with a traceback and exit 1 instead of one line
and exit 3.  Nothing here imports :mod:`tda.cli`, which is what makes that
impossible rather than merely fixed.
"""
from __future__ import annotations

import argparse
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional

from tda import pipeline as P
from tda.core.db import Db

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ORDER = 2  # the pipeline order was not respected
EXIT_LOCKED = 3  # another annotator holds the single-user lock

__all__ = [
    "ConfigError", "EXIT_ERROR", "EXIT_LOCKED", "EXIT_OK", "EXIT_ORDER", "Locked",
    "desktops", "load_paths", "safety_backup", "session",
]


class Locked(RuntimeError):
    """Another annotator holds the single-user lock (spec 3.5)."""


class ConfigError(RuntimeError):
    """``paths.yaml`` is missing, unreadable or not valid YAML.

    Carries the whole message the command prints, ``cannot read <path>: ...``,
    so every entry point renders it the same way.
    """


def load_paths(path: str) -> dict:
    """``paths.yaml`` as a dict, or :class:`ConfigError` with one line to print.

    A configuration file the annotator mistyped, deleted or half-edited is the
    most ordinary way to start a command, and the three ways it fails --
    ``FileNotFoundError``, any other ``OSError`` (a directory, a permission), and
    ``yaml.YAMLError`` -- all used to reach the terminal as a traceback.  YAML's
    own message spans several lines, so it is folded onto one: the interesting
    part is which file and roughly why.
    """
    import yaml

    try:
        loaded = P.load_paths(path)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read {path}: {' '.join(str(exc).split())}") from None
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"cannot read {path}: it holds a {type(loaded).__name__}, not a mapping "
            f"of settings"
        )
    return loaded


def desktops(args: argparse.Namespace) -> Optional[set[int]]:
    """Parse ``--desktops``; ``None`` means every desktop the source offers."""
    return P.parse_desktops(getattr(args, "desktops", None))


@contextmanager
def session(args: argparse.Namespace, lock: bool = False) -> Iterator[tuple[dict, Db]]:
    """Open ``paths.yaml`` + the database, optionally holding the single-user lock.

    The lock is released only when this call took it, so a refused command never
    unlocks the annotator who is actually working.
    """
    paths = load_paths(args.paths)
    db = P.open_db(paths, args.db)
    held = False
    try:
        if lock:
            try:
                db.acquire_lock(f"cli:{args.command}")
            except RuntimeError as exc:
                raise Locked(str(exc)) from None
            held = True
        yield paths, db
    finally:
        if held:
            db.release_lock()
        db.close()


def safety_backup(paths: dict, db: Db, command: str, why: str) -> bool:
    """Back the database up before a destructive run; ``False`` when it failed.

    A destructive command that could not make its safety copy must stop before
    it writes anything, so this swallows the three ways the copy can fail --
    ``OSError`` (a full, missing or read-only ``backup_dir``), ``sqlite3.Error``
    (the copy itself) and ``ValueError`` (``paths.yaml`` defines no
    ``backup_dir``) -- and turns each into the same single line. The caller
    returns :data:`EXIT_ERROR` immediately; the lock is released by
    :func:`session` on the way out either way. No traceback ever reaches the
    annotator: there is nothing in it they could act on.
    """
    try:
        out = db.backup(P.backup_dest(paths), P.backup_keep(paths))
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"[{command}] backup failed: {exc}; nothing was written")
        return False
    print(f"[{command}] {why}: backed the database up first -> {out}")
    return True
