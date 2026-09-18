"""Where the annotator's application state lives, and how it survives a crash.

Three things the main window needs that have nothing to do with annotation:

* **Locations.** Everything the app writes goes under ``<cache_dir>/../.cache``
  -- the window geometry and dock layout (an INI file, never the Windows
  registry and never ``%APPDATA%``), the rotating log, and the crash sidecar.
  On this machine that resolves to ``D:/DataSet/.cache``; nothing is ever
  written to ``C:``.
* **Robustness.** :func:`guard` wraps a Qt slot so no exception can escape into
  the event loop, and :func:`install_excepthook` catches the ones that get out
  anyway.  Both end in the log plus a one-line message in the status bar.
* **Crash safety.** Every committed edit is already on disk (the database
  commits per operation), so the only thing a crash can lose is the editing
  layer that has not been committed yet.  :class:`EditSidecar` writes it to a
  small JSON file at the end of every stroke and hands it back on the next open
  of the same frame and instance.
"""
from __future__ import annotations

import functools
import json
import logging
import re
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PySide6.QtCore import QSettings

from tda.core import masks
from tda.core.model import FrameKey

__all__ = [
    "APP_SUBDIR",
    "LOG_NAME",
    "SETTINGS_NAME",
    "EditSidecar",
    "app_dir",
    "get_logger",
    "guard",
    "install_excepthook",
    "log_path",
    "make_settings",
    "settings_path",
    "sidecar_dir",
]

#: Sits next to ``cache_dir`` rather than inside it: ``D:/DataSet/cache`` holds
#: the frame images and their manifests, which no application state may touch.
APP_SUBDIR = ".cache"
SETTINGS_NAME = "tda_app.ini"
LOG_NAME = "tda_app.log"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUPS = 3
_DEFAULT_CACHE = "D:/DataSet/cache"


def app_dir(paths: dict) -> Path:
    """``<cache_dir>/../.cache`` -- the only directory the app writes to."""
    cache = Path(str((paths or {}).get("cache_dir") or _DEFAULT_CACHE))
    return cache.parent / APP_SUBDIR


def settings_path(paths: dict) -> str:
    """The INI file the window geometry and the last frame are stored in."""
    return str(app_dir(paths) / SETTINGS_NAME)


def log_path(paths: dict) -> str:
    """The rotating application log."""
    return str(app_dir(paths) / "logs" / LOG_NAME)


def sidecar_dir(paths: dict) -> Path:
    """Where :class:`EditSidecar` keeps the uncommitted editing layers."""
    return app_dir(paths) / "sidecar"


def make_settings(paths: dict) -> QSettings:
    """A ``QSettings`` bound to :func:`settings_path`, in INI format.

    Constructed from an explicit file name, so Qt never falls back to the
    registry (Windows) or to ``%APPDATA%`` -- spec 3.5 keeps every byte we
    produce on ``D:``.
    """
    target = Path(settings_path(paths))
    target.parent.mkdir(parents=True, exist_ok=True)
    return QSettings(str(target), QSettings.Format.IniFormat)


def get_logger(paths: dict) -> logging.Logger:
    """The application logger, with one rotating handler per log file."""
    target = Path(log_path(paths))
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tda.app")
    logger.setLevel(logging.INFO)
    resolved = str(target.resolve())
    for handler in logger.handlers:
        if getattr(handler, "_tda_target", None) == resolved:
            return logger
    handler = RotatingFileHandler(
        str(target), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._tda_target = resolved  # type: ignore[attr-defined]
    logger.addHandler(handler)
    return logger


def install_excepthook(
    logger: logging.Logger, report: Optional[Callable[[str], None]] = None
) -> Callable:
    """Log (and optionally show) anything that still escapes; returns the old hook.

    A ``KeyboardInterrupt`` keeps the default behaviour so ``Ctrl+C`` in the
    launching console still stops the process.
    """
    previous = sys.excepthook

    def hook(kind, value, tb) -> None:
        if issubclass(kind, KeyboardInterrupt):
            previous(kind, value, tb)
            return
        logger.error("unhandled exception\n%s",
                     "".join(traceback.format_exception(kind, value, tb)))
        if report is not None:
            try:
                report(f"{kind.__name__}: {value}")
            except Exception:  # pragma: no cover - the reporter must never loop
                pass

    sys.excepthook = hook
    return previous


def guard(method: Callable) -> Callable:
    """Wrap a Qt slot so that an exception is reported instead of escaping.

    Qt has no way to propagate a Python exception out of a slot: it prints it
    and carries on with whatever half-applied state the slot left behind.  Every
    slot the window connects goes through here, which turns that into one log
    entry, one status-bar line and -- via ``report_exception`` -- a rollback of
    any database transaction the failure interrupted.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - that is the whole point
            handler = getattr(self, "report_exception", None)
            if handler is None:
                raise
            handler(exc, method.__name__)
            return None

    return wrapper


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)) or "anon"


class EditSidecar:
    """The uncommitted editing layer of one annotator, on disk.

    One file per annotator rather than per frame: only one edit can be in
    progress at a time, and a stale file for a frame nobody is on any more is
    worse than none -- :meth:`pending_for` simply ignores it.
    """

    def __init__(self, paths: dict, annotator: str) -> None:
        self._path = sidecar_dir(paths) / f"edit_{_slug(annotator)}.json"

    def path(self) -> str:
        """The sidecar file (it may not exist)."""
        return str(self._path)

    def save(self, key: FrameKey, instance: str, mask: Optional[np.ndarray]) -> None:
        """Record ``mask`` as the layer being edited on ``key``.

        An empty or missing mask clears the file instead of writing one: there
        is nothing to offer back, and a stale file would make the next launch
        ask a pointless question.
        """
        if mask is None or not np.asarray(mask).any():
            self.clear()
            return
        payload = {
            "desktop": int(key.desktop),
            "step": int(key.step),
            "view": str(key.view),
            "instance": str(instance),
            "rle": masks.encode_rle(np.asarray(mask, dtype=bool)),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    def clear(self) -> None:
        """Forget the stored layer (the edit was committed or abandoned)."""
        self._path.unlink(missing_ok=True)

    def load(self) -> Optional[dict]:
        """The stored layer as ``{"key", "instance", "mask"}``, or ``None``.

        A file that cannot be read or decoded is treated as absent: a crash
        during the write must not stop the next launch.
        """
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            mask = masks.decode_rle(payload["rle"])
            key = FrameKey(int(payload["desktop"]), int(payload["step"]),
                           str(payload["view"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return {"key": key, "instance": str(payload["instance"]), "mask": mask}

    def pending_for(self, key: FrameKey, hw: Optional[tuple] = None) -> Optional[dict]:
        """The stored layer when it belongs to ``key`` (and fits ``hw``)."""
        found = self.load()
        if found is None or found["key"] != key:
            return None
        if hw is not None and tuple(found["mask"].shape) != tuple(hw):
            return None
        return found


def describe_lock(held: Optional[dict[str, Any]]) -> str:
    """A human sentence for a lock file's contents (used by the message box)."""
    if not held:
        return "another annotator holds the database lock"
    return (f"another annotator holds the database lock: "
            f"{held.get('annotator')!r} since {held.get('ts')}")
