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
from typing import Callable, Optional

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
    "close_logger",
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


def _default_cache_dir() -> str:
    """``cache_dir`` from ``configs/paths.yaml``; the repo is the only fallback.

    A literal ``D:/DataSet/cache`` in library code would be a second, silent
    source of truth for a path the configuration already owns -- and it would
    write to this machine's real cache from a test that forgot its ``paths``.
    """
    from tda import pipeline as P

    try:
        return str(P.require(P.load_paths(P.DEFAULT_PATHS_PATH), "cache_dir"))
    except Exception:  # noqa: BLE001 - no config: stay inside the checkout
        return str(Path(__file__).resolve().parents[2] / "cache")


def app_dir(paths: dict) -> Path:
    """``<cache_dir>/../.cache`` -- the only directory the app writes to.

    An explicit ``app_dir`` in ``paths`` overrides it, which is how a tool that
    drives the window (the smoke run) keeps its INI, log and sidecars out of the
    annotator's own state instead of moving their last frame.
    """
    override = (paths or {}).get("app_dir")
    if override:
        return Path(str(override))
    cache = Path(str((paths or {}).get("cache_dir") or _default_cache_dir()))
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
    """The application logger, with exactly one rotating handler.

    Handlers pointing at any *other* file are closed and removed: a window
    opened on a second workspace (or a test using ``tmp_path``) would otherwise
    leave the previous run's file open for the life of the process, and every
    later message would be written to both.
    """
    target = Path(log_path(paths))
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tda.app")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # the root logger is not ours to write to
    resolved = str(target.resolve())
    for handler in list(logger.handlers):
        if getattr(handler, "_tda_target", None) == resolved:
            return logger
        logger.removeHandler(handler)
        handler.close()
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


def close_logger(logger: logging.Logger) -> None:
    """Close and drop the handlers a window added (called from ``shutdown``)."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


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
    """The uncommitted editing layers of one annotator, on disk.

    **One file per ``(desktop, view, step, instance)``**, all under one
    directory per annotator.  A single shared file lost an edit as soon as the
    annotator opened a second instance on the same frame -- the second
    ``save()`` overwrote the first, and the first was never committed.

    The files are small (a part silhouette's RLE is a few kB) and the window
    debounces the writes, so the cost of a stroke is a memcpy, not I/O.
    """

    def __init__(self, paths: dict, annotator: str) -> None:
        self._dir = sidecar_dir(paths) / _slug(annotator)

    def directory(self) -> str:
        """Where this annotator's sidecars live (it may not exist yet)."""
        return str(self._dir)

    def path_for(self, key: FrameKey, instance: str) -> str:
        """The file one ``(frame, instance)`` pair is stored in."""
        return str(self._file(key, instance))

    def _file(self, key: FrameKey, instance: str) -> Path:
        name = (f"d{int(key.desktop):02d}_{_slug(str(key.view))}"
                f"_s{int(key.step):03d}_{_slug(str(instance))}.json")
        return self._dir / name

    def save(self, key: FrameKey, instance: str, mask: Optional[np.ndarray]) -> None:
        """Record ``mask`` as the layer being edited on ``(key, instance)``.

        An empty or missing mask removes the file instead of writing one: there
        is nothing to offer back, and a stale file would make the next launch
        ask a pointless question.
        """
        if mask is None or not np.asarray(mask).any():
            self.clear(key, instance)
            return
        payload = {
            "desktop": int(key.desktop),
            "step": int(key.step),
            "view": str(key.view),
            "instance": str(instance),
            "rle": masks.encode_rle(np.asarray(mask, dtype=bool)),
        }
        target = self._file(key, instance)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)

    def clear(self, key: Optional[FrameKey] = None,
              instance: Optional[str] = None) -> None:
        """Forget one stored layer, or every one of them when nothing is named."""
        if key is None or instance is None:
            for stored in self._dir.glob("*.json"):
                stored.unlink(missing_ok=True)
            return
        self._file(key, instance).unlink(missing_ok=True)

    def _read(self, path: Path) -> Optional[dict]:
        """One file as ``{"key", "instance", "mask"}``; ``None`` when unusable.

        A file that cannot be read or decoded counts as absent: a crash during
        the write must not stop the next launch.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            mask = masks.decode_rle(payload["rle"])
            key = FrameKey(int(payload["desktop"]), int(payload["step"]),
                           str(payload["view"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return {"key": key, "instance": str(payload["instance"]), "mask": mask,
                "path": path}

    def entries_for(self, key: FrameKey,
                    hw: Optional[tuple] = None) -> list[dict]:
        """Every stored layer that belongs to ``key`` (and fits ``hw``)."""
        found = []
        for path in sorted(self._dir.glob("*.json")):
            entry = self._read(path)
            if entry is None or entry["key"] != key:
                continue
            if hw is not None and tuple(entry["mask"].shape) != tuple(hw):
                continue
            found.append(entry)
        return found

    def pending_for(self, key: FrameKey, instance: Optional[str] = None,
                    hw: Optional[tuple] = None) -> Optional[dict]:
        """The stored layer of one frame (and instance), or ``None``."""
        if instance is not None:
            entry = self._read(self._file(key, instance))
            if entry is None or entry["key"] != key:
                return None
            if hw is not None and tuple(entry["mask"].shape) != tuple(hw):
                return None
            return entry
        entries = self.entries_for(key, hw)
        return entries[0] if entries else None

    def drop(self, entry: dict) -> None:
        """Delete the file an entry came from (its instance is gone, or it was used)."""
        path = entry.get("path")
        if isinstance(path, Path):
            path.unlink(missing_ok=True)
