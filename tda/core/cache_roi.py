"""Where a thumbnail's crop comes from: the annotation database, read-only.

:mod:`tda.core.cache_thumbs` cuts each frame to a chassis box.  Which box that
is, for a frame whose ROI a human has already drawn, is this module's job: the
``pose_segment.roi_json`` of the segment the frame belongs to.  It is split out
so that "how a thumbnail is made" and "where its ROI comes from" stay separate
concerns -- the batch job works perfectly well without a database.

Nothing here writes: an offline batch has no business creating, migrating or
even locking the file the annotator is working in.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Callable, Optional
from urllib.parse import quote

from tda.core.model import FrameKey

__all__ = ["DbRoiLookup", "RoiLookup"]

_MISSING_SQL_OBJECT = "no such "  # sqlite's wording for an absent table or column

#: What a ``roi_lookup`` is: a frame in, its chassis box ``(x0, y0, x1, y1)``
#: in *original image pixels* out, or ``None`` for "nothing recorded".
RoiLookup = Callable[[FrameKey], Optional[tuple[int, int, int, int]]]


def _norm(path) -> str:
    """Forward-slash form of ``path``."""
    return str(path).replace("\\", "/")


def _as_box(raw) -> Optional[tuple[int, int, int, int]]:
    """Parse a stored ROI into ``(x0, y0, x1, y1)``; ``None`` when it is not one.

    Accepts the shapes :func:`tda.core.export.coco.roi_of` accepts:
    ``[x0, y0, x1, y1]`` and a mapping with ``x0/y0/x1/y1`` or ``x/y/w/h``.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        x0, y0, x1, y1 = (int(round(float(v))) for v in raw)
    elif isinstance(raw, dict) and {"x0", "y0", "x1", "y1"} <= set(raw):
        x0, y0, x1, y1 = (int(raw[n]) for n in ("x0", "y0", "x1", "y1"))
    elif isinstance(raw, dict) and {"x", "y", "w", "h"} <= set(raw):
        x0, y0 = int(raw["x"]), int(raw["y"])
        x1, y1 = x0 + int(raw["w"]), y0 + int(raw["h"])
    else:
        return None
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


class DbRoiLookup:
    """``roi_lookup`` reading ``pose_segment.roi_json`` from a read-only database.

    Which segment a frame is in is decided exactly as
    :meth:`tda.core.db.Db.pose_segment_for` decides it: the frame's own
    ``frame.pose_segment`` wins, and only when that is NULL (or names a segment
    that no longer exists) does the ``[start_step, end_step]`` range decide.  A
    frame outside every segment, or in one without a ROI, gets ``None`` and
    therefore an uncropped thumbnail.

    The file is opened through a ``file:...?mode=ro`` URI and *not* through
    :class:`~tda.core.db.Db`: the schema bootstrap would create, migrate and
    re-stamp it, and a database written by a newer build would be refused
    outright - none of which a thumbnail batch may do to the annotator's working
    copy.  Read-only here means the database itself is untouched (same bytes,
    same ``schema_version``, same tables); sqlite may still create the ``-wal``
    and ``-shm`` side files of a WAL database, which is harmless.  A
    ``<db>.lock`` left by a running annotator is reported through ``notify`` and
    otherwise ignored - reading alongside it is safe.

    Segments are read once per ``(desktop, view)``, so a whole desktop costs two
    queries.  Use it as a context manager, or call :meth:`close` when done.
    """

    def __init__(self, db_path, notify: Callable[[str], None] = print) -> None:
        self.path = _norm(db_path)
        if os.path.exists(f"{self.path}.lock"):
            notify(f"note: {self.path}.lock exists (the annotator may be open); reading anyway")
        self.conn = sqlite3.connect(self._read_only_uri(db_path), uri=True)
        self.conn.row_factory = sqlite3.Row
        self._segments: dict[tuple[int, str], dict[int, tuple[int, int, Any]]] = {}
        self._overrides: dict[tuple[int, str], dict[int, int]] = {}

    @staticmethod
    def _read_only_uri(db_path) -> str:
        """``file:`` URI of ``db_path`` that sqlite may only read."""
        absolute = _norm(os.path.abspath(str(db_path)))
        return f"file:{quote(absolute, safe='/:')}?mode=ro"

    def __call__(self, key: FrameKey) -> Optional[tuple[int, int, int, int]]:
        found = self._segment_for(key)
        return _as_box(found[1]) if found is not None else None

    def segment_of(self, key: FrameKey) -> Optional[int]:
        """Which pose segment the frame is in, so its thumbnails can be grouped.

        :func:`build_thumbs` records that id in ``thumbs.json``, which is what
        lets a later run notice that *this* segment's ROI was adjusted and
        rebuild only its steps.
        """
        found = self._segment_for(key)
        return found[0] if found is not None else None

    def _segment_for(self, key: FrameKey) -> Optional[tuple[int, Any]]:
        """``(seg, roi_json)`` of the frame's segment, exactly as ``Db`` picks it."""
        segments = self._segments_of(key.desktop, key.view)
        seg = self._overrides_of(key.desktop, key.view).get(key.step)
        if seg is not None and seg in segments:
            return (seg, segments[seg][2])
        return next(((s, v[2]) for s, v in segments.items() if v[0] <= key.step <= v[1]), None)

    def _segments_of(self, desktop: int, view: str) -> dict[int, tuple[int, int, Any]]:
        """``{seg: (start, end, roi_json)}`` of one view, read at most once."""
        cached = self._segments.get((desktop, view))
        if cached is None:
            rows = self.conn.execute(
                "SELECT seg, start_step, end_step, roi_json FROM pose_segment "
                "WHERE desktop=? AND view=? AND start_step IS NOT NULL "
                "AND end_step IS NOT NULL ORDER BY seg", (desktop, view)).fetchall()
            cached = {int(r["seg"]): (int(r["start_step"]), int(r["end_step"]), r["roi_json"])
                      for r in rows}
            self._segments[(desktop, view)] = cached
        return cached

    def _overrides_of(self, desktop: int, view: str) -> dict[int, int]:
        """``{step: seg}`` for the frames that name a segment themselves."""
        cached = self._overrides.get((desktop, view))
        if cached is None:
            try:
                rows = self.conn.execute(
                    "SELECT step, pose_segment FROM frame WHERE desktop=? AND view=? "
                    "AND pose_segment IS NOT NULL", (desktop, view)).fetchall()
            except sqlite3.OperationalError as exc:  # only "no such table/column"
                if _MISSING_SQL_OBJECT not in str(exc):
                    raise  # a locked or corrupt database is not a missing override
                rows = []  # an older file without the column: the ranges decide
            cached = {int(r["step"]): int(r["pose_segment"]) for r in rows}
            self._overrides[(desktop, view)] = cached
        return cached

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "DbRoiLookup":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


