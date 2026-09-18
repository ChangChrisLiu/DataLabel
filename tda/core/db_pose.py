"""Pose-segment accessors of :class:`tda.core.db.Db`, kept out of ``db.py`` for size.

``db.py`` already carries :meth:`~tda.core.db.Db.set_pose_segment` (define one
segment) and :meth:`~tda.core.db.Db.pose_segment_for` (which segment is a frame
in). The segment *list* of a view -- read, partially update, trim -- is what the
pipeline needs when it re-cuts the segments at the ``reorient`` steps (spec
2.5), and lives here.

A segment's ``corners``/``homography``/``roi`` are anchored to its ``ref_step``:
whoever moves that boundary has to decide what happens to the geometry, so
:meth:`PoseSegmentMixin.update_pose_segment` changes only the fields it is
given and :meth:`PoseSegmentMixin.clear_pose_geometry` is the explicit way to
throw it away.
"""
from __future__ import annotations

from typing import Any, Optional

from tda.core import dbrows as R

__all__ = ["POSE_GEOMETRY_COLUMNS", "PoseSegmentMixin"]

#: The columns that only make sense relative to a segment's reference frame.
POSE_GEOMETRY_COLUMNS = ("corners_json", "homography_json", "roi_json")
#: What :meth:`PoseSegmentMixin.update_pose_segment` accepts.
POSE_UPDATABLE = ("start_step", "end_step", "ref_step")


class PoseSegmentMixin:
    """Read and reshape the pose segments of one ``(desktop, view)``."""

    def pose_segments(self, desktop: int, view: Optional[str] = None) -> list[dict]:
        """Stored segments of one view (or of every view), ordered by view and seg.

        Each row comes back with ``corners``/``homography``/``roi`` decoded, the
        same shape :meth:`~tda.core.db.Db.pose_segment_for` returns.
        """
        sql = "SELECT * FROM pose_segment WHERE desktop=?"
        args: list[Any] = [desktop]
        if view is not None:
            sql += " AND view=?"
            args.append(view)
        rows = self.conn.execute(sql + " ORDER BY view, seg", args).fetchall()
        return [R.pose_row(r) for r in rows]

    def update_pose_segment(self, desktop: int, view: str, seg: int, **fields) -> None:
        """Change ``start_step``/``end_step``/``ref_step`` of one segment only.

        Unlike :meth:`~tda.core.db.Db.set_pose_segment`, which rewrites the whole
        row, this leaves every other column -- the chassis corners above all --
        exactly as it was. Unknown field names raise.
        """
        unknown = set(fields) - set(POSE_UPDATABLE)
        if unknown:
            raise ValueError(f"cannot update pose segment fields {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f'"{c}"=?' for c in fields)
        with self._tx():
            self.conn.execute(
                f"UPDATE pose_segment SET {assignments} "
                "WHERE desktop=? AND view=? AND seg=?",
                (*fields.values(), desktop, view, seg),
            )

    def clear_pose_geometry(self, desktop: int, view: str, seg: int) -> None:
        """Drop the corners, homography and ROI of one segment (they are stale)."""
        assignments = ", ".join(f'"{c}"=NULL' for c in POSE_GEOMETRY_COLUMNS)
        with self._tx():
            self.conn.execute(
                f"UPDATE pose_segment SET {assignments} "
                "WHERE desktop=? AND view=? AND seg=?",
                (desktop, view, seg),
            )

    def delete_pose_segments_from(self, desktop: int, view: str, first_seg: int) -> int:
        """Delete segment ``first_seg`` and every segment after it; returns the count."""
        with self._tx():
            cur = self.conn.execute(
                "DELETE FROM pose_segment WHERE desktop=? AND view=? AND seg>=?",
                (desktop, view, first_seg),
            )
        return int(cur.rowcount or 0)
