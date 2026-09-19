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

from typing import Any, Optional, Sequence

from tda.core import dbrows as R

__all__ = ["POSE_GEOMETRY_COLUMNS", "PoseSegmentMixin", "clean_roi"]


def _clean_roi(roi: Sequence[float],
               hw: Optional[tuple[int, int]] = None) -> list[int]:
    """Validate ``[x0, y0, x1, y1]`` and clamp it into a frame of size ``hw``.

    The one validator for every rectangle stored against a pose segment -- the
    chassis crop an export may cut to, and the staging area that decides whether
    a view is asked for bench boxes at all.
    """
    values = list(roi)
    if len(values) != 4:
        raise ValueError(f"an ROI is four numbers (x0, y0, x1, y1), got {roi!r}")
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"an ROI must be numeric, got {roi!r}") from None
    if hw is not None:
        height, width = int(hw[0]), int(hw[1])
        x0, x1 = min(max(x0, 0), width), min(max(x1, 0), width)
        y0, y1 = min(max(y0, 0), height), min(max(y1, 0), height)
    else:
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = max(x1, 0), max(y1, 0)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"an ROI must be non-empty with x0 < x1 and y0 < y1, got {roi!r}")
    return [x0, y0, x1, y1]


#: Public name of the validator, so the window can check a draft before storing.
clean_roi = _clean_roi

#: The columns that only make sense relative to a segment's reference frame.
POSE_GEOMETRY_COLUMNS = ("corners_json", "homography_json", "roi_json",
                         "bench_roi_json")
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

    def set_pose_segment_roi(self, desktop: int, view: str, seg: int,
                             roi: Optional[Sequence[float]],
                             annotator: str = "system",
                             hw: Optional[tuple[int, int]] = None) -> Optional[list[int]]:
        """Store (or clear with ``None``) one segment's region of interest.

        The ROI is ``[x0, y0, x1, y1]`` in the reference frame's coordinates --
        the chassis rectangle the annotator accepts on the first open of a
        desktop/view (spec 2.4). :meth:`~tda.core.db.Db.set_pose_segment`
        rewrites the whole row and never wrote this column, so the window needs
        a setter that leaves the corners and the homography alone.

        It is **validated and logged** rather than written blind, because the
        ROI leaves the tool: ``export_coco(roi_crop=True)`` crops every mask to
        it, so a transposed or degenerate rectangle silently truncates a
        release, and "who decided this crop, and when" has to be answerable.
        ``hw`` clamps the rectangle to a frame of that size when it is known.

        Returns the rectangle as stored, or ``None`` when it was cleared.

        Raises:
            ValueError: ``roi`` is not four numbers, or is empty/inverted, or
                the segment does not exist.
        """
        payload = None if roi is None else _clean_roi(roi, hw)
        before = self._roi_of(desktop, view, seg)
        with self.transaction():
            cur = self.conn.execute(
                "UPDATE pose_segment SET roi_json=? "
                "WHERE desktop=? AND view=? AND seg=?",
                (R.dumps(payload), desktop, view, seg),
            )
            if not cur.rowcount:
                raise ValueError(
                    f"no pose segment {seg} for desktop {desktop} view {view!r}"
                )
            # in the same transaction as the write, like the bench ROI next
            # door: a crop that leaves the tool with no record of who chose it
            # is the half of this pair that must not survive alone
            self.log_op(
                desktop, view, "set_pose_roi",
                {"seg": int(seg), "roi": payload}, {"seg": int(seg), "roi": before},
                annotator,
            )
        return payload

    def _has_segment(self, desktop: int, view: str, seg: int) -> bool:
        """Does this segment exist? Asked before a write decides it has nothing to do."""
        return self.conn.execute(
            "SELECT 1 FROM pose_segment WHERE desktop=? AND view=? AND seg=?",
            (desktop, view, seg),
        ).fetchone() is not None

    def _roi_of(self, desktop: int, view: str, seg: int) -> Optional[list[int]]:
        row = self.conn.execute(
            "SELECT roi_json FROM pose_segment WHERE desktop=? AND view=? AND seg=?",
            (desktop, view, seg),
        ).fetchone()
        return None if row is None else R.loads(row["roi_json"])

    def clear_pose_geometry(self, desktop: int, view: str, seg: int) -> None:
        """Drop the corners, homography and ROI of one segment (they are stale).

        Every one of those columns is a compiler input -- the registration this
        view's shapes are drawn against, and the staging area that decides
        whether bench parts are in the frame at all -- so the frozen frames of
        the view are queued for a re-check in the same transaction. Nothing else
        would ever compare them: this runs from the pipeline, not from the
        session's edit path.
        """
        assignments = ", ".join(f'"{c}"=NULL' for c in POSE_GEOMETRY_COLUMNS)
        with self.transaction():
            self.conn.execute(
                f"UPDATE pose_segment SET {assignments} "
                "WHERE desktop=? AND view=? AND seg=?",
                (desktop, view, seg),
            )
            self.queue_rechecks_for_view(desktop, view)

    def set_pose_segment_bench_roi(self, desktop: int, view: str, seg: int,
                                   roi: Optional[Sequence[float]],
                                   annotator: str = "system",
                                   hw: Optional[tuple[int, int]] = None
                                   ) -> Optional[list[int]]:
        """Record (or clear with ``None``) the staging area this view can see.

        Spec 4.2 asks for a part on the bench to be boxed only 若该视角有堆放区
        ROI -- *if this view has a staging area*. The scanner looks straight down
        at the board and never will, so this stays NULL on most views, and the
        task card asks for no bench work until somebody draws one.

        The rectangle goes through :func:`clean_roi`, the same validator the
        chassis ROI uses, and the change is logged the same way. A frame has one
        notion of "a rectangle in its coordinates": a second validator that
        rounded differently and clamped nothing is how the two drifted apart, and
        which of them a bench box was measured against is as answerable a
        question as it is for the chassis crop.

        Drawing (or clearing) a staging area changes what **every frame of the
        view** compiles to: a part on the bench becomes an instance of the frame
        that was not in it before, so ``needs``, ``placements`` and the digest
        all move (spec 3.3 step 2). The view's frozen frames are therefore
        queued for a re-check, in the same transaction as the write and the op
        log -- without it they kept rows describing a machine with nothing on
        the bench, and an export published them.

        Returns the rectangle as stored, or ``None`` when it was cleared.

        Raises:
            ValueError: ``roi`` is malformed, or the segment does not exist.
        """
        payload = None if roi is None else _clean_roi(roi, hw)
        before = self.bench_roi(desktop, view, seg)
        if payload == before and self._has_segment(desktop, view, seg):
            # re-confirming the rectangle that is already there is not a change
            # to the view: nothing compiles differently, so nothing is queued
            # and there is nothing to log
            return payload
        with self.transaction():
            cur = self.conn.execute(
                "UPDATE pose_segment SET bench_roi_json=? WHERE desktop=? AND view=? "
                "AND seg=?",
                (R.dumps(payload), desktop, view, seg),
            )
            if not cur.rowcount:
                raise ValueError(
                    f"no pose segment {seg} for desktop {desktop} view {view!r}"
                )
            self.queue_rechecks_for_view(desktop, view)
            self.log_op(
                desktop, view, "set_bench_roi",
                {"seg": int(seg), "roi": payload}, {"seg": int(seg), "roi": before},
                annotator,
            )
        return payload

    def bench_roi(self, desktop: int, view: str, seg: int) -> Optional[list]:
        """The staging area of one pose segment, or ``None`` when it has none."""
        row = self.conn.execute(
            "SELECT bench_roi_json FROM pose_segment WHERE desktop=? AND view=? AND seg=?",
            (desktop, view, seg),
        ).fetchone()
        return None if row is None else R.loads(row["bench_roi_json"])

    def delete_pose_segments_from(self, desktop: int, view: str, first_seg: int) -> int:
        """Delete segment ``first_seg`` and every segment after it; returns the count.

        The frames of those segments fall back to another segment's reference
        frame, which is a different set of compiler inputs, so the view's frozen
        frames are queued for a re-check in the same transaction.
        """
        with self.transaction():
            cur = self.conn.execute(
                "DELETE FROM pose_segment WHERE desktop=? AND view=? AND seg>=?",
                (desktop, view, first_seg),
            )
            removed = int(cur.rowcount or 0)
            if removed:
                self.queue_rechecks_for_view(desktop, view)
        return removed
