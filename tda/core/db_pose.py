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

Schema version 4 adds the view's *own* breaks (spec 2.5 v1.5) -- the
``pose_break`` rows -- and with them the one routine that may move a boundary
once there is annotator work on both sides of it:
:meth:`PoseSegmentMixin.apply_recut`. It is a single transaction by
construction: every row keyed by a segment number is re-keyed, copied or
carried, the frozen frames it reaches are queued for a re-check, and the whole
thing is logged -- or none of it happens.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from tda.core import dbrows as R
from tda.core.pose_breaks import (
    ACCEPTED,
    CARRIED,
    STATUSES,
    RecutPlan,
    recut_plan,
    straddling,
)

__all__ = ["POSE_GEOMETRY_COLUMNS", "RECUT_OP", "PoseSegmentMixin", "clean_roi"]

#: ``op_log.kind`` of a re-cut. It is **not** on the undo stack: a re-cut is a
#: structural edit like applying the step table, and its undo is to reject the
#: break and re-cut again (which merges the segments back).
RECUT_OP = "pose_recut"


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


def _has_ref_geometry(row: dict) -> bool:
    """Does this segment carry anything drawn against its **reference frame**?

    The two rectangles are not: they are windows in the view's own image, and a
    re-cut keeps them (see :data:`FRAME_BOX_COLUMNS`).
    """
    return any(row.get(name) is not None for name in ("corners", "homography"))


def _needs_repair(rows: list[dict]) -> bool:
    """Is a stored segment row inconsistent with itself?

    A reference step outside the segment's own range (or missing) means the
    shapes of that segment are drawn against a frame that is not in it. It used
    to be repaired as a side effect of a cut always rewriting every row; now
    that an unchanged cut writes nothing, it is a reason to write on its own.
    """
    return any(r.get("start_step") is None or r.get("end_step") is None
               or r.get("ref_step") is None
               or not (int(r["start_step"]) <= int(r["ref_step"]) <= int(r["end_step"]))
               for r in rows)


def _keeper_of(by_seg: dict, sources: list[int], start: int, end: int) -> Optional[int]:
    """Which of several merging segments gives the merged one its reference frame.

    Undoing a split has to give back what the split took: the piece whose
    reference step is still inside the merged range wins, the latest one of
    those first, because that is the frame a new segment would reference anyway
    (spec 4.2 annotates the most disassembled frame first) and it is the piece
    the chassis corners were left with.
    """
    inside = [s for s in sources
              if (by_seg.get(s) or {}).get("ref_step") is not None
              and start <= int(by_seg[s]["ref_step"]) <= end]
    with_geometry = [s for s in inside if _has_ref_geometry(by_seg[s])]
    for candidates in (with_geometry, inside):
        if candidates:
            return max(candidates, key=lambda s: int(by_seg[s]["ref_step"]))
    return sources[-1] if sources else None


def _empty_recut(plan: "RecutPlan") -> dict:
    """The summary of a re-cut that has not written anything (yet, or at all)."""
    return {
        "changed": False,
        "ranges": list(plan.ranges),
        "renumber": dict(plan.renumber),
        "split": {k: list(v) for k, v in plan.split.items()},
        "ref_moves": [],
        "discarded": [],
        "carried": [],
        "uncarried": [],
        "rechecked": [],
    }

#: The columns that only make sense relative to a segment's reference frame.
POSE_GEOMETRY_COLUMNS = ("corners_json", "homography_json", "roi_json",
                         "bench_roi_json")
#: Of those, the two that are drawn *against* the reference frame and are wrong
#: the moment it leaves the segment: the chassis corners and the homography
#: derived from them. A re-cut drops these, and only these (spec 2.4/2.5).
REF_BOUND_COLUMNS = ("corners_json", "homography_json")
#: And the two that are rectangles in the view's own image and survive a re-cut:
#: masks are stored full-frame (spec 2.4), so an ROI is a window for display,
#: diffing and inference, never a coordinate system. Dropping the chassis
#: rectangle because the camera was nudged 3 px would throw away a decision the
#: annotator made and make them take it again; it is copied to every piece a
#: segment is split into and re-confirmed there instead.
FRAME_BOX_COLUMNS = ("roi_json", "bench_roi_json")
#: Every column of a ``pose_segment`` row apart from its key.
POSE_ROW_COLUMNS = ("start_step", "end_step", "ref_step", *POSE_GEOMETRY_COLUMNS)
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

    # ------------------------------------------------------------ pose breaks
    def pose_breaks(self, desktop: int, view: Optional[str] = None,
                    status: Optional[str] = None) -> list[dict]:
        """This desktop's per-view breaks, ordered by view and step.

        ``view`` and ``status`` narrow it; ``status='accepted'`` is what
        :func:`tda.pipeline.split_pose_segments` asks for, because a proposal
        nobody has looked at must not move a boundary under an annotator.
        """
        sql = "SELECT * FROM pose_break WHERE desktop=?"
        args: list[Any] = [int(desktop)]
        if view is not None:
            sql += " AND view=?"
            args.append(str(view))
        if status is not None:
            sql += " AND status=?"
            args.append(str(status))
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY view, step", args)]

    def pose_break(self, desktop: int, view: str, step: int) -> Optional[dict]:
        """One break row, or ``None``."""
        row = self.conn.execute(
            "SELECT * FROM pose_break WHERE desktop=? AND view=? AND step=?",
            (int(desktop), str(view), int(step)),
        ).fetchone()
        return None if row is None else dict(row)

    def add_pose_break(self, desktop: int, view: str, step: int, *,
                       status: str = ACCEPTED, kind: Optional[str] = None,
                       magnitude_px: Optional[float] = None, source: str = "manual",
                       note: str = "") -> None:
        """Record one break; an existing row of that ``(desktop, view, step)`` wins.

        Never an overwrite: the audit import proposes the same six camera moves
        on every run, and a proposal the annotator has already accepted -- or
        rejected after flashing the two frames -- is a decision, not something
        for a re-import to take back. Changing a stored row's verdict is
        :meth:`set_pose_break_status`'s job and nothing else's.
        """
        if status not in STATUSES:
            raise ValueError(f"a pose break's status is one of {STATUSES}, got {status!r}")
        with self._tx():
            self._ensure_desktop(int(desktop))
            self.conn.execute(
                "INSERT OR IGNORE INTO pose_break(desktop, view, step, status, kind, "
                "magnitude_px, source, note) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (int(desktop), str(view), int(step), status, kind,
                 None if magnitude_px is None else float(magnitude_px), str(source),
                 str(note or "")),
            )

    def set_pose_break_status(self, desktop: int, view: str, step: int,
                              status: str) -> bool:
        """Accept or reject one break; ``False`` when there is no such row.

        The status is the whole of the decision: rejecting does **not** delete
        the row, because the audit would propose it again on the next import and
        the annotator would be asked to look at the same two frames a second
        time.
        """
        if status not in STATUSES:
            raise ValueError(f"a pose break's status is one of {STATUSES}, got {status!r}")
        with self._tx():
            cur = self.conn.execute(
                "UPDATE pose_break SET status=? WHERE desktop=? AND view=? AND step=?",
                (str(status), int(desktop), str(view), int(step)),
            )
            return bool(cur.rowcount)

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

    # --------------------------------------------------------------- the re-cut
    def apply_recut(self, desktop: int, view: str, new_bounds: Iterable[int],
                    n_steps: Optional[int] = None, *, carry_at: Iterable[int] = (),
                    annotator: str = "system", note: str = "") -> dict:
        """Re-cut one view's pose segments at ``new_bounds``, in ONE transaction.

        ``new_bounds`` is the whole boundary list the view should have
        (:func:`tda.core.pose_breaks.boundaries`), so a re-cut is a
        re-derivation and running it twice is a no-op -- which is what lets
        ``load-index`` and ``import-logs --force`` run it again over segments a
        human has since added breaks to.

        Nothing an annotator made disappears:

        * a **keyframe** goes to the new segment that holds its own
          ``anchor_step``.  The frames on the other side of a new boundary lose
          it and become ``missing_shape``, which is the intended "redraw it in
          the new pose" signal (spec 3.3 step 3) -- unless the boundary is in
          ``carry_at``, in which case every keyframe whose coverage straddles it
          is **duplicated** into the earlier segment with
          ``anchor_step = boundary - 1`` and ``source='carried'``;
        * **per-segment** rows -- the layer order, the pair overrides, the
          chassis ROI and the staging-area ROI -- are *copied* into every piece
          a segment was split into: after a 3 px camera nudge they are still the
          best first guess, and the annotator re-confirms the ROI;
        * the **chassis corners** and the homography drawn against them follow
          the reference step and are dropped only where it no longer is;
        * **verified** frames of every affected segment are queued for a
          re-check (``recheck_queue`` -> the conflict queue).  Not one compiled
          row is written here: a frozen row changes only through that queue (I1).

        Removing a break -- ``set_pose_break_status(..., 'rejected')`` and re-cut
        -- merges the two segments back and is the undo of a split; the
        untouched ``carried`` duplicates it created are deleted again, an edited
        one is kept and reported.

        Returns a summary dict; ``changed`` is ``False`` when the view already
        looked exactly like this and nothing at all was written.
        """
        desktop, view = int(desktop), str(view)
        rows = self.pose_segments(desktop, view)
        old = [(int(r["seg"]), int(r["start_step"]), int(r["end_step"])) for r in rows
               if r["start_step"] is not None and r["end_step"] is not None]
        plan = recut_plan(old, [int(b) for b in new_bounds],
                          int(n_steps) if n_steps else 0)
        summary = _empty_recut(plan)
        if not old or not (plan.changed or _needs_repair(rows)):
            # A view with no segments (nothing photographed it) and a view that
            # is already cut like this are the same answer: there is nothing to
            # re-key, so nothing is written and nothing is queued. A stored row
            # whose reference step is not inside its own range is the exception:
            # the ranges may be right while the frame the shapes are drawn
            # against is not, and that is repaired here as it always was.
            return summary

        old_starts = {start for _seg, start, _end in plan.old if start > 1}
        new_starts = {start for _seg, start, _end in plan.ranges if start > 1}
        carry = sorted({int(b) for b in carry_at} & (new_starts - old_starts))
        gone = sorted(old_starts - new_starts)

        with self.transaction():
            carried = self._carry_keyframes(desktop, view, plan, carry)
            summary["ref_moves"], summary["discarded"] = self._recut_segment_rows(
                desktop, view, rows, plan)
            summary["discarded"] += self._recut_layer_rows(desktop, view, plan)
            self._recut_keyframes(desktop, view, plan)
            self._recut_frame_overrides(desktop, view, plan)
            summary["carried"] = [self._insert_carried(desktop, view, plan, *c)
                                  for c in carried]
            summary["uncarried"] = self._drop_carried(desktop, view, gone)
            affected = plan.affected_steps()
            for move in summary["ref_moves"]:
                # dropping the corners changes the registration of every frame
                # of that segment, whether or not its range moved
                if move["dropped"]:
                    affected.update(range(move["start"], move["end"] + 1))
            summary["rechecked"] = self.add_rechecks(
                desktop, view,
                [s for s in self.frozen_steps(desktop, view) if s in affected])
            summary["changed"] = True
            payload = {"ranges": [list(r) for r in plan.ranges],
                       "renumber": {str(k): v for k, v in plan.renumber.items()},
                       "split": {str(k): v for k, v in plan.split.items()},
                       "carried": summary["carried"], "carry_at": carry,
                       "discarded": summary["discarded"],
                       "rechecked": summary["rechecked"], "note": str(note or "")}
            self.log_op(desktop, view, RECUT_OP, payload,
                        {"ranges": [list(r) for r in plan.old]}, annotator)
        return summary

    # -- the five row groups a re-cut touches ---------------------------------
    def _recut_segment_rows(self, desktop: int, view: str, rows: list[dict],
                            plan: RecutPlan) -> tuple[list[dict], list[dict]]:
        """Rewrite the segment table itself; returns (reference moves, discarded).

        The columns are split by what they are drawn against: ``ref_step`` is
        re-chosen the way a new cut has always chosen it (keep the stored one
        while it is still inside the segment, else its last step -- the most
        disassembled frame, which spec 4.2 annotates first), the corners and the
        homography survive only where that reference step did, and the two
        rectangles survive everywhere because they are windows in the view's own
        image, not a coordinate system (spec 2.4).
        """
        by_seg = {int(r["seg"]): r for r in rows}
        ref_moves: list[dict] = []
        discarded: list[dict] = []
        new_rows: list[tuple] = []
        for seg, start, end in plan.ranges:
            sources = plan.merged.get(seg) or []
            keeper = _keeper_of(by_seg, sources, start, end) if len(sources) > 1 else (
                sources[0] if sources else plan.old_segment_of(start))
            src = by_seg.get(keeper if keeper is not None else -1) or {}
            stored = src.get("ref_step")
            ref = stored if (stored is not None and start <= int(stored) <= end) else end
            kept_ref = stored is not None and ref == stored
            data = {
                "start_step": start, "end_step": end, "ref_step": int(ref),
                "corners_json": R.dumps(src.get("corners")) if kept_ref else None,
                "homography_json": R.dumps(src.get("homography")) if kept_ref else None,
                "roi_json": R.dumps(src.get("roi")),
                "bench_roi_json": R.dumps(src.get("bench_roi")),
            }
            new_rows.append((desktop, view, seg, *data.values()))
            if not kept_ref and src:
                ref_moves.append({"seg": seg, "start": start, "end": end,
                                  "ref_step": int(ref), "old": src,
                                  "dropped": _has_ref_geometry(src)})
            for extra in [s for s in sources if s != keeper]:
                # only what the merge really loses: the pieces of a split carry
                # copies of one another's rectangles, and reporting those back
                # as "discarded" would be a warning about nothing
                lost = {n: (by_seg.get(extra) or {}).get(n)
                        for n in ("corners", "homography", "roi", "bench_roi")
                        if (by_seg.get(extra) or {}).get(n) is not None
                        and (by_seg.get(extra) or {}).get(n) != src.get(n)}
                if lost:
                    discarded.append({"table": "pose_segment", "pose_segment": extra,
                                      "into": seg, "row": lost})
        self.conn.execute("DELETE FROM pose_segment WHERE desktop=? AND view=?",
                          (desktop, view))
        self.conn.executemany(
            "INSERT INTO pose_segment(desktop, view, seg, start_step, end_step, ref_step, "
            "corners_json, homography_json, roi_json, bench_roi_json) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", new_rows)
        return ref_moves, discarded

    def _recut_layer_rows(self, desktop: int, view: str, plan: RecutPlan) -> list[dict]:
        """Re-key the layer order and the pair overrides; returns what was dropped.

        A **split** copies both into every piece.  A **merge** unions the pair
        overrides -- they are a set of exceptions, so nothing has to give way --
        but a z-order is a total order and two of them cannot be merged: the
        earlier segment's wins, the later one's is recorded here and in the op
        log, and the caller says so.
        """
        dropped: list[dict] = []
        targets = {seg for seg, _s, _e in plan.ranges}
        touched = set(plan.renumber) | targets

        z_rows = self._rows_in("zorder", desktop, view, touched)
        p_rows = self._rows_in("pair_override", desktop, view, touched)
        keys = ", ".join("?" * len(touched))
        for table in ("zorder", "pair_override"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE desktop=? AND view=? AND pose_segment IN ({keys})",
                (desktop, view, *sorted(touched)))

        placed: set[int] = set()
        for row in z_rows:                      # in old-segment order: earliest wins
            for target in self._targets_of(plan, int(row["pose_segment"])):
                if target in placed:
                    dropped.append({"table": "zorder",
                                    "pose_segment": int(row["pose_segment"]),
                                    "into": target,
                                    "row": R.loads(row["order_json"])})
                    continue
                placed.add(target)
                self.conn.execute(
                    "INSERT INTO zorder(desktop, view, pose_segment, order_json, version) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (desktop, view, target, row["order_json"], row["version"]))
        for row in p_rows:
            for target in self._targets_of(plan, int(row["pose_segment"])):
                self.conn.execute(
                    "INSERT OR IGNORE INTO pair_override(desktop, view, pose_segment, "
                    "above, below) VALUES(?, ?, ?, ?, ?)",
                    (desktop, view, target, row["above"], row["below"]))
        return dropped

    def _rows_in(self, table: str, desktop: int, view: str, segs: set[int]) -> list:
        """Every row of ``table`` for this view in one of ``segs``, oldest segment first."""
        keys = ", ".join("?" * len(segs))
        return self.conn.execute(
            f"SELECT * FROM {table} WHERE desktop=? AND view=? AND pose_segment IN ({keys}) "
            f"ORDER BY pose_segment", (desktop, view, *sorted(segs))).fetchall()

    @staticmethod
    def _targets_of(plan: RecutPlan, seg: int) -> list[int]:
        """The new segments one old segment's per-segment rows belong to.

        A segment number the plan does not know is an orphan left by an earlier
        shrink; it is dropped rather than re-adopted by whichever segment now
        happens to carry that number.
        """
        if seg not in plan.renumber:
            return []
        return plan.split.get(seg) or [plan.renumber[seg]]

    def _recut_keyframes(self, desktop: int, view: str, plan: RecutPlan) -> None:
        """Move every shape keyframe to the new segment that holds its anchor step.

        Rows whose ``pose_segment`` the plan does not know are left exactly as
        they are -- which is how the Label Studio drafts (segment 0, never
        compiled, spec 5) come through a re-cut untouched.
        """
        keys = ", ".join("?" * len(plan.renumber))
        rows = self.conn.execute(
            f"SELECT id, pose_segment, anchor_step FROM shape_keyframe "
            f"WHERE desktop=? AND view=? AND pose_segment IN ({keys})",
            (desktop, view, *sorted(plan.renumber))).fetchall()
        for row in rows:
            target = plan.segment_of(int(row["anchor_step"]))
            if target is None:      # an anchor outside every range: keep it with
                target = plan.renumber[int(row["pose_segment"])]   # its own segment
            if target != int(row["pose_segment"]):
                self.conn.execute("UPDATE shape_keyframe SET pose_segment=? WHERE id=?",
                                  (int(target), int(row["id"])))

    def _recut_frame_overrides(self, desktop: int, view: str, plan: RecutPlan) -> None:
        """Follow the frames that name a pose segment themselves.

        ``frame.pose_segment`` beats the step ranges everywhere it is not NULL
        (:meth:`tda.core.db.Db.pose_segment_for`), so leaving it behind would
        point those frames -- and the thumbnails cut for them -- at a segment
        number that now means something else.
        """
        rows = self.conn.execute(
            "SELECT step, pose_segment FROM frame WHERE desktop=? AND view=? "
            "AND pose_segment IS NOT NULL", (desktop, view)).fetchall()
        old_ranges = {seg: (start, end) for seg, start, end in plan.old}
        for row in rows:
            seg, step = int(row["pose_segment"]), int(row["step"])
            if seg not in plan.renumber:
                continue
            target = plan.renumber[seg]
            start, end = old_ranges[seg]
            if seg in plan.split and start <= step <= end:
                target = plan.segment_of(step) or target
            if target != seg:
                self.conn.execute(
                    "UPDATE frame SET pose_segment=? WHERE desktop=? AND view=? AND step=?",
                    (target, desktop, view, step))

    # -- carrying shapes across a new boundary --------------------------------
    def _carry_keyframes(self, desktop: int, view: str, plan: RecutPlan,
                         carry: list[int]) -> list[tuple[int, int]]:
        """``(keyframe id, boundary)`` for every shape a carried boundary cuts through.

        Read **before** anything is re-keyed, because "which shapes straddle
        this boundary" is a question about the segment as it still is.
        """
        if not carry:
            return []
        keys = ", ".join("?" * len(plan.renumber))
        rows = self.conn.execute(
            f"SELECT id, instance, placement, pose_segment, anchor_step, source "
            f"FROM shape_keyframe WHERE desktop=? AND view=? AND pose_segment IN ({keys}) "
            f"ORDER BY anchor_step, id", (desktop, view, *sorted(plan.renumber))).fetchall()
        starts = {seg: start for seg, start, _end in plan.old}
        chains: dict[tuple, list] = {}
        for row in rows:
            chains.setdefault(
                (row["instance"], row["placement"], int(row["pose_segment"])), []).append(row)
        out: list[tuple[int, int]] = []
        for boundary in carry:
            for (_inst, _place, seg), chain in chains.items():
                anchors = [int(r["anchor_step"]) for r in chain]
                if boundary - 1 in anchors:
                    continue            # this chain already ends at the boundary
                wanted = set(straddling(anchors, starts.get(seg, 1), boundary))
                out += [(int(r["id"]), boundary) for r in chain
                        if int(r["anchor_step"]) in wanted]
        return out

    def _insert_carried(self, desktop: int, view: str, plan: RecutPlan,
                        keyframe_id: int, boundary: int) -> int:
        """Duplicate one keyframe into the segment before ``boundary``; its new id."""
        segment = plan.segment_of(boundary - 1)
        self.conn.execute(
            "INSERT INTO shape_keyframe(desktop, view, instance, pose_segment, anchor_step, "
            "placement, geom_type, amodal_complete, source, draft_id, version, edit_count, "
            "edit_time_ms) SELECT desktop, view, instance, ?, ?, placement, geom_type, "
            "amodal_complete, ?, draft_id, 1, 0, 0 FROM shape_keyframe WHERE id=?",
            (segment, int(boundary) - 1, CARRIED, int(keyframe_id)))
        new_id = int(self.conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        self.conn.execute(
            "INSERT INTO shape_part(keyframe_id, idx, name, rle_json, box_json) "
            "SELECT ?, idx, name, rle_json, box_json FROM shape_part WHERE keyframe_id=?",
            (new_id, int(keyframe_id)))
        return new_id

    def _drop_carried(self, desktop: int, view: str, gone: list[int]) -> list[int]:
        """Undo the duplicates of the boundaries that have just disappeared.

        Only the untouched ones: a carried keyframe the annotator has since
        redrawn carries a version above 1 and is their work, not bookkeeping, so
        it stays and its id is returned for the caller to mention.
        """
        kept: list[int] = []
        for boundary in gone:
            rows = self.conn.execute(
                "SELECT id, version, edit_count FROM shape_keyframe WHERE desktop=? AND "
                "view=? AND source=? AND anchor_step=?",
                (desktop, view, CARRIED, int(boundary) - 1)).fetchall()
            for row in rows:
                if int(row["version"]) > 1 or int(row["edit_count"]) > 0:
                    kept.append(int(row["id"]))
                    continue
                self.conn.execute("DELETE FROM shape_part WHERE keyframe_id=?", (row["id"],))
                self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (row["id"],))
        return kept
