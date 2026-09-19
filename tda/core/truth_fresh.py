"""The cheap fingerprint of a frame's inputs, and what it is used for.

Two jobs in this codebase need to know whether a frame's *inputs* have moved,
without paying for the pixel work that would prove it:

* the truth sweeper, which takes the fingerprint before compiling a frozen frame
  and checks it again inside the write transaction -- if the annotator edited in
  between, the comparison it just made describes inputs nobody has any more;
* every batch pass over a view -- an export, a quality check -- which otherwise
  compiles each frame only to discover from the ``input_hash`` afterwards that
  nothing had changed.  Fourteen frames, fourteen compilations, on an export of
  a desktop nobody had touched.

:func:`digest_of` is that fingerprint: identities and versions of everything
:class:`~tda.core.truth_inputs.FrameInputs` carries, and not one mask decoded.
:class:`~tda.core.db_digest.DigestMixin` stores it per frame, written in the
same transaction as the ``compiled_mask`` rows it describes, so the two can
never disagree.

:func:`ensure_fresh` is the batch entry point: run the pending re-checks,
recompile whatever actually moved, and refuse to hand out a view that still has
a frozen frame nobody has compared.
"""
from __future__ import annotations

import hashlib

from tda.core import masks
from tda.core.compiler import select_keyframe
from tda.core.model import FrameKey, Placement
from tda.core.truth_inputs import FrameInputs, annotatable_steps

__all__ = ["FreshMixin", "digest_of"]

IN_CHASSIS = Placement.IN_CHASSIS.value


def digest_of(inputs: FrameInputs, compiler_version: str) -> str:
    """Fingerprint one frame's compiler inputs; no mask is decoded.

    Every field of :class:`~tda.core.truth_inputs.FrameInputs` is in here, by
    identity rather than by value: a keyframe by its id and version (which
    :meth:`~tda.core.db.Db.update_keyframe` always bumps), a mask by the
    ``counts`` string it is stored as, the layer order by its version and its
    contents.  Two different input sets therefore cannot look the same without a
    sha1 collision -- but it is a fingerprint, not an equality test, and callers
    treat it as one.

    The keyframes enter as the ones the compiler would **select**, not as every
    chain the view has: re-tracing one part's shape must not move the digest of
    the frames that shape does not reach, or a batch pass would recompile the
    whole view after every edit.  Selecting is :func:`select_keyframe` over a
    filtered chain, which is a comparison of integers -- the expensive half is
    the pixels, and none are touched here.
    """
    key = inputs.key
    parts = [
        f"{key.desktop}/{key.view}/{key.step}", str(inputs.hw),
        str(inputs.pose_segment), str(inputs.bench_roi),
        str(sorted(inputs.needs.items())),
        str(sorted(inputs.placements.items())),
        str((inputs.transform.scale, inputs.transform.theta,
             inputs.transform.tx, inputs.transform.ty)),
        str((inputs.zorder.version, inputs.zorder.order)),
        str([(p.above, p.below) for p in inputs.overrides]),
        str(_selected(inputs)),
        str(sorted((o.occluder_type, masks.rle_counts(o.rle))
                   for o in inputs.occluders)),
        str(sorted((i, masks.rle_counts(o.visible_rle), o.visibility)
                   for i, o in inputs.frame_overrides.items())),
        compiler_version,
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _selected(inputs: FrameInputs) -> list[tuple]:
    """``(instance, keyframe identity)`` for the shapes this frame actually uses.

    The same narrowing :func:`tda.core.compiler.compile_frame` does in its step
    3 -- view, desktop, placement, pose segment, then the smallest anchor at or
    after this step -- so the digest moves exactly when the geometry would.

    Two of the compiler's branches are deliberately not mirrored, because
    :func:`tda.core.truth_inputs.gather` cannot produce them: it always resolves
    a concrete ``pose_segment`` (so the ``None`` fallback that picks the newest
    segment and reports ``pose_segment_ambiguous`` is unreachable), and it
    always fills ``placements`` from the state machine for every instance it put
    in ``needs`` (so the ``in_chassis`` default never applies). The assert says
    so rather than leaving the reader to check.
    """
    key = inputs.key
    assert inputs.pose_segment is not None, "gather always resolves a pose segment"
    out: list[tuple] = []
    for instance in sorted(inputs.needs):
        assert instance in inputs.placements, "gather fills placements for every need"
        placement = inputs.placements.get(instance, IN_CHASSIS)
        chain = [
            kf for kf in inputs.keyframes.get(instance, ())
            if kf.view == key.view and kf.desktop == key.desktop
            and kf.placement == placement and kf.pose_segment == inputs.pose_segment
        ]
        chosen = select_keyframe(chain, key.step)
        out.append((instance, placement,
                    None if chosen is None
                    else (chosen.id, chosen.version, chosen.geom_type)))
    return out


class FreshMixin:
    """Bringing a whole view's truth table up to date, for :class:`TruthService`."""

    def inputs_digest(self, key: FrameKey, cache=None) -> str:
        """:func:`digest_of` for one frame, gathering its inputs first."""
        from tda.core.truth_inputs import gather

        return digest_of(gather(self.db, self.tax, key, cache), self.compiler_version)

    def _digest_is_current(self, key: FrameKey, digest: str) -> bool:
        """Do the stored rows already describe exactly these inputs?

        Both halves matter: the digest says the inputs are the ones the rows
        were made from, and the row count says the rows are still there. The
        two are written together, so they only come apart if somebody deleted
        rows behind the truth service's back -- which the exports do not, but a
        repair script might.
        """
        stored = self.db.frame_digest(key)
        if stored is None or stored["digest"] != digest:
            return False
        if stored["compiler_version"] != self.compiler_version:
            return False
        return int(stored["n_rows"]) == len(self.db.compiled(key))

    def ensure_fresh(self, desktop: int, view: str, only_verified: bool = False,
                     force: bool = False) -> dict:
        """Make a whole view's truth table complete before it is read out.

        The compiled rows of an unverified frame are a cache the annotator's
        commits deliberately leave stale (spec 3.4), so anything that reads the
        table as a whole -- an export, a quality check -- has to fill it first,
        or it would silently publish a frame as it was several edits ago, or
        drop one that was never visited at all.  Frames whose digest still
        matches cost nothing.

        Frozen frames are re-checked whatever ``only_verified`` says, because
        that is where conflicts come from; the rest of the view is recompiled
        unless only the frozen rows are wanted.

        **This writes**, so the caller needs the single-user lock (spec 3.5).
        The totals are the two halves added together, and each half is also
        reported on its own under ``rechecked`` and ``refreshed``, so a caller
        can say which of them found something.

        ``open_conflicts`` counts the disagreements of this view that nobody has
        settled. They are **not** an error here -- the truth table is as fresh
        as it can be made while they stand, and the frozen rows deliberately
        keep their values -- but a caller publishing the view has to know: that
        is what :func:`tda.core.export.coco.export_coco`'s ``allow_conflicts``
        decides, and what a quality check reports.

        Raises ``RuntimeError`` when a re-check is still outstanding afterwards
        -- an export must not run on a frozen frame nobody has compared -- and
        when the stored digests were written by another ``compiler_version``,
        because recompiling the view under a different one would move every
        digest and mass-queue the frozen frames; ``force=True`` accepts that.
        """
        if not force:
            self._refuse_foreign_digests(desktop, view)
        drained = self.run_pending_rechecks(desktop, view)
        total = dict(drained)
        total["rechecked"] = drained
        total["refreshed"] = {}
        if not only_verified:
            steps = annotatable_steps(
                self.db, desktop, view,
                [row["step"] for row in self.db.frames_for(desktop, view)],
            )
            swept = self.refresh_range(desktop, view, steps)
            total["refreshed"] = swept
            for counter in ("updated", "conflicts", "skipped"):
                total[counter] = total.get(counter, 0) + swept[counter]
            total["problems"] = list(total.get("problems") or []) + list(swept["problems"])
        left = self.pending_rechecks(desktop, view)
        if left:
            raise RuntimeError(
                f"desktop {desktop} view {view}: {len(left)} frozen frame(s) still "
                f"await a truth re-check ({left[:5]}...); run them before exporting"
            )
        total["open_conflicts"] = len(self.open_conflicts(desktop, view))
        return total

    def _refuse_foreign_digests(self, desktop: int, view: str) -> None:
        """Refuse a view whose digests this compiler did not write."""
        foreign = {
            row["compiler_version"]
            for row in self.db.frame_digests(desktop, view).values()
            if row["compiler_version"] != self.compiler_version
        }
        if foreign:
            raise RuntimeError(
                f"desktop {desktop} view {view}: the stored digests were written by "
                f"compiler version {sorted(foreign)}, this service is "
                f"{self.compiler_version!r}. Refreshing would move every digest and "
                f"queue every frozen frame for a re-check; pass force=True if that "
                f"is what you mean"
            )
