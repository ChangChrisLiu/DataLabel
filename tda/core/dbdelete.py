"""The inverse of the geometry setters of :class:`tda.core.db.Db`.

Every edit the annotator makes is logged with the patch that takes it back
(spec 4.6), and an undo that *created* something has to be able to remove it
again: a freshly traced shape, a pairwise layering exception, a single-frame
override, an occluder layer.  :mod:`tda.core.db` has one setter per table and
no deletes, because nothing outside undo ever removes annotation data -- so the
four deletes live here, mixed into ``Db`` next to
:class:`~tda.core.dbconn.ConnectionMixin`.

Each one is idempotent: removing a row that is not there is not an error, which
is what makes an undo safe to replay.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey, PairOverride

__all__ = ["DeleteMixin"]


def _scoped(sql: str, args: list, desktops) -> tuple[str, list]:
    """Narrow a delete to ``desktops``; ``None`` means every desktop."""
    if desktops is None:
        return sql, args
    ids = sorted({int(d) for d in desktops})
    if not ids:
        return sql + " AND 0", args  # an empty scope deletes nothing
    return sql + f" AND desktop IN ({','.join('?' * len(ids))})", args + ids


class DeleteMixin:
    """Row removals for a repository holding ``self.conn``."""

    def delete_keyframe(self, keyframe_id: int) -> None:
        """Drop one shape keyframe; its parts go with it (``ON DELETE CASCADE``)."""
        with self._tx():
            self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (int(keyframe_id),))

    def delete_keyframes_by_source(self, source: str, desktops=None,
                                   instance: Optional[str] = None) -> int:
        """Drop imported shapes by provenance; returns how many rows went.

        The second exception to "nothing outside undo removes annotation data",
        and a narrow one: only rows carrying the *importer's* ``source``.
        Two callers, one query:

        * :func:`tda.core.ls_import.import_ls_export` re-runs itself, which
          means replacing everything a previous run wrote for the desktops the
          export covers (``desktops=None`` is the deliberate full reset);
        * S1 deletes one Label Studio draft key together with the shapes it was
          made of -- the two are one thing, and a key nobody adopted would
          otherwise leave orphaned pixels behind. Anything a human drew onto
          that key has a different ``source`` and stays.

        ``shape_part`` rows go with their keyframe (``ON DELETE CASCADE``). The
        caller is responsible for the re-checks this owes
        (:meth:`~tda.core.db_recheck.RecheckMixin.queue_rechecks_for_view`):
        only it knows which views it has just changed.
        """
        sql = "DELETE FROM shape_keyframe WHERE source=?"
        args: list = [str(source)]
        sql, args = _scoped(sql, args, desktops)
        if instance is not None:
            sql += " AND instance=?"
            args.append(str(instance))
        with self._tx():
            return int(self.conn.execute(sql, args).rowcount or 0)

    def delete_relations_by_source(self, source: str, desktops=None) -> int:
        """The same for constraint edges an importer proposed; returns how many.

        ``relation`` rows are desktop-scoped and cannot be narrowed by view, so
        a re-import covering one view of a desktop replaces that desktop's edges
        whole. They need a real delete rather than an upsert: an edge whose
        provisional key shifted between runs would upsert to a *different* row
        and the old one would linger.
        """
        sql, args = _scoped("DELETE FROM relation WHERE source=?", [str(source)], desktops)
        with self._tx():
            return int(self.conn.execute(sql, args).rowcount or 0)

    def delete_pair_override(self, po: PairOverride) -> None:
        """Drop one "above beats below" exception of a (view, pose segment)."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM pair_override WHERE desktop=? AND view=? AND pose_segment=? "
                "AND above=? AND below=?",
                (po.desktop, po.view, po.pose_segment, po.above, po.below),
            )

    def delete_frame_override(self, key: FrameKey, instance: str) -> None:
        """Drop one instance's frame-local mask/visibility override."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM frame_override WHERE desktop=? AND step=? AND view=? AND instance=?",
                (key.desktop, key.step, key.view, instance),
            )

    def delete_occluder(self, key: FrameKey, occluder_type: str) -> None:
        """Drop one occluder layer of one frame."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM occluder_mask WHERE desktop=? AND step=? AND view=? "
                "AND occluder_type=?",
                (key.desktop, key.step, key.view, occluder_type),
            )
