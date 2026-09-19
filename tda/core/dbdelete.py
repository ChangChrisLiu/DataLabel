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

from tda.core.model import FrameKey, PairOverride

__all__ = ["DeleteMixin"]


class DeleteMixin:
    """Row removals for a repository holding ``self.conn``."""

    def delete_relation(self, desktop: int, rel_type: str, target: str,
                        blocker: str) -> None:
        """Drop one constraint edge; an edge that is not there is not an error.

        ``python -m tda.cli constraints`` re-derives a desktop's rule edges from
        scratch, so an edge the rules no longer propose has to go -- otherwise a
        corrected ``screw.fastens`` would leave the old ``fastened_by`` behind
        forever, and the graph would only ever grow.
        """
        with self._tx():
            self.conn.execute(
                'DELETE FROM relation WHERE desktop=? AND "type"=? AND target=? '
                "AND blocker=?",
                (int(desktop), str(rel_type), str(target), str(blocker)),
            )

    def delete_keyframe(self, keyframe_id: int) -> None:
        """Drop one shape keyframe; its parts go with it (``ON DELETE CASCADE``)."""
        with self._tx():
            self.conn.execute("DELETE FROM shape_keyframe WHERE id=?", (int(keyframe_id),))

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
