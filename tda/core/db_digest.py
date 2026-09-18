"""What a frame's compiled rows were derived from, stored beside them.

Deriving one frame's geometry is the expensive half of the truth table, and
until now the only way to find out whether it had to be done again was to do it:
:meth:`~tda.core.truth.TruthService.refresh` compiled the frame and *then*
compared the resulting ``input_hash`` with the stored one.  A batch pass over a
view nobody had touched -- an export, a quality check -- therefore cost one full
compilation per frame, every time.

The digest here is the cheap fingerprint of the same inputs
(:func:`tda.core.truth_fresh.digest_of`), written in the **same transaction** as
the ``compiled_mask`` rows it describes, so the two can never disagree: a frame
whose digest still matches needs no pixel work at all.  ``n_rows`` is the second
half of that promise -- it catches rows removed behind the service's back -- and
``compiler_version`` stops one build trusting digests another build wrote.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey

__all__ = ["DigestMixin"]


class DigestMixin:
    """Per-frame input digests, for a repository holding ``self.conn``."""

    def set_frame_digest(self, key: FrameKey, digest: str, compiler_version: str,
                         n_rows: int) -> None:
        """Record what the frame's compiled rows were derived from.

        Always called inside the transaction that writes those rows, so a
        digest can never describe a frame the database does not actually hold.
        """
        self._upsert(
            "frame_digest", self._fk(key),
            {"digest": str(digest), "compiler_version": str(compiler_version),
             "n_rows": int(n_rows)},
            desktop=key.desktop,
        )

    def frame_digest(self, key: FrameKey) -> Optional[dict]:
        """``{"digest", "compiler_version", "n_rows"}`` of one frame, or ``None``."""
        row = self.conn.execute(
            "SELECT digest, compiler_version, n_rows FROM frame_digest "
            "WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchone()
        return None if row is None else dict(row)

    def frame_digests(self, desktop: int, view: str) -> dict[int, dict]:
        """Every stored digest of one view, keyed by logical step."""
        rows = self.conn.execute(
            "SELECT step, digest, compiler_version, n_rows FROM frame_digest "
            "WHERE desktop=? AND view=?",
            (desktop, view),
        ).fetchall()
        return {int(r["step"]): dict(r) for r in rows}

    def clear_frame_digest(self, key: FrameKey) -> None:
        """Forget one frame's digest, so the next pass compiles it again."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM frame_digest WHERE desktop=? AND step=? AND view=?",
                (key.desktop, key.step, key.view),
            )
