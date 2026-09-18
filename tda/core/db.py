"""SQLite persistence layer for TDA: schema bootstrap, repository, backup, lock.

One :class:`Db` instance owns one connection (WAL, ``foreign_keys=ON``) to
``annotations/tda.sqlite``. The DDL lives next to this module in ``schema.sql``
and is replayed on every open, so re-opening an existing database is a no-op.
Row/dataclass mapping and SQL building live in :mod:`tda.core.dbrows`.

Conventions
-----------
* Every JSON column is TEXT holding ``json.dumps(obj, ensure_ascii=False)``;
  COCO RLE dicts are stored that way too.
* Geometry is stored in the coordinates the dataclasses define; this module
  only serialises, it never transforms.
* No Qt import belongs in this file (``tda.core`` must stay head-less).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from tda.core import dbrows as R
from tda.core.db_pose import PoseSegmentMixin
from tda.core.db_recheck import RecheckMixin
from tda.core.db_status import StatusMixin
from tda.core.dbconn import ConnectionMixin
from tda.core.dbdelete import DeleteMixin
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    OccluderMask,
    PairOverride,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    StateEvent,
    StepRec,
    ZOrderRec,
)

SCHEMA_VERSION = 2
LOCK_TTL = timedelta(hours=12)
#: How a conflict may be closed. The first three are a human's decision;
#: ``superseded`` is what the truth service records when the inputs moved on
#: before anybody got to the conflict (spec 3.4).
RESOLUTIONS = ("keep_old", "accept_new", "edited", "superseded")


class Db(ConnectionMixin, PoseSegmentMixin, StatusMixin, DeleteMixin, RecheckMixin):
    """Repository over the TDA SQLite file. Every write commits immediately --
    unless it runs inside :meth:`~tda.core.dbconn.ConnectionMixin.transaction`;
    :mod:`tda.core.db_pose` and :mod:`tda.core.db_status` mix in more readers,
    :mod:`tda.core.dbdelete` the undo-side row removals and
    :mod:`tda.core.db_recheck` the queue of frames awaiting a truth re-check."""

    def __init__(self, path: str):
        self.path = str(path)
        parent = Path(self.path).parent
        if str(parent):
            parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # the truth sweeper writes through a second connection to this same
        # file, so a writer must wait rather than fail (spec 3.5, WAL)
        self.conn.execute("PRAGMA busy_timeout=5000")
        self._lock_path = Path(self.path + ".lock")
        self._lock_annotator: Optional[str] = None
        self._tx_depth = 0
        try:
            self.init_schema(Path(__file__).with_name("schema.sql"), SCHEMA_VERSION)
        except Exception:
            self.conn.close()  # an unusable file leaves no connection behind
            raise

    # ------------------------------------------------------------------ setup

    def close(self) -> None:
        """Close the connection; safe to call more than once."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]

    def _ensure_desktop(self, desktop: int) -> None:
        """Create the parent desktop row so foreign keys never block a write."""
        self.conn.execute("INSERT OR IGNORE INTO desktop(id) VALUES(?)", (desktop,))

    def _upsert(self, table: str, keys: dict, data: dict, desktop: int | None = None) -> None:
        """Insert-or-update one row in its own transaction."""
        sql, params = R.upsert_sql(table, keys, data)
        with self._tx():
            if desktop is not None:
                self._ensure_desktop(desktop)
            self.conn.execute(sql, params)

    def _insert(self, table: str, data: dict, desktop: int | None = None) -> int:
        """Insert one row in its own transaction; returns the new rowid."""
        sql, params = R.insert_sql(table, data)
        with self._tx():
            if desktop is not None:
                self._ensure_desktop(desktop)
            cur = self.conn.execute(sql, params)
        return int(cur.lastrowid)

    # ------------------------------------------------------- desktops / frames

    def upsert_desktop(self, desktop: int, meta: dict) -> None:
        """Insert or update one desktop; keys outside the columns go to meta_json.

        The write is a full replacement: a known column missing from ``meta`` is
        set back to NULL, and the meta_json overflow is rewritten wholesale.
        """
        rest = dict(meta or {})
        data: dict[str, Any] = {c: rest.pop(c, None) for c in R.DESKTOP_COLUMNS}
        data["meta_json"] = R.dumps(rest) if rest else None
        self._upsert("desktop", {"id": desktop}, data)

    def get_desktop(self, desktop: int) -> Optional[dict]:
        """Desktop meta as a dict (columns + meta_json extras), or None.

        Deliberately asymmetric with :meth:`upsert_desktop`: columns that are
        NULL are left out instead of coming back as ``None``, and the returned
        dict always carries ``id``. A key stored as ``None`` therefore does not
        reappear; overflow keys do, unwrapped from meta_json.
        """
        row = self.conn.execute("SELECT * FROM desktop WHERE id=?", (desktop,)).fetchone()
        if row is None:
            return None
        out: dict[str, Any] = {"id": row["id"]}
        out.update({c: row[c] for c in R.DESKTOP_COLUMNS if row[c] is not None})
        out.update(R.loads(row["meta_json"]) or {})
        return out

    def upsert_frame(
        self, key: FrameKey, path: str, aux: dict, ts: str | None, flags: dict | None = None
    ) -> None:
        """Insert or update one image row; ``flags`` may carry any frame flag."""
        data: dict[str, Any] = {"path": path, "aux_json": R.dumps(aux or {}), "ts": ts}
        data.update(R.flag_data(flags or {}))
        self._upsert("frame", self._fk(key), data, desktop=key.desktop)

    def set_frame_flags(self, key: FrameKey, **flags) -> None:
        """Update the given frame flags only, leaving path/aux/ts and others intact."""
        if flags:
            self._upsert("frame", self._fk(key), R.flag_data(flags), desktop=key.desktop)

    @staticmethod
    def _fk(key: FrameKey) -> dict:
        """Primary-key columns of a frame-scoped table."""
        return {"desktop": key.desktop, "step": key.step, "view": key.view}

    def get_frame(self, key: FrameKey) -> dict | None:
        """One frame row as a dict with decoded aux/burst, or None."""
        row = self.conn.execute(
            "SELECT * FROM frame WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchone()
        return None if row is None else R.frame_row(row)

    def frames_for(self, desktop: int, view: str) -> list[dict]:
        """All frames of one view, ordered by logical step."""
        rows = self.conn.execute(
            "SELECT * FROM frame WHERE desktop=? AND view=? ORDER BY step", (desktop, view)
        ).fetchall()
        return [R.frame_row(r) for r in rows]

    # -------------------------------------------- steps / actions / instances

    def replace_steps(self, desktop: int, steps: list[StepRec], actions: list[ActionRec]) -> None:
        """Replace the whole step table (and its actions) of one desktop atomically."""
        with self._tx():
            self._ensure_desktop(desktop)
            self.conn.execute("DELETE FROM action WHERE desktop=?", (desktop,))
            self.conn.execute("DELETE FROM step WHERE desktop=?", (desktop,))
            self.conn.executemany(
                "INSERT INTO step(desktop, step, step_type, raw_name, dupli, notes, duration_s) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                [(s.desktop, s.step, s.step_type, s.raw_name, int(s.dupli), s.notes, s.duration_s)
                 for s in steps],
            )
            self.conn.executemany(
                "INSERT INTO action(desktop, step, idx, target, verb, tool, direction, result, "
                "failure_reason, difficulty) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(a.desktop, a.step, a.idx, a.target, a.verb, a.tool, a.direction, a.result,
                  a.failure_reason, a.difficulty) for a in actions],
            )

    def steps(self, desktop: int) -> list[StepRec]:
        """Steps of one desktop ordered by logical step."""
        rows = self.conn.execute(
            "SELECT * FROM step WHERE desktop=? ORDER BY step", (desktop,)
        ).fetchall()
        return [R.step_row(r) for r in rows]

    def actions(self, desktop: int, step: int | None = None) -> list[ActionRec]:
        """Actions of one desktop (optionally one step), ordered by (step, idx)."""
        sql = "SELECT * FROM action WHERE desktop=?"
        args: list[Any] = [desktop]
        if step is not None:
            sql += " AND step=?"
            args.append(step)
        return [R.action_row(r) for r in self.conn.execute(sql + " ORDER BY step, idx", args)]

    def upsert_instance(self, inst: InstanceRec) -> None:
        """Insert or update one instance identity row."""
        self._upsert("instance", {"desktop": inst.desktop, "key": inst.key},
                     R.instance_data(inst), desktop=inst.desktop)

    def delete_instance(self, desktop: int, key: str) -> None:
        """Drop one instance identity row; unknown keys are a no-op.

        Only the identity row goes: the caller decides what to do with the
        geometry, events and relations that may still name the key (stage S1
        refuses the deletion outright while any of them do, see
        :meth:`instance_reference_counts`).
        """
        with self._tx():
            self.conn.execute(
                'DELETE FROM instance WHERE desktop=? AND "key"=?', (desktop, key)
            )

    def _count(self, sql: str, args: tuple) -> int:
        return int(self.conn.execute(sql, args).fetchone()[0])

    def instance_reference_counts(self, desktop: int, key: str) -> dict[str, int]:
        """Rows still naming one instance, per table; empty when nothing does.

        Covers the tables that have no per-instance query of their own, so a
        caller about to delete an identity row can refuse in one call. Tables
        left out on purpose: ``shape_keyframe`` and ``relation``, which
        :meth:`keyframes` and :meth:`relations` already answer for a single
        instance and with more to say (which view, which edge type), and
        ``action``, which the step table owns. Only *manual* state events count
        -- the automatic ones are derived and go with
        :meth:`delete_auto_events`.
        """
        counts = {
            "frame_override": self._count(
                "SELECT COUNT(*) FROM frame_override WHERE desktop=? AND instance=?",
                (desktop, key)),
            "pair_override": self._count(
                "SELECT COUNT(*) FROM pair_override WHERE desktop=? AND (above=? OR below=?)",
                (desktop, key, key)),
            "compiled_mask": self._count(
                "SELECT COUNT(*) FROM compiled_mask WHERE desktop=? AND instance=?",
                (desktop, key)),
            "conflict": self._count(
                "SELECT COUNT(*) FROM conflict WHERE desktop=? AND instance=? AND status='open'",
                (desktop, key)),
            "state_event": self._count(
                "SELECT COUNT(*) FROM state_event WHERE desktop=? AND target=? AND auto=0",
                (desktop, key)),
        }
        return {table: n for table, n in counts.items() if n}

    def delete_auto_events(self, desktop: int, target: str) -> None:
        """Drop the derived state events of one target; hand-written ones stay."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM state_event WHERE desktop=? AND target=? AND auto=1",
                (desktop, target),
            )

    def instances(self, desktop: int) -> dict[str, InstanceRec]:
        """All instances of one desktop keyed by ``instance_key``."""
        rows = self.conn.execute(
            'SELECT * FROM instance WHERE desktop=? ORDER BY "key"', (desktop,)
        ).fetchall()
        return {r["key"]: R.instance_row(r) for r in rows}

    def replace_events(self, desktop: int, events: list[StateEvent], auto_only=True) -> None:
        """Replace state events; with ``auto_only`` the hand-written ones survive."""
        with self._tx():
            self._ensure_desktop(desktop)
            sql = "DELETE FROM state_event WHERE desktop=?"
            self.conn.execute(sql + (" AND auto=1" if auto_only else ""), (desktop,))
            self.conn.executemany(
                'INSERT INTO state_event(desktop, step, target, attr, "old", "new", '
                "evidence_view, auto) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                [(e.desktop, e.step, e.target, e.attr, e.old, e.new, e.evidence_view, int(e.auto))
                 for e in events],
            )

    def events(self, desktop: int) -> list[StateEvent]:
        """State events of one desktop ordered by (step, insertion order)."""
        rows = self.conn.execute(
            "SELECT * FROM state_event WHERE desktop=? ORDER BY step, id", (desktop,)
        ).fetchall()
        return [R.event_row(r) for r in rows]

    # --------------------------------------------------------------- geometry

    def _write_parts(self, keyframe_id: int, parts: Iterable[ShapePart]) -> None:
        """Replace the part rows of one keyframe (caller holds the transaction)."""
        self.conn.execute("DELETE FROM shape_part WHERE keyframe_id=?", (keyframe_id,))
        self.conn.executemany(
            "INSERT INTO shape_part(keyframe_id, idx, name, rle_json, box_json) "
            "VALUES(?, ?, ?, ?, ?)",
            [(keyframe_id, i, p.name, R.dumps(p.rle), R.dumps(list(p.box) if p.box else None))
             for i, p in enumerate(parts)],
        )

    def add_keyframe(self, kf: ShapeKeyframe, keep_id: int | None = None) -> int:
        """Insert a new shape keyframe with its parts; returns (and sets) its id.

        ``keep_id`` re-inserts a row under the id it had before: redoing an
        operation that an undo removed must bring the *same* keyframe back, or
        every other operation naming it would find nothing and insert a
        duplicate of its own (spec 4.6).
        """
        data = R.keyframe_data(kf)
        if keep_id is not None:
            data = {"id": int(keep_id)} | data
        sql, params = R.insert_sql("shape_keyframe", data)
        with self._tx():
            self._ensure_desktop(kf.desktop)
            kf.id = int(self.conn.execute(sql, params).lastrowid)
            self._write_parts(kf.id, kf.parts)
        return kf.id

    def update_keyframe(self, kf: ShapeKeyframe) -> None:
        """Overwrite a keyframe and its parts, bumping the stored version by one."""
        if kf.id is None:
            raise ValueError("update_keyframe needs kf.id; use add_keyframe for new shapes")
        with self._tx():
            row = self.conn.execute(
                "SELECT version FROM shape_keyframe WHERE id=?", (kf.id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no shape_keyframe with id={kf.id}")
            kf.version = int(row["version"]) + 1
            data = R.keyframe_data(kf)
            assignments = ", ".join(f'"{c}"=?' for c in data)
            self.conn.execute(f"UPDATE shape_keyframe SET {assignments} WHERE id=?",
                              (*data.values(), kf.id))
            self._write_parts(kf.id, kf.parts)

    def keyframes(self, desktop: int, view: str, instance: str | None = None) -> list[ShapeKeyframe]:
        """Keyframes of one view, ordered by (instance, pose segment, anchor step)."""
        sql = "SELECT * FROM shape_keyframe WHERE desktop=? AND view=?"
        args: list[Any] = [desktop, view]
        if instance is not None:
            sql += " AND instance=?"
            args.append(instance)
        rows = self.conn.execute(
            sql + " ORDER BY instance, pose_segment, anchor_step, id", args
        ).fetchall()
        out = []
        for r in rows:
            parts = self.conn.execute(
                "SELECT * FROM shape_part WHERE keyframe_id=? ORDER BY idx", (r["id"],)
            ).fetchall()
            out.append(R.keyframe_row(r, parts))
        return out

    def set_zorder(self, z: ZOrderRec) -> None:
        """Store the (instance, part) total order of one (view, pose segment)."""
        self._upsert(
            "zorder", {"desktop": z.desktop, "view": z.view, "pose_segment": z.pose_segment},
            {"order_json": R.dumps([list(p) for p in z.order]), "version": z.version},
            desktop=z.desktop,
        )

    def zorder(self, desktop: int, view: str, pose_segment: int) -> ZOrderRec:
        """Stored order, or an empty version-1 record when none was saved yet."""
        row = self.conn.execute(
            "SELECT * FROM zorder WHERE desktop=? AND view=? AND pose_segment=?",
            (desktop, view, pose_segment),
        ).fetchone()
        if row is None:
            return ZOrderRec(desktop, view, pose_segment, [])
        order = [tuple(p) for p in (R.loads(row["order_json"]) or [])]
        return ZOrderRec(desktop, view, pose_segment, order, version=row["version"])

    def set_pair_override(self, po: PairOverride) -> None:
        """Record "above beats below" for one pair; repeated calls are no-ops."""
        with self._tx():
            self._ensure_desktop(po.desktop)
            self.conn.execute(
                "INSERT OR IGNORE INTO pair_override(desktop, view, pose_segment, above, below) "
                "VALUES(?, ?, ?, ?, ?)",
                (po.desktop, po.view, po.pose_segment, po.above, po.below),
            )

    def pair_overrides(self, desktop: int, view: str, pose_segment: int) -> list[PairOverride]:
        """Pairwise layering exceptions of one (view, pose segment)."""
        rows = self.conn.execute(
            "SELECT * FROM pair_override WHERE desktop=? AND view=? AND pose_segment=? "
            "ORDER BY above, below", (desktop, view, pose_segment),
        ).fetchall()
        return [PairOverride(r["desktop"], r["view"], r["pose_segment"], r["above"], r["below"])
                for r in rows]

    def set_occluder(self, om: OccluderMask) -> None:
        """Store (replacing) the occluder layer of one type on one frame."""
        keys = self._fk(om.frame) | {"occluder_type": om.occluder_type}
        self._upsert("occluder_mask", keys, {"rle_json": R.dumps(om.rle)},
                     desktop=om.frame.desktop)

    def occluders(self, key: FrameKey) -> list[OccluderMask]:
        """Occluder layers of one frame, ordered by type."""
        rows = self.conn.execute(
            "SELECT * FROM occluder_mask WHERE desktop=? AND step=? AND view=? "
            "ORDER BY occluder_type", (key.desktop, key.step, key.view),
        ).fetchall()
        return [OccluderMask(key, r["occluder_type"], R.loads(r["rle_json"])) for r in rows]

    def set_frame_override(self, fo: FrameOverride) -> None:
        """Store a frame-local mask/visibility override for one instance."""
        self._upsert(
            "frame_override", self._fk(fo.frame) | {"instance": fo.instance},
            {"visible_rle_json": R.dumps(fo.visible_rle), "visibility": fo.visibility},
            desktop=fo.frame.desktop,
        )

    def frame_overrides(self, key: FrameKey) -> dict[str, FrameOverride]:
        """Frame-local overrides keyed by instance."""
        rows = self.conn.execute(
            "SELECT * FROM frame_override WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchall()
        return {r["instance"]: FrameOverride(key, r["instance"],
                                             R.loads(r["visible_rle_json"]), r["visibility"])
                for r in rows}

    def set_pose_segment(self, desktop: int, view: str, seg: int, start: int, end: int,
                         ref_step: int, corners: list | None, homography: list | None) -> None:
        """Define one pose segment (step range, reference frame, chassis corners)."""
        self._upsert(
            "pose_segment", {"desktop": desktop, "view": view, "seg": seg},
            {"start_step": start, "end_step": end, "ref_step": ref_step,
             "corners_json": R.dumps(corners), "homography_json": R.dumps(homography)},
            desktop=desktop,
        )

    def pose_segment_for(self, key: FrameKey) -> dict:
        """Segment of a frame: its own ``pose_segment`` wins, else the step range.

        Falls back to a bare segment 0 descriptor when nothing is recorded yet.
        """
        frame = self.conn.execute(
            "SELECT pose_segment FROM frame WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchone()
        row = None
        if frame is not None and frame["pose_segment"] is not None:
            row = self.conn.execute(
                "SELECT * FROM pose_segment WHERE desktop=? AND view=? AND seg=?",
                (key.desktop, key.view, frame["pose_segment"]),
            ).fetchone()
        if row is None:
            row = self.conn.execute(
                "SELECT * FROM pose_segment WHERE desktop=? AND view=? AND start_step<=? "
                "AND end_step>=? ORDER BY seg LIMIT 1",
                (key.desktop, key.view, key.step, key.step),
            ).fetchone()
        if row is None:
            return {"desktop": key.desktop, "view": key.view, "seg": 0, "start_step": None,
                    "end_step": None, "ref_step": None, "corners": None, "homography": None,
                    "roi": None}
        return R.pose_row(row)

    def set_transform(self, key: FrameKey, sim: Similarity) -> None:
        """Store the frame's similarity transform to its pose-segment reference."""
        self._upsert("frame_transform", self._fk(key),
                     {"scale": sim.scale, "theta": sim.theta, "tx": sim.tx, "ty": sim.ty},
                     desktop=key.desktop)

    def transform(self, key: FrameKey) -> Similarity:
        """Stored transform, or the identity when the frame has none."""
        row = self.conn.execute(
            "SELECT * FROM frame_transform WHERE desktop=? AND step=? AND view=?",
            (key.desktop, key.step, key.view),
        ).fetchone()
        return Similarity() if row is None else Similarity(row["scale"], row["theta"],
                                                           row["tx"], row["ty"])

    # ------------------------------------------------------------------ truth

    def put_compiled(self, key: FrameKey, instance: str, visible_rle: dict | None,
                     occlusion_ratio: float, visibility: str, placement: str, status: str,
                     input_hash: str, verified_by: str | None = None,
                     geom_type: str = "mask", box: tuple | list | None = None) -> None:
        """Insert or refresh one compiled-truth row of (desktop, view, step, instance).

        ``geom_type`` says which column carries the geometry: ``"mask"`` rows
        store ``visible_rle``, ``"box"`` rows (a bench part tracked by a
        rectangle) store ``box`` as ``[x0, y0, x1, y1]``.
        """
        verified_at = R.now_iso() if (verified_by is not None or status == "verified") else None
        self._upsert(
            "compiled_mask", self._fk(key) | {"instance": instance},
            {"visible_rle_json": R.dumps(visible_rle), "occlusion_ratio": occlusion_ratio,
             "visibility": visibility, "placement": placement, "status": status,
             "input_hash": input_hash, "verified_by": verified_by, "verified_at": verified_at,
             "geom_type": geom_type,
             "box_json": R.dumps(None if box is None else [float(v) for v in box])},
            desktop=key.desktop,
        )

    def compiled(self, key: FrameKey) -> dict[str, dict]:
        """Compiled truth of one frame keyed by instance (``visible_rle`` and ``box`` decoded)."""
        rows = self.conn.execute(
            "SELECT * FROM compiled_mask WHERE desktop=? AND step=? AND view=? ORDER BY instance",
            (key.desktop, key.step, key.view),
        ).fetchall()
        return {r["instance"]: R.json_row(r, "visible_rle", "box") for r in rows}

    def delete_compiled(self, key: FrameKey, instance: str) -> None:
        """Drop one compiled-truth row; a row that is not there is not an error."""
        with self._tx():
            self.conn.execute(
                "DELETE FROM compiled_mask WHERE desktop=? AND step=? AND view=? AND instance=?",
                (key.desktop, key.step, key.view, instance),
            )

    def add_conflict(self, key: FrameKey, instance: str, old_rle: dict | None,
                     new_rle: dict | None, sym_diff_px: int) -> int:
        """Queue a frozen-vs-recompiled disagreement; returns the conflict id."""
        data = self._fk(key) | {
            "instance": instance, "old_rle_json": R.dumps(old_rle),
            "new_rle_json": R.dumps(new_rle), "sym_diff_px": int(sym_diff_px),
            "status": "open", "created_at": R.now_iso(),
        }
        return self._insert("conflict", data, desktop=key.desktop)

    def conflicts(self, desktop: int, view: str | None = None, open_only=True) -> list[dict]:
        """Conflict queue of one desktop, oldest first."""
        sql = "SELECT * FROM conflict WHERE desktop=?"
        args: list[Any] = [desktop]
        if view is not None:
            sql += " AND view=?"
            args.append(view)
        if open_only:
            sql += " AND status='open'"
        rows = self.conn.execute(sql + " ORDER BY id", args).fetchall()
        return [R.json_row(r, "old_rle", "new_rle") for r in rows]

    def get_conflict(self, cid: int) -> Optional[dict]:
        """One conflict by id, or ``None``."""
        row = self.conn.execute("SELECT * FROM conflict WHERE id=?", (cid,)).fetchone()
        return None if row is None else R.json_row(row, "old_rle", "new_rle")

    def resolve_conflict(self, cid: int, resolution: str) -> None:
        """Close a conflict with keep_old / accept_new / edited."""
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution must be one of {RESOLUTIONS}, got {resolution!r}")
        with self._tx():
            self.conn.execute(
                "UPDATE conflict SET status='resolved', resolution=?, resolved_at=? WHERE id=?",
                (resolution, R.now_iso(), cid),
            )

    # -------------------------------------------------------------- relations

    def add_relation(self, desktop: int, rel_type: str, target: str, blocker: str,
                     necessity: str = "required", mode: str | None = None,
                     reason: str | None = None, source: str = "manual",
                     evidence_step: int | None = None, status: str = "active") -> int:
        """Insert or update one hard-constraint edge; returns its id.

        ``target`` is the node being acted on and ``blocker`` the node that must
        change state first; ``source`` is the edge's provenance (manual /
        template / derived).
        """
        keys = {"desktop": desktop, "type": rel_type, "target": target, "blocker": blocker}
        self._upsert("relation", keys,
                     {"necessity": necessity, "mode": mode, "reason": reason, "source": source,
                      "evidence_step": evidence_step, "status": status},
                     desktop=desktop)
        row = self.conn.execute(
            'SELECT id FROM relation WHERE desktop=? AND "type"=? AND target=? AND blocker=?',
            (desktop, rel_type, target, blocker),
        ).fetchone()
        return int(row["id"])

    def relations(self, desktop: int) -> list[dict]:
        """All constraint edges of one desktop."""
        rows = self.conn.execute(
            "SELECT * FROM relation WHERE desktop=? ORDER BY id", (desktop,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------------- op log

    def log_op(self, desktop: int, view: str, kind: str, payload: dict, inverse: dict,
               annotator: str) -> int:
        """Append one source-level operation with its inverse patch; returns its id."""
        data = {"desktop": desktop, "view": view, "kind": kind, "payload_json": R.dumps(payload),
                "inverse_json": R.dumps(inverse), "annotator": annotator, "ts": R.now_iso()}
        return self._insert("op_log", data, desktop=desktop)

    def ops(self, desktop: int, view: str, limit: int = 100) -> list[dict]:
        """Most recent operations of one (desktop, view), newest first."""
        rows = self.conn.execute(
            "SELECT * FROM op_log WHERE desktop=? AND view=? ORDER BY id DESC LIMIT ?",
            (desktop, view, limit),
        ).fetchall()
        return [R.json_row(r, "payload", "inverse") for r in rows]

    # ------------------------------------------------------------ backup/lock

    def backup(self, dest_dir: str) -> str:
        """Copy the live database with the SQLite backup API; returns the new path."""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = dest / f"tda_{stamp}.sqlite"
        serial = 1
        while out.exists():
            out = dest / f"tda_{stamp}_{serial}.sqlite"
            serial += 1
        target = sqlite3.connect(str(out))
        try:
            self.conn.backup(target)
        finally:
            target.close()
        return str(out)

    def _read_lock(self) -> Optional[dict]:
        """Parse the lock file, or None when it is missing, unreadable or not an object.

        A lock whose content is not a JSON object carries no annotator, so it is
        treated like a stale one and the next ``acquire_lock`` takes it over.
        """
        try:
            held = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return held if isinstance(held, dict) else None

    def acquire_lock(self, annotator: str) -> None:
        """Take the single-user lock; raises if another annotator holds a fresh one."""
        held = self._read_lock()
        if held and held.get("annotator") != annotator:
            try:
                ts = datetime.fromisoformat(str(held.get("ts")))
            except ValueError:
                ts = None  # unreadable timestamp: treat the lock as stale
            if ts is not None:
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - ts < LOCK_TTL:
                    raise RuntimeError(
                        f"database locked by {held.get('annotator')!r} since {held.get('ts')}"
                    )
        self._lock_path.write_text(
            json.dumps({"annotator": annotator, "ts": R.now_iso()}, ensure_ascii=False),
            encoding="utf-8",
        )
        self._lock_annotator = annotator

    def release_lock(self) -> None:
        """Remove the lock file if present; safe to call more than once."""
        self._lock_path.unlink(missing_ok=True)
        self._lock_annotator = None
