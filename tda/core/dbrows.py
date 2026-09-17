"""Row <-> dataclass mapping and SQL statement builders used by ``tda.core.db``.

Kept apart from the repository itself so ``db.py`` holds only query logic. No Qt
and no I/O here: everything is a pure function over ``sqlite3.Row`` objects.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from tda.core.model import (
    ActionRec,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StateEvent,
    StepRec,
)

#: Frame flag name -> (column, stored as 0/1). ``burst`` holds the scanner burst
#: metrics plus the chosen-frame reason.
FRAME_FLAGS: dict[str, tuple[str, bool]] = {
    "hand_or_tool_in_frame": ("hand_or_tool_in_frame", True),
    "in_progress": ("in_progress", True),
    "image_quality": ("image_quality", False),
    "missing": ("missing", True),
    "bench_annotated": ("bench_annotated", True),
    "review_status": ("review_status", False),
    "burst": ("burst_json", False),
    "pose_segment": ("pose_segment", False),
}
BOOL_FRAME_COLUMNS = ("hand_or_tool_in_frame", "in_progress", "missing", "bench_annotated")
DESKTOP_COLUMNS = (
    "brand", "model_family", "chassis_platform", "chassis_type",
    "size", "date", "split", "notes",
)


def dumps(obj: Any) -> Optional[str]:
    """JSON-encode ``obj`` for a TEXT column; ``None`` stays ``None``."""
    return None if obj is None else json.dumps(obj, ensure_ascii=False)


def loads(text: Optional[str]) -> Any:
    """Decode a JSON TEXT column; ``None``/empty stays ``None``."""
    return None if text is None or text == "" else json.loads(text)


def flag_int(value: Any) -> Optional[int]:
    """Encode a tri-state flag (``None`` means "not recorded")."""
    return None if value is None else int(bool(value))


def now_iso() -> str:
    """Current UTC timestamp, second resolution, ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------- SQL builders
# Table and column names always come from this package, never from user input.


def upsert_sql(table: str, keys: dict, data: dict) -> tuple[str, tuple]:
    """Build ``INSERT .. ON CONFLICT(keys) DO UPDATE SET data`` and its parameters."""
    cols = list(keys) + list(data)
    names = ", ".join(f'"{c}"' for c in cols)
    placeholders = ", ".join("?" * len(cols))
    conflict = ", ".join(f'"{c}"' for c in keys)
    sets = ", ".join(f'"{c}"=excluded."{c}"' for c in data)
    if not sets:  # nothing to update: keep the row as it is
        first = next(iter(keys))
        sets = f'"{first}"=excluded."{first}"'
    sql = (f"INSERT INTO {table}({names}) VALUES({placeholders}) "
           f"ON CONFLICT({conflict}) DO UPDATE SET {sets}")
    return sql, (*keys.values(), *data.values())


def insert_sql(table: str, data: dict) -> tuple[str, tuple]:
    """Build a plain ``INSERT`` and its parameters."""
    names = ", ".join(f'"{c}"' for c in data)
    placeholders = ", ".join("?" * len(data))
    return f"INSERT INTO {table}({names}) VALUES({placeholders})", tuple(data.values())


def flag_data(flags: dict) -> dict:
    """Translate frame flag kwargs into column values; unknown names raise."""
    out: dict[str, Any] = {}
    for name, value in flags.items():
        if name not in FRAME_FLAGS:
            raise ValueError(f"unknown frame flag: {name}")
        column, is_bool = FRAME_FLAGS[name]
        if is_bool:
            out[column] = flag_int(value)
        else:
            out[column] = dumps(value) if column.endswith("_json") else value
    return out


# -------------------------------------------------------------- row -> python


def json_row(row: sqlite3.Row, *names: str) -> dict:
    """Row as a dict with ``<name>_json`` columns decoded into ``<name>``."""
    out = dict(row)
    for name in names:
        out[name] = loads(out.pop(f"{name}_json"))
    return out


def frame_row(row: sqlite3.Row) -> dict:
    """Frame row with decoded aux/burst and tri-state flags as bools."""
    out = json_row(row, "aux", "burst")
    out["aux"] = out["aux"] or {}
    for column in BOOL_FRAME_COLUMNS:
        out[column] = None if out[column] is None else bool(out[column])
    return out


def pose_row(row: sqlite3.Row) -> dict:
    """Pose-segment row with decoded corners/homography/roi."""
    return json_row(row, "corners", "homography", "roi")


def step_row(row: sqlite3.Row) -> StepRec:
    return StepRec(row["desktop"], row["step"], row["step_type"], row["raw_name"],
                   dupli=bool(row["dupli"]), notes=row["notes"] or "",
                   duration_s=row["duration_s"])


def action_row(row: sqlite3.Row) -> ActionRec:
    return ActionRec(row["desktop"], row["step"], row["idx"], row["target"], row["verb"],
                     tool=row["tool"], direction=row["direction"], result=row["result"],
                     failure_reason=row["failure_reason"], difficulty=row["difficulty"])


def event_row(row: sqlite3.Row) -> StateEvent:
    return StateEvent(row["desktop"], row["step"], row["target"], row["attr"], row["old"],
                      row["new"], evidence_view=row["evidence_view"], auto=bool(row["auto"]))


def instance_row(row: sqlite3.Row) -> InstanceRec:
    return InstanceRec(
        key=row["key"], desktop=row["desktop"], cls=row["cls"],
        attrs=loads(row["attrs_json"]) or {}, parent=row["parent"],
        attached=bool(row["attached"]), mounted_on=row["mounted_on"], fastens=row["fastens"],
        socket_host=row["socket_host"], cable=row["cable"], slot_id=row["slot_id"],
        group_id=row["group_id"], group_order=row["group_order"],
        removal_direction=row["removal_direction"], raw_names=loads(row["raw_names_json"]) or [],
    )


def part_row(row: sqlite3.Row) -> ShapePart:
    box = loads(row["box_json"])
    return ShapePart(row["name"], rle=loads(row["rle_json"]), box=tuple(box) if box else None)


def keyframe_row(row: sqlite3.Row, parts: Iterable[sqlite3.Row]) -> ShapeKeyframe:
    return ShapeKeyframe(
        id=row["id"], instance=row["instance"], desktop=row["desktop"], view=row["view"],
        pose_segment=row["pose_segment"], anchor_step=row["anchor_step"],
        placement=row["placement"], geom_type=row["geom_type"],
        parts=[part_row(p) for p in parts], amodal_complete=bool(row["amodal_complete"]),
        source=row["source"], draft_id=row["draft_id"], version=row["version"],
        edit_count=row["edit_count"], edit_time_ms=row["edit_time_ms"],
    )


def instance_data(inst: InstanceRec) -> dict:
    """Column values of an instance row (without its primary key)."""
    return {
        "cls": inst.cls, "attrs_json": dumps(inst.attrs or {}), "parent": inst.parent,
        "attached": int(inst.attached), "mounted_on": inst.mounted_on, "fastens": inst.fastens,
        "socket_host": inst.socket_host, "cable": inst.cable, "slot_id": inst.slot_id,
        "group_id": inst.group_id, "group_order": inst.group_order,
        "removal_direction": inst.removal_direction,
        "raw_names_json": dumps(list(inst.raw_names or [])),
    }


def keyframe_data(kf: ShapeKeyframe) -> dict:
    """Column values of a shape_keyframe row (without its id)."""
    return {
        "desktop": kf.desktop, "view": kf.view, "instance": kf.instance,
        "pose_segment": kf.pose_segment, "anchor_step": kf.anchor_step,
        "placement": kf.placement, "geom_type": kf.geom_type,
        "amodal_complete": int(kf.amodal_complete), "source": kf.source,
        "draft_id": kf.draft_id, "version": kf.version, "edit_count": kf.edit_count,
        "edit_time_ms": kf.edit_time_ms,
    }
