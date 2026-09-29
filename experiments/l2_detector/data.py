"""The drafts as detection data: every ``ls:`` keyframe of every desktop.

Source: ``shape_keyframe`` rows with ``source='labelstudio'`` (instance key
``ls:<Label>#<n>``), geometry in ``shape_part.rle_json`` (COCO RLE at the
view's native resolution), taxonomy class from the ``instance`` table -- the
same join ``experiments/plan_b_probe/transfer/common.py`` and L1's
``siblings.py`` use, so ``Motherboard Screw`` / ``Heatsink Screw`` / ... ->
``screw``, ``RAM Module Retention Clip (open|closed)`` -> ``ram_latch`` etc.

Read-only: the database is ``D:/DataSet/.cache/tmp/l2/l2.sqlite`` (a copy of
``u2b3_copy.sqlite``) opened ``mode=ro``; frames are read from the raw drive
through :func:`tda.core.rawroot.resolve_raw` (the scanner frame is
``frame.path`` = ``P_0.png``, the image the drafts were drawn on).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.l2_detector import env  # noqa: E402,F401  (pins caches first)

DB = "D:/DataSet/.cache/tmp/l2/l2.sqlite"
DB_URI = f"file:{DB}?mode=ro"
GEOM = env.TMP / "geom" / "ls_boxes_all.json"

NATIVE_HW = {"scan": (1600, 1600), "oak1": (3040, 4032), "oak2": (3040, 4032),
             "rs": (720, 1280)}
VIEWS = ("scan", "oak1", "oak2", "rs")
EVENT_DESKTOPS = (13, 24, 33)

#: Detector classes.  The first six are the small parts the guess cannot find
#: (screws, connectors, latches and clips); ``ram_module`` and ``cpu`` are two
#: cheap context classes (they fit in a 640 tile at native resolution).
DET_CLASSES = ("screw", "connector", "ram_latch", "psu_latch",
               "cpu_socket_lever", "drive_latch", "ram_module", "cpu")
SMALL_CLASSES = DET_CLASSES[:6]
CLS_ID = {c: i for i, c in enumerate(DET_CLASSES)}


@dataclass
class Box:
    desktop: int
    view: str
    step: int
    instance: str
    cls: str
    label: str
    area: int                        # mask pixels (visible-only draft)
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive), native px

    @property
    def width(self) -> float:
        """sqrt(area) -- B2 / L1's "part width"."""
        return float(np.sqrt(max(1, self.area)))


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_URI, uri=True)
    con.row_factory = sqlite3.Row
    return con


def _label_of(instance: str) -> str:
    body = instance[3:] if instance.startswith("ls:") else instance
    return body.rpartition("#")[0] if "#" in body else body


def build_boxes(path: Path = GEOM) -> None:
    """Every draft's bbox + area, straight from the compressed RLE."""
    from pycocotools import mask as mask_utils

    con = connect()
    rows = con.execute(
        """SELECT k.desktop, k.view, k.anchor_step AS step, k.instance,
                  k.source, i.cls AS cls, p.rle_json
           FROM shape_keyframe k
           JOIN shape_part p ON p.keyframe_id = k.id
           LEFT JOIN instance i ON i.desktop = k.desktop AND i."key" = k.instance
           WHERE k.instance LIKE 'ls:%'
           ORDER BY k.desktop, k.view, k.anchor_step, k.instance""").fetchall()
    out = []
    for r in rows:
        rle = json.loads(r["rle_json"])
        rle_c = {"size": rle["size"], "counts": rle["counts"].encode("ascii")
                 if isinstance(rle["counts"], str) else rle["counts"]}
        area = int(mask_utils.area(rle_c))
        if area <= 0:
            continue
        x, y, w, h = (float(v) for v in mask_utils.toBbox(rle_c))
        out.append({
            "desktop": int(r["desktop"]), "view": r["view"], "step": int(r["step"]),
            "instance": r["instance"], "cls": r["cls"] or "",
            "label": _label_of(r["instance"]), "source": r["source"],
            "area": area,
            "box": [int(round(x)), int(round(y)), int(round(x + w)),
                    int(round(y + h))],
        })
    con.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"[l2] wrote {len(out)} draft boxes to {path}")


def load_boxes() -> list[Box]:
    if not GEOM.exists():
        build_boxes()
    raw = json.loads(GEOM.read_text(encoding="utf-8"))
    return [Box(desktop=d["desktop"], view=d["view"], step=d["step"],
                instance=d["instance"], cls=d["cls"], label=d["label"],
                area=d["area"], box=tuple(d["box"]))
            for d in raw if d.get("source", "labelstudio") == "labelstudio"]


def index(boxes: list[Box]) -> dict[tuple[int, str, int], list[Box]]:
    by: dict[tuple[int, str, int], list[Box]] = defaultdict(list)
    for b in boxes:
        by[(b.desktop, b.view, b.step)].append(b)
    return by


def roi_of(boxes: list[Box], desktop: int, view: str) -> tuple[int, int, int, int]:
    """Union box of the view's drafts, padded 10 % (L1 / B2's stand-in ROI)."""
    bs = [b.box for b in boxes if b.desktop == desktop and b.view == view]
    h, w = NATIVE_HW[view]
    x0 = min(b[0] for b in bs)
    y0 = min(b[1] for b in bs)
    x1 = max(b[2] for b in bs)
    y1 = max(b[3] for b in bs)
    px, py = int(0.10 * (x1 - x0)), int(0.10 * (y1 - y0))
    return (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
_RAW_READY = False


def _raw() -> None:
    global _RAW_READY
    if _RAW_READY:
        return
    import yaml

    from tda.core import rawroot

    paths = yaml.safe_load((REPO / "configs" / "paths.yaml").read_text("utf-8"))
    raw = rawroot.configure(paths)
    if not raw.connected:
        raise SystemExit(f"raw drive not found: {raw.status}")
    _RAW_READY = True


def frame_path(con, desktop: int, view: str, step: int) -> Optional[str]:
    """``frame.path`` resolved to today's raw drive letter (read-only)."""
    from tda.core.rawroot import resolve_raw

    _raw()
    r = con.execute("SELECT path FROM frame WHERE desktop=? AND view=? AND step=?",
                    (desktop, view, step)).fetchone()
    if r is None or not r["path"]:
        return None
    p = resolve_raw(r["path"])
    return p if p and os.path.exists(p) else None


def read_rgb(path: str) -> Optional[np.ndarray]:
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def md_table(df) -> str:
    """A pandas frame as a Markdown table (``tabulate`` is not installed)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and v != v)
                                       else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# label statistics
# --------------------------------------------------------------------------- #
def label_stats(boxes: list[Box]) -> dict:
    """Counts per class x view x desktop and width quantiles per class x view."""
    cnt = Counter((b.cls, b.view, b.desktop) for b in boxes)
    frames = defaultdict(set)
    for b in boxes:
        frames[(b.view, b.desktop)].add(b.step)
    widths = defaultdict(list)
    for b in boxes:
        widths[(b.cls, b.view)].append(b.width)
    return {"count": cnt, "frames": frames, "widths": widths}


def main() -> int:
    boxes = load_boxes()
    st = label_stats(boxes)
    desks = sorted({b.desktop for b in boxes})
    lines = ["# L2 label counts (ls: drafts, source=labelstudio)", ""]
    lines.append("## Frames with drafts per view x desktop")
    lines.append("")
    lines.append("| view | " + " | ".join(f"D{d}" for d in desks) + " |")
    lines.append("|---" * (len(desks) + 1) + "|")
    for v in VIEWS:
        lines.append(f"| {v} | " + " | ".join(
            str(len(st["frames"].get((v, d), ()))) or "" for d in desks) + " |")
    for v in VIEWS:
        lines += ["", f"## {v}: draft count per class x desktop", ""]
        dv = [d for d in desks if st["frames"].get((v, d))]
        lines.append("| class | " + " | ".join(f"D{d}" for d in dv)
                     + " | total | width p10/p50/p90 px |")
        lines.append("|---" * (len(dv) + 3) + "|")
        classes = sorted({b.cls for b in boxes if b.view == v})
        for c in classes:
            tot = sum(st["count"][(c, v, d)] for d in dv)
            w = np.asarray(st["widths"][(c, v)])
            q = (f"{np.percentile(w, 10):.0f} / {np.percentile(w, 50):.0f} / "
                 f"{np.percentile(w, 90):.0f}" if w.size else "-")
            mark = "**" if c in SMALL_CLASSES else ""
            lines.append(f"| {mark}{c}{mark} | " + " | ".join(
                str(st["count"][(c, v, d)] or "") for d in dv)
                + f" | {tot} | {q} |")
    out = env.OUT / "label_counts.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
