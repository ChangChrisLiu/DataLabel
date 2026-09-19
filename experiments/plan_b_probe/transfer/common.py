"""Shared plumbing for the Plan B cross-view transfer probe (read-only).

Everything here reads the *experiment copy* of the annotation database
(``D:/DataSet/.cache/tmp/exp_copy.sqlite``) through a ``file:...?mode=ro`` URI
and the raw frames on ``F:`` -- nothing in this package ever writes to either.

What the probe needs and where it comes from
--------------------------------------------
* The provisional Label Studio instances are ``shape_keyframe`` rows whose
  ``instance`` starts with ``ls:``, one row per (view, anchor_step, label
  ordinal), with the geometry in ``shape_part.rle_json``.
* ``tda.core.ls_import`` rasterises those polygons into the view's **native**
  resolution (:data:`tda.core.ls_export.NATIVE_HW`) and *skips* the
  depth-aligned 1280x800 OAK uploads outright, so every stored OAK mask is in
  4032x3040 coordinates and belongs on ``frame.path`` (the 12 MP still), not on
  ``aux_json["aligned"]``. ``m0_overlays.py`` proves that on real frames.
* The ``#<ordinal>`` of a label is **not** a physical identity: the importer
  numbers a frame's shapes by ascending bbox ``min x`` *within that frame*
  (``tda.core.ls_export.geometry_of`` sorts on ``_pct_anchor``), so the same
  screw carries different ordinals in two views -- and even in two steps of one
  view once a neighbour has been removed.

Geometry cache
--------------
Decoding ~7 900 RLEs, a third of them 12 MP, costs minutes, so
:func:`load_geometry` builds a JSON summary (area, bbox, centroid, contour
polygons) once under ``D:/DataSet/.cache/tmp/e2`` and reads it back after that.
The full masks are still decoded on demand by :func:`frame_masks`, which does it
once per (desktop, view, step) so a whole frame's hit tests share the work.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from tda.core.masks import decode_rle

DB_URI = "file:D:/DataSet/.cache/tmp/exp_copy.sqlite?mode=ro"
TMP = Path("D:/DataSet/.cache/tmp/e2")
OUT = Path("D:/DataSet/experiments_out/plan_b_probe/transfer")
HERE = Path(__file__).resolve().parent

#: Native ``(h, w)`` per view -- the resolution the LS drafts were rasterised at.
NATIVE_HW = {"scan": (1600, 1600), "oak1": (3040, 4032), "oak2": (3040, 4032),
             "rs": (720, 1280)}
DESKTOPS = (13, 24, 33)
VIEWS = ("scan", "oak1", "oak2", "rs")
TARGETS = ("oak1", "oak2", "rs")

#: Which LS class group a part belongs to, for the M2 plane comparison.
#: ``flat`` sits on the board, ``tall`` stands well above it, ``rim`` lives on
#: the chassis opening itself.
CLASS_GROUP = {
    "ram_module": "flat", "cpu": "flat", "connector": "flat", "screw": "flat",
    "ram_latch": "flat", "cpu_socket_lever": "flat", "motherboard": "flat",
    "cpu_cooler": "tall", "heatsink": "tall", "psu": "tall",
    "drive_cage": "tall", "storage_drive": "tall", "optical_drive": "tall",
    "expansion_card": "tall", "gpu": "tall", "cooler_bracket": "tall",
    "cover": "rim", "psu_latch": "rim", "drive_latch": "rim",
}


def connect() -> sqlite3.Connection:
    """Read-only handle on the experiment copy of the database."""
    con = sqlite3.connect(DB_URI, uri=True)
    con.row_factory = sqlite3.Row
    return con


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass
class Shape:
    """One LS provisional instance on one frame, summarised."""

    desktop: int
    view: str
    step: int
    instance: str          # "ls:<Label>#<ordinal>"
    cls: str               # taxonomy class from the instance table
    label: str             # <Label> without the ordinal
    ordinal: int
    area: int
    box: tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)
    centroid: tuple[float, float]
    polys: list[list[float]]         # contours, flat [x0,y0,x1,y1,...]

    @property
    def group(self) -> str:
        return CLASS_GROUP.get(self.cls, "other")

    @property
    def size(self) -> float:
        """sqrt(area) -- the length scale distances are normalised by."""
        return float(np.sqrt(max(1, self.area)))


def _split_key(instance: str) -> tuple[str, int]:
    body = instance[3:] if instance.startswith("ls:") else instance
    if "#" in body:
        label, _, num = body.rpartition("#")
        try:
            return label, int(num)
        except ValueError:
            return body, 1
    return body, 1


def _raw_rows(con: sqlite3.Connection, desktops: Iterable[int]) -> list[sqlite3.Row]:
    marks = ",".join("?" * len(tuple(desktops)))
    return con.execute(
        f"""SELECT k.desktop, k.view, k.anchor_step AS step, k.instance,
                   i.cls AS cls, p.rle_json
            FROM shape_keyframe k
            JOIN shape_part p ON p.keyframe_id = k.id
            LEFT JOIN instance i ON i.desktop = k.desktop AND i."key" = k.instance
            WHERE k.instance LIKE 'ls:%' AND k.desktop IN ({marks})
            ORDER BY k.desktop, k.view, k.anchor_step, k.instance""",
        tuple(desktops),
    ).fetchall()


def build_geometry_cache(path: Path) -> None:
    """Decode every LS mask once and write the per-shape summary as JSON."""
    con = connect()
    rows = _raw_rows(con, DESKTOPS)
    out = []
    for n, r in enumerate(rows):
        rle = json.loads(r["rle_json"])
        mask = decode_rle(rle)
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            continue
        label, ordinal = _split_key(r["instance"])
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = []
        for c in contours:
            approx = cv2.approxPolyDP(c, 2.0, True)
            if len(approx) >= 3:
                polys.append([float(v) for v in approx.reshape(-1)])
        out.append({
            "desktop": int(r["desktop"]), "view": r["view"], "step": int(r["step"]),
            "instance": r["instance"], "cls": r["cls"] or "", "label": label,
            "ordinal": ordinal, "area": int(xs.size),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "polys": polys,
        })
        if n % 500 == 0:
            print(f"  ... {n}/{len(rows)}", flush=True)
    con.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {len(out)} shapes to {path}")


def load_geometry() -> list[Shape]:
    """Every LS shape of the three probe desktops, building the cache if needed."""
    path = TMP / "ls_geometry.json"
    if not path.exists():
        build_geometry_cache(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Shape(desktop=d["desktop"], view=d["view"], step=d["step"],
              instance=d["instance"], cls=d["cls"], label=d["label"],
              ordinal=d["ordinal"], area=d["area"], box=tuple(d["box"]),
              centroid=tuple(d["centroid"]), polys=d["polys"])
        for d in raw
    ]


def index_shapes(shapes: list[Shape]) -> dict[tuple[int, str, int], dict[str, Shape]]:
    """``(desktop, view, step) -> {instance: Shape}``."""
    out: dict[tuple[int, str, int], dict[str, Shape]] = {}
    for s in shapes:
        out.setdefault((s.desktop, s.view, s.step), {})[s.instance] = s
    return out


def frame_masks(con: sqlite3.Connection, desktop: int, view: str,
                step: int) -> dict[str, np.ndarray]:
    """Decode every LS mask of one frame (one decode pass, shared by callers)."""
    rows = con.execute(
        """SELECT k.instance, p.rle_json FROM shape_keyframe k
           JOIN shape_part p ON p.keyframe_id = k.id
           WHERE k.instance LIKE 'ls:%' AND k.desktop=? AND k.view=? AND k.anchor_step=?""",
        (desktop, view, step),
    ).fetchall()
    return {r["instance"]: decode_rle(json.loads(r["rle_json"])) for r in rows}


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def frame_path(con: sqlite3.Connection, desktop: int, view: str,
               step: int) -> Optional[str]:
    """The native still of one frame -- for OAK the 12 MP jpg, never the aligned png."""
    r = con.execute("SELECT path FROM frame WHERE desktop=? AND view=? AND step=?",
                    (desktop, view, step)).fetchone()
    if r is None or not r["path"]:
        return None
    return r["path"] if os.path.exists(r["path"]) else None


def read_rgb(path: str) -> Optional[np.ndarray]:
    """Load a frame as ``HxWx3`` RGB uint8."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def steps_with_ls(shapes: list[Shape], desktop: int, view: str) -> list[int]:
    return sorted({s.step for s in shapes if s.desktop == desktop and s.view == view})


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
_PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 214, 0), (166, 86, 40), (247, 129, 191),
    (0, 206, 209), (154, 205, 50),
]


def colour_of(key: str) -> tuple[int, int, int]:
    return _PALETTE[hash(key) % len(_PALETTE)]


def draw_shapes(img: np.ndarray, shapes: Iterable[Shape], thickness: int = 3,
                label: bool = True, font_scale: float = 1.0) -> np.ndarray:
    """Outline LS shapes on a copy of ``img`` (RGB in, RGB out)."""
    out = img.copy()
    for s in shapes:
        col = colour_of(s.label)
        for poly in s.polys:
            pts = np.asarray(poly, np.float64).reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], True, col, thickness, cv2.LINE_AA)
        cx, cy = int(round(s.centroid[0])), int(round(s.centroid[1]))
        cv2.drawMarker(out, (cx, cy), col, cv2.MARKER_CROSS, 18, thickness)
        if label:
            text = f"{s.label[:22]}#{s.ordinal}"
            cv2.putText(out, text, (s.box[0], max(14, s.box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        thickness + 2, cv2.LINE_AA)
            cv2.putText(out, text, (s.box[0], max(14, s.box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, col,
                        thickness, cv2.LINE_AA)
    return out


def draw_grid(img: np.ndarray, spacing: int, colour=(255, 255, 0),
              thickness: int = 1, font_scale: float = 0.8) -> np.ndarray:
    """Overlay a labelled pixel grid so coordinates can be read off the image."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, spacing):
        major = (x // spacing) % 5 == 0
        cv2.line(out, (x, 0), (x, h), colour, thickness + (1 if major else 0))
        if major:
            cv2.putText(out, str(x), (x + 4, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, str(x), (x + 4, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, colour, 2, cv2.LINE_AA)
    for y in range(0, h, spacing):
        major = (y // spacing) % 5 == 0
        cv2.line(out, (0, y), (w, y), colour, thickness + (1 if major else 0))
        if major:
            cv2.putText(out, str(y), (4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out, str(y), (4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, colour, 2, cv2.LINE_AA)
    return out


def save_jpg(path: Path, rgb: np.ndarray, max_side: int = 1400,
             quality: int = 82) -> Path:
    """Write an RGB array as a JPEG no larger than ``max_side`` on its long edge."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        rgb = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return path


def hstack_pad(images: list[np.ndarray], gap: int = 12) -> np.ndarray:
    """Place images side by side on a white strip, top-aligned."""
    h = max(im.shape[0] for im in images)
    total = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    canvas = np.full((h, total, 3), 255, np.uint8)
    x = 0
    for im in images:
        canvas[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return canvas


def banner(img: np.ndarray, text: str, height: int = 46) -> np.ndarray:
    """Add a white caption strip above ``img``."""
    strip = np.full((height, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (10, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([strip, img])
