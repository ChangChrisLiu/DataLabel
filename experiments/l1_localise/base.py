"""Shared plumbing for L1: the 201 events, their frames, their ground truth.

Everything that decides *which* events are measured and *what* counts as the
part is reused verbatim from the B2 harness
(``experiments/plan_b_probe/diff_eval.py``) and the E2 transfer probe it builds
on (``experiments/plan_b_probe/transfer``):

* event resolution -- :func:`experiments.plan_b_probe.diff_eval.resolve_events`
  (an action whose target class has exactly one ``ls:*`` draft at ``step - 1``
  that is gone at ``step``);
* ground truth -- that draft's mask at ``step - 1``, decoded by
  :func:`experiments.plan_b_probe.transfer.common.frame_masks`;
* ROI -- :func:`experiments.plan_b_probe.diff_eval.roi_of` (the view's LS union
  box padded 10 %), and the per-view ``min_area`` diff_eval uses.

Only three things differ, and all three are plumbing:

* the database is ``D:/DataSet/.cache/tmp/l1/l1.sqlite``, a copy of
  ``u2b3_copy.sqlite`` opened ``mode=ro`` (never the live database);
* the raw drive letter has moved (``F:`` -> today's letter), so a stored frame
  path goes through :func:`tda.core.rawroot.resolve_raw` before it is opened;
* the geometry cache lives under ``D:/DataSet/.cache/tmp/l1/geom``.

The population is **pinned**: :func:`load_events` checks the resolved events
against ``events_baseline.csv`` of the B2 run and refuses to continue if the
set differs.
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.plan_b_probe import diff_eval as E  # noqa: E402
from experiments.plan_b_probe.transfer import common  # noqa: E402

DB = "D:/DataSet/.cache/tmp/l1/l1.sqlite"
TMP = Path("D:/DataSet/.cache/tmp/l1")
OUT = Path("D:/DataSet/experiments_out/l1_localise")
B2_BASELINE = Path(
    "D:/DataSet/experiments_out/plan_b_probe/diff_eval/events_baseline.csv")

VIEWS = ("scan", "oak1", "oak2", "rs")
#: Part-width buckets on ``sqrt(area)`` of the draft -- B2's "part width".
BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<40", 0.0, 40.0),
    ("40-100", 40.0, 100.0),
    ("100-250", 100.0, 250.0),
    (">250", 250.0, 1e12),
)

Box = tuple[int, int, int, int]


def bucket_of(width: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= width < hi:
            return name
    return BUCKETS[-1][0]


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
_READY = False


def setup() -> None:
    """Point the reused modules at the L1 copy and locate the raw drive."""
    global _READY
    if _READY:
        return
    import yaml

    from tda.core import rawroot

    common.DB_URI = f"file:{DB}?mode=ro"
    common.TMP = TMP / "geom"
    common.TMP.mkdir(parents=True, exist_ok=True)
    paths = yaml.safe_load((REPO / "configs" / "paths.yaml").read_text("utf-8"))
    raw = rawroot.configure(paths)
    if not raw.connected:
        raise SystemExit(f"raw drive not found: {raw.status}")
    _READY = True


def frame_path(con, desktop: int, view: str, step: int) -> Optional[str]:
    """``common.frame_path`` with the stored ``F:`` path resolved to today's drive."""
    from tda.core.rawroot import resolve_raw

    r = con.execute("SELECT path FROM frame WHERE desktop=? AND view=? AND step=?",
                    (desktop, view, step)).fetchone()
    if r is None or not r["path"]:
        return None
    path = resolve_raw(r["path"])
    return path if path and os.path.exists(path) else None


class Frames:
    """A tiny LRU of decoded RGB frames for one (desktop, view)."""

    def __init__(self, con, desktop: int, view: str, keep: int = 6) -> None:
        self.con, self.desktop, self.view, self.keep = con, desktop, view, keep
        self._cache: dict[int, Optional[np.ndarray]] = {}
        self._order: list[int] = []

    def get(self, step: int) -> Optional[np.ndarray]:
        if step in self._cache:
            self._order.remove(step)
            self._order.append(step)
            return self._cache[step]
        path = frame_path(self.con, self.desktop, self.view, step)
        img = None if path is None else common.read_rgb(path)
        self._cache[step] = img
        self._order.append(step)
        while len(self._order) > self.keep:
            self._cache.pop(self._order.pop(0), None)
        return img


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
@dataclass
class Ev:
    """One of the 201 events, with everything a method may legitimately use."""

    desktop: int
    view: str
    step: int          # action step k; the annotated frame is j = k - 1
    verb: str
    target: str
    cls: str
    instance: str      # GT draft key at k - 1
    label: str
    group: str
    part_area: int
    part_width: float
    gt_box: Box
    roi: Box
    min_area: int

    @property
    def j(self) -> int:
        return self.step - 1

    @property
    def key(self) -> tuple:
        return (self.desktop, self.view, self.step, self.instance)

    @property
    def bucket(self) -> str:
        return bucket_of(self.part_width)


def load_events(views=VIEWS, check: bool = True):
    """``(events, shapes, by_frame, con)`` -- the pinned B2 population."""
    setup()
    shapes = common.load_geometry()
    by_frame = common.index_shapes(shapes)
    con = common.connect()
    classes = {r["cls"] for r in con.execute("SELECT DISTINCT cls FROM instance")}
    events: list[Ev] = []
    for desktop in common.DESKTOPS:
        for view in views:
            if not any(s.desktop == desktop and s.view == view for s in shapes):
                continue
            roi = E.roi_of(shapes, desktop, view)
            h, w = common.NATIVE_HW[view]
            min_area = max(30, int(round(80 * (h * w) / (1600 * 1600))))
            evs, _unres = E.resolve_events(con, shapes, by_frame, classes,
                                           desktop, view)
            for ev in evs:
                s = ev.shape
                events.append(Ev(
                    desktop=desktop, view=view, step=ev.step, verb=ev.verb,
                    target=ev.target, cls=ev.cls, instance=s.instance,
                    label=s.label, group=s.group, part_area=int(s.area),
                    part_width=float(s.size), gt_box=tuple(int(v) for v in s.box),
                    roi=tuple(int(v) for v in roi), min_area=min_area))
    if check:
        want = set()
        with B2_BASELINE.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                want.add((int(r["desktop"]), r["view"], int(r["step"]),
                          r["instance"]))
        got = {e.key for e in events if e.view in views}
        want = {k for k in want if k[1] in views}
        if want - got:
            raise SystemExit(f"B2 events that no longer resolve: "
                             f"{sorted(want - got)[:10]}")
        extra = sorted(got - want)
        if extra:
            # The copy post-dates B2: the controller's D13 decisions added a
            # psu_latch removal at step 33 and D24's misfiled last step now
            # resolves on rs.  Those events are new, not B2's; drop them.
            print(f"[l1] {len(extra)} events resolve now that B2 did not "
                  f"have; excluded to keep the population pinned: {extra}")
            events = [e for e in events if e.key in want]
    return events, shapes, by_frame, con


def gt_mask(con, ev: Ev) -> Optional[np.ndarray]:
    return common.frame_masks(con, ev.desktop, ev.view, ev.j).get(ev.instance)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def box_iou(a, b) -> float:
    return E.box_iou(a, b)


def point_in(mask: np.ndarray, point) -> bool:
    x, y = int(round(point[0])), int(round(point[1]))
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])


class GtGeom:
    """Cheap per-event lookups on the GT mask (pixel list, centroid, size)."""

    def __init__(self, mask: np.ndarray) -> None:
        self.mask = mask
        ys, xs = np.nonzero(mask)
        self.xs, self.ys = xs.astype(np.float32), ys.astype(np.float32)
        self.cx, self.cy = float(xs.mean()), float(ys.mean())
        self.size = float(np.sqrt(max(1, xs.size)))

    def dist(self, point) -> float:
        """Distance from ``point`` to the nearest part pixel (0 inside)."""
        if point_in(self.mask, point):
            return 0.0
        d2 = (self.xs - point[0]) ** 2 + (self.ys - point[1]) ** 2
        return float(np.sqrt(d2.min()))

    def centre_dist(self, point) -> float:
        return float(np.hypot(point[0] - self.cx, point[1] - self.cy))

    def box_touches(self, box) -> bool:
        """Does the box contain any part pixel?"""
        x0, y0, x1, y1 = box
        inside = ((self.xs >= x0) & (self.xs < x1) & (self.ys >= y0)
                  & (self.ys < y1))
        return bool(inside.any())


def crop_roi(img: np.ndarray, roi: Box) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return img[y0:y1, x0:x1]


def save_rgb(path: Path, rgb: np.ndarray, quality: int = 88) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), quality])
