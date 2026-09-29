"""Tiled inference: one frame + its ROI -> detections in native frame pixels.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.infer --fold hold13

Runs the fold's ``last.pt`` on every frame of the held-out desktop that has
drafts (every event's frame ``j`` has them by construction) and writes
``detections/<fold>.json``: ``{"<desktop>:<view>:<step>": [[x0, y0, x1, y1,
cls, conf], ...]}`` at confidence >= 0.001, so every later threshold is a
filter on this file.

Merging the tiles: a detection touching an *interior* tile edge (one that is
not the crop's edge) is dropped when a same-class detection from another tile
contains >= 60 % of it (the part is whole there, 128 px overlap); then
class-wise NMS at IoU 0.5 over all tiles.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402

RUNS = env.OUT / "runs"
DETS = env.OUT / "detections"
EDGE_PX = 2.0
CONTAIN = 0.6
NMS_IOU = 0.5


def weights_of(fold: str) -> Path:
    return RUNS / fold / "weights" / "last.pt"


@dataclass
class Det:
    box: tuple[float, float, float, float]   # native frame px
    cls: str
    conf: float

    @property
    def centre(self) -> tuple[float, float]:
        return (0.5 * (self.box[0] + self.box[2]), 0.5 * (self.box[1] + self.box[3]))

    def row(self) -> list:
        return [round(v, 1) for v in self.box] + [self.cls, round(self.conf, 4)]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box ``a`` against rows of ``b``."""
    ix0 = np.maximum(a[0], b[:, 0])
    iy0 = np.maximum(a[1], b[:, 1])
    ix1 = np.minimum(a[2], b[:, 2])
    iy1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa + ab - inter, 1e-6)


def merge(boxes: np.ndarray, confs: np.ndarray, clss: np.ndarray,
          edge: np.ndarray) -> np.ndarray:
    """Indices kept after the edge/containment rule and class-wise NMS."""
    n = len(boxes)
    if n == 0:
        return np.zeros(0, int)
    alive = np.ones(n, bool)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in np.nonzero(edge)[0]:
        same = np.nonzero((clss == clss[i]) & (np.arange(n) != i))[0]
        if same.size == 0:
            continue
        b = boxes[same]
        ix0 = np.maximum(boxes[i, 0], b[:, 0])
        iy0 = np.maximum(boxes[i, 1], b[:, 1])
        ix1 = np.minimum(boxes[i, 2], b[:, 2])
        iy1 = np.minimum(boxes[i, 3], b[:, 3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        frac = inter / max(areas[i], 1e-6)
        # contained in a detection that is not itself a cut-off one
        if np.any((frac >= CONTAIN) & ~edge[same]):
            alive[i] = False
    keep = []
    for c in np.unique(clss):
        idx = np.nonzero(alive & (clss == c))[0]
        idx = idx[np.argsort(-confs[idx])]
        while idx.size:
            i = idx[0]
            keep.append(i)
            if idx.size == 1:
                break
            ious = _iou(boxes[i], boxes[idx[1:]])
            idx = idx[1:][ious < NMS_IOU]
    return np.asarray(sorted(keep, key=lambda i: -confs[i]), int)


class Detector:
    """A fold's YOLO26n, run tile-wise on the ROI crop of a frame."""

    def __init__(self, weights: Path, half: bool = True, conf: float = 0.001):
        env.limit_threads()
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.half = half
        self.conf = conf
        # warm up (cudnn autotune, lazy init)
        dummy = [np.full((T.TILE, T.TILE, 3), 114, np.uint8)] * 4
        for _ in range(2):
            self.model.predict(dummy, imgsz=T.TILE, conf=self.conf, **self._prec(),
                               verbose=False, device=0, max_det=300)

    def _prec(self) -> dict:
        # ultralytics 8.4 replaced ``half=True`` by ``quantize=16``
        return {"quantize": 16} if self.half else {}

    def _run(self, tiles: list[T.Tile]):
        """Per tile ``(xyxy, conf, cls)`` numpy arrays, tile pixel coordinates."""
        batch = [cv2.cvtColor(t.rgb, cv2.COLOR_RGB2BGR) for t in tiles]
        res = self.model.predict(batch, imgsz=T.TILE, conf=self.conf, **self._prec(),
                                 verbose=False, device=0, max_det=300)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append(None)
                continue
            out.append((r.boxes.xyxy.cpu().numpy().astype(np.float64),
                        r.boxes.conf.cpu().numpy().astype(np.float64),
                        r.boxes.cls.cpu().numpy().astype(int)))
        return out

    def detect(self, img_rgb: np.ndarray, roi, view: str,
               timings: dict | None = None) -> list[Det]:
        t0 = time.perf_counter()
        crop = T.work_crop(img_rgb, roi, view)
        tiles = T.make_tiles(crop)
        H, W = crop.shape[:2]
        t1 = time.perf_counter()
        res = self._run(tiles)
        t2 = time.perf_counter()
        allb, allc, allk, alle = [], [], [], []
        for t, r in zip(tiles, res):
            if r is None or len(r[0]) == 0:
                continue
            xyxy, cf, cl = r
            # the padded area is not part of the image
            xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, t.w)
            xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, t.h)
            ok = (xyxy[:, 2] - xyxy[:, 0] >= 1) & (xyxy[:, 3] - xyxy[:, 1] >= 1)
            xyxy, cf, cl = xyxy[ok], cf[ok], cl[ok]
            e = np.zeros(len(xyxy), bool)
            if t.ox > 0:
                e |= xyxy[:, 0] <= EDGE_PX
            if t.oy > 0:
                e |= xyxy[:, 1] <= EDGE_PX
            if t.ox + t.w < W:
                e |= xyxy[:, 2] >= t.w - EDGE_PX
            if t.oy + t.h < H:
                e |= xyxy[:, 3] >= t.h - EDGE_PX
            xyxy[:, [0, 2]] += t.ox
            xyxy[:, [1, 3]] += t.oy
            allb.append(xyxy)
            allc.append(cf)
            allk.append(cl)
            alle.append(e)
        out: list[Det] = []
        if allb:
            b = np.concatenate(allb)
            c = np.concatenate(allc)
            k = np.concatenate(allk)
            e = np.concatenate(alle)
            for i in merge(b, c, k, e):
                nb = T.to_native(b[i], roi, view)
                out.append(Det(box=tuple(float(v) for v in nb),
                               cls=self.names[int(k[i])], conf=float(c[i])))
        t3 = time.perf_counter()
        if timings is not None:
            timings.update(prep_s=t1 - t0, net_s=t2 - t1, merge_s=t3 - t2,
                           total_s=t3 - t0, n_tiles=len(tiles))
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    args = ap.parse_args(argv)
    held = int(args.fold.replace("hold", ""))
    boxes = D.load_boxes()
    by = D.index(boxes)
    det = Detector(weights_of(args.fold))
    tag = args.fold
    con = D.connect()
    out: dict[str, list] = {}
    tim = []
    for view in D.VIEWS:
        steps = sorted(s for (d, v, s) in by if d == held and v == view)
        if not steps:
            continue
        roi = D.roi_of(boxes, held, view)
        for step in steps:
            path = D.frame_path(con, held, view, step)
            img = D.read_rgb(path) if path else None
            if img is None:
                print(f"  missing D{held} {view} s{step}")
                continue
            t = {}
            dets = det.detect(img, roi, view, timings=t)
            tim.append({"view": view, **t})
            out[f"{held}:{view}:{step}"] = [d.row() for d in dets]
        print(f"  D{held} {view}: {len(steps)} frames", flush=True)
    con.close()
    DETS.mkdir(parents=True, exist_ok=True)
    (DETS / f"{tag}.json").write_text(json.dumps(out), encoding="utf-8")
    (DETS / f"{tag}_timing.json").write_text(json.dumps(tim), encoding="utf-8")
    print(f"wrote {DETS / (tag + '.json')} ({len(out)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
