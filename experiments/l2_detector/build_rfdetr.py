"""The same tiles as YOLO-segmentation labels, laid out per fold for RF-DETR.

Ran once; its output fed nothing, because RF-DETR training stops at a missing
dependency (see ``train_rfdetr.py``).  Kept with that script.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.build_rfdetr

The only RF-DETR checkpoint on disk is ``rf-detr-seg-preview.pt`` (a
segmentation model), so the tiles get polygon labels: each kept draft's mask,
cut to the tile at the work scale, largest external contour.  Tiles, the
keep-rule and the folds are exactly ``build_dataset``'s; images are hard links
(same volume), so the per-fold ``train/valid`` folders RF-DETR's YOLO loader
wants cost no disk.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402
from experiments.l2_detector.build_dataset import DS, FOLDS, KEEP_FRAC  # noqa: E402

RF = env.OUT / "dataset_rfdetr"


def frame_masks(con, desktop, view, step, roi):
    """``[(cls, work-scale mask of the ROI crop, work box)]`` of one frame."""
    from pycocotools import mask as mask_utils

    rows = con.execute(
        """SELECT k.instance, i.cls, p.rle_json FROM shape_keyframe k
           JOIN shape_part p ON p.keyframe_id = k.id
           LEFT JOIN instance i ON i.desktop = k.desktop AND i."key" = k.instance
           WHERE k.instance LIKE 'ls:%' AND k.desktop=? AND k.view=? AND k.anchor_step=?""",
        (desktop, view, step)).fetchall()
    out = []
    x0, y0, x1, y1 = roi
    sc = T.WORK_SCALE[view]
    for r in rows:
        if r["cls"] not in D.CLS_ID:
            continue
        rle = json.loads(r["rle_json"])
        rle = {"size": rle["size"], "counts": rle["counts"].encode("ascii")}
        m = mask_utils.decode(rle)[y0:y1, x0:x1]
        if sc != 1.0:
            m = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        out.append((r["cls"], np.ascontiguousarray(m),
                    (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return out


def tile_polys(masks, ox, oy, w, h) -> list[str]:
    lines = []
    for cls, m, (bx0, by0, bx1, by1) in masks:
        area = (bx1 - bx0) * (by1 - by0)
        cx0, cy0 = max(bx0, ox), max(by0, oy)
        cx1, cy1 = min(bx1, ox + w), min(by1, oy + h)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2 or (cx1 - cx0) * (cy1 - cy0) < KEEP_FRAC * area:
            continue
        sub = m[oy:oy + h, ox:ox + w]
        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) < 3:
            x, y, ww, hh = cv2.boundingRect(sub)
            c = np.array([[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]])
        pts = " ".join(f"{px / T.TILE:.5f} {py / T.TILE:.5f}" for px, py in c)
        lines.append(f"{D.CLS_ID[cls]} {pts}\n")
    return lines


def main() -> int:
    manifest = json.loads((DS / "manifest.json").read_text(encoding="utf-8"))
    boxes = D.load_boxes()
    by_frame = defaultdict(list)
    for m in manifest:
        by_frame[(m["desktop"], m["view"], m["step"])].append(m)
    con = D.connect()
    seg_lab = RF / "labels_all"
    seg_lab.mkdir(parents=True, exist_ok=True)
    n = 0
    for (d, v, s), tiles in sorted(by_frame.items()):
        roi = D.roi_of(boxes, d, v)
        masks = frame_masks(con, d, v, s, roi)
        crop_h = int(round((roi[3] - roi[1]) * T.WORK_SCALE[v]))
        crop_w = int(round((roi[2] - roi[0]) * T.WORK_SCALE[v]))
        for t in tiles:
            w = min(T.TILE, crop_w - t["ox"])
            h = min(T.TILE, crop_h - t["oy"])
            lines = tile_polys(masks, t["ox"], t["oy"], w, h)
            name = Path(t["image"]).stem
            (seg_lab / f"{name}.txt").write_text("".join(lines), encoding="utf-8")
            n += 1
        if n % 500 < len(tiles):
            print(f"  {n} tiles", flush=True)
    con.close()
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(D.DET_CLASSES))
    for fold, held in FOLDS.items():
        root = RF / fold
        train = [m for m in manifest if m["desktop"] != held]
        val_list = set(Path(p).stem for p in
                       (DS / f"{fold}_val.txt").read_text(encoding="utf-8").split())
        for split, items in (("train", train),
                             ("valid", [m for m in train if Path(m["image"]).stem in val_list])):
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            for m in items:
                src = Path(m["image"])
                dst = root / split / "images" / src.name
                if not dst.exists():
                    os.link(src, dst)
                lab = seg_lab / f"{src.stem}.txt"
                (root / split / "labels" / f"{src.stem}.txt").write_text(
                    lab.read_text(encoding="utf-8"), encoding="utf-8")
        (root / "data.yaml").write_text(
            f"path: {root.as_posix()}\ntrain: train/images\nval: valid/images\n"
            f"nc: {len(D.DET_CLASSES)}\nnames:\n{names}", encoding="utf-8")
        print(f"  {fold}: {len(train)} train tiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
