"""Write the tile dataset (YOLO format) and the leave-one-desktop-out folds.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.build_dataset

Frames used for training: every (desktop, view, step) with at least one draft
of a detector class, on a desktop-view whose drafts include at least
``MIN_SMALL`` screws + connectors.  The rule drops the desktop-views where the
drafts are clearly partial (D2, D11, D15, D16, D64, D65 scan: 2-20 drafts in
total, no screws), whose unlabeled screws would otherwise be taught as
background.

Tile labels: a draft's bbox, at the work scale, clipped to the tile; kept when
at least half of it lies inside (a part cut by a tile edge is whole in the
neighbouring tile, 128 px overlap).

Folds: ``hold13``, ``hold24``, ``hold33`` -- train on every other desktop's
tiles, all views (including the other two event desktops' drafts); the held-
out desktop is never trained on.  ``val`` lists 150 random *training* tiles
and exists only because ultralytics needs one: ``last.pt`` is used, never
``best.pt``, so nothing is selected on it.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402

MIN_SMALL = 20
KEEP_FRAC = 0.5
DS = env.OUT / "dataset"
FOLDS = {f"hold{d}": d for d in D.EVENT_DESKTOPS}


def included_desktop_views(boxes) -> list[tuple[int, str]]:
    n = Counter((b.desktop, b.view) for b in boxes
                if b.cls in ("screw", "connector"))
    return sorted(k for k, v in n.items() if v >= MIN_SMALL)


def tile_labels(frame_boxes, roi, view, tile: T.Tile) -> list[tuple[int, float, float, float, float]]:
    out = []
    for b in frame_boxes:
        if b.cls not in D.CLS_ID:
            continue
        x0, y0, x1, y1 = T.to_work(b.box, roi, view)
        x0 -= tile.ox
        x1 -= tile.ox
        y0 -= tile.oy
        y1 -= tile.oy
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        cx0, cy0 = max(0.0, x0), max(0.0, y0)
        cx1, cy1 = min(float(tile.w), x1), min(float(tile.h), y1)
        if cx1 - cx0 < 2 or cy1 - cy0 < 2 or area <= 0:
            continue
        if (cx1 - cx0) * (cy1 - cy0) < KEEP_FRAC * area:
            continue
        out.append((D.CLS_ID[b.cls], (cx0 + cx1) / 2 / T.TILE,
                    (cy0 + cy1) / 2 / T.TILE, (cx1 - cx0) / T.TILE,
                    (cy1 - cy0) / T.TILE))
    return out


def main() -> int:
    boxes = D.load_boxes()
    by = D.index(boxes)
    dvs = included_desktop_views(boxes)
    print(f"[l2] desktop-views used: {dvs}")
    con = D.connect()
    manifest = []
    lab_count = Counter()
    t0 = time.perf_counter()
    for desktop, view in dvs:
        roi = D.roi_of(boxes, desktop, view)
        steps = sorted(s for (d, v, s) in by if d == desktop and v == view
                       and any(b.cls in D.CLS_ID for b in by[(d, v, s)]))
        for step in steps:
            path = D.frame_path(con, desktop, view, step)
            if path is None:
                print(f"  missing frame D{desktop} {view} s{step}")
                continue
            img = D.read_rgb(path)
            if img is None:
                print(f"  unreadable frame D{desktop} {view} s{step}")
                continue
            crop = T.work_crop(img, roi, view)
            for tile in T.make_tiles(crop):
                labels = tile_labels(by[(desktop, view, step)], roi, view, tile)
                name = f"D{desktop:02d}_{view}_s{step:03d}_{tile.oy:04d}_{tile.ox:04d}"
                ip = DS / "images" / view / f"{name}.jpg"
                lp = DS / "labels" / view / f"{name}.txt"
                ip.parent.mkdir(parents=True, exist_ok=True)
                lp.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ip), cv2.cvtColor(tile.rgb, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                lp.write_text("".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
                                      for c, x, y, w, h in labels), encoding="utf-8")
                for c, *_ in labels:
                    lab_count[(desktop, view, D.DET_CLASSES[c])] += 1
                manifest.append({"image": str(ip).replace("\\", "/"),
                                 "desktop": desktop, "view": view, "step": step,
                                 "ox": tile.ox, "oy": tile.oy, "n": len(labels)})
            print(f"  D{desktop} {view} s{step}: {len(manifest)} tiles, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
    con.close()
    (DS / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    names = {i: c for i, c in enumerate(D.DET_CLASSES)}
    rng = random.Random(0)
    summary = defaultdict(dict)
    for fold, held in FOLDS.items():
        train = [m["image"] for m in manifest if m["desktop"] != held]
        val = rng.sample(train, 150)
        (DS / f"{fold}_train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (DS / f"{fold}_val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
        yaml = (f"path: {DS.as_posix()}\ntrain: {fold}_train.txt\nval: {fold}_val.txt\n"
                "names:\n" + "".join(f"  {i}: {c}\n" for i, c in names.items()))
        (DS / f"{fold}.yaml").write_text(yaml, encoding="utf-8")
        per_view = Counter(m["view"] for m in manifest if m["desktop"] != held)
        summary[fold] = {"tiles": len(train), "per_view": dict(per_view),
                         "labels": {c: sum(n for (d, v, cc), n in lab_count.items()
                                           if d != held and cc == c)
                                    for c in D.DET_CLASSES}}
    (DS / "folds.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    tot = Counter(m["view"] for m in manifest)
    print(f"[l2] {len(manifest)} tiles, per view {dict(tot)}, "
          f"{sum(1 for m in manifest if m['n'] == 0)} without labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
