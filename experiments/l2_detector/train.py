"""Train one leave-one-desktop-out fold of the tile detector.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.train --fold hold13

Model: YOLO26n, initialised from the COCO ``yolo26n.pt`` already on disk (the
only pretrained YOLO26 detection weight present; ``yolo26s/m/l`` and every
RF-DETR detection checkpoint are absent and were not downloaded).

The recipe is fixed a priori and identical for the three folds -- nothing is
tuned on a held-out desktop: 640 tiles, 60 epochs, batch 32, SGD/auto, mosaic
(off for the last 10 epochs), flips both ways (top-down views), scale jitter
+-25 % (not the default +-50 %: a 15 px screw should not be trained at 7 px),
4 loader workers.  ``last.pt`` is the fold's model; ``best.pt`` is never used.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402

WEIGHTS = r"D:\DataSet\models\ultralytics\weights\yolo26n.pt"
RUNS = env.OUT / "runs"
DS = env.OUT / "dataset"

RECIPE = dict(
    epochs=60, imgsz=640, batch=32, workers=env.MAX_WORKERS, device=0,
    # cache: hold24 ran with cache="ram"; hold13 (4 015 tiles, ~4.9 GB) then
    # crashed at close_mosaic, when Windows re-spawns the loader workers and
    # the pickled RAM cache exceeds what the pipe takes (OSError 22).  The
    # other folds read the JPEG tiles instead -- same pixels, same recipe.
    seed=0, deterministic=True, cache=False, plots=False, val=False,
    close_mosaic=10, mosaic=1.0, fliplr=0.5, flipud=0.5, scale=0.25,
    degrees=0.0, translate=0.1, hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
    patience=0, amp=True, exist_ok=True, verbose=True,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    ap.add_argument("--name", default="")
    args = ap.parse_args(argv)
    env.limit_threads()
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    rec = dict(RECIPE)
    rec["epochs"] = args.epochs
    t0 = time.perf_counter()
    model.train(data=str(DS / f"{args.fold}.yaml"), project=str(RUNS),
                name=args.name or args.fold, **rec)
    print(f"[l2] fold {args.fold} trained in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
