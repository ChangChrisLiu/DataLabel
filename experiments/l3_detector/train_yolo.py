"""Train one fold of a larger YOLO26 with L2's recipe, unchanged.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.train_yolo --model yolo26s --fold hold13

Only the initial weights change (COCO ``yolo26s.pt`` / ``yolo26m.pt`` /
``yolo26l.pt`` from ``D:/DataSet/models/weights``).  The recipe is L2's
``train.RECIPE`` imported verbatim -- 60 epochs, batch 32, imgsz 640, mosaic
off for the last 10, h/v flips, scale +-25 %, 4 workers, seed 0,
``cache=False``, ``last.pt`` -- fixed a priori and identical for every fold;
``--batch`` exists only in case a model does not fit (the report says so).

``--fold all`` trains on every desktop (the production candidate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402
from experiments.l2_detector.train import RECIPE as L2_RECIPE  # noqa: E402

YDS = env.OUT / "dataset_yolo"
MODELS = {"yolo26n": r"D:\DataSet\models\ultralytics\weights\yolo26n.pt",
          "yolo26s": str(env.WEIGHTS / "yolo26s.pt"),
          "yolo26m": str(env.WEIGHTS / "yolo26m.pt"),
          "yolo26l": str(env.WEIGHTS / "yolo26l.pt")}


def runs_dir(model: str) -> Path:
    return env.OUT / model / "runs"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--batch", type=int, default=L2_RECIPE["batch"])
    args = ap.parse_args(argv)
    env.limit_threads()
    from ultralytics import YOLO

    rec = dict(L2_RECIPE)
    rec["batch"] = args.batch
    rec["workers"] = env.MAX_WORKERS
    model = YOLO(MODELS[args.model])
    out = runs_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model.train(data=str(YDS / f"{args.fold}.yaml"), project=str(out), name=args.fold, **rec)
    dt = time.perf_counter() - t0
    (out / args.fold / "l3_train.json").write_text(json.dumps(
        {"model": args.model, "init": MODELS[args.model], "fold": args.fold,
         "recipe": rec, "seconds": round(dt, 1)}, indent=1), encoding="utf-8")
    print(f"[l3] {args.model} fold {args.fold} trained in {dt:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
