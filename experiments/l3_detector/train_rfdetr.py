"""Fine-tune an RF-DETR detection checkpoint on L2's tiles, one fold.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.train_rfdetr --model rfdetr_small --fold hold13

Initial weights: Roboflow's COCO detection checkpoints (``rf-detr-small.pth``
etc., Apache-2.0) in ``D:/DataSet/models/weights``.  Data: L3's
``dataset_rfdetr/<fold>`` -- L2's tiles and box labels, byte for byte.

Recipe -- RF-DETR's own fine-tuning defaults, fixed a priori, one recipe for
every fold and every size, nothing looked at on a held-out desktop:

* AdamW, lr 1e-4, encoder lr 1.5e-4 (layer decay 0.8), weight decay 1e-4,
  grad clip 0.1, no warm-up, EMA (0.993, tau 100), AMP -- all rfdetr defaults;
* effective batch 16 (RF-DETR's recommended effective batch): 8 x 2
  gradient-accumulation steps;
* resolution 640 = the tile, so a tile is never shrunk at inference (small's
  COCO default is 512: that would shrink a 15 px scanner screw to 12 px).
  RF-DETR's default training resize is kept as it is: with ``multi_scale`` +
  ``expanded_scales`` and ``do_random_resize_via_padding=False`` (default)
  rfdetr 1.10.1 keeps only the *largest* scale, so every training tile is
  resized to 800 x 800 (x1.25), half of them after a resize-to-400/500/600 +
  random 384-600 crop (x1.25 to ~x2 net), plus horizontal flip; validation
  and inference run at 640 (``predict``'s default = the model resolution).
  This train-larger / infer-at-resolution scheme is RF-DETR's own default and
  was not changed (the throughput probe revealed it; nothing was tuned);
* ``EPOCHS`` epochs (see below), CSV logging only, 4 loader workers, seed 0;
* the model is ``last_ema.pth`` (the last epoch's EMA weights); the
  ``valid`` split is 150 *training* tiles and selects nothing.

Budget: ``EPOCHS = 30`` at effective batch 16 is the same number of optimizer
steps as L2's YOLO recipe (60 epochs at batch 32), ~0.5x its image passes;
the choice was fixed from the RF-DETR throughput probe only (``--probe``),
before any fold was trained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402

RF = env.OUT / "dataset_rfdetr"
MODELS = {"rfdetr_small": ("RFDETRSmall", "rf-detr-small.pth"),
          "rfdetr_medium": ("RFDETRMedium", "rf-detr-medium.pth"),
          "rfdetr_base": ("RFDETRBase", "rf-detr-base.pth")}
RESOLUTION = 640
EPOCHS = 30
RECIPE = dict(
    epochs=EPOCHS, batch_size=8, grad_accum_steps=2, lr=1e-4, lr_encoder=1.5e-4,
    weight_decay=1e-4, use_ema=True, multi_scale=True, expanded_scales=True,
    num_workers=env.MAX_WORKERS, seed=0, tensorboard=False, wandb=False, mlflow=False,
    progress_bar=None, checkpoint_interval=1000, run_test=False,
)


def runs_dir(model: str) -> Path:
    return env.OUT / model / "runs"


def weights_of(model: str, fold: str) -> Path:
    return runs_dir(model) / fold / "last_ema.pth"


def build(model: str, resolution: int = RESOLUTION, pretrain: str | None = None):
    import rfdetr

    cls_name, ckpt = MODELS[model]
    cls = getattr(rfdetr, cls_name)
    return cls(pretrain_weights=pretrain or str(env.WEIGHTS / ckpt), resolution=resolution)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--fold", required=True)
    ap.add_argument("--batch", type=int, default=RECIPE["batch_size"])
    ap.add_argument("--probe", action="store_true",
                    help="throughput probe: 1 epoch, written to runs/_probe (budget only)")
    args = ap.parse_args(argv)
    env.limit_threads()
    rec = dict(RECIPE)
    if args.batch != rec["batch_size"]:
        # keep the effective batch at 16
        rec["grad_accum_steps"] = max(1, 16 // args.batch)
        rec["batch_size"] = args.batch
    name = args.fold
    if args.probe:
        rec["epochs"] = 1
        name = f"_probe_{args.fold}"
    model = build(args.model)
    out = runs_dir(args.model) / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model.train(dataset_dir=str(RF / args.fold), dataset_file="yolo", output_dir=str(out),
                resolution=RESOLUTION, **rec)
    dt = time.perf_counter() - t0
    (out / "l3_train.json").write_text(json.dumps(
        {"model": args.model, "init": MODELS[args.model][1], "fold": args.fold,
         "resolution": RESOLUTION, "recipe": rec, "seconds": round(dt, 1)}, indent=1),
        encoding="utf-8")
    print(f"[l3] {args.model} fold {name} trained in {dt:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
