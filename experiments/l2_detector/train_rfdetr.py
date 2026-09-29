"""Second detector: fine-tune RF-DETR-Seg-Preview on the same tiles and folds.

**Status: stopped, not run.**  ``rfdetr`` 1.10.1 trains through PyTorch
Lightning, which is not installed in the ``tda`` env (``import
pytorch_lightning`` fails inside ``RFDETR.train``; the package asks for
``pip install "rfdetr[train,loggers]"``).  Installing it would mean a download,
so the branch stops here; the weight itself (``rf-detr-seg-preview.pt``) is on
disk and loads offline.  Kept as the record of what would have run.

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.train_rfdetr --fold hold24

``rf-detr-seg-preview.pt`` is the only RF-DETR checkpoint on disk (the
detection checkpoints ``rf-detr-nano/small/base/medium.pth`` are absent and
were not downloaded), so this is the segmentation model trained on the drafts'
polygons; only its boxes are used downstream.  Fixed recipe, identical per
fold: resolution 624 (the nearest multiple of patch 12 x 2 windows to the 640
tile, so a tile is shrunk by 2.5 % only), no multi-scale (it would shrink the
15 px screws), 20 epochs, batch 8, 4 workers, EMA weights of the last epoch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402

WEIGHTS = r"D:\DataSet\models\rfdetr\rf-detr-seg-preview.pt"
RF = env.OUT / "dataset_rfdetr"
RUNS = env.OUT / "runs_rfdetr"
RESOLUTION = 624


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args(argv)
    env.limit_threads()
    from rfdetr import RFDETRSegPreview

    model = RFDETRSegPreview(pretrain_weights=WEIGHTS, resolution=RESOLUTION)
    out = RUNS / args.fold
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model.train(dataset_dir=str(RF / args.fold), dataset_file="yolo",
                output_dir=str(out), epochs=args.epochs, batch_size=8,
                grad_accum_steps=2, num_workers=env.MAX_WORKERS,
                multi_scale=False, expanded_scales=False, tensorboard=False,
                checkpoint_interval=100, progress_bar=None, seed=0)
    print(f"[l2] rfdetr fold {args.fold} trained in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
