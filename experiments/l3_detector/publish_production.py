"""Copy the all-desktop model to ``D:/DataSet/models/weights`` with a manifest.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.publish_production --model yolo26s

Source: ``<OUT>/<model>/runs/all/`` (trained by ``train_yolo`` /
``train_rfdetr`` with ``--fold all``, the fold recipe unchanged).  Target:
``tda_det_<model>_all.pt`` and ``tda_det_<model>_all.json`` (training tiles
hash, class list, recipe, tiling, source commit, date, hashes).  The weight is
never committed to git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402
from experiments.l2_detector import data as D  # noqa: E402
from experiments.l2_detector import tiles as T  # noqa: E402
from experiments.l3_detector.infer import YOLO_MODELS, weights_of  # noqa: E402


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(env.REPO), text=True).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)
    src = weights_of(args.model, "all")
    if not src.exists():
        raise SystemExit(f"missing {src}")
    run = src.parent.parent if args.model in YOLO_MODELS else src.parent
    train_meta = json.loads((run / "l3_train.json").read_text(encoding="utf-8"))
    tiles = json.loads((env.OUT / "tiles_hash.json").read_text(encoding="utf-8"))["all"]
    ledger = json.loads((env.OUT / "downloads.json").read_text(encoding="utf-8"))
    init_key = {"yolo26s": "yolo26s", "yolo26m": "yolo26m", "yolo26l": "yolo26l",
                "rfdetr_small": "rfdetr_small", "rfdetr_medium": "rfdetr_medium",
                "rfdetr_base": "rfdetr_base"}.get(args.model)
    dst = env.WEIGHTS / f"tda_det_{args.model}_all.pt"
    shutil.copyfile(src, dst)
    status = git("status", "--porcelain", "--", "experiments/l3_detector", "experiments/l2_detector")
    manifest = {
        "model": args.model,
        "file": str(dst),
        "sha256": sha256(dst),
        "bytes": dst.stat().st_size,
        "source_run": str(run),
        "init_weights": ledger.get(init_key, {"file": train_meta.get("init")}),
        "training": {
            "desktops": "all (no hold-out): every desktop-view with >= 20 screw+connector "
                        "drafts, L2's rule",
            "tiles": tiles["tiles"],
            "tiles_sha256": tiles["sha256"],
            "tiles_hash_rule": "sha256 over tiles sorted by name of (name, sha256(image), "
                               "sha256(label))",
            "tile_source": str(env.L2OUT / "dataset" / "manifest.json"),
            "recipe": train_meta.get("recipe"),
            "resolution": train_meta.get("resolution", T.TILE),
            "seconds": train_meta.get("seconds"),
        },
        "classes": list(D.DET_CLASSES),
        "inference": {
            "tile": T.TILE, "stride": T.STRIDE, "pad_value": T.PAD_VALUE,
            "work_scale": T.WORK_SCALE,
            "roi": "view's region of interest (L2 used the drafts' union box + 10 %)",
            "merge": "drop cut-off boxes contained >= 60 % in a whole one, class-wise NMS 0.5",
            "guess": "screw only (scan, oak1): conf >= 0.10, skip parts already drawn "
                     "(centre +-2 px or IoU >= 0.3), rank by native mean dE j->k in the box",
            "precision": "fp16",
        },
        "source_commit": git("rev-parse", "HEAD"),
        "source_dirty": bool(status),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluation": str(env.OUT / "compare.md") + " (leave-one-desktop-out fold models; "
                      "this all-desktop model has no held-out score of its own)",
        "licence": (ledger.get(init_key) or {}).get("licence", ""),
    }
    (env.WEIGHTS / f"tda_det_{args.model}_all.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
