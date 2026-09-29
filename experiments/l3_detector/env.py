"""Pin every cache, temp dir and thread pool before torch / ultralytics load.

Import this module **first** in every L3 entry point.  It imports L2's
``env`` (so L2's modules can be reused unchanged) and then overrides every
variable with L3's own locations: all caches under ``D:/DataSet/.cache/l3``,
weights under ``D:/DataSet/models/weights``, outputs under
``D:/DataSet/experiments_out/l3_detector``.  Nothing lands on C:.

Downloads stay off (``YOLO_OFFLINE`` / ``HF_HUB_OFFLINE``) except in
``download_weights.py``, which fetches the user-approved weights once.

Threads / loader workers stay at 4: another agent runs a timing-sensitive UI
suite on this machine.  Run with the ``tda_l3`` env (a clone of ``tda``),
never ``tda`` itself: ``D:\\Anaconda\\envs\\tda_l3\\python.exe -m ...``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.l2_detector import env as l2env  # noqa: E402  (sets L2's values first)

CACHE = Path("D:/DataSet/.cache/l3")
TMP = CACHE / "tmp"
DBTMP = Path("D:/DataSet/.cache/tmp/l3")
DB = str(DBTMP / "l3.sqlite")           # a copy of u2b3_copy.sqlite, opened mode=ro
OUT = Path("D:/DataSet/experiments_out/l3_detector")
WEIGHTS = Path("D:/DataSet/models/weights")
L2OUT = l2env.OUT                        # L2's tiles, folds, detections (read only)
MAX_THREADS = 4
MAX_WORKERS = 4

_ENV = {
    "TMP": str(TMP),
    "TEMP": str(TMP),
    "TMPDIR": str(TMP),
    "PIP_CACHE_DIR": str(CACHE / "pip"),
    "TORCH_HOME": str(CACHE / "torch"),
    "HF_HOME": str(CACHE / "hf"),
    "HUGGINGFACE_HUB_CACHE": str(CACHE / "hf" / "hub"),
    "TRANSFORMERS_CACHE": str(CACHE / "hf" / "transformers"),
    "XDG_CACHE_HOME": str(CACHE / "xdg"),
    "YOLO_CONFIG_DIR": str(CACHE / "ultralytics_cfg"),
    "RF_HOME": str(WEIGHTS),
    "ROBOFLOW_HOME": str(WEIGHTS),
    "MPLCONFIGDIR": str(TMP / "mpl"),
    "WANDB_MODE": "disabled",
    "WANDB_DIR": str(TMP / "wandb"),
    # never download (download_weights.py lifts these for its one job)
    "YOLO_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # CPU threads
    "OMP_NUM_THREADS": str(MAX_THREADS),
    "MKL_NUM_THREADS": str(MAX_THREADS),
    "OPENBLAS_NUM_THREADS": str(MAX_THREADS),
    "NUMEXPR_NUM_THREADS": str(MAX_THREADS),
}

for _k, _v in _ENV.items():
    os.environ[_k] = _v
for _p in (TMP, TMP / "mpl", DBTMP, OUT, WEIGHTS):
    _p.mkdir(parents=True, exist_ok=True)


def limit_threads() -> None:
    """Cap torch and OpenCV CPU threads (call after importing torch)."""
    l2env.limit_threads()


def bind_l2(tag: str | None = None) -> Path:
    """Point L2's reused modules at L3's database copy, temp dir and outputs.

    ``tag`` is the model's output folder (``yolo26s``, ``rfdetr_small``, ...):
    L2's evaluation code reads ``<tag>/detections/<fold>.json`` and writes its
    tables into ``<tag>/`` exactly as it did for YOLO26n.  Returns that folder.
    """
    from experiments.l2_detector import data as D

    D.DB = DB
    D.DB_URI = f"file:{DB}?mode=ro"
    D.GEOM = DBTMP / "geom" / "ls_boxes_all.json"
    if not D.GEOM.exists():
        # build_boxes' default path was bound to L2's GEOM at import: pass ours
        D.build_boxes(D.GEOM)
    l2env.TMP = DBTMP
    out = OUT / tag if tag else OUT
    out.mkdir(parents=True, exist_ok=True)
    l2env.OUT = out
    # module-level constants computed from L2's OUT at import time
    from experiments.l2_detector import det_eval, guess_eval

    det_eval.DETS = out / "detections"
    guess_eval.DETS = out / "detections"
    return out
