"""Pin every cache, temp dir and thread pool before torch / ultralytics load.

Import this module **first** in every L2 entry point.  Nothing may land on C:
(user rule), nothing may be downloaded (task rule: the offline switches make a
missing weight fail loudly instead of fetching it), and CPU threads / loader
workers stay at 4 because another worker runs a timing-sensitive UI suite on
this machine.
"""
from __future__ import annotations

import os
from pathlib import Path

TMP = Path("D:/DataSet/.cache/tmp/l2")
OUT = Path("D:/DataSet/experiments_out/l2_detector")
MAX_THREADS = 4
MAX_WORKERS = 4

_ENV = {
    "TMP": str(TMP),
    "TEMP": str(TMP),
    "TMPDIR": str(TMP),
    "TORCH_HOME": r"D:\DataSet\models\torch_home",
    "HF_HOME": r"D:\DataSet\models\hf",
    "HUGGINGFACE_HUB_CACHE": r"D:\DataSet\models\hf\hub",
    "TRANSFORMERS_CACHE": r"D:\DataSet\models\hf\transformers",
    "XDG_CACHE_HOME": r"D:\DataSet\.cache",
    "YOLO_CONFIG_DIR": r"D:\DataSet\.cache\ultralytics_cfg",
    "RF_HOME": r"D:\DataSet\models\rfdetr",
    "ROBOFLOW_HOME": r"D:\DataSet\models\rfdetr",
    "MPLCONFIGDIR": str(TMP / "mpl"),
    # never download
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
TMP.mkdir(parents=True, exist_ok=True)
(TMP / "mpl").mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def limit_threads() -> None:
    """Cap torch and OpenCV CPU threads (call after importing torch)."""
    try:
        import torch

        torch.set_num_threads(MAX_THREADS)
        torch.set_num_interop_threads(min(2, MAX_THREADS))
    except Exception:  # noqa: BLE001 - torch absent or already initialised
        pass
    try:
        import cv2

        cv2.setNumThreads(MAX_THREADS)
    except Exception:  # noqa: BLE001
        pass
