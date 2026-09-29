"""Which detector weights exist locally, and do they load with downloads off?

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.probe_weights

Nothing is downloaded (``env`` sets ``YOLO_OFFLINE`` / ``HF_HUB_OFFLINE``).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402

CANDIDATES = [
    r"D:\DataSet\models\ultralytics\weights\yolo26n.pt",
    r"D:\DataSet\models\weights\yolo26n-seg.pt",
    r"D:\DataSet\models\rfdetr\rf-detr-seg-preview.pt",
]


def main() -> int:
    env.limit_threads()
    for p in CANDIDATES:
        print(p, "exists" if Path(p).exists() else "MISSING")
    from ultralytics import YOLO

    m = YOLO(CANDIDATES[0])
    print("yolo26n task:", m.task, "params:", sum(x.numel() for x in m.model.parameters()))
    try:
        from rfdetr import RFDETRSegPreview

        r = RFDETRSegPreview(pretrain_weights=CANDIDATES[2])
        print("RFDETRSegPreview loaded; resolution:", r.model_config.resolution)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
