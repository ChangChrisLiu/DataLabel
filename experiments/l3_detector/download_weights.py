"""Fetch the user-approved pretrained weights (2026-09-29: "可以用更大的模型").

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.download_weights yolo26s yolo26m rfdetr_small

Only these names are accepted; each lands in ``D:/DataSet/models/weights`` and
is recorded (URL, bytes, SHA-256, licence) in ``downloads.json`` next to the
L3 outputs.  A file already present is not fetched again.

* ``yolo26{s,m,l}.pt`` -- Ultralytics COCO detection weights, GitHub release
  ``ultralytics/assets`` v8.4.0 (AGPL-3.0, or an Ultralytics Enterprise licence);
* ``rf-detr-{small,medium,base}.pth`` -- Roboflow's COCO detection checkpoints
  from ``storage.googleapis.com/rfdetr`` (Apache-2.0), MD5-checked by rfdetr.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402

YOLO = {"yolo26s": "yolo26s.pt", "yolo26m": "yolo26m.pt", "yolo26l": "yolo26l.pt"}
RFDETR = {"rfdetr_small": "rf-detr-small.pth", "rfdetr_medium": "rf-detr-medium.pth",
          "rfdetr_base": "rf-detr-base.pth"}
LEDGER = env.OUT / "downloads.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    names = list(argv if argv is not None else sys.argv[1:])
    bad = [n for n in names if n not in YOLO and n not in RFDETR]
    if bad or not names:
        raise SystemExit(f"approved names only: {sorted(YOLO) + sorted(RFDETR)}; got {bad or names}")
    for k in ("YOLO_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(k, None)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {}
    for n in names:
        if n in YOLO:
            from ultralytics.utils.downloads import attempt_download_asset

            dst = env.WEIGHTS / YOLO[n]
            url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{YOLO[n]}"
            lic = "AGPL-3.0 (Ultralytics; Enterprise licence for closed use)"
            fetched = not dst.exists()
            t0 = time.perf_counter()
            attempt_download_asset(str(dst), release="v8.4.0")
        else:
            from rfdetr.assets.model_weights import ModelWeights, download_pretrain_weights

            dst = env.WEIGHTS / RFDETR[n]
            asset = ModelWeights.from_filename(RFDETR[n])
            url = asset.url
            lic = "Apache-2.0 (Roboflow RF-DETR)"
            fetched = not dst.exists()
            t0 = time.perf_counter()
            download_pretrain_weights(str(dst))
        if not dst.exists():
            raise SystemExit(f"{n}: download failed, {dst} missing")
        rec = {"file": str(dst), "url": url, "bytes": dst.stat().st_size,
               "sha256": sha256(dst), "licence": lic,
               "fetched_now": fetched, "seconds": round(time.perf_counter() - t0, 1),
               "date": time.strftime("%Y-%m-%d %H:%M:%S")}
        if n in ledger and not fetched:
            rec["date"] = ledger[n].get("date", rec["date"])
        ledger[n] = rec
        print(f"[l3] {n}: {dst} {rec['bytes'] / 1e6:.1f} MB sha256 {rec['sha256'][:16]}...")
    LEDGER.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
