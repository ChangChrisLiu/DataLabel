"""Smoke test: does SAM 3 load and run on this Windows box at all?

Time-boxed gate for the experiment. If this fails (triton, numpy pin, an
Application Control block on a fresh .pyd), the SAM 3 half is reported as
blocked and only the SAM 2.1 baseline is produced.
"""
from __future__ import annotations

import traceback

import numpy as np

from experiments.sam_compare import _env  # noqa: F401  (env side effects first)
from experiments.sam_compare import common


def _demo_crop() -> np.ndarray:
    frame = common.load_frame_rgb(10, 1)
    if frame is None:
        raise RuntimeError("no cached scan frame D10/s001.png to smoke-test on")
    return np.ascontiguousarray(frame[500:1012, 500:1012])


def main() -> int:
    crop = _demo_crop()
    ok: dict[str, bool] = {}

    from experiments.sam_compare.sam3_service import Sam3Service

    print("[smoke] building SAM 3 ...", flush=True)
    svc = Sam3Service()
    print(f"[smoke] loaded in {svc.load_ms / 1000:.1f} s", flush=True)

    try:
        res = svc.predict_interactive(
            crop,
            points=np.array([[256.0, 256.0]], np.float32),
            labels=np.array([1], np.int32),
            multimask=True,
        )
        print(f"[smoke] point -> {int(res.mask.sum())} px, score {res.score:.3f}, "
              f"{res.ms:.0f} ms (set_image {svc.last_set_image_ms:.0f} ms)", flush=True)
        res = svc.predict_interactive(crop, box=np.array([200, 200, 320, 320], np.float32))
        print(f"[smoke] box   -> {int(res.mask.sum())} px, score {res.score:.3f}, "
              f"{res.ms:.0f} ms (cached embedding)", flush=True)
        ok["interactive"] = True
    except Exception:
        traceback.print_exc()
        ok["interactive"] = False

    try:
        for text in ("screw", "RAM module"):
            out = svc.detect_text(crop, text, threshold=0.4)
            print(f"[smoke] text {text!r:14} -> {len(out['scores'])} instances, "
                  f"{out['ms']:.0f} ms", flush=True)
        ok["concept"] = True
    except Exception:
        traceback.print_exc()
        ok["concept"] = False

    import torch

    print(f"[smoke] peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print(f"\n[smoke] result: {ok}")
    return 0 if all(ok.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
