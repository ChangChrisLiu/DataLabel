"""Score one model's held-out detections with L2's evaluation code, unchanged.

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.evaluate --model yolo26s

Runs L2's ``det_eval``, ``guess_eval`` and ``analyse`` with their inputs and
outputs re-pointed (``env.bind_l2``) at ``<OUT>/<model>/`` and at L3's
read-only copy of the database.  Nothing in L2's code changes: detector
metrics against the drafts, the reverse-walk guess on B2's 201 events scored
against L1's GT-B (``guess_eval`` still refuses to run if GT-B differs from
L1's ``gt_check.csv``), the ceiling, the tables.

``--model yolo26n`` copies L2's own detection files first, so the baseline is
re-scored by the same code path (a reproduction check against L2's CSVs).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--skip-guess", action="store_true")
    args = ap.parse_args(argv)
    out = env.bind_l2(args.model)
    dd = out / "detections"
    if args.model == "yolo26n":
        dd.mkdir(parents=True, exist_ok=True)
        for f in ("hold13", "hold24", "hold33"):
            shutil.copyfile(env.L2OUT / "detections" / f"{f}.json", dd / f"{f}.json")
    missing = [f for f in ("hold13", "hold24", "hold33") if not (dd / f"{f}.json").exists()]
    if missing:
        raise SystemExit(f"{args.model}: detections missing for {missing}")
    from experiments.l2_detector import analyse, det_eval, guess_eval

    det_eval.main()
    if not args.skip_guess:
        guess_eval.main()
        analyse.main()
    print(f"[l3] {args.model}: tables in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
