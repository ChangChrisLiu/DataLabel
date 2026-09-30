"""L2's overlay sheets for one L3 model's guess (L2's ``overlays`` re-pointed).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.overlays --model rfdetr_small 24:oak2:40 ...

Writes to ``<OUT>/<model>/overlays/``; the legend is L2's (green = removed
part, yellow = kept candidates with dE, grey = skipped as already drawn,
red/cyan = top-1, magenta = today's guess).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("events", nargs="+")
    args = ap.parse_args(argv)
    out = env.bind_l2(args.model)
    from experiments.l2_detector import overlays as O

    O.OUTDIR = out / "overlays"
    O.OUTDIR.mkdir(parents=True, exist_ok=True)
    return O.main(args.events)


if __name__ == "__main__":
    raise SystemExit(main())
