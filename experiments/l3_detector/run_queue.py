"""Run L3 jobs one after another (one GPU job at a time, <= 4 loader workers).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.run_queue \\
        train_yolo:yolo26s:hold24 infer:yolo26s:hold24 ...

A job is ``<script>:<model>:<fold>`` (``evaluate`` / ``timing`` take no fold).
Each job's output goes to ``logs/<script>_<model>[_<fold>].log``; a failed job
is reported and the queue goes on with the next one.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402

PY = r"D:\Anaconda\envs\tda_l3\python.exe"
LOGS = env.OUT / "logs"


def main(argv=None) -> int:
    jobs = list(argv if argv is not None else sys.argv[1:])
    LOGS.mkdir(parents=True, exist_ok=True)
    status = []
    for job in jobs:
        parts = job.split(":")
        script, model = parts[0], parts[1]
        cmd = [PY, "-m", f"experiments.l3_detector.{script}", "--model", model]
        tag = f"{script}_{model}"
        rest = parts[2:]
        if rest and not rest[0].startswith("-"):
            cmd += ["--fold", rest[0]]
            tag += f"_{rest[0]}"
            rest = rest[1:]
        # extra flags, e.g. ``--conf=0.1`` (use ``=``: ``:`` separates the fields)
        cmd += rest
        tag += "".join(f"_{r.lstrip('-').replace('=', '')}" for r in rest)
        t0 = time.perf_counter()
        with (LOGS / f"{tag}.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(env.REPO))
        dt = time.perf_counter() - t0
        status.append((job, rc, dt))
        print(f"[queue] {job}: rc={rc} {dt:.0f} s", flush=True)
    bad = [j for j, rc, _ in status if rc != 0]
    print(f"[queue] done; failed: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
