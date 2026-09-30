"""Paired comparison of two models: the guess (over events) and screw hit (over frames).

::

    D:\\Anaconda\\envs\\tda_l3\\python.exe -m experiments.l3_detector.pairwise rfdetr_small rfdetr_medium

* guess: top-1 of B minus A on GT-B events (< 40 px per view; motherboard
  screws on scan + oak1), paired bootstrap over events (10 000, seed 0);
* detector: screw hit@0.10 (a same-class box centre inside the draft box,
  L2's ``det_eval`` rule) of B minus A, pooled over scan + oak1 held-out
  frames, bootstrap over frames (the drafts of one frame move together).

Writes ``<OUT>/pairwise_<A>_vs_<B>.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l3_detector import env  # noqa: E402

LOW = 0.10


def boot(d: np.ndarray, n: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), *np.percentile(b, [2.5, 97.5])


def frame_hits(model: str, by, views, cls: str = "screw"):
    """``{frame key: (n_gt, n_hit)}`` for the held-out frames of ``views``."""
    out = {}
    for fold in ("hold13", "hold24", "hold33"):
        dets = json.loads((env.OUT / model / "detections" / f"{fold}.json").read_text("utf-8"))
        for key, rows in dets.items():
            d, view, step = key.split(":")
            if view not in views:
                continue
            g = [b.box for b in by.get((int(d), view, int(step)), []) if b.cls == cls]
            if not g:
                continue
            c = [(0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])) for r in rows
                 if r[4] == cls and r[5] >= LOW]
            hit = sum(1 for (x0, y0, x1, y1) in g
                      if any(x0 <= cx < x1 and y0 <= cy < y1 for cx, cy in c))
            out[key] = (len(g), hit)
    return out


def main(argv=None) -> int:
    a, b = (argv if argv is not None else sys.argv[1:])[:2]
    env.bind_l2(a)
    from experiments.l2_detector import data as D

    by = D.index(D.load_boxes())
    ea = pd.read_csv(env.OUT / a / "guess_events.csv")
    eb = pd.read_csv(env.OUT / b / "guess_events.csv")
    assert list(ea.ev) == list(eb.ev)
    lines = [f"# Paired: {b} minus {a}", "", "## Guess top-1 (points, 95 % CI, wins / losses)", "",
             "| events | n | " + a + " | " + b + " | B - A | CI | wins / losses |",
             "|---|---|---|---|---|---|---|"]
    mb = ea.target.astype(str).str.startswith("screw.motherboard")
    sets = [(f"{v} < 40 px", (ea["view"] == v) & (ea.bucketB == "<40"))
            for v in ("scan", "oak1", "oak2", "rs")]
    sets += [("scan + oak1 < 40 px", ea["view"].isin(["scan", "oak1"]) & (ea.bucketB == "<40")),
             ("scan + oak1 motherboard screws", ea["view"].isin(["scan", "oak1"]) & mb),
             ("all < 40 px", ea.bucketB == "<40")]
    for name, sel in sets:
        x, y = ea[sel].L2small_in1.to_numpy(float), eb[sel].L2small_in1.to_numpy(float)
        m, lo, hi = boot(y - x)
        lines.append(f"| {name} | {len(x)} | {100 * x.mean():.1f} | {100 * y.mean():.1f} | "
                     f"{100 * m:+.1f} | {100 * lo:+.1f} .. {100 * hi:+.1f} | "
                     f"{int(((y == 1) & (x == 0)).sum())} / {int(((y == 0) & (x == 1)).sum())} |")
    lines += ["", "## Screw hit@0.10 on held-out frames (bootstrap over frames)", "",
              "| views | frames | drafts | " + a + " | " + b + " | B - A | CI |",
              "|---|---|---|---|---|---|---|"]
    for views in (("scan",), ("oak1",), ("scan", "oak1"), ("oak2",), ("rs",)):
        ha, hb = frame_hits(a, by, views), frame_hits(b, by, views)
        keys = sorted(set(ha) & set(hb))
        n = np.array([ha[k][0] for k in keys], float)
        xa = np.array([ha[k][1] for k in keys], float)
        xb = np.array([hb[k][1] for k in keys], float)
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(10000):
            i = rng.integers(0, len(keys), len(keys))
            diffs.append((xb[i].sum() - xa[i].sum()) / n[i].sum())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        lines.append(f"| {' + '.join(views)} | {len(keys)} | {int(n.sum())} | "
                     f"{100 * xa.sum() / n.sum():.1f} | {100 * xb.sum() / n.sum():.1f} | "
                     f"{100 * (xb.sum() - xa.sum()) / n.sum():+.1f} | {100 * lo:+.1f} .. {100 * hi:+.1f} |")
    txt = "\n".join(lines) + "\n"
    (env.OUT / f"pairwise_{a}_vs_{b}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
