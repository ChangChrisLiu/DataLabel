"""Tables and the gate from ``guess_events.csv`` (+ ``det_metrics.csv``).

::

    D:\\Anaconda\\envs\\tda\\python.exe -m experiments.l2_detector.analyse
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l2_detector import env  # noqa: E402
from experiments.l2_detector.data import md_table  # noqa: E402

VIEWS = ["scan", "oak1", "oak2", "rs"]
BUCKETS = ["<40", "40-100", "100-250", ">250"]


def pct(s) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{100 * s.mean():.1f}" if len(s) else "-"


def grid(d: pd.DataFrame, methods: dict[str, str], k: str = "in1") -> pd.DataFrame:
    rows = []
    for v in VIEWS + ["all"]:
        sel = d if v == "all" else d[d["view"] == v]
        for label, col in methods.items():
            row = {"view": v, "method": label}
            for b in BUCKETS:
                s = sel[sel.bucketB == b][f"{col}_{k}" if not col.startswith("ceil") else col]
                row[b] = f"{pct(s)} ({len(s)})" if len(s) else "-"
            s = sel[f"{col}_{k}" if not col.startswith("ceil") else col]
            row["all"] = f"{pct(s)} ({len(s)})"
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    d = pd.read_csv(env.OUT / "guess_events.csv")
    out = ["# L2 guess evaluation (GT-B, held-out desktops)", ""]
    # reproduction check against L1
    rep = (d["M0app_in1"] == d["L1_M0app_in1"]).mean()
    out.append(f"M0app recomputed here vs L1's per-event M0app (GT-B top-1): "
               f"{100 * rep:.1f} % identical over {len(d)} events.")
    out.append("")
    top1 = {"M0app": "M0app", "L1 M3js_rr10|M0": "L1_M3js", "L2small": "L2small",
            "L2all": "L2all"}
    out += ["## Point-in-part top-1 % (n)", "", md_table(grid(d, top1, "in1")), ""]
    out += ["## Any of the top 3 % (n)", "", md_table(grid(d, top1, "in3")), ""]
    ceil = {"ceiling conf>=0.10 (after skip)": "ceilsup_c10",
            "ceiling conf>=0.10 (before skip)": "ceil_c10",
            "ceiling conf>=0.01 (after skip)": "ceilsup_c01"}
    dd = d[d.cls.isin(["screw", "connector", "ram_latch", "psu_latch",
                       "cpu_socket_lever", "drive_latch"])]
    out += ["## Ceiling: removed part among the detector's candidates in frame j "
            "(small classes only) % (n)", "", md_table(grid(dd, ceil, "in1")), ""]
    var = {"L2small": "L2small", "no prior": "L2small_nop", "no skip": "L2small_nosup",
           "conf 0.05": "L2small_c05", "conf 0.25": "L2small_c25"}
    out += ["## Sensitivity, top-1 % (n)", "", md_table(grid(d, var, "in1")), ""]

    # small-class events only: how often the detector armed, and its precision
    s = dd.copy()
    s["det_armed"] = s["L2small_src"] == "det"
    rows = []
    for v in VIEWS + ["all"]:
        x = s if v == "all" else s[s["view"] == v]
        armed = x[x.det_armed]
        rows.append({
            "view": v, "small-class events": len(x),
            "detector armed %": pct(x.det_armed.astype(int)),
            "hit when armed %": pct(armed.L2small_in1),
            "M0app on the same events %": pct(armed.M0app_in1),
            "candidates after skip (median)": int(np.median(armed.L2small_n)) if len(armed) else "-",
            "true part rank 1 / 2-3 / >3 / absent": (
                f"{(armed.L2small_true_rank == 1).sum()} / "
                f"{armed.L2small_true_rank.between(2, 3).sum()} / "
                f"{(armed.L2small_true_rank > 3).sum()} / "
                f"{(armed.L2small_true_rank == 0).sum()}") if len(armed) else "-",
        })
    out += ["## Small-class events: when the detector arms", "", md_table(pd.DataFrame(rows)), ""]

    # per desktop, < 40 px
    rows = []
    for dk in (13, 24, 33):
        for v in VIEWS:
            x = d[(d.desktop == dk) & (d["view"] == v) & (d.bucketB == "<40")]
            if len(x) == 0:
                continue
            rows.append({"desktop": dk, "view": v, "n": len(x),
                         "M0app": pct(x.M0app_in1), "L1 M3js": pct(x.L1_M3js_in1),
                         "L2small": pct(x.L2small_in1), "L2small top-3": pct(x.L2small_in3),
                         "ceiling": pct(x.ceilsup_c10)})
    out += ["## Parts < 40 px per held-out desktop, top-1 %", "",
            md_table(pd.DataFrame(rows)), ""]

    # by what was removed (screw role / connector), small-class events
    s = dd.copy()
    s["what"] = np.where(s.cls == "screw", s.target.str.split(".").str[:2].str.join("."),
                         s.cls)
    rows = []
    for (v, w), x in s.groupby([s["view"], "what"]):
        rows.append({"view": v, "removed": w, "n": len(x), "M0app": pct(x.M0app_in1),
                     "L2small": pct(x.L2small_in1), "ceiling": pct(x.ceilsup_c10)})
    out += ["## By what was removed, top-1 %", "", md_table(pd.DataFrame(rows)), ""]

    # gate, with a paired bootstrap over events
    rng = np.random.default_rng(0)
    g = []
    for v in ("scan", "oak1"):
        x = d[(d["view"] == v) & (d.bucketB == "<40")]
        a, b = 100 * x.M0app_in1.mean(), 100 * x.L2small_in1.mean()
        diff = (x.L2small_in1 - x.M0app_in1).to_numpy(float)
        boots = [100 * diff[rng.integers(0, len(diff), len(diff))].mean()
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        g.append(f"- < 40 px, {v}: M0app {a:.1f} -> L2small {b:.1f} "
                 f"(**{b - a:+.1f}** points, 95 % bootstrap CI {lo:+.1f} .. {hi:+.1f}, "
                 f"n={len(x)}; gate +20)")
    worst = []
    for v in VIEWS + ["all"]:
        for bk in ("100-250", ">250"):
            x = d[(d.bucketB == bk) & ((d["view"] == v) if v != "all" else True)]
            if len(x) == 0:
                continue
            delta = 100 * (x.L2small_in1.mean() - x.M0app_in1.mean())
            worst.append((delta, v, bk, len(x)))
    worst.sort()
    g.append("- >= 100 px buckets, worst change: " + ", ".join(
        f"{v} {bk} {dl:+.1f} (n={n})" for dl, v, bk, n in worst[:3]))
    out += ["## Gate", ""] + g + [""]

    # D13 scan 38 / 39 (frames 37 / 38)
    x = d[(d.desktop == 13) & (d["view"] == "scan") & (d.step.isin([38, 39]))]
    cols = ["ev", "frame_j", "cls", "partB_width", "gtB_box", "M0app_in1", "M0app_p1",
            "L2small_in1", "L2small_in3", "L2small_src", "L2small_p1", "L2small_dist1",
            "L2small_n", "L2small_true_rank", "ceil_c10", "ceilsup_c10", "L2small_top"]
    out += ["## D13 / scan frames 37 and 38", "", md_table(x[cols]), ""]
    txt = "\n".join(out)
    (env.OUT / "guess_tables.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
