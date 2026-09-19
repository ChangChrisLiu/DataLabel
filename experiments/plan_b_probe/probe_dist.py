"""Look at the distribution of measured motion, to pick thresholds honestly."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    for v in C.VIEWS:
        d = load(v)
        ev = [e for e in d["events"] if e["ok"]]
        bad = [e for e in d["events"] if not e["ok"]]
        px = np.array([e["table_px"] for e in ev])
        print("== %s  step-pairs ok=%d undetermined=%d" % (v, len(ev), len(bad)))
        qs = [50, 90, 99, 99.5, 99.9, 100]
        print("   table_px percentiles:",
              "  ".join("p%s=%.2f" % (q, np.percentile(px, q)) for q in qs))
        for t in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
            print("      >= %4.1f px: %4d pairs" % (t, int((px >= t).sum())))
        reasons = {}
        for e in bad:
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
        print("   undetermined reasons:", reasons)
        b = [x for x in d["bounds"] if x["ok"]]
        bp = sorted(x["px"] for x in b)
        print("   boundaries ok=%d undetermined=%d" % (len(b), 65 - len(b)))
        print("   boundary px sorted:", " ".join("%.2f" % x for x in bp))
        print()
