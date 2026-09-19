"""Run the tape-overlap check over flagged events and a matched control set."""
import os
import pickle
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import verify as VF       # noqa: E402
import validate as V      # noqa: E402

FLAG_PX = 2.0


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


def run(view, rows, tag):
    fs = V.frames(view)
    out = []
    for e in rows:
        d, s0, s1 = e["desktop"], e["step_from"], e["step_to"]
        if (d, s0) not in fs or (d, s1) not in fs:
            continue
        S = VF.transform_for(view, d, s0, s1)
        if S is None:
            continue
        r = VF.check(fs[(d, s0)].image_path(), fs[(d, s1)].image_path(), view, S)
        if not r["usable"]:
            continue
        r.update({"view": view, "desktop": d, "step_from": s0, "step_to": s1,
                  "table_px": e["table_px"], "tag": tag})
        out.append(r)
    return out


def main():
    rnd = random.Random(11)
    allr = []
    for v in C.VIEWS:
        d = load(v)
        flags = [e for e in d["events"] if e["ok"] and e["table_px"] >= FLAG_PX]
        quiet = [e for e in d["events"] if e["ok"] and e["table_px"] < 0.5]
        allr += run(v, flags, "flag")
        allr += run(v, rnd.sample(quiet, min(25, len(quiet))), "still")
    with open(os.path.join(C.TMP, "verify.pkl"), "wb") as f:
        pickle.dump(allr, f)

    print("%-5s %-6s %5s %8s %9s %9s %8s" %
          ("view", "tag", "d", "steps", "table_px", "iou_before", "iou_after"))
    for r in sorted(allr, key=lambda r: (r["tag"], r["view"], r["desktop"])):
        print("%-5s %-6s %5d %4d->%-3d %8.2f %9.3f %9.3f" % (
            r["view"], r["tag"], r["desktop"], r["step_from"], r["step_to"],
            r["table_px"], r["iou_before"], r["iou_after"]))
    for tag in ("flag", "still"):
        s = [r for r in allr if r["tag"] == tag]
        if not s:
            continue
        gain = np.array([r["iou_after"] - r["iou_before"] for r in s])
        print("\n%s: n=%d  median iou_before=%.3f iou_after=%.3f gain=%.3f  "
              "gain>0.02 in %d/%d" % (
                  tag, len(s),
                  float(np.median([r["iou_before"] for r in s])),
                  float(np.median([r["iou_after"] for r in s])),
                  float(np.median(gain)), int((gain > 0.02).sum()), len(s)))


if __name__ == "__main__":
    main()
