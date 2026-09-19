"""Scratch probe: does the table match across desktops / capture days?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common as C
import tablemask as T
import estimate as E
from probe_noise import feats, ORIG

if __name__ == "__main__":
    view = sys.argv[1] if len(sys.argv) > 1 else "oak1"
    up = ORIG[view] / 640.0
    reps = []
    for d in range(1, 67):
        fr = [f for f in C.load_frames(view, [d]) if f.missing == 0]
        if not fr:
            reps.append((d, None, None))
            continue
        a = feats(fr[0], view)
        b = feats(fr[-1], view)
        reps.append((d, a, b, fr[0].step, fr[-1].step))
    print("view", view, "-- last frame of desktop k  ->  first frame of desktop k+1")
    for i in range(len(reps) - 1):
        d0 = reps[i]
        d1 = reps[i + 1]
        if len(d0) < 4 or len(d1) < 4 or d0[2] is None or d1[1] is None:
            print(" %2d -> %2d  UNDETERMINED (no frames)" % (d0[0], d1[0]))
            continue
        r = E.estimate(d0[2][0], d0[2][1], d1[1][0], d1[1][1], up)
        print(" %2d -> %2d  ok=%s m=%4d in=%4d px=%s rot=%s scale=%s %s" % (
            d0[0], d1[0], r["ok"], r["n_match"], r["n_inlier"],
            None if r["px"] is None else round(r["px"], 2),
            None if r["rot_deg"] is None else round(r["rot_deg"], 3),
            None if r["scale"] is None else round(r["scale"], 4), r["reason"]))
