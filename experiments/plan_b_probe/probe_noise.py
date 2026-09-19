"""Scratch probe: what does a static camera look like, numerically?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import common as C
import tablemask as T
import estimate as E

ORIG = {"oak1": 1280, "oak2": 1280, "rs": 1280, "scan": 1600}


def feats(f, view, W=640):
    img = C.imread(f.image_path(), W)
    if img is None:
        return None
    kp, des, safe, yel = T.detect(img, view, 800)
    return np.float32([k.pt for k in kp]), des, img


if __name__ == "__main__":
    for view in ("oak1", "scan", "oak2", "rs"):
        fr = [f for f in C.load_frames(view, [30]) if f.missing == 0][:10]
        up = ORIG[view] / 640.0
        F = [feats(f, view) for f in fr]
        print("==", view)
        for i in range(len(F) - 1):
            r = E.estimate(F[i][0], F[i][1], F[i + 1][0], F[i + 1][1], up)
            print("  step %d->%d" % (fr[i].step, fr[i + 1].step), "ok", r["ok"],
                  "m", r["n_match"], "in", r["n_inlier"],
                  "px", None if r["px"] is None else round(r["px"], 2),
                  "rot", None if r["rot_deg"] is None else round(r["rot_deg"], 3),
                  r["reason"])
