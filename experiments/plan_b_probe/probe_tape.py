"""Try the tape-ECC tier on the cases SIFT still could not resolve."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import tape_ecc as TE     # noqa: E402
import validate as V      # noqa: E402
from extract import ORIG  # noqa: E402

if __name__ == "__main__":
    with open(os.path.join(C.TMP, "bigmove.pkl"), "rb") as f:
        big = pickle.load(f)
    res = {}
    for key, r in sorted(big.items(), key=lambda kv: str(kv[0])):
        if r["ok"]:
            continue
        v = key[1]
        fs = V.frames(v)
        if key[0] == "bound":
            d0, d1 = key[2], key[3]
            sa, sb = V.valid_steps(v, d0), V.valid_steps(v, d1)
            if not sa or not sb:
                continue
            pa, pb = fs[(d0, sa[-1])].image_path(), fs[(d1, sb[0])].image_path()
            label = "bound %s d%02d->d%02d" % (v, d0, d1)
        else:
            d, s0, s1 = key[2], key[3], key[4]
            if (d, s0) not in fs or (d, s1) not in fs:
                continue
            pa, pb = fs[(d, s0)].image_path(), fs[(d, s1)].image_path()
            label = "step  %s d%02d %3d->%3d" % (v, d, s0, s1)
        t = TE.estimate_tape(pa, pb, v, ORIG[v])
        res[key] = t
        print("%-28s ok=%-5s px=%-8s rot=%-7s scale=%-7s cc=%-5s iou=%s(from %s) %s" % (
            label, t["ok"],
            None if t["px"] is None else round(t["px"], 1),
            None if t["rot_deg"] is None else round(t["rot_deg"], 2),
            None if t["scale"] is None else round(t["scale"], 3),
            None if t["cc"] is None else round(t["cc"], 2),
            None if t["iou"] is None else round(t["iou"], 2),
            None if t["iou0"] is None else round(t["iou0"], 2),
            t["reason"]), flush=True)
    with open(os.path.join(C.TMP, "tape.pkl"), "wb") as f:
        pickle.dump(res, f)
