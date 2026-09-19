"""Re-estimate every undetermined boundary and step pair with the SIFT fallback."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import bigmove as BM      # noqa: E402
import validate as V      # noqa: E402
from extract import ORIG  # noqa: E402


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


def main():
    out = {}
    for v in C.VIEWS:
        d = load(v)
        fs = V.frames(v)
        print("==", v, flush=True)
        for b in d["bounds"]:
            if b["ok"]:
                continue
            d0, d1 = b["desktop_from"], b["desktop_to"]
            sa, sb = V.valid_steps(v, d0), V.valid_steps(v, d1)
            if not sa or not sb:
                continue
            r = BM.estimate_big(fs[(d0, sa[-1])].image_path(),
                                fs[(d1, sb[0])].image_path(), v, ORIG[v])
            out[("bound", v, d0, d1)] = r
            print("   bound d%02d->d%02d  ok=%s px=%s rot=%s scale=%s in=%d/%d %s" % (
                d0, d1, r["ok"],
                None if r["px"] is None else round(r["px"], 1),
                None if r["rot_deg"] is None else round(r["rot_deg"], 2),
                None if r["scale"] is None else round(r["scale"], 4),
                r["n_inlier"], r["n_match"], r["reason"]), flush=True)
        for e in d["events"]:
            if e["ok"]:
                continue
            a, b2 = fs.get((e["desktop"], e["step_from"])), fs.get((e["desktop"], e["step_to"]))
            if not a or not b2:
                continue
            r = BM.estimate_big(a.image_path(), b2.image_path(), v, ORIG[v])
            out[("step", v, e["desktop"], e["step_from"], e["step_to"])] = r
            print("   step  d%02d %3d->%3d  ok=%s px=%s rot=%s scale=%s in=%d/%d %s" % (
                e["desktop"], e["step_from"], e["step_to"], r["ok"],
                None if r["px"] is None else round(r["px"], 1),
                None if r["rot_deg"] is None else round(r["rot_deg"], 2),
                None if r["scale"] is None else round(r["scale"], 4),
                r["n_inlier"], r["n_match"], r["reason"]), flush=True)
    with open(os.path.join(C.TMP, "bigmove.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("saved", len(out))


if __name__ == "__main__":
    main()
