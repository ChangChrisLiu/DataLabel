"""Q1 first look: between-desktop table motion per view, from the cache."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C     # noqa: E402
import analyze as A    # noqa: E402

if __name__ == "__main__":
    views = sys.argv[1:] or list(C.VIEWS)
    for v in views:
        print("==", v)
        for d in range(1, 66):
            r = A.boundary_estimate(v, d, d + 1)
            flag = ""
            if not r["ok"]:
                flag = "  <== UNDETERMINED"
            elif r["px"] >= A.MOVE_PX:
                flag = "  <== MOVE"
            if flag or (r["px"] or 0) > 0.5:
                print("  d%02d->d%02d px=%s rot=%s scale=%s pairs=%d inl=%d %s%s" % (
                    d, d + 1,
                    None if r["px"] is None else round(r["px"], 2),
                    None if r["rot_deg"] is None else round(r["rot_deg"], 3),
                    None if r["scale"] is None else round(r["scale"], 4),
                    r["n_pairs"], r["n_inlier"], r["reason"], flag), flush=True)
