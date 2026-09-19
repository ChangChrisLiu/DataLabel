"""List the candidate events and boundaries with their identities."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0


def load(view):
    with open(os.path.join(C.TMP, "raw_%s.pkl" % view), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    for v in C.VIEWS:
        d = load(v)
        print("== %s step-pairs >= %.1f px" % (v, T))
        for e in d["events"]:
            if e["ok"] and e["table_px"] >= T:
                print("   d%02d %3d->%3d  table=%7.2f rot=%7.3f scale=%.4f "
                      "inl=%4d/%4d chassis=%s area=%s" % (
                          e["desktop"], e["step_from"], e["step_to"],
                          e["table_px"], e["rot_deg"], e["scale"],
                          e["n_inlier"], e["n_match"],
                          "  n/a" if e["chassis_px"] is None else "%6.1f" % e["chassis_px"],
                          "n/a" if e["chassis_area_ratio"] is None
                          else "%.2f" % e["chassis_area_ratio"]))
        print("   undetermined step-pairs:")
        for e in d["events"]:
            if not e["ok"]:
                print("      d%02d %3d->%3d  %s (m=%d in=%d)" % (
                    e["desktop"], e["step_from"], e["step_to"], e["reason"],
                    e["n_match"], e["n_inlier"]))
        print("   boundaries >= %.1f px or undetermined:" % T)
        for b in d["bounds"]:
            if not b["ok"] or b["px"] >= T:
                print("      d%02d->d%02d px=%s rot=%s scale=%s pairs=%d %s" % (
                    b["desktop_from"], b["desktop_to"],
                    None if b["px"] is None else round(b["px"], 2),
                    None if b["rot_deg"] is None else round(b["rot_deg"], 3),
                    None if b["scale"] is None else round(b["scale"], 4),
                    b["n_pairs"], b["reason"]))
        print()
