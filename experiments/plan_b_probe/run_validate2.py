"""Render the validation set driven by the final decisions.

Four groups, so the report can state precision on flags *and* show that the
rejections and the quiet pairs were checked too:
  conf  - reported as a camera move
  rej   - ORB proposed a move, the tape test threw it out
  undet - could not be resolved either way
  still - quiet pairs, biased towards steps where the chassis changed a lot
"""
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C        # noqa: E402
import validate as V      # noqa: E402
import decide as D        # noqa: E402


def main():
    with open(os.path.join(C.TMP, "decided.pkl"), "rb") as f:
        dec = pickle.load(f)
    os.makedirs(C.OVERLAYS, exist_ok=True)
    for f in os.listdir(C.OVERLAYS):
        os.remove(os.path.join(C.OVERLAYS, f))

    rnd = random.Random(23)
    n = 0
    for b in dec["bounds"]:
        v, d0, d1 = b["view"], b["desktop_from"], b["desktop_to"]
        if b["verdict"] == "confirmed" and (b["px"] or 0) >= D.REPORT_PX:
            tag, kind = "conf", "camera move"
        elif b["verdict"] == "undetermined":
            tag, kind = "undet", "undetermined"
        elif b["verdict"] == "rejected" and (b["px"] or 0) >= 2.0:
            tag, kind = "rej", "rejected by tape test"
        else:
            continue
        if V.render_boundary(v, d0, d1, tag, b["px"], kind):
            n += 1

    for e in dec["events"]:
        v, d = e["view"], e["desktop"]
        if e["kind"] in ("camera", "both"):
            tag, kind = "conf", e["kind"]
        elif e["verdict"] == "undetermined":
            tag, kind = "undet", "undetermined"
        elif e["verdict"] == "rejected" and (e["table_px"] or 0) >= 2.0:
            tag, kind = "rej", "rejected by tape test"
        else:
            continue
        if V.render_step_pair(v, d, e["step_from"], e["step_to"], tag,
                              e["table_px"], kind):
            n += 1

    for v in C.VIEWS:
        cands = [e for e in dec["events"]
                 if e["view"] == v and e["verdict"] == "static"
                 and e["chassis_px"] is not None]
        cands.sort(key=lambda e: -(e["chassis_px"] or 0))
        pick = cands[:2] + rnd.sample(cands[2:], min(2, max(0, len(cands) - 2)))
        for e in pick:
            if V.render_step_pair(v, e["desktop"], e["step_from"], e["step_to"],
                                  "still", e["table_px"], "non-event"):
                n += 1

    sizes = sorted(((os.path.getsize(os.path.join(C.OVERLAYS, f)), f)
                    for f in os.listdir(C.OVERLAYS)), reverse=True)
    print("rendered %d overlays; largest %.2f MB (%s)"
          % (n, sizes[0][0] / 1e6, sizes[0][1]))


if __name__ == "__main__":
    main()
