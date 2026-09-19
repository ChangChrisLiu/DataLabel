"""Dump the adjudicated boundaries and the surviving step events."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C   # noqa: E402
import decide as D   # noqa: E402

if __name__ == "__main__":
    with open(os.path.join(C.TMP, "decided.pkl"), "rb") as f:
        dec = pickle.load(f)
    for v in C.VIEWS:
        print("== %s boundaries (candidates and failures)" % v)
        for b in dec["bounds"]:
            if b["view"] != v:
                continue
            if b["verdict"] == "static":
                continue
            print("   d%02d->d%02d %-13s px=%-8s iou %-6s->%-6s gain=%-7s fb=%s" % (
                b["desktop_from"], b["desktop_to"], b["verdict"],
                None if b["px"] is None else round(b["px"], 2),
                None if b["iou_before"] is None else round(b["iou_before"], 3),
                None if b["iou_after"] is None else round(b["iou_after"], 3),
                None if b["iou_gain"] is None else round(b["iou_gain"], 3),
                None if b.get("fallback_px") is None else round(b["fallback_px"], 1)))
        print("   -- step events (non-static)")
        for e in dec["events"]:
            if e["view"] != v or e["verdict"] in ("static",):
                continue
            if e["verdict"] == "rejected" and (e["table_px"] or 0) < 2.0:
                continue
            print("   d%02d %3d->%-3d %-13s px=%-8s iou %-6s->%-6s gain=%s" % (
                e["desktop"], e["step_from"], e["step_to"], e["verdict"],
                None if e["table_px"] is None else round(e["table_px"], 2),
                None if e["iou_before"] is None else round(e["iou_before"], 3),
                None if e["iou_after"] is None else round(e["iou_after"], 3),
                None if e["iou_gain"] is None else round(e["iou_gain"], 3)))
        print()
