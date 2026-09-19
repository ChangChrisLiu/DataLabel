"""Q2: what does the stored pose_segment actually say, and does it agree?"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C   # noqa: E402

if __name__ == "__main__":
    con = C.connect()
    print("frame.pose_segment non-null:",
          con.execute("select count(*) from frame where pose_segment is not null")
          .fetchone()[0])
    print("\npose_segment table, per view:")
    for v in C.VIEWS:
        rows = con.execute(
            "select desktop, seg, start_step, end_step, ref_step, "
            "homography_json is not null, corners_json is not null "
            "from pose_segment where view=? order by desktop, seg", (v,)).fetchall()
        multi = {}
        for d, seg, s0, s1, rs, h, cj in rows:
            multi.setdefault(d, []).append((seg, s0, s1, rs, h, cj))
        n_multi = sum(1 for d, v2 in multi.items() if len(v2) > 1)
        print("  %-5s rows=%d desktops=%d desktops_with_more_than_one_segment=%d"
              % (v, len(rows), len(multi), n_multi))
        for d, segs in sorted(multi.items()):
            if len(segs) > 1:
                print("      desktop %d: %s" % (d, segs))
    print("\nsample rows (scan):")
    for r in con.execute("select desktop, seg, start_step, end_step, ref_step "
                         "from pose_segment where view='scan' order by desktop limit 8"):
        print("   ", r)
    con.close()

    p = os.path.join(C.TMP, "decided.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            dec = pickle.load(f)
        print("\nmy confirmed camera moves within a sequence:")
        for e in dec["events"]:
            if e["kind"] in ("camera", "both"):
                print("   %-5s desktop %d step %d->%d  %.2f px"
                      % (e["view"], e["desktop"], e["step_from"], e["step_to"],
                         e["table_px"]))
