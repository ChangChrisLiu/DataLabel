"""Are the calibration captures taken inside their desktop's capture window?

If they are not, the board tells you about the rig at calibration time, not
about the rig during the teardown -- which changes how Q3 may be read.
"""
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C   # noqa: E402
import board as B    # noqa: E402


def cal_ts(desktop, cam=1):
    jpg, _al = B.oak_calib_capture(desktop, cam)
    if not jpg:
        return None
    m = re.match(r"(\d{8})_(\d{6})_(\d{3})", os.path.basename(jpg))
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


if __name__ == "__main__":
    con = C.connect()
    print("%-5s %-21s %-21s %-21s %s" %
          ("desk", "seq_start", "seq_end", "calibration", "inside?"))
    for d in B.oak_calib_desktops():
        row = con.execute("select min(ts), max(ts) from frame where view='oak1' "
                          "and desktop=? and missing=0", (d,)).fetchone()
        if not row or not row[0]:
            continue
        s = datetime.fromisoformat(row[0])
        e = datetime.fromisoformat(row[1])
        c = cal_ts(d)
        if c is None:
            continue
        inside = "IN" if s <= c <= e else ("before %.0f min" % ((s - c).total_seconds() / 60)
                                           if c < s else
                                           "after %.0f min" % ((c - e).total_seconds() / 60))
        print("%-5d %-21s %-21s %-21s %s" % (d, s, e, c, inside))
    con.close()
