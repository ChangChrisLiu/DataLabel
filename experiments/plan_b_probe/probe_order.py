"""Is 'order by capture timestamp' the same as 'order by desktop number'?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

if __name__ == "__main__":
    con = C.connect()
    for view in C.VIEWS:
        rows = con.execute(
            "select desktop, min(ts) from frame where view=? and missing=0 "
            "group by desktop order by min(ts)", (view,)).fetchall()
        order = [r[0] for r in rows]
        print("%-5s first=%s last=%s  monotonic_in_desktop_number=%s" % (
            view, rows[0][1], rows[-1][1], order == sorted(order)))
        if order != sorted(order):
            print("      order:", order)
    con.close()
