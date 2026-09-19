"""Render the validation overlays for a chosen list of pairs.

Nothing in the report may claim a visual confirmation that was not actually
looked at, so this only renders; the verdicts are recorded by hand afterwards.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C       # noqa: E402
import overlays as O     # noqa: E402

_cache: dict = {}


def frames(view):
    if view not in _cache:
        _cache[view] = {(f.desktop, f.step): f
                        for f in C.load_frames(view) if not f.missing and f.path}
    return _cache[view]


def valid_steps(view, desktop):
    fs = frames(view)
    return sorted(s for (d, s) in fs if d == desktop)


def render_step_pair(view, desktop, sa, sb, tag, px, kind):
    fs = frames(view)
    a, b = fs.get((desktop, sa)), fs.get((desktop, sb))
    if not a or not b:
        return None
    name = "%s_%s_d%02d_s%03d-%03d.png" % (tag, view, desktop, sa, sb)
    return O.make(
        a.image_path(), b.image_path(), view,
        "%s  %s  desktop %d  step %d -> %d   table motion = %s px  [%s]"
        % (tag.upper(), view, desktop, sa, sb,
           "n/a" if px is None else "%.2f" % px, kind),
        ("d%d step %d" % (desktop, sa), "d%d step %d" % (desktop, sb),
         "table should %s" % ("MOVE" if kind == "flag" else "stay still")),
        os.path.join(C.OVERLAYS, name))


def render_boundary(view, d0, d1, tag, px, kind):
    fs = frames(view)
    sa = valid_steps(view, d0)
    sb = valid_steps(view, d1)
    if not sa or not sb:
        return None
    a, b = fs[(d0, sa[-1])], fs[(d1, sb[0])]
    name = "%s_%s_d%02d-d%02d.png" % (tag, view, d0, d1)
    return O.make(
        a.image_path(), b.image_path(), view,
        "%s  %s  desktop %d (last) -> desktop %d (first)   table motion = %s px  [%s]"
        % (tag.upper(), view, d0, d1, "n/a" if px is None else "%.2f" % px, kind),
        ("d%d step %d" % (d0, sa[-1]), "d%d step %d" % (d1, sb[0]),
         "table should %s" % ("MOVE" if kind == "flag" else "stay still")),
        os.path.join(C.OVERLAYS, name))
