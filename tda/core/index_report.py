"""Markdown summary of a built frame index (see :mod:`tda.core.index`)."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from tda.core.model import VIEWS

_HEADER = (
    "| desktop | n_steps | frames | missing scan | missing oak1 | "
    "missing oak2 | missing rs | issues |"
)


def compact(steps: list[int]) -> str:
    """``[1, 2, 3, 7]`` -> ``"1-3, 7 (n=4)"``."""
    out: list[str] = []
    start = prev = None
    for s in steps:
        if start is None:
            start = prev = s
        elif s == prev + 1:
            prev = s
        else:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = prev = s
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"{', '.join(out)} (n={len(steps)})"


def write_report(idx: dict[int, Any], path: str) -> None:
    """Write a per-desktop markdown report: step counts, missing frames, issues."""
    n_steps = sum(di.n_steps for di in idx.values())
    n_frames = sum(len(di.frames) for di in idx.values())
    lines = [
        "# Frame index report",
        "",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- desktops: {len(idx)}",
        f"- logical steps: {n_steps}",
        f"- frames: {n_frames}",
        "",
        "## Per desktop",
        "",
        _HEADER,
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {v: 0 for v in VIEWS}
    for d, di in sorted(idx.items()):
        per = {v: sum(1 for k in di.missing if k.view == v) for v in VIEWS}
        for v in VIEWS:
            totals[v] += per[v]
        lines.append(
            f"| {d} | {di.n_steps} | {len(di.frames)} | {per['scan']} | {per['oak1']} | "
            f"{per['oak2']} | {per['rs']} | {len(di.issues)} |"
        )
    lines.append(
        f"| **total** | {n_steps} | {n_frames} | {totals['scan']} | {totals['oak1']} | "
        f"{totals['oak2']} | {totals['rs']} | "
        f"{sum(len(di.issues) for di in idx.values())} |"
    )

    lines += ["", "## Missing frames", ""]
    for d, di in sorted(idx.items()):
        if not di.missing:
            continue
        lines.append(f"- **Desktop {d}**")
        for view in VIEWS:
            steps = sorted(k.step for k in di.missing if k.view == view)
            if steps:
                lines.append(f"  - `{view}`: {compact(steps)}")

    lines += ["", "## Issues and applied fixes", ""]
    for d, di in sorted(idx.items()):
        if not di.issues:
            continue
        lines.append(f"- **Desktop {d}**")
        lines += [f"  - {msg}" for msg in di.issues]
    lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
