"""Resolution and verification of the declared anomaly fixes.

The rules live in ``configs/index_fixes.yaml`` (spec 2.2).  OAK ``Camera_2`` is
normally paired to ``Camera_1`` by identical filename timestamp, so a declared
fix serves two purposes: it is the *fallback* when no timestamp partner exists,
and it is an *assertion* on the pairing the timestamps produced.  Every declared
fix therefore emits a line into ``DesktopIndex.issues`` - a confirmation naming
the resolved logical step, or a ``MISMATCH``.  Nothing is applied silently.

Folder-name vocabulary: ``from_dir`` / ``to_cam1_dir`` / the ``cam2_swap`` pairs
are Camera_*folder* names (``"004"``), never logical step numbers.  A folder maps
to a logical step through ``step_of_dir``, which already accounts for
``cam1_reorder_by_ts``.
"""
from __future__ import annotations

from typing import Optional

from tda.core.model import FrameKey


def cam2_fix_step(
    oak_fix: dict, src_dir: str, step_of_dir: dict[str, int]
) -> Optional[int]:
    """Logical step a Camera_2 folder is declared to hold, or ``None``.

    Consulted only when timestamp pairing found no Camera_1 partner.
    """
    shift = oak_fix.get("cam2_shift") or {}
    if shift.get("from_dir") == src_dir:
        return step_of_dir.get(str(shift["to_cam1_dir"]))
    for pair in oak_fix.get("cam2_swap") or []:
        if src_dir == pair[0]:
            return step_of_dir.get(pair[1])
        if src_dir == pair[1]:
            return step_of_dir.get(pair[0])
    if oak_fix.get("cam2_extra_from") == src_dir and "as_step" in oak_fix:
        return step_of_dir.get(f"{int(oak_fix['as_step']):03d}")
    return None


def verify_cam2_fixes(
    desktop: int,
    oak_fix: dict,
    step_of_dir: dict[str, int],
    assigned: dict[str, list[int]],
) -> list[str]:
    """Check the resulting pairing against every declared Camera_2 fix."""
    issues: list[str] = []

    def check(label: str, src_dir: str, want_dir: str) -> None:
        want = step_of_dir.get(want_dir)
        got = assigned.get(src_dir, [])
        head = f"fix:oak[{desktop}] {label}"
        if want is not None and want in got:
            issues.append(
                f"{head}: cam2 {src_dir} -> cam1 dir {want_dir} = logical step {want}"
            )
        else:
            issues.append(
                f"{head} MISMATCH: cam2 {src_dir} expected -> cam1 dir {want_dir} = "
                f"logical step {want}, got steps {got}"
            )

    shift = oak_fix.get("cam2_shift") or {}
    if shift:
        check("cam2_shift", shift["from_dir"], str(shift["to_cam1_dir"]))
    for pair in oak_fix.get("cam2_swap") or []:
        check(f"cam2_swap {pair[0]}<->{pair[1]}", pair[0], pair[1])
        check(f"cam2_swap {pair[0]}<->{pair[1]}", pair[1], pair[0])
    extra = oak_fix.get("cam2_extra_from")
    if extra and "as_step" in oak_fix:
        check("cam2_extra_from", extra, f"{int(oak_fix['as_step']):03d}")
    return issues


def verify_declared_missing(
    desktop: int, fixes: dict, step_of_dir: dict[str, int], missing: list[FrameKey]
) -> list[str]:
    """Check the ``scan``/``rs`` gaps declared in the yaml against the real gaps.

    ``missing_steps`` / ``missing_steps_range`` are *nominal* step numbers, so
    each is resolved through its Camera_1 folder first.  A declared-missing frame
    that turns out to exist is a ``MISMATCH`` - the yaml is then stale.
    """
    issues: list[str] = []
    for view in ("scan", "rs"):
        entry = (fixes.get(view) or {}).get(desktop)
        if not isinstance(entry, dict):
            continue
        nominal = set(entry.get("missing_steps") or [])
        rng = entry.get("missing_steps_range")
        if rng:
            nominal |= set(range(int(rng[0]), int(rng[1]) + 1))
        if not nominal:
            continue
        resolved = {n: step_of_dir.get(f"{n:03d}") for n in sorted(nominal)}
        declared = {s for s in resolved.values() if s is not None}
        unresolved = [n for n, s in resolved.items() if s is None]
        actual = {k.step for k in missing if k.view == view}
        present = sorted(declared - actual)
        if present:
            issues.append(
                f"fix:{view}[{desktop}] MISMATCH: declared-missing logical steps "
                f"present in index: {present}"
            )
        else:
            note = (
                f" ({len(unresolved)} declared step(s) have no cam1 folder: {unresolved})"
                if unresolved
                else ""
            )
            issues.append(
                f"fix:{view}[{desktop}] declared missing steps "
                f"{[n for n, s in resolved.items() if s is not None]} confirmed absent"
                + note
            )
    return issues
