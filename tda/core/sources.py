"""Layout of the four raw capture trees on the read-only F: drive.

This module knows *where the files are* and nothing about logical steps: it is
the only place that encodes folder and filename conventions.  Everything is
enumerated with :func:`os.scandir` at the known depths - never a recursive walk,
because the source drive is a slow external disk - and nothing here writes.

Layout (spec 2.1)::

    oak_root/Desktop <n>/<Disassemble>/Camera_<1|2>/<NNN>/<ts>_camera_<c>_<role>
    scanner_root/TAMU_B2.3_<a>-<b>_RGB/<subdir>/<n>/RGB<step>1/P_<k>.png
    rs_root/Desktop <n>/<Disassemble|Disassembly>/<NNN>/original_color.png
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Iterable, Optional

OAK_FILE_RE = re.compile(r"^(\d{8}_\d{6}_\d{3})_camera_(\d)_(.+)$")
STEP_DIR_RE = re.compile(r"^\d{3}$")
SCAN_GROUP_RE = re.compile(r"^TAMU_B2\.3_(\d+)-(\d+)_RGB$")
BURST_RE = re.compile(r"^P_(\d+)\.png$", re.IGNORECASE)

#: OAK filename suffix -> ``FrameFile.aux`` key ("main" goes to ``FrameFile.path``).
OAK_ROLES = {
    "rgb_12mp.jpg": "main",
    "rgb_aligned.png": "aligned",
    "depth_raw.npy": "depth_npy",
    "depth_raw.png": "depth_png",
    "pointcloud.ply": "pointcloud",
}
#: The disassembly section folder is spelled both ways across the dataset.
DEFAULT_FOLDER_ALIASES = ("Disassemble", "Disassembly")


def iso_from_oak_token(token: str) -> str:
    """``20250531_104736_629`` -> ``2025-05-31T10:47:36.629000``."""
    base = datetime.strptime(token[:15], "%Y%m%d_%H%M%S")
    return base.replace(microsecond=int(token[16:19]) * 1000).isoformat()


def iso_from_mtime(mtime: float) -> str:
    """File modification time as an ISO string (used where no capture time exists)."""
    return datetime.fromtimestamp(mtime).isoformat(timespec="milliseconds")


def norm(path: str) -> str:
    """Store every path with forward slashes so the JSON index is platform-neutral."""
    return path.replace("\\", "/")


def dirs(path: str) -> list[os.DirEntry]:
    """``os.scandir`` one level, directories only; an unreadable path yields []."""
    try:
        return [e for e in os.scandir(path) if e.is_dir()]
    except OSError:
        return []


def files(path: str) -> list[os.DirEntry]:
    """``os.scandir`` one level, files only; an unreadable path yields []."""
    try:
        return [e for e in os.scandir(path) if e.is_file()]
    except OSError:
        return []


def first_existing(base: str, names: Iterable[str]) -> Optional[str]:
    """First of ``names`` that exists as a sub-directory of ``base``."""
    for name in names:
        candidate = os.path.join(base, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def scan_oak_step_dir(path: str) -> dict[str, dict[str, str]]:
    """Group one OAK ``NNN`` folder's files by capture timestamp token.

    Returns ``{ts_token: {role: abs_path}}``; a folder holding two captures
    (D10 step 1, D60 cam2 folder 005) yields two entries.
    """
    out: dict[str, dict[str, str]] = {}
    for entry in files(path):
        m = OAK_FILE_RE.match(entry.name)
        if not m:
            continue
        role = OAK_ROLES.get(m.group(3))
        if role is None:
            continue
        out.setdefault(m.group(1), {})[role] = norm(entry.path)
    return out


def scan_oak_camera(cam_dir: str) -> dict[str, dict[str, dict[str, str]]]:
    """``{step_dir: {ts_token: {role: path}}}`` for one ``Camera_<n>`` folder."""
    return {
        e.name: scan_oak_step_dir(e.path)
        for e in dirs(cam_dir)
        if STEP_DIR_RE.match(e.name)
    }


def scanner_desktop_dir(scanner_root: str, desktop: int) -> Optional[str]:
    """Locate ``<group>/<subdir>/<desktop>`` for one desktop.

    The group folder is ``TAMU_B2.3_<a>-<b>_RGB`` whose range contains the
    desktop.  Inside it the payload sub-folder usually repeats the group name,
    but for 50-66 it is ``tamu_color_50-66_Bright2.3``; ``new``/``New`` staging
    folders and ``ext.py`` are ignored.
    """
    groups = []
    for entry in dirs(scanner_root):
        m = SCAN_GROUP_RE.match(entry.name)
        if m and int(m.group(1)) <= desktop <= int(m.group(2)):
            groups.append(entry)
    for group in groups:
        candidates = [e for e in dirs(group.path) if e.name.lower() != "new"]
        same_name = [e for e in candidates if e.name == group.name]
        for pool in (same_name, candidates):
            for cand in pool:
                target = os.path.join(cand.path, str(desktop))
                if os.path.isdir(target):
                    return target
    return None
