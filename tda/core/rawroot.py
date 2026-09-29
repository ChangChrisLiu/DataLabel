"""Where the raw dataset is *now* -- one answer for a drive letter that moves.

The raw data (``PHD Data Backup/Desktop Dataset`` with ``OAKD Capture``,
``UGA DATA`` and ``Realsense Capture``) lives on an external drive, and Windows
hands out its letter at plug-in time. Every ``frame.path`` and every ``aux``
path was recorded as an absolute ``F:/PHD Data Backup/...`` string. On the day
the drive came back as ``G:`` three of the four views went blank -- ``scan``
only survived because it reads its local copy on ``D:`` -- and the exit backup
created ``F:/PHD Data Backup/Desktop Dataset/TDA_backups`` on the *other* drive
that happened to be ``F:`` that day.

So this module owns the question, once:

* :func:`locate` finds the root: the recorded one when it still holds the
  dataset's three folders, otherwise the one drive letter (``A:``-``Z:``, cheap
  ``isdir`` checks only) that has ``<X>:/<raw_marker>`` with those folders.
  Several candidates are told apart by ``raw_volume_label``, or not at all.
* :func:`resolve_raw` turns a **stored** raw path into the file to open *now*:
  the recorded root prefix -- any spelling, any case, ``/`` or ``\\`` -- is
  swapped for the resolved one. The stored value is never rewritten: it is the
  provenance record, and a drive letter is not a fact about the data.
* :func:`backup_target` is where a backup may go without ever creating a
  folder on a volume that is not the raw-data one.

The module keeps one :class:`RawRoot` for the process (:func:`configure`).
Until something configures it, :func:`resolve_raw` is the identity -- which is
what every test scene, whose paths never name the marker, wants anyway.
"""
from __future__ import annotations

import logging
import os
import re
import string
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = [
    "AMBIGUOUS", "BackupUnavailable", "DEFAULT_MARKER", "FOUND", "MISSING", "MOVED",
    "RAW_CHILDREN", "ROOT_KEYS", "RawRoot", "UNCONFIGURED", "backup_root", "backup_target",
    "checked", "configure", "current", "locate", "reset", "resolve_raw",
]

log = logging.getLogger(__name__)

#: The folder that *is* the raw dataset, relative to a drive root.
DEFAULT_MARKER = "PHD Data Backup/Desktop Dataset"
#: What a folder must contain to be the dataset rather than, say, a drive that
#: only holds our own backups under the same name (today's ``F:``).
RAW_CHILDREN = ("OAKD Capture", "UGA DATA", "Realsense Capture")
#: ``paths.yaml`` keys whose values are *recorded* raw locations.
ROOT_KEYS = ("f_root", "scanner_root", "oak_root", "rs_root")

#: :attr:`RawRoot.status` values.
FOUND = "found"                # the recorded root still holds the dataset
MOVED = "moved"                # found on another drive letter
MISSING = "missing"            # no drive holds it: "raw data drive not connected"
AMBIGUOUS = "ambiguous"        # several drives hold it and nothing says which
UNCONFIGURED = "unconfigured"  # the configuration names no raw root at all


#: ``\\?\`` (and ``\\.\``), the Win32 long-path prefixes, once slashes are forward.
_DEVICE_PREFIX = re.compile(r"^//[?.]/(?=[A-Za-z]:)")


def _norm(path) -> str:
    """One spelling of a path: forward slashes, no doubled separator, no trailing one.

    ``F://x``, ``F:\\\\x`` and ``\\\\?\\F:\\x`` are all ``F:/x``; a UNC share keeps
    its leading ``//``, and a bare drive keeps its ``X:/``.
    """
    text = _DEVICE_PREFIX.sub("", str(path).replace("\\", "/"))
    lead = "//" if text.startswith("//") else ""
    text = lead + re.sub(r"/{2,}", "/", text[len(lead):])
    while len(text) > 1 and text.endswith("/") and not re.fullmatch(r"[A-Za-z]:/", text):
        text = text[:-1]
    return text


def _fold(path: str) -> str:
    return _norm(path).casefold()


def _under(path: str, root: str) -> Optional[str]:
    """The part of ``path`` below ``root``, or ``None`` when it is not below it.

    Always ``""`` (the root itself) or a string starting with ``/``, so that
    ``new_root + tail`` is a path whatever the two spellings were.
    """
    p, r = _norm(path), _norm(root)
    if not r:
        return None
    pf, rf = p.casefold(), r.casefold()
    if pf == rf:
        return ""
    prefix = rf if rf.endswith("/") else rf + "/"
    if pf.startswith(prefix):
        return "/" + p[len(prefix):]
    return None


# --------------------------------------------------------------------------- #
# the platform: which drives exist, and what they are called
# --------------------------------------------------------------------------- #
#: ``GetDriveTypeW`` answers (winbase.h).
DRIVE_UNKNOWN, DRIVE_NO_ROOT_DIR, DRIVE_REMOVABLE, DRIVE_FIXED = 0, 1, 2, 3
DRIVE_REMOTE, DRIVE_CDROM, DRIVE_RAMDISK = 4, 5, 6
#: The only kinds of drive the scan looks at: an external USB disk is reported
#: as one or the other. A network letter is never probed -- ``isdir`` on a
#: dead share can block for the length of an SMB timeout -- nor is an optical
#: drive, which spins up to answer.
PROBED_TYPES = frozenset({DRIVE_REMOVABLE, DRIVE_FIXED})


def _logical_drives() -> list[str]:
    """``["C:/", "D:/", ...]``: one ``GetLogicalDrives`` call, a bitmask, no I/O."""
    if sys.platform != "win32":
        return []
    try:
        import ctypes

        mask = int(ctypes.windll.kernel32.GetLogicalDrives())
    except Exception:  # noqa: BLE001 - no enumeration is "nothing to scan"
        return []
    return [f"{c}:/" for i, c in enumerate(string.ascii_uppercase) if mask & (1 << i)]


def _drive_type(root: str) -> int:
    """``GetDriveTypeW`` of one drive root; it reads the mount table, not the disk."""
    try:
        import ctypes

        return int(ctypes.windll.kernel32.GetDriveTypeW(ctypes.c_wchar_p(
            str(root).replace("/", "\\"))))
    except Exception:  # noqa: BLE001 - an unknown kind is not probed
        return DRIVE_UNKNOWN


def _drive_roots() -> list[str]:
    """The drive roots the scan may probe: local fixed and removable disks only.

    The letters come from one ``GetLogicalDrives`` bitmask and their kind from
    ``GetDriveTypeW``, neither of which touches a disk. Only then does
    :func:`locate` call ``isdir`` -- and only on the drives of
    :data:`PROBED_TYPES`, so a disconnected network letter or an empty optical
    drive cannot stall the start-up. The *recorded* root is checked directly
    whatever its kind: it is where the data was. Anywhere but Windows there
    are no letters to scan.
    """
    return [root for root in _logical_drives() if _drive_type(root) in PROBED_TYPES]


def _volume_label(root: str) -> Optional[str]:
    """The volume label of the drive holding ``root`` -- any path on it -- or ``None``."""
    if sys.platform != "win32":
        return None
    drive = os.path.splitdrive(str(root))[0]
    if not drive:
        return None
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(261)
        ok = ctypes.windll.kernel32.GetVolumeInformationW(
            ctypes.c_wchar_p(drive + "\\"), buffer, len(buffer),
            None, None, None, None, 0)
    except Exception:  # noqa: BLE001 - an unreadable label is "no label"
        return None
    return buffer.value if ok else None


def _holds_dataset(root: str) -> bool:
    """Does ``root`` have the three folders of the raw dataset? ``isdir`` only."""
    try:
        return all(os.path.isdir(os.path.join(root, child)) for child in RAW_CHILDREN)
    except (OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# the answer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RawRoot:
    """Where the raw dataset was recorded, and where it is now."""

    marker: str = DEFAULT_MARKER
    #: Recorded root prefixes (forward slashes), most authoritative first.
    recorded: tuple[str, ...] = ()
    #: The root now, or ``None`` when it is not connected (or is ambiguous).
    resolved: Optional[str] = None
    status: str = UNCONFIGURED
    #: Every drive root that held the dataset, for the ambiguous case.
    candidates: tuple[str, ...] = ()
    label: Optional[str] = None
    #: Why this root rather than another, when that was a choice (the label).
    note: str = ""
    _marker_re: re.Pattern = field(default=None, repr=False, compare=False)  # type: ignore

    def __post_init__(self) -> None:
        pattern = re.compile(r"^[A-Za-z]:/" + re.escape(_norm(self.marker)) + r"(?=/|$)",
                             re.IGNORECASE)
        object.__setattr__(self, "_marker_re", pattern)

    @property
    def connected(self) -> bool:
        return self.resolved is not None

    @property
    def configured(self) -> bool:
        return self.status != UNCONFIGURED

    def tail(self, path) -> Optional[str]:
        """The part of a stored path below the raw root, or ``None`` if not raw.

        A path is raw when it starts with a recorded root, or when the marker
        sits directly at a drive root under *any* letter -- the index rebuilt
        on another day records another letter, and it is the same data.
        """
        if not self.configured or path is None:
            return None
        text = _norm(path)
        for root in self.recorded:
            below = _under(text, root)
            if below is not None:
                return below
        found = self._marker_re.match(text)
        if found is not None:
            return text[found.end():]
        return None

    def resolve(self, path) -> Optional[str]:
        """The file to open for a stored path; ``None`` when its drive is not here.

        A path that is not on the raw drive at all -- the local cache on
        ``D:``, a test's temporary file -- comes back exactly as it was given.
        """
        if path is None or str(path) == "":
            return None
        below = self.tail(path)
        if below is None:
            return str(path)
        if self.resolved is None:
            return None
        if self.status == FOUND and _under(_norm(path), self.resolved) is not None:
            return str(path)
        return _norm(self.resolved) + below

    def _resolved_is_recorded(self) -> bool:
        resolved = (self.resolved or "").casefold()
        return any(_norm(r).casefold() == resolved for r in self.recorded)

    def to_recorded(self, path) -> str:
        """A path found on today's drive, spelled the way the dataset was recorded.

        The inverse of :meth:`resolve`, for whatever *writes* a path it found:
        ``build-index`` scans ``G:/...`` today, and what it stores has to stay
        ``F:/...`` -- the stored value is the provenance record, and a re-index
        followed by ``load-index`` rewrote all 168 D13 frame paths to ``G:``.
        Anything not under the resolved root comes back unchanged.
        """
        if not self.connected or not self.recorded or self._resolved_is_recorded():
            return str(path)
        below = _under(_norm(path), self.resolved or "")
        if below is None:
            return str(path)
        return _norm(self.recorded[0]) + below

    def respell(self, text: str) -> str:
        """:meth:`to_recorded` for free text: every mention of today's root.

        An index issue names a folder the way ``os.path.join`` built it --
        today's root, then whatever separators followed -- so the root is
        matched with either separator and the rest of the text is left alone.
        """
        if not self.connected or not self.recorded or self._resolved_is_recorded():
            return str(text)
        parts = [re.escape(p) for p in _norm(self.resolved or "").split("/") if p]
        pattern = re.compile(r"[\\/]+".join(parts) + r"(?=[\\/]|$|[\s'\",;:)\]])",
                             re.IGNORECASE)
        recorded = _norm(self.recorded[0])
        return pattern.sub(lambda _m: recorded, str(text))

    @property
    def log_line(self) -> str:
        """The one line a log gets: ``raw root: recorded F:/... resolved G:/...``.

        :func:`locate` itself logs nothing -- it runs before the application
        log exists, and again on every ``F5`` -- so whoever owns a log says it,
        once, at :attr:`log_level`.
        """
        if self.status == UNCONFIGURED:
            return ""
        recorded = self.recorded[0] if self.recorded else "?"
        if self.connected:
            note = f" ({self.note})" if self.note else ""
            return f"raw root: recorded {recorded} resolved {self.resolved}{note}"
        where = f" ({', '.join(self.candidates)})" if self.candidates else ""
        return f"raw root: recorded {recorded} {self.status}{where}"

    @property
    def log_level(self) -> int:
        return logging.INFO if self.connected or not self.configured else logging.WARNING

    @property
    def message(self) -> str:
        """One bilingual line for the window and the command line."""
        marker = _norm(self.marker)
        if self.status == MISSING:
            plug = f"插上 {self.label} 盘后按 F5" if self.label else "插上原始数据盘后按 F5"
            plug_en = (f"plug in the {self.label} drive, then press F5" if self.label
                       else "plug in the raw data drive, then press F5")
            return (f"原始数据盘没连上：找不到 {marker}（{plug}）/ raw data drive not "
                    f"found: no drive has {marker} ({plug_en})")
        if self.status == AMBIGUOUS:
            where = ", ".join(self.candidates)
            return (f"原始数据盘不止一个：{where} 都有 {marker}，没有自动选（在 paths.yaml "
                    f"里设 raw_volume_label）/ several drives hold {marker} ({where}); "
                    f"none was chosen -- set raw_volume_label in paths.yaml")
        if self.status == MOVED:
            return (f"原始数据盘在 {self.resolved}（记录的是 {self.recorded[0]}）/ raw "
                    f"data found at {self.resolved} (recorded {self.recorded[0]})")
        if self.status == FOUND:
            return f"原始数据盘 / raw data: {self.resolved}"
        return ""


def _recorded_roots(paths: dict, marker: str) -> tuple[str, ...]:
    """The recorded raw roots of a configuration, ``f_root`` first."""
    out: list[str] = []
    needle = "/" + _norm(marker).casefold()
    for key in ROOT_KEYS:
        value = paths.get(key)
        if not value:
            continue
        text = _norm(value)
        at = (text.casefold() + "/").find(needle + "/")
        if at >= 0:
            root = text[:at + len(needle)]
        elif key == "f_root":
            root = text        # an f_root that does not follow the marker still counts
        else:
            continue
        if root.casefold() not in {r.casefold() for r in out}:
            out.append(root)
    return tuple(out)


def locate(paths: dict, drives: Optional[Iterable[str]] = None) -> RawRoot:
    """Find the raw dataset for this configuration (see the module docstring).

    ``drives`` overrides the enumeration (tests); otherwise it is
    :func:`_drive_roots`, read only when the recorded root does not answer.
    """
    paths = paths or {}
    marker = _norm(paths.get("raw_marker") or DEFAULT_MARKER)
    label = paths.get("raw_volume_label") or None
    recorded = _recorded_roots(paths, marker)
    if not recorded:
        return RawRoot(marker=marker, status=UNCONFIGURED, label=label)
    wanted = str(label).casefold() if label else ""

    def labelled(root: str) -> bool:
        return bool(wanted) and (_volume_label(root) or "").casefold() == wanted

    here = next((_norm(r) for r in recorded if _holds_dataset(r)), None)
    if here is not None and (not wanted or labelled(here)):
        return RawRoot(marker=marker, recorded=recorded, resolved=here,
                       status=FOUND, candidates=(here,), label=label)
    found: list[str] = []
    for drive in (_drive_roots() if drives is None else drives):
        candidate = _norm(os.path.join(str(drive), marker))
        if _holds_dataset(candidate) and candidate.casefold() not in {
                c.casefold() for c in found}:
            found.append(candidate)
    if here is not None:
        # The recorded root holds a full copy, but not on the volume the
        # configuration names: a labelled volume elsewhere is the dataset, the
        # copy is not. With no such volume the copy is what there is.
        others = [c for c in found if c.casefold() != here.casefold() and labelled(c)]
        if len(others) == 1:
            note = (f"the volume labelled {label} was chosen over the copy at {here} "
                    f"on volume {_volume_label(here) or '(no label)'}")
            return RawRoot(marker=marker, recorded=recorded, resolved=others[0],
                           status=MOVED, candidates=(others[0], here), label=label,
                           note=note)
        return RawRoot(marker=marker, recorded=recorded, resolved=here,
                       status=FOUND, candidates=(here,), label=label)
    if len(found) > 1 and wanted:
        chosen = [c for c in found if labelled(c)]
        if len(chosen) == 1:
            found = chosen
    if len(found) == 1:
        return RawRoot(marker=marker, recorded=recorded, resolved=found[0],
                       status=MOVED, candidates=tuple(found), label=label)
    return RawRoot(marker=marker, recorded=recorded, resolved=None,
                   status=AMBIGUOUS if found else MISSING,
                   candidates=tuple(found), label=label)


# --------------------------------------------------------------------------- #
# the process-wide answer
# --------------------------------------------------------------------------- #
_current: Optional[RawRoot] = None


def configure(paths: dict, drives: Optional[Iterable[str]] = None) -> RawRoot:
    """Locate the raw root for ``paths`` and make it the process's answer."""
    global _current
    _current = locate(paths, drives)
    return _current


def current() -> Optional[RawRoot]:
    """The configured answer, or ``None`` before anything configured one."""
    return _current


def reset() -> None:
    """Forget the answer: :func:`resolve_raw` is the identity again (tests)."""
    global _current
    _current = None


def resolve_raw(path) -> Optional[str]:
    """**The** way a stored raw path becomes a file to open.

    ``None`` for an empty path, and for a path on the raw drive while that
    drive is not connected; any other path is returned unchanged when it is
    not under a recorded raw root, and with the root swapped when it is.
    """
    if path is None or str(path) == "":
        return None
    if _current is None:
        return str(path)
    return _current.resolve(path)


# --------------------------------------------------------------------------- #
# backups
# --------------------------------------------------------------------------- #
class BackupUnavailable(OSError):
    """There is nowhere a backup may go right now; the message says why."""


def checked(paths: dict, raw: Optional[RawRoot] = None) -> RawRoot:
    """An answer that is still true *now*: re-located unless its root still holds the data.

    The process-wide answer is from start-up (or the last ``F5``). A drive
    unplugged since then, and another volume that took its letter, would
    otherwise receive the next backup -- so anything that is about to *write*
    asks again. Only ``isdir`` checks; the process-wide answer is not changed.
    """
    raw = raw or current() or locate(paths)
    if not raw.configured:
        return raw
    if raw.connected and _holds_dataset(raw.resolved or ""):
        return raw
    return locate(paths)


def backup_root(paths: dict, raw: Optional[RawRoot] = None) -> str:
    """Where ``backup_dir`` is, given an answer about the raw drive.

    A ``backup_dir`` under a recorded raw root follows the raw drive to its
    current letter; any other one is taken as configured. Raises
    :class:`BackupUnavailable` when it is on the raw drive and that drive is
    not connected, and ``ValueError`` when there is no ``backup_dir`` at all.
    Without ``raw`` the answer is :func:`checked` first.
    """
    configured = paths.get("backup_dir")
    if not configured:
        raise ValueError("paths.yaml defines no backup_dir")
    raw = raw or checked(paths)
    below = raw.tail(configured)
    if below is None:
        return os.path.abspath(str(configured))
    if raw.resolved is None:
        raise BackupUnavailable(
            f"原始数据盘没连上，没有备份 / no backup: the raw data drive is not "
            f"connected, and backup_dir ({_norm(configured)}) lives on it"
            + (f" -- {raw.message}" if raw.status == AMBIGUOUS else ""))
    return _norm(raw.resolved) + below


def backup_target(paths: dict, raw: Optional[RawRoot] = None) -> str:
    """:func:`backup_root`, made usable -- or :class:`BackupUnavailable`.

    **Never creates a folder on a volume that is not the raw-data one.** On the
    raw drive only the leaf (``TDA_backups``) may be created, and only inside
    the resolved raw root; a ``backup_dir`` anywhere else has to exist already.
    The answer about the raw drive -- the one handed in, or the process's -- is
    re-checked first (:func:`checked`): a cached ``G:`` is not trusted to still
    be the dataset.
    """
    raw = checked(paths, raw)
    target = backup_root(paths, raw)
    if os.path.isdir(target):
        return target
    if raw.tail(paths.get("backup_dir")) is None:
        raise BackupUnavailable(
            f"备份目录不存在，没有备份 / no backup: backup_dir {_norm(target)} does "
            f"not exist -- create it first (it is outside the raw data drive, so it "
            f"is never created automatically)")
    parent = os.path.dirname(_norm(target).rstrip("/"))
    inside = _under(parent, raw.resolved or "") is not None
    if not inside or not os.path.isdir(parent):
        raise BackupUnavailable(
            f"没有备份：{_norm(target)} 的上级目录不存在 / no backup: the folder above "
            f"{_norm(target)} does not exist, and only the last folder is ever created")
    try:
        os.mkdir(target)
    except FileExistsError:
        pass
    except OSError as exc:
        raise BackupUnavailable(f"没有备份：建不了 {_norm(target)} / no backup: "
                                f"cannot create {_norm(target)}: {exc}") from None
    return target
