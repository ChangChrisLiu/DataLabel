"""Tests for the unified frame index.

Every test builds a *synthetic* source tree under ``tmp_path`` (real, empty
files with the real naming conventions) and passes a ``roots`` dict pointing at
it, so no test ever touches the read-only F: drive.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from tda.core.index import (
    DesktopIndex,
    FrameFile,
    build_index,
    load_fixes,
    load_index,
    save_index,
    scan_desktop,
)
from tda.core.model import FrameKey
from tda.core.sources import dirs, files

REPO_ROOT = Path(__file__).resolve().parents[1]
OAK_ROLES = (
    "rgb_12mp.jpg",
    "rgb_aligned.png",
    "depth_raw.npy",
    "depth_raw.png",
    "pointcloud.ply",
)


# --------------------------------------------------------------------------
# fixture builder
# --------------------------------------------------------------------------
def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _epoch(ts_token: str) -> float:
    """``20250101_100000_000`` -> POSIX timestamp."""
    base = datetime.strptime(ts_token[:15], "%Y%m%d_%H%M%S")
    return base.replace(microsecond=int(ts_token[16:19]) * 1000).timestamp()


def build_tree(
    root: Path,
    desktop: int,
    oak1: dict[str, list[str]] | None = None,
    oak2: dict[str, list[str]] | None = None,
    scan_steps: dict[int, int] | None = None,
    rs_steps: list[int] | None = None,
    rs_alias: str = "Disassemble",
    components_c2: dict[str, list[str]] | None = None,
    rs_mtimes: dict[int, str] | None = None,
) -> dict:
    """Create a mini copy of the real source layout and return a ``roots`` dict.

    ``oak1``/``oak2`` map a 3-digit step folder name to the list of capture
    timestamp tokens (``YYYYMMDD_HHMMSS_mmm``) stored in it.  ``scan_steps``
    maps a *nominal* scanner step number to its burst length.  ``rs_steps`` is
    the list of nominal RealSense step numbers, and ``rs_mtimes`` optionally
    stamps a step's ``original_color.png`` with a given timestamp token (the
    RealSense carries no capture time in its filenames, so the real index reads
    the file mtime).  ``components_c2`` adds Camera_2 captures under
    ``Components/C2`` (the D24 anomaly).
    """
    oak_root = root / "oak"
    scanner_root = root / "scan"
    rs_root = root / "rs"

    for cam, spec in ((1, oak1), (2, oak2)):
        for step_dir, tokens in (spec or {}).items():
            for ts in tokens:
                for role in OAK_ROLES:
                    _touch(
                        oak_root
                        / f"Desktop {desktop}"
                        / "Disassemble"
                        / f"Camera_{cam}"
                        / step_dir
                        / f"{ts}_camera_{cam}_{role}"
                    )
    for step_dir, tokens in (components_c2 or {}).items():
        for ts in tokens:
            for role in OAK_ROLES:
                _touch(
                    oak_root
                    / f"Desktop {desktop}"
                    / "Components"
                    / "C2"
                    / step_dir
                    / f"{ts}_camera_2_{role}"
                )

    if scan_steps:
        group = f"TAMU_B2.3_{desktop}-{desktop}_RGB"
        base = scanner_root / group / group / str(desktop)
        _touch(scanner_root / group / group / "ext.py")
        (scanner_root / group / group / "New").mkdir(parents=True, exist_ok=True)
        for step, n_burst in scan_steps.items():
            for k in range(n_burst):
                _touch(base / f"RGB{step}1" / f"P_{k}.png")

    for step in rs_steps or []:
        d = rs_root / f"Desktop {desktop}" / rs_alias / f"{step:03d}"
        _touch(d / "original_color.png")
        _touch(d / "depth_raw.npy")
        _touch(d / "capture_info.txt")
        if rs_mtimes and step in rs_mtimes:
            stamp = _epoch(rs_mtimes[step])
            os.utime(d / "original_color.png", (stamp, stamp))

    return {
        "oak_root": str(oak_root),
        "scanner_root": str(scanner_root),
        "rs_root": str(rs_root),
    }


TS99 = [
    "20250101_100000_000",
    "20250101_100100_000",
    "20250101_100200_000",
    "20250101_100300_000",
]


@pytest.fixture
def roots99(tmp_path: Path) -> dict:
    """Desktop 99: 4 logical steps, scanner step 3 missing, RS alias 'Disassembly'."""
    return build_tree(
        tmp_path,
        99,
        oak1={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        oak2={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        scan_steps={1: 10, 2: 9, 4: 10},
        rs_steps=[1, 2, 3, 4],
        rs_alias="Disassembly",
        rs_mtimes={i + 1: TS99[i] for i in range(4)},
    )


@pytest.fixture
def fixes() -> dict:
    return load_fixes(str(REPO_ROOT / "configs" / "index_fixes.yaml"))


# --------------------------------------------------------------------------
# basic behaviour
# --------------------------------------------------------------------------
def test_n_steps_from_oak_cam1(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    assert isinstance(idx, DesktopIndex)
    assert idx.n_steps == 4
    for step in range(1, 5):
        ff = idx.frames[FrameKey(99, step, "oak1")]
        assert isinstance(ff, FrameFile)
        assert ff.path.endswith("_camera_1_rgb_12mp.jpg")
        assert ff.src_step_dir == f"{step:03d}"
        assert ff.aux["aligned"].endswith("_camera_1_rgb_aligned.png")
        assert ff.aux["depth_npy"].endswith("_camera_1_depth_raw.npy")


def test_missing_scanner_step(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    assert FrameKey(99, 3, "scan") in idx.missing
    assert FrameKey(99, 1, "scan") not in idx.missing
    assert FrameKey(99, 3, "scan") not in idx.frames


def test_oak2_paired_by_timestamp(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    for step in range(1, 5):
        assert (
            idx.frames[FrameKey(99, step, "oak2")].ts
            == idx.frames[FrameKey(99, step, "oak1")].ts
        )
    assert idx.frames[FrameKey(99, 1, "oak1")].ts == "2025-01-01T10:00:00"


def test_rs_folder_alias_resolved(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    for step in range(1, 5):
        ff = idx.frames[FrameKey(99, step, "rs")]
        assert "Disassembly" in ff.path.replace("\\", "/")
        assert ff.path.endswith("original_color.png")
        assert ff.aux["depth_npy"].endswith("depth_raw.npy")
        assert ff.ts is not None


def test_scanner_burst_and_main_image(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    s1 = idx.frames[FrameKey(99, 1, "scan")]
    assert s1.path.endswith("P_0.png")
    assert s1.src_step_dir == "RGB11"
    assert len(s1.aux["burst"]) == 10
    assert len(idx.frames[FrameKey(99, 2, "scan")].aux["burst"]) == 9


def test_save_load_roundtrip(roots99, fixes, tmp_path):
    idx = build_index([99], roots99, fixes_path=str(REPO_ROOT / "configs" / "index_fixes.yaml"))
    out = tmp_path / "out" / "index.json"
    save_index(idx, str(out))
    assert out.is_file()
    assert load_index(str(out)) == idx


# --------------------------------------------------------------------------
# anomaly fixes (§2.2 / configs/index_fixes.yaml)
# --------------------------------------------------------------------------
def test_fixes_file_matches_spec(fixes):
    assert fixes["general"]["step_order_source"] == "oak1_timestamp"
    assert fixes["oak"][60]["cam1_reorder_by_ts"] is True
    assert fixes["oak"][60]["cam2_shift"]["from_dir"] == "004"
    # a Camera_1 *folder* name, not a logical step number
    assert fixes["oak"][60]["cam2_shift"]["to_cam1_dir"] == "003"
    assert fixes["oak"][42]["cam2_swap"] == [["016", "017"]]
    assert fixes["oak"][24]["cam2_extra_from"] == "Components/C2/043"
    assert fixes["oak"][24]["as_step"] == 43
    assert fixes["oak"][10]["step1_keep"] == "latest"
    assert fixes["scan"][1]["missing_steps"] == [2, 3, 4, 28, 29, 30, 31, 32, 34, 35]
    assert fixes["rs"][42]["missing_steps_range"] == [1, 12]
    assert fixes["rs"]["folder_aliases"] == ["Disassemble", "Disassembly"]
    for view in ("oak", "scan", "rs"):
        for desktop, entry in fixes[view].items():
            if isinstance(desktop, int):
                assert entry.get("reason"), f"{view}/{desktop} has no reason string"


def test_d60_cam1_reordered_and_cam2_split(tmp_path, fixes):
    """cam1 003/004 timestamps are swapped; cam2 lacks 003 and 005 holds two captures."""
    roots = build_tree(
        tmp_path,
        60,
        oak1={
            "001": ["20250609_101820_447"],
            "002": ["20250609_101857_869"],
            "003": ["20250609_102048_178"],
            "004": ["20250609_101931_616"],
            "005": ["20250609_102207_233"],
        },
        oak2={
            "001": ["20250609_101820_447"],
            "002": ["20250609_101857_869"],
            "004": ["20250609_102048_178"],
            "005": ["20250609_101931_616", "20250609_102207_233"],
        },
    )
    idx = scan_desktop(60, roots, fixes)
    assert idx.n_steps == 5
    # logical order follows cam1 timestamps, so folder 004 is step 3.
    assert idx.frames[FrameKey(60, 3, "oak1")].src_step_dir == "004"
    assert idx.frames[FrameKey(60, 4, "oak1")].src_step_dir == "003"
    # cam2 folder 004 pairs with cam1 folder 003 == logical step 4.
    assert idx.frames[FrameKey(60, 4, "oak2")].src_step_dir == "004"
    # cam2 folder 005 is split by timestamp across logical steps 3 and 5.
    assert idx.frames[FrameKey(60, 3, "oak2")].src_step_dir == "005"
    assert idx.frames[FrameKey(60, 5, "oak2")].src_step_dir == "005"
    for step in range(1, 6):
        assert (
            idx.frames[FrameKey(60, step, "oak2")].ts
            == idx.frames[FrameKey(60, step, "oak1")].ts
        )
    assert any("cam1_reorder_by_ts" in i for i in idx.issues)
    # the confirmation names the resolved logical step, not just the folder
    assert any(
        "cam2_shift: cam2 004 -> cam1 dir 003 = logical step 4" in i for i in idx.issues
    )


def test_d42_cam2_swap(tmp_path, fixes):
    oak1 = {f"{k:03d}": [f"20250607_1058{k:02d}_407"] for k in range(1, 18)}
    oak2 = dict(oak1)
    oak2["016"], oak2["017"] = oak1["017"], oak1["016"]
    roots = build_tree(tmp_path, 42, oak1=oak1, oak2=oak2)
    idx = scan_desktop(42, roots, fixes)
    assert idx.n_steps == 17
    assert idx.frames[FrameKey(42, 16, "oak2")].src_step_dir == "017"
    assert idx.frames[FrameKey(42, 17, "oak2")].src_step_dir == "016"
    assert any("cam2_swap" in i for i in idx.issues)


def test_d24_cam2_extra_from_components(tmp_path, fixes):
    oak1 = {f"{k:03d}": [f"20250604_1713{k:02d}_514"] for k in range(1, 44)}
    oak2 = {k: v for k, v in oak1.items() if k != "043"}
    roots = build_tree(
        tmp_path,
        24,
        oak1=oak1,
        oak2=oak2,
        components_c2={"043": ["20250604_171343_514"]},
    )
    idx = scan_desktop(24, roots, fixes)
    assert idx.n_steps == 43
    ff = idx.frames[FrameKey(24, 43, "oak2")]
    assert ff.src_step_dir == "Components/C2/043"
    assert ff.ts == idx.frames[FrameKey(24, 43, "oak1")].ts
    assert FrameKey(24, 43, "oak2") not in idx.missing
    assert any("cam2_extra_from" in i for i in idx.issues)


def test_d10_step1_keep_latest(tmp_path, fixes):
    roots = build_tree(
        tmp_path,
        10,
        oak1={
            "001": ["20250603_095715_196", "20250603_095842_144"],
            "002": ["20250603_100000_000"],
        },
        oak2={
            "001": ["20250603_095715_196", "20250603_095842_144"],
            "002": ["20250603_100000_000"],
        },
    )
    idx = scan_desktop(10, roots, fixes)
    assert idx.n_steps == 2
    assert idx.frames[FrameKey(10, 1, "oak1")].ts == "2025-06-03T09:58:42.144000"
    assert idx.frames[FrameKey(10, 1, "oak2")].ts == "2025-06-03T09:58:42.144000"
    assert any("step1_keep" in i for i in idx.issues)


def test_cam2_swap_mismatch_when_timestamps_contradict(tmp_path, fixes):
    """D42 declares a 016/017 swap; a tree where cam2 is *not* swapped must complain."""
    oak1 = {f"{k:03d}": [f"20250607_1058{k:02d}_407"] for k in range(1, 18)}
    roots = build_tree(tmp_path, 42, oak1=oak1, oak2=dict(oak1))
    idx = scan_desktop(42, roots, fixes)
    mismatches = [i for i in idx.issues if "cam2_swap" in i and "MISMATCH" in i]
    assert len(mismatches) == 2, idx.issues
    assert any("cam2 016 expected -> cam1 dir 017" in i for i in mismatches)
    assert any("cam2 017 expected -> cam1 dir 016" in i for i in mismatches)
    # the frames themselves still follow the timestamps
    assert idx.frames[FrameKey(42, 16, "oak2")].src_step_dir == "016"


def test_declared_missing_scan_steps_confirmed(tmp_path, fixes):
    """D1 declares scanner steps 2/3/4 missing; a tree without them confirms it."""
    roots = build_tree(
        tmp_path,
        1,
        oak1={f"{k:03d}": [f"20250531_1047{k:02d}_000"] for k in range(1, 7)},
        scan_steps={1: 10, 5: 10, 6: 10},
    )
    idx = scan_desktop(1, roots, fixes)
    for step in (2, 3, 4):
        assert FrameKey(1, step, "scan") in idx.missing
    confirmed = [i for i in idx.issues if i.startswith("fix:scan[1]")]
    assert len(confirmed) == 1, idx.issues
    assert "confirmed absent" in confirmed[0]
    assert "[2, 3, 4]" in confirmed[0]
    # the declared steps 28..35 have no cam1 folder in this 6-step tree
    assert "have no cam1 folder" in confirmed[0]


def test_declared_missing_scan_step_present_is_mismatch(tmp_path, fixes):
    """A declared-missing scanner step that actually exists must raise MISMATCH."""
    roots = build_tree(
        tmp_path,
        1,
        oak1={f"{k:03d}": [f"20250531_1047{k:02d}_000"] for k in range(1, 7)},
        scan_steps={1: 10, 2: 10, 5: 10, 6: 10},
    )
    idx = scan_desktop(1, roots, fixes)
    assert FrameKey(1, 2, "scan") not in idx.missing
    mismatch = [i for i in idx.issues if i.startswith("fix:scan[1]") and "MISMATCH" in i]
    assert len(mismatch) == 1, idx.issues
    assert "[2]" in mismatch[0]


def test_declared_missing_rs_range_confirmed(tmp_path, fixes):
    """D42 declares RealSense steps 1-12 missing."""
    roots = build_tree(
        tmp_path,
        42,
        oak1={f"{k:03d}": [f"20250607_1058{k:02d}_407"] for k in range(1, 15)},
        rs_steps=[13, 14],
        rs_mtimes={13: "20250607_105813_407", 14: "20250607_105814_407"},
    )
    idx = scan_desktop(42, roots, fixes)
    assert {k.step for k in idx.missing if k.view == "rs"} == set(range(1, 13))
    confirmed = [i for i in idx.issues if i.startswith("fix:rs[42]")]
    assert len(confirmed) == 1, idx.issues
    assert "confirmed absent" in confirmed[0]


def test_declared_missing_rs_step_present_is_mismatch(tmp_path, fixes):
    roots = build_tree(
        tmp_path,
        42,
        oak1={f"{k:03d}": [f"20250607_1058{k:02d}_407"] for k in range(1, 15)},
        rs_steps=[12, 13, 14],
        rs_mtimes={k: f"20250607_1058{k:02d}_407" for k in (12, 13, 14)},
    )
    idx = scan_desktop(42, roots, fixes)
    mismatch = [i for i in idx.issues if i.startswith("fix:rs[42]") and "MISMATCH" in i]
    assert len(mismatch) == 1, idx.issues
    assert "[12]" in mismatch[0]


def test_rs_time_matching_oak1_raises_no_issue(roots99, fixes):
    idx = scan_desktop(99, roots99, fixes)
    assert not [i for i in idx.issues if "rs time mismatch" in i], idx.issues


def test_rs_time_mismatch_is_reported(tmp_path, fixes):
    """A RealSense file time far from its OAK cam1 capture time must be reported."""
    roots = build_tree(
        tmp_path,
        99,
        oak1={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        rs_steps=[1, 2, 3, 4],
        # step 3's image is stamped 10:05, over an hour from the oak1 capture
        rs_mtimes={1: TS99[0], 2: TS99[1], 3: "20250101_110500_000", 4: TS99[3]},
    )
    idx = scan_desktop(99, roots, fixes)
    bad = [i for i in idx.issues if "rs time mismatch" in i]
    assert len(bad) == 1, idx.issues
    assert "folder 003 -> logical step 3" in bad[0]
    assert "tolerance 30s" in bad[0]
    # the frame is still indexed - the issue flags it, it does not drop it
    assert FrameKey(99, 3, "rs") in idx.frames


def test_unreadable_source_directory_is_reported(tmp_path, fixes):
    """A failed listing must surface as an issue, not as a silent empty folder."""
    roots = build_tree(
        tmp_path,
        99,
        oak1={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        oak2={},  # Camera_2 never created
    )
    idx = scan_desktop(99, roots, fixes)
    failed = [i for i in idx.issues if i.startswith("scandir failed:")]
    assert any("Camera_2" in i and "FileNotFoundError" in i for i in failed), idx.issues
    assert idx.n_steps == 4  # cam1 is unaffected


def test_scandir_helpers_record_the_exception(tmp_path):
    missing = str(tmp_path / "nope")
    errors: list[str] = []
    assert dirs(missing, errors) == []
    assert files(missing, errors) == []
    assert len(errors) == 2
    assert all(e.startswith("scandir failed:") and "nope" in e for e in errors)
    assert all("FileNotFoundError" in e for e in errors)
    # without an errors list the helpers stay quiet (back-compatible)
    assert dirs(missing) == [] and files(missing) == []


def test_orphan_cam2_capture_is_reported_and_skipped(tmp_path, fixes):
    roots = build_tree(
        tmp_path,
        98,
        oak1={"001": ["20250101_100000_000"], "002": ["20250101_100100_000"]},
        oak2={"001": ["20250101_100000_000"], "002": ["20250101_999999_999"]},
    )
    idx = scan_desktop(98, roots, fixes)
    assert idx.n_steps == 2
    assert FrameKey(98, 2, "oak2") in idx.missing
    assert any("no cam1 partner" in i for i in idx.issues)


def test_scan_and_rs_nominal_steps_map_through_cam1_folders(tmp_path, fixes):
    """Scanner/RealSense folder numbers are nominal steps -> cam1 folder -> logical step."""
    roots = build_tree(
        tmp_path,
        60,
        oak1={
            "001": ["20250609_101820_447"],
            "002": ["20250609_101857_869"],
            "003": ["20250609_102048_178"],
            "004": ["20250609_101931_616"],
        },
        oak2={},
        scan_steps={1: 10, 2: 10, 3: 10, 4: 10},
        rs_steps=[1, 2, 3, 4],
    )
    idx = scan_desktop(60, roots, fixes)
    # nominal scanner step 3 (RGB31) belongs to cam1 folder 003 == logical step 4.
    assert idx.frames[FrameKey(60, 4, "scan")].src_step_dir == "RGB31"
    assert idx.frames[FrameKey(60, 3, "scan")].src_step_dir == "RGB41"
    assert idx.frames[FrameKey(60, 4, "rs")].src_step_dir == "003"
    assert idx.frames[FrameKey(60, 3, "rs")].src_step_dir == "004"


def test_missing_lists_every_absent_view(tmp_path, fixes):
    roots = build_tree(tmp_path, 97, oak1={"001": ["20250101_100000_000"]}, oak2={})
    idx = scan_desktop(97, roots, fixes)
    assert idx.n_steps == 1
    assert sorted(idx.missing) == sorted(
        [FrameKey(97, 1, v) for v in ("oak2", "rs", "scan")]
    )


def test_build_index_over_several_desktops(tmp_path, fixes):
    roots = build_tree(
        tmp_path,
        99,
        oak1={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        oak2={f"{i + 1:03d}": [TS99[i]] for i in range(4)},
        scan_steps={1: 10, 2: 9, 4: 10},
        rs_steps=[1, 2, 3, 4],
        rs_alias="Disassembly",
    )
    build_tree(Path(roots["oak_root"]).parent, 98, oak1={"001": ["20250101_110000_000"]})
    idx = build_index([98, 99], roots)
    assert set(idx) == {98, 99}
    assert idx[98].n_steps == 1
    assert idx[99].n_steps == 4


def test_absent_desktop_yields_empty_index_with_issue(tmp_path, fixes):
    roots = build_tree(tmp_path, 99, oak1={"001": ["20250101_100000_000"]})
    idx = scan_desktop(1, roots, fixes)
    assert idx.n_steps == 0
    assert idx.frames == {}
    assert idx.issues


def test_no_writes_to_source_tree(roots99, fixes):
    before = {
        p: os.stat(os.path.join(dp, p)).st_mtime
        for dp, _, fs in os.walk(roots99["oak_root"])
        for p in fs
    }
    scan_desktop(99, roots99, fixes)
    after = {
        p: os.stat(os.path.join(dp, p)).st_mtime
        for dp, _, fs in os.walk(roots99["oak_root"])
        for p in fs
    }
    assert before == after


def test_paths_yaml_provides_the_three_roots():
    with open(REPO_ROOT / "configs" / "paths.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("oak_root", "scanner_root", "rs_root"):
        assert cfg[k]
