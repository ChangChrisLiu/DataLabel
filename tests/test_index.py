"""Tests for the unified frame index.

Every test builds a *synthetic* source tree under ``tmp_path`` (real, empty
files with the real naming conventions) and passes a ``roots`` dict pointing at
it, so no test ever touches the read-only F: drive.
"""
from __future__ import annotations

import os
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


def build_tree(
    root: Path,
    desktop: int,
    oak1: dict[str, list[str]] | None = None,
    oak2: dict[str, list[str]] | None = None,
    scan_steps: dict[int, int] | None = None,
    rs_steps: list[int] | None = None,
    rs_alias: str = "Disassemble",
    components_c2: dict[str, list[str]] | None = None,
) -> dict:
    """Create a mini copy of the real source layout and return a ``roots`` dict.

    ``oak1``/``oak2`` map a 3-digit step folder name to the list of capture
    timestamp tokens (``YYYYMMDD_HHMMSS_mmm``) stored in it.  ``scan_steps``
    maps a *nominal* scanner step number to its burst length.  ``rs_steps`` is
    the list of nominal RealSense step numbers.  ``components_c2`` adds
    Camera_2 captures under ``Components/C2`` (the D24 anomaly).
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
    assert fixes["oak"][60]["cam2_shift"]["to_step"] == 3
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
    assert any("cam2_shift" in i for i in idx.issues)


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
