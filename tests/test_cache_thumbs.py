"""Tests for the offline thumbnail tier of the local cache (spec 2.4).

Every fixture is synthetic and lives under ``tmp_path``: a miniature cache
(``<cache>/<view>/D<nn>/s<kkk>.<ext>`` plus its ``manifest.json``) and, where a
ROI is needed, a throw-away annotation database built straight from
``schema.sql``.  No test reads the real cache, the real database or F:.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

import tda.core.cache_thumbs as ct
from tda.core.cache import main
from tda.core.cache_thumbs import DbRoiLookup, build_thumbs, thumb_path
from tda.core.index import DesktopIndex, FrameFile, save_index
from tda.core.model import FrameKey

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "tda" / "core" / "schema.sql"
RED_BOX = (600, 100, 800, 300)  # the "chassis" of the synthetic 1000x1000 frame


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _frame(width: int = 1000, height: int = 1000, box=None) -> np.ndarray:
    """A flat gray BGR frame with an optional red rectangle at ``box``."""
    img = np.full((height, width, 3), 120, np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        img[y0:y1, x0:x1] = (0, 0, 220)  # BGR: red
    return img


def _red_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that survived JPEG as clearly red."""
    return float(((img[:, :, 2] > 150) & (img[:, :, 0] < 100)).mean())


def _make_cache(tmp_path: Path, view: str = "scan", desktop: int = 7, steps=(1, 2),
                images: dict | None = None) -> str:
    """A cache dir holding full-size frames for ``steps`` plus their manifest."""
    cache = tmp_path / "cache"
    ddir = cache / view / f"D{desktop:02d}"
    ddir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if view.startswith("oak") else "png"
    manifest = {}
    for step in steps:
        img = (images or {}).get(step)
        assert cv2.imwrite(str(ddir / f"s{step:03d}.{ext}"),
                           _frame() if img is None else img)
        manifest[str(step)] = {"chosen": 0, "reason": "p0", "metrics": [],
                               "src": f"F:/fake/D{desktop}/s{step}.{ext}"}
    (ddir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(cache)


def _age(path, seconds: float) -> int:
    """Shift a file's mtime by ``seconds``; returns the new mtime in ns."""
    stamp = os.stat(path).st_mtime_ns + int(seconds * 1e9)
    os.utime(path, ns=(stamp, stamp))
    return stamp


def _make_db(path: str, *, version: int = 2, desktop: int = 7, view: str = "scan",
             roi=RED_BOX, start: int = 1, end: int = 50) -> None:
    """A database with one pose segment carrying a ROI, stamped ``version``."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO desktop(id) VALUES(?)", (desktop,))
    conn.execute(
        "INSERT INTO pose_segment(desktop, view, seg, start_step, end_step, ref_step, roi_json)"
        " VALUES(?,?,?,?,?,?,?)",
        (desktop, view, 0, start, end, start, json.dumps(list(roi))),
    )
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(version),))
    conn.commit()
    conn.close()


def _oak_index(tmp_path: Path) -> str:
    """A one-frame index of a single OAK-style image, saved to disk."""
    src = tmp_path / "src" / "frame.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(src), _frame(400, 300))
    key = FrameKey(7, 1, "oak1")
    index = {7: DesktopIndex(desktop=7, n_steps=1,
                             frames={key: FrameFile(key=key, path=str(src))})}
    path = str(tmp_path / "index.json")
    save_index(index, path)
    return path


# --------------------------------------------------------------------------
# thumb_path
# --------------------------------------------------------------------------
def test_thumb_path_is_the_layout_the_ui_hard_codes():
    assert (thumb_path("D:/DataSet/cache", FrameKey(13, 42, "scan"))
            == "D:/DataSet/cache/thumbs/scan/D13/s042.jpg")
    assert thumb_path("D:/c/", FrameKey(1, 5, "oak1")) == "D:/c/thumbs/oak1/D01/s005.jpg"
    assert thumb_path(Path("D:/c"), FrameKey(66, 123, "rs")) == "D:/c/thumbs/rs/D66/s123.jpg"


# --------------------------------------------------------------------------
# build_thumbs: size, aspect, format
# --------------------------------------------------------------------------
def test_build_thumbs_scales_every_step_to_the_max_side_keeping_the_aspect(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2),
                        images={1: _frame(800, 400), 2: _frame(300, 900)})

    stats = build_thumbs(cache, max_side=192)

    assert stats["written"] == 2
    assert (stats["skipped"], stats["missing_source"], stats["failed"]) == (0, 0, 0)
    wide = cv2.imread(thumb_path(cache, FrameKey(7, 1, "scan")))
    tall = cv2.imread(thumb_path(cache, FrameKey(7, 2, "scan")))
    assert wide.shape[:2] == (96, 192)
    assert tall.shape[:2] == (192, 64)


def test_build_thumbs_writes_real_jpegs_under_the_thumbs_dir_only(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,))
    full = Path(cache) / "scan" / "D07" / "s001.png"
    before = full.stat().st_mtime_ns

    build_thumbs(cache)

    dest = Path(thumb_path(cache, FrameKey(7, 1, "scan")))
    assert dest.read_bytes()[:2] == b"\xff\xd8"  # JPEG SOI
    assert dest.stat().st_size < full.stat().st_size
    assert full.stat().st_mtime_ns == before
    assert sorted(p.name for p in Path(cache).iterdir()) == ["scan", "thumbs"]
    assert sorted(p.name for p in dest.parent.iterdir()) == ["s001.jpg"]


def test_build_thumbs_honours_the_view_and_desktop_filters(tmp_path):
    cache = _make_cache(tmp_path, desktop=7, steps=(1,))
    _make_cache(tmp_path, desktop=8, steps=(1,))
    _make_cache(tmp_path, view="oak1", desktop=7, steps=(1,))

    stats = build_thumbs(cache, views=("scan",), desktops=[7])

    assert stats["written"] == 1
    assert Path(thumb_path(cache, FrameKey(7, 1, "scan"))).is_file()
    assert not Path(thumb_path(cache, FrameKey(8, 1, "scan"))).exists()
    assert not (Path(cache) / "thumbs" / "oak1").exists()


# --------------------------------------------------------------------------
# build_thumbs: ROI crop
# --------------------------------------------------------------------------
def test_build_thumbs_crops_to_the_roi_so_the_chassis_fills_the_thumbnail(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(1000, 1000, box=RED_BOX)})
    key = FrameKey(7, 1, "scan")

    build_thumbs(cache)
    whole = cv2.imread(thumb_path(cache, key))
    build_thumbs(cache, force=True, roi_lookup=lambda k: RED_BOX)
    cropped = cv2.imread(thumb_path(cache, key))

    assert _red_fraction(whole) < 0.10  # 200x200 of 1000x1000
    assert _red_fraction(cropped) > 0.70  # the ROI, padded by 4%
    assert cropped.shape[:2] == (192, 192)


def test_build_thumbs_falls_back_to_the_whole_frame_when_the_lookup_has_no_roi(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(1000, 1000, box=RED_BOX)})

    stats = build_thumbs(cache, roi_lookup=lambda k: None)

    assert stats["written"] == 1
    assert _red_fraction(cv2.imread(thumb_path(cache, FrameKey(7, 1, "scan")))) < 0.10


# --------------------------------------------------------------------------
# build_thumbs: up-to-date check
# --------------------------------------------------------------------------
def test_build_thumbs_skips_an_up_to_date_thumb_rebuilds_a_stale_one_and_obeys_force(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,))
    key = FrameKey(7, 1, "scan")
    src = Path(cache) / "scan" / "D07" / "s001.png"
    dest = Path(thumb_path(cache, key))

    assert build_thumbs(cache)["written"] == 1
    stamp = _age(dest, +60)  # unambiguously newer than its source

    second = build_thumbs(cache)
    assert (second["written"], second["skipped"]) == (0, 1)
    assert dest.stat().st_mtime_ns == stamp

    forced = build_thumbs(cache, force=True)
    assert (forced["written"], forced["skipped"]) == (1, 0)
    assert dest.stat().st_mtime_ns != stamp

    _age(src, +600)  # the cached frame was rebuilt after the thumbnail
    assert build_thumbs(cache)["written"] == 1


# --------------------------------------------------------------------------
# build_thumbs: failures
# --------------------------------------------------------------------------
def test_build_thumbs_leaves_no_partial_file_when_the_encoder_fails(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))
    monkeypatch.setattr(ct.cv2, "imwrite", lambda *a, **k: False)

    stats = build_thumbs(cache)

    assert (stats["written"], stats["failed"]) == (0, 1)
    assert stats["failures"][0]["path"].endswith("s001.png")
    assert stats["failures"][0]["error"]
    assert list((Path(cache) / "thumbs" / "scan" / "D07").iterdir()) == []


def test_build_thumbs_leaves_no_partial_file_when_the_encoder_raises(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))

    def boom(*args, **kwargs):
        raise cv2.error("simulated encoder crash")

    monkeypatch.setattr(ct.cv2, "imwrite", boom)
    stats = build_thumbs(cache)

    assert (stats["written"], stats["failed"]) == (0, 1)
    assert list((Path(cache) / "thumbs" / "scan" / "D07").iterdir()) == []


def test_build_thumbs_counts_a_corrupt_source_and_still_writes_the_others(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2))
    (Path(cache) / "scan" / "D07" / "s001.png").write_bytes(b"not a PNG at all")

    stats = build_thumbs(cache)

    assert (stats["written"], stats["failed"], stats["missing_source"]) == (1, 1, 0)
    assert stats["failures"][0]["path"].endswith("s001.png")
    assert not Path(thumb_path(cache, FrameKey(7, 1, "scan"))).exists()
    assert Path(thumb_path(cache, FrameKey(7, 2, "scan"))).is_file()


def test_build_thumbs_reports_a_step_whose_cached_frame_is_missing(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2))
    os.remove(Path(cache) / "scan" / "D07" / "s001.png")

    stats = build_thumbs(cache)

    assert (stats["written"], stats["missing_source"], stats["failed"]) == (1, 1, 0)
    assert stats["failures"] == []


def test_build_thumbs_on_an_empty_cache_does_nothing(tmp_path):
    stats = build_thumbs(str(tmp_path / "nope"))
    assert stats["written"] == 0 and stats["failed"] == 0


# --------------------------------------------------------------------------
# DbRoiLookup
# --------------------------------------------------------------------------
def test_db_roi_lookup_reads_the_segment_that_contains_the_step(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db), start=4, end=9)

    with DbRoiLookup(str(db)) as lookup:
        assert lookup(FrameKey(7, 4, "scan")) == RED_BOX
        assert lookup(FrameKey(7, 9, "scan")) == RED_BOX
        assert lookup(FrameKey(7, 3, "scan")) is None   # before the segment
        assert lookup(FrameKey(7, 10, "scan")) is None  # after it
        assert lookup(FrameKey(7, 4, "oak1")) is None   # another view
        assert lookup(FrameKey(8, 4, "scan")) is None   # another desktop


def test_db_roi_lookup_never_writes_to_the_database_or_migrates_a_newer_schema(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db), version=99)  # written by a build this code does not know
    before, stamp = db.read_bytes(), db.stat().st_mtime_ns
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(1000, 1000, box=RED_BOX)})

    with DbRoiLookup(str(db)) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert stats["written"] == 1
    assert _red_fraction(cv2.imread(thumb_path(cache, FrameKey(7, 1, "scan")))) > 0.70
    assert db.read_bytes() == before
    assert db.stat().st_mtime_ns == stamp
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache", "tda.sqlite"]
    with pytest.raises(sqlite3.OperationalError):
        DbRoiLookup(str(db)).conn.execute("INSERT INTO meta(key, value) VALUES('x', 'y')")


def test_db_roi_lookup_ignores_a_segment_without_a_roi(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE pose_segment SET roi_json=NULL")
    conn.commit()
    conn.close()

    with DbRoiLookup(str(db)) as lookup:
        assert lookup(FrameKey(7, 1, "scan")) is None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def test_main_thumbs_only_builds_the_tier_and_prints_the_counts(tmp_path, capsys):
    cache = _make_cache(tmp_path, steps=(1, 2))

    code = main(["--cache", cache, "--thumbs-only", "--views", "scan",
                 "--first", "7", "--last", "7", "--thumb-side", "64"])

    assert code == 0
    assert "written=2" in capsys.readouterr().out
    assert max(cv2.imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2]) == 64
    assert not (Path(cache) / "index.json").exists()  # no index was needed


def test_main_thumbs_only_skips_then_rebuilds_with_force_thumbs(tmp_path, capsys):
    cache = _make_cache(tmp_path, steps=(1,))
    argv = ["--cache", cache, "--thumbs-only", "--first", "7", "--last", "7"]

    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv) == 0
    assert "skipped=1" in capsys.readouterr().out
    assert main(argv + ["--force-thumbs"]) == 0
    assert "written=1" in capsys.readouterr().out


def test_main_builds_the_cache_and_then_the_thumbnails(tmp_path, capsys):
    index_path = _oak_index(tmp_path)
    cache = str(tmp_path / "cache")

    code = main(["--index", index_path, "--cache", cache, "--views", "oak1",
                 "--first", "7", "--last", "7", "--thumbs"])

    assert code == 0
    assert (Path(cache) / "oak1" / "D07" / "s001.jpg").is_file()
    assert Path(thumb_path(cache, FrameKey(7, 1, "oak1"))).is_file()
    out = capsys.readouterr().out
    assert "copied=1" in out and "written=1" in out


def test_main_thumbs_only_returns_non_zero_when_a_thumbnail_fails(tmp_path, capsys):
    cache = _make_cache(tmp_path, steps=(1,))
    (Path(cache) / "scan" / "D07" / "s001.png").write_bytes(b"broken")

    code = main(["--cache", cache, "--thumbs-only", "--first", "7", "--last", "7"])

    assert code != 0
    assert "failed=1" in capsys.readouterr().out
