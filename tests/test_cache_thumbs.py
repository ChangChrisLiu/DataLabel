"""Tests for the offline thumbnail tier of the local cache (spec 2.4).

Every fixture is synthetic and lives under ``tmp_path``: a miniature cache
(``<cache>/<view>/D<nn>/s<kkk>.<ext>`` plus its ``manifest.json``) and, where a
ROI is needed, a throw-away WAL annotation database built straight from
``schema.sql``.  No test reads the real cache, the real database or F:.

Images are written and read through ``imencode``/``imdecode`` here too, so the
fixtures work under the non-ASCII path one of the tests uses.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import tda.core.cache as cache_module
import tda.core.cache_thumbs as ct
from tda.core.cache import main, suggest_roi
from tda.core.cache_thumbs import DbRoiLookup, build_thumbs, thumb_path
from tda.core.index import DesktopIndex, FrameFile, save_index
from tda.core.model import FrameKey

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "tda" / "core" / "schema.sql"
RED_BOX = (600, 100, 800, 300)  # the "chassis" of the synthetic 1000x1000 frame
GREEN_BOX = (100, 600, 300, 800)  # ... and where it sits after a reorient
CHASSIS = (300, 250, 700, 800)  # a dark chassis suggest_roi() really finds


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _imwrite(path, img: np.ndarray) -> None:
    """Write an image without cv2 touching the path (non-ASCII safe)."""
    ok, buf = cv2.imencode(Path(path).suffix, img)
    assert ok
    buf.tofile(str(path))


def _imread(path) -> np.ndarray:
    """Read an image without cv2 touching the path (non-ASCII safe)."""
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _frame(width: int = 1000, height: int = 1000, box=None) -> np.ndarray:
    """A flat gray BGR frame with an optional red rectangle at ``box``."""
    img = np.full((height, width, 3), 120, np.uint8)
    if box is not None:
        x0, y0, x1, y1 = box
        img[y0:y1, x0:x1] = (0, 0, 220)  # BGR: red
    return img


def _scan_frame(chassis, size: int = 1000, tape=(120, 880), band: int = 30) -> np.ndarray:
    """A scanner-like frame: white board, yellow tape square, dark chassis.

    This is the shape :func:`tda.core.cache.suggest_roi` was written for, so the
    automatic ROI really is measured here rather than faked.
    """
    img = np.full((size, size, 3), 240, np.uint8)
    lo, hi = tape
    img[lo:hi, lo:hi] = (255, 255, 255)
    for band_slice in ((slice(lo, lo + band), slice(lo, hi)),
                       (slice(hi - band, hi), slice(lo, hi)),
                       (slice(lo, hi), slice(lo, lo + band)),
                       (slice(lo, hi), slice(hi - band, hi))):
        img[band_slice] = (0, 200, 255)  # BGR: the yellow/orange tape
    x0, y0, x1, y1 = chassis
    img[y0:y1, x0:x1] = (45, 45, 45)
    return img


def _two_box_frame(size: int = 1000) -> np.ndarray:
    """A frame with a red patch and a green one, to tell two ROIs apart."""
    img = np.full((size, size, 3), 120, np.uint8)
    img[RED_BOX[1]:RED_BOX[3], RED_BOX[0]:RED_BOX[2]] = (0, 0, 220)
    img[GREEN_BOX[1]:GREEN_BOX[3], GREEN_BOX[0]:GREEN_BOX[2]] = (0, 200, 0)
    return img


def _red_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that survived JPEG as clearly red."""
    return float(((img[:, :, 2] > 150) & (img[:, :, 0] < 100)).mean())


def _green_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that survived JPEG as clearly green."""
    return float(((img[:, :, 1] > 130) & (img[:, :, 2] < 110)).mean())


def _groups(stats: dict, index: int = 0) -> list[dict]:
    """The ROI groups one desktop+view was cut with."""
    return stats["rois"][index]["groups"]


def _dark_fraction(img: np.ndarray) -> float:
    """Fraction of pixels that survived JPEG as clearly dark (the chassis)."""
    return float((img.max(axis=2) < 90).mean())


def _reference_thumb(img: np.ndarray, box, max_side: int = 192) -> np.ndarray:
    """What the module should produce for ``img`` cropped to ``box``."""
    height, width = img.shape[:2]
    x0, y0, x1, y1 = box
    px, py = int(round((x1 - x0) * 0.04)), int(round((y1 - y0) * 0.04))
    crop = img[max(0, y0 - py):min(height, y1 + py), max(0, x0 - px):min(width, x1 + px)]
    h, w = crop.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    return cv2.resize(crop, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _close(got: np.ndarray, want: np.ndarray) -> bool:
    """Same picture, allowing for JPEG."""
    return (got.shape == want.shape
            and float(np.abs(got.astype(int) - want.astype(int)).mean()) < 6.0)


def _make_cache(root: Path, view: str = "scan", desktop: int = 7, steps=(1, 2),
                images: dict | None = None) -> str:
    """A cache dir holding full-size frames for ``steps`` plus their manifest."""
    cache = root / "cache"
    ddir = cache / view / f"D{desktop:02d}"
    ddir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if view.startswith("oak") else "png"
    manifest = {}
    for step in steps:
        img = (images or {}).get(step)
        _imwrite(ddir / f"s{step:03d}.{ext}", _frame() if img is None else img)
        manifest[str(step)] = {"chosen": 0, "reason": "p0", "metrics": [],
                               "src": f"F:/fake/D{desktop}/s{step}.{ext}"}
    (ddir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(cache)


def _sidecar(cache: str, view: str = "scan", desktop: int = 7) -> dict:
    """The recorded ROI plan of one desktop+view."""
    path = Path(cache) / "thumbs" / view / f"D{desktop:02d}" / "thumbs.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _age(path, seconds: float) -> int:
    """Shift a file's mtime by ``seconds``; returns the new mtime in ns."""
    stamp = os.stat(path).st_mtime_ns + int(seconds * 1e9)
    os.utime(path, ns=(stamp, stamp))
    return stamp


def _make_db(path: str, *, version: int = 2, desktop: int = 7, view: str = "scan",
             roi=RED_BOX, start: int = 1, end: int = 50, segments=None) -> None:
    """A WAL database (like the real one) with pose segments carrying ROIs.

    ``segments`` is ``[(seg, start_step, end_step, roi or None), ...]``; without
    it one segment 0 covering ``start..end`` with ``roi`` is written.
    """
    rows = segments if segments is not None else [(0, start, end, roi)]
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.execute("INSERT INTO desktop(id) VALUES(?)", (desktop,))
    for seg, first, last, box in rows:
        conn.execute(
            "INSERT INTO pose_segment(desktop, view, seg, start_step, end_step, ref_step,"
            " roi_json) VALUES(?,?,?,?,?,?,?)",
            (desktop, view, seg, first, last, first,
             None if box is None else json.dumps(list(box))),
        )
    conn.execute("INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(version),))
    conn.commit()
    conn.close()


def _set_roi(path: str, seg: int, roi, desktop: int = 7, view: str = "scan") -> None:
    """Change one segment's ROI, the way the annotator would."""
    conn = sqlite3.connect(path)
    conn.execute("UPDATE pose_segment SET roi_json=? WHERE desktop=? AND view=? AND seg=?",
                 (None if roi is None else json.dumps(list(roi)), desktop, view, seg))
    conn.commit()
    conn.close()


def _sql(path: str, *statements: str) -> None:
    """Reshape the segments the way ``update_pose_segment``/``delete`` would."""
    conn = sqlite3.connect(path)
    for statement in statements:
        conn.execute(statement)
    conn.commit()
    conn.close()


def _tables(path: Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        rows = conn.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _schema_version(path: Path) -> str | None:
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    finally:
        conn.close()
    return None if row is None else row[0]


def _oak_index(tmp_path: Path) -> str:
    """A one-frame index of a single OAK-style image, saved to disk."""
    src = tmp_path / "src" / "frame.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    _imwrite(src, _frame(400, 300))
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


@pytest.mark.parametrize("order", [
    "tda.core.cache_roi, tda.core.cache_thumbs, tda.core.cache",
    "tda.core.cache_thumbs, tda.core.cache, tda.core.cache_roi",
    "tda.core.cache, tda.core.cache_roi, tda.core.cache_thumbs",
])
def test_the_three_modules_import_in_any_order(order):
    # cache imports cache_thumbs, which imports cache back from inside its
    # functions: a fresh interpreter is the only honest test of that
    done = subprocess.run([sys.executable, "-c", f"import {order}; print('ok')"],
                          cwd=str(REPO_ROOT), capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert "ok" in done.stdout


def test_the_tier_is_reachable_through_the_cache_module():
    from tda.core.cache import DbRoiLookup as ReDb
    from tda.core.cache import build_thumbs as ReBuild
    from tda.core.cache import thumb_path as ReThumb

    assert (ReThumb, ReBuild, ReDb) == (thumb_path, build_thumbs, DbRoiLookup)
    names = {"DbRoiLookup", "build_thumbs", "thumb_path"}
    assert names <= set(cache_module.__all__)
    assert names <= set(dir(cache_module))


# --------------------------------------------------------------------------
# build_thumbs: size, aspect, format
# --------------------------------------------------------------------------
def test_build_thumbs_scales_every_step_to_the_max_side_keeping_the_aspect(tmp_path):
    # one desktop per frame shape: within a desktop every cached frame is the same size
    cache = _make_cache(tmp_path, desktop=7, steps=(1,), images={1: _frame(800, 400)})
    _make_cache(tmp_path, desktop=8, steps=(1,), images={1: _frame(300, 900)})

    stats = build_thumbs(cache, max_side=192)

    assert stats["written"] == 2
    assert (stats["skipped"], stats["missing_source"], stats["failed"]) == (0, 0, 0)
    assert _imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2] == (96, 192)
    assert _imread(thumb_path(cache, FrameKey(8, 1, "scan"))).shape[:2] == (192, 64)


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
    assert sorted(p.name for p in dest.parent.iterdir()) == ["s001.jpg", "thumbs.json"]


def test_build_thumbs_honours_the_view_and_desktop_filters(tmp_path):
    cache = _make_cache(tmp_path, desktop=7, steps=(1,))
    _make_cache(tmp_path, desktop=8, steps=(1,))
    _make_cache(tmp_path, view="oak1", desktop=7, steps=(1,))

    stats = build_thumbs(cache, views=("scan",), desktops=[7])

    assert stats["written"] == 1
    assert Path(thumb_path(cache, FrameKey(7, 1, "scan"))).is_file()
    assert not Path(thumb_path(cache, FrameKey(8, 1, "scan"))).exists()
    assert not (Path(cache) / "thumbs" / "oak1").exists()


def test_build_thumbs_works_under_a_non_ascii_path(tmp_path):
    cache = _make_cache(tmp_path / "测试_桌面拆解", steps=(1,), images={1: _frame(800, 400)})

    stats = build_thumbs(cache, auto_roi=False)

    assert (stats["written"], stats["failed"]) == (1, 0)
    assert _imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2] == (96, 192)


# --------------------------------------------------------------------------
# build_thumbs: the ROI
# --------------------------------------------------------------------------
def test_build_thumbs_crops_to_the_roi_so_the_chassis_fills_the_thumbnail(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(1000, 1000, box=RED_BOX)})
    key = FrameKey(7, 1, "scan")

    build_thumbs(cache, auto_roi=False)
    whole = _imread(thumb_path(cache, key))
    build_thumbs(cache, force=True, roi_lookup=lambda k: RED_BOX)
    cropped = _imread(thumb_path(cache, key))

    assert _red_fraction(whole) < 0.10  # 200x200 of 1000x1000
    assert _red_fraction(cropped) > 0.70  # the ROI, padded by 4%
    assert cropped.shape[:2] == (192, 192)


def test_build_thumbs_uses_one_stable_auto_roi_for_every_step_of_a_desktop(tmp_path):
    first, later = _scan_frame(CHASSIS), _scan_frame((250, 300, 650, 700))
    cache = _make_cache(tmp_path, steps=(1, 2), images={1: first, 2: later})
    box = suggest_roi(first, "scan")
    assert suggest_roi(later, "scan") != box  # per-frame suggestions really do differ

    stats = build_thumbs(cache)

    assert _groups(stats) == [{"source": "auto", "segment": None, "steps": [[1, 2]],
                               "box": list(box)}]
    for step, img in ((1, first), (2, later)):
        got = _imread(thumb_path(cache, FrameKey(7, step, "scan")))
        assert _close(got, _reference_thumb(img, box))  # the reference box, not their own
    assert _dark_fraction(_imread(thumb_path(cache, FrameKey(7, 1, "scan")))) > 0.60


def test_build_thumbs_prefers_the_looked_up_roi_over_the_automatic_one(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _scan_frame(CHASSIS)})

    stats = build_thumbs(cache, roi_lookup=lambda k: RED_BOX)

    assert _groups(stats) == [{"source": "db", "segment": None, "steps": [[1, 1]],
                               "box": list(RED_BOX)}]
    assert _sidecar(cache)["groups"] == _groups(stats)


def test_build_thumbs_without_auto_roi_keeps_the_whole_frame(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _scan_frame(CHASSIS)})

    stats = build_thumbs(cache, auto_roi=False, roi_lookup=lambda k: None)

    assert _groups(stats) == [{"source": "none", "segment": None, "steps": [[1, 1]], "box": None}]
    assert _close(_imread(thumb_path(cache, FrameKey(7, 1, "scan"))),
                  _reference_thumb(_scan_frame(CHASSIS), (0, 0, 1000, 1000)))


def test_build_thumbs_falls_back_to_the_whole_frame_when_no_roi_can_be_measured(tmp_path):
    # a frame suggest_roi() cannot read at all: the plan has to degrade, not raise
    cache = _make_cache(tmp_path, steps=(1,))
    (Path(cache) / "scan" / "D07" / "s001.png").write_bytes(b"not a PNG")

    stats = build_thumbs(cache)

    assert _groups(stats) == [{"source": "none", "segment": None, "steps": [[1, 1]], "box": None}]
    assert (stats["written"], stats["failed"]) == (0, 1)


def test_build_thumbs_degrades_a_chassis_box_that_is_too_small_to_the_central_crop(tmp_path):
    # a light-coloured chassis leaves suggest_roi() on the motherboard (~13% of
    # the frame, the real D64): below ROI_MIN_AREA_FRAC, so the central box wins
    tiny, real = _scan_frame((450, 450, 800, 800)), _scan_frame((350, 350, 850, 850))
    cache = _make_cache(tmp_path, desktop=7, steps=(1,), images={1: tiny})
    _make_cache(tmp_path, desktop=8, steps=(1,), images={1: real})

    stats = build_thumbs(cache)

    assert _groups(stats, 0) == [{"source": "auto", "segment": None, "steps": [[1, 1]],
                                  "box": [150, 150, 850, 850]}]  # the central 70%
    assert _groups(stats, 1)[0]["box"] == list(suggest_roi(real, "scan"))  # its own chassis


@pytest.mark.parametrize("bad", [(300, 300, 300, 400),  # zero width
                                 (700, 300, 200, 400),  # inverted
                                 (2000, 2000, 2400, 2400)])  # entirely outside
def test_build_thumbs_ignores_a_degenerate_roi(tmp_path, bad):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(800, 400)})

    stats = build_thumbs(cache, auto_roi=False, roi_lookup=lambda k: bad)

    assert stats["written"] == 1
    assert _imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2] == (96, 192)


# --------------------------------------------------------------------------
# build_thumbs: the recorded plan and the up-to-date check
# --------------------------------------------------------------------------
def test_build_thumbs_records_the_plan_it_used(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2), images={1: _scan_frame(CHASSIS)})

    build_thumbs(cache, max_side=64, quality=70)

    record = _sidecar(cache)
    assert record["groups"] == [{"source": "auto", "segment": None, "steps": [[1, 2]],
                                 "box": list(suggest_roi(_scan_frame(CHASSIS), "scan"))}]
    assert (record["max_side"], record["quality"]) == (64, 70)
    assert record["built_at"]


@pytest.mark.parametrize("older", ["single_box", "first_last_pair"])
def test_build_thumbs_treats_a_sidecar_in_an_older_format_as_stale(tmp_path, older):
    cache = _make_cache(tmp_path, steps=(1,))
    assert build_thumbs(cache)["written"] == 1
    sidecar = Path(cache) / "thumbs" / "scan" / "D07" / ct.SIDECAR_NAME
    box = _sidecar(cache)["groups"][0]["box"]
    record = {"source": "auto", "box": box} if older == "single_box" else {
        "groups": [{"source": "auto", "segment": None, "steps": [1, 1], "box": box}]}
    record.update(max_side=192, quality=85, built_at="old")
    sidecar.write_text(json.dumps(record), encoding="utf-8")
    _age(Path(thumb_path(cache, FrameKey(7, 1, "scan"))), +60)

    stats = build_thumbs(cache)  # unreadable plan: rebuild rather than trust it

    assert stats["written"] == 1
    assert _sidecar(cache)["groups"][0]["steps"] == [[1, 1]]


def test_build_thumbs_rebuilds_an_uncropped_group_that_was_never_recorded(tmp_path):
    # "no box recorded" must not read as "the recorded box was None, so unchanged":
    # that is how a desktop whose crop was withdrawn kept its old, cropped thumbs
    cache = _make_cache(tmp_path, steps=(1,), images={1: _scan_frame(CHASSIS)})
    assert build_thumbs(cache)["written"] == 1  # cropped to the measured chassis
    _age(Path(thumb_path(cache, FrameKey(7, 1, "scan"))), +60)
    old = {"source": "auto", "box": list(suggest_roi(_scan_frame(CHASSIS), "scan")),
           "max_side": 192, "quality": 85, "built_at": "old"}  # same size, no groups
    (Path(cache) / "thumbs" / "scan" / "D07" / ct.SIDECAR_NAME).write_text(
        json.dumps(old), encoding="utf-8")

    assert build_thumbs(cache, auto_roi=False)["written"] == 1  # the crop was withdrawn


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


def test_build_thumbs_rebuilds_a_desktop_whose_recorded_roi_changed(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2), images={1: _scan_frame(CHASSIS)})
    assert build_thumbs(cache)["written"] == 2
    for step in (1, 2):
        _age(Path(thumb_path(cache, FrameKey(7, step, "scan"))), +60)
    assert build_thumbs(cache)["skipped"] == 2  # same plan, nothing to do

    changed = build_thumbs(cache, roi_lookup=lambda k: RED_BOX)  # a --db run arrives

    assert (changed["written"], changed["skipped"]) == (2, 0)
    assert _sidecar(cache)["groups"][0]["box"] == list(RED_BOX)


# --------------------------------------------------------------------------
# build_thumbs: per-pose-segment ROIs
# --------------------------------------------------------------------------
def _reoriented(tmp_path: Path, steps=(1, 2, 3, 4)) -> tuple[str, str]:
    """A cache of identical two-patch frames plus a DB cut into two segments."""
    cache = _make_cache(tmp_path, steps=steps, images={s: _two_box_frame() for s in steps})
    db = str(tmp_path / "tda.sqlite")
    _make_db(db, segments=[(0, 1, 2, RED_BOX), (1, 3, 4, GREEN_BOX)])
    return cache, db


def test_build_thumbs_cuts_every_pose_segment_with_its_own_roi(tmp_path):
    cache, db = _reoriented(tmp_path)

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert stats["written"] == 4
    assert _groups(stats) == [
        {"source": "db", "segment": 0, "steps": [[1, 2]], "box": list(RED_BOX)},
        {"source": "db", "segment": 1, "steps": [[3, 4]], "box": list(GREEN_BOX)},
    ]
    for step in (1, 2):
        assert _red_fraction(_imread(thumb_path(cache, FrameKey(7, step, "scan")))) > 0.70
    for step in (3, 4):
        assert _green_fraction(_imread(thumb_path(cache, FrameKey(7, step, "scan")))) > 0.70


def test_build_thumbs_follows_a_frames_pose_segment_override(tmp_path):
    cache, db = _reoriented(tmp_path)
    conn = sqlite3.connect(db)  # step 2 was reassigned to the second segment
    conn.execute("INSERT INTO frame(desktop, step, view, pose_segment) VALUES(7, 2, 'scan', 1)")
    conn.commit()
    conn.close()

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert [(g["segment"], g["steps"]) for g in _groups(stats)] == [(0, [[1, 1]]), (1, [[2, 4]])]
    assert _green_fraction(_imread(thumb_path(cache, FrameKey(7, 2, "scan")))) > 0.70


def test_build_thumbs_rebuilds_only_the_segment_whose_roi_was_adjusted(tmp_path):
    cache, db = _reoriented(tmp_path)
    with DbRoiLookup(db) as lookup:
        assert build_thumbs(cache, roi_lookup=lookup)["written"] == 4
    stamps = {s: _age(Path(thumb_path(cache, FrameKey(7, s, "scan"))), +60) for s in (1, 2, 3, 4)}
    _set_roi(db, 1, (120, 620, 320, 820))  # the operator nudges segment 1 only

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    def stamp(step: int) -> int:
        return Path(thumb_path(cache, FrameKey(7, step, "scan"))).stat().st_mtime_ns

    assert (stats["written"], stats["skipped"]) == (2, 2)
    assert [stamp(s) for s in (1, 2)] == [stamps[1], stamps[2]]  # untouched segment
    assert all(stamp(s) != stamps[s] for s in (3, 4))  # the adjusted one


def test_build_thumbs_rebuilds_a_step_that_moved_to_another_segment(tmp_path):
    # update_pose_segment() moved the boundary; neither roi_json was touched, so
    # nothing but step 3's *membership* changed - and its file is cut with the
    # wrong box until that alone is noticed
    cache, db = _reoriented(tmp_path)
    with DbRoiLookup(db) as lookup:
        assert build_thumbs(cache, roi_lookup=lookup)["written"] == 4
    stamps = {s: _age(Path(thumb_path(cache, FrameKey(7, s, "scan"))), +60) for s in (1, 2, 3, 4)}
    _sql(db, "UPDATE pose_segment SET end_step=3 WHERE seg=0",
         "UPDATE pose_segment SET start_step=4 WHERE seg=1")

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    def stamp(step: int) -> int:
        return Path(thumb_path(cache, FrameKey(7, step, "scan"))).stat().st_mtime_ns

    assert (stats["written"], stats["skipped"]) == (1, 3)
    assert _red_fraction(_imread(thumb_path(cache, FrameKey(7, 3, "scan")))) > 0.70
    assert [stamp(s) for s in (1, 2, 4)] == [stamps[1], stamps[2], stamps[4]]
    assert _sidecar(cache)["groups"] == [
        {"source": "db", "segment": 0, "steps": [[1, 3]], "box": list(RED_BOX)},
        {"source": "db", "segment": 1, "steps": [[4, 4]], "box": list(GREEN_BOX)},
    ]


def test_build_thumbs_rebuilds_the_steps_of_a_segment_that_was_deleted(tmp_path):
    cache, db = _reoriented(tmp_path)
    with DbRoiLookup(db) as lookup:
        assert build_thumbs(cache, roi_lookup=lookup)["written"] == 4
    stamps = {s: _age(Path(thumb_path(cache, FrameKey(7, s, "scan"))), +60) for s in (1, 2, 3, 4)}
    _sql(db, "DELETE FROM pose_segment WHERE seg=1",
         "UPDATE pose_segment SET end_step=4 WHERE seg=0")  # absorbed by its neighbour

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert (stats["written"], stats["skipped"]) == (2, 2)
    for step in (3, 4):
        thumb = Path(thumb_path(cache, FrameKey(7, step, "scan")))
        assert _red_fraction(_imread(thumb)) > 0.70  # the absorbing segment's box
        assert thumb.stat().st_mtime_ns != stamps[step]
    assert _sidecar(cache)["groups"] == [
        {"source": "db", "segment": 0, "steps": [[1, 4]], "box": list(RED_BOX)}]


def test_build_thumbs_records_a_non_contiguous_membership_exactly(tmp_path):
    cache, db = _reoriented(tmp_path)
    _sql(db, "INSERT INTO frame(desktop, step, view, pose_segment)"
             " VALUES(7, 1, 'scan', 1)")  # step 1 belongs to the second segment

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert _groups(stats) == [
        {"source": "db", "segment": 1, "steps": [[1, 1], [3, 4]], "box": list(GREEN_BOX)},
        {"source": "db", "segment": 0, "steps": [[2, 2]], "box": list(RED_BOX)},
    ]
    assert _sidecar(cache)["groups"] == _groups(stats)


def test_build_thumbs_rewrites_the_plan_when_only_the_membership_changed(tmp_path):
    # both segments carry the same box, so no thumbnail needs rebuilding when one
    # absorbs the other - but the record must still say who owns which steps
    cache = _make_cache(tmp_path, steps=(1, 2, 3, 4),
                        images={s: _two_box_frame() for s in (1, 2, 3, 4)})
    db = str(tmp_path / "tda.sqlite")
    _make_db(db, segments=[(0, 1, 2, RED_BOX), (1, 3, 4, RED_BOX)])
    with DbRoiLookup(db) as lookup:
        assert build_thumbs(cache, roi_lookup=lookup)["written"] == 4
    for step in (1, 2, 3, 4):
        _age(Path(thumb_path(cache, FrameKey(7, step, "scan"))), +60)
    _sql(db, "DELETE FROM pose_segment WHERE seg=1",
         "UPDATE pose_segment SET end_step=4 WHERE seg=0")

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    assert (stats["written"], stats["skipped"]) == (0, 4)
    assert _sidecar(cache)["groups"] == [
        {"source": "db", "segment": 0, "steps": [[1, 4]], "box": list(RED_BOX)}]


def test_build_thumbs_leaves_the_plan_untouched_when_nothing_changed(tmp_path):
    # every run would otherwise dirty 66 sidecars with a new built_at
    cache = _make_cache(tmp_path, steps=(1, 2))
    assert build_thumbs(cache)["written"] == 2
    for step in (1, 2):
        _age(Path(thumb_path(cache, FrameKey(7, step, "scan"))), +60)
    sidecar = Path(cache) / "thumbs" / "scan" / "D07" / ct.SIDECAR_NAME
    before, stamp = sidecar.read_bytes(), sidecar.stat().st_mtime_ns

    stats = build_thumbs(cache)

    assert (stats["written"], stats["skipped"]) == (0, 2)
    assert sidecar.read_bytes() == before
    assert sidecar.stat().st_mtime_ns == stamp


def test_build_thumbs_gives_a_segment_without_a_roi_the_automatic_box(tmp_path):
    cache, db = _reoriented(tmp_path)
    _set_roi(db, 1, None)  # the second segment has no ROI yet

    with DbRoiLookup(db) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    groups = _groups(stats)
    assert [(g["source"], g["steps"]) for g in groups] == [("db", [[1, 2]]), ("auto", [[3, 4]])]
    assert groups[1]["box"] == list(suggest_roi(_two_box_frame(), "scan"))  # not the red box
    for step in (3, 4):
        thumb = _imread(thumb_path(cache, FrameKey(7, step, "scan")))
        assert _red_fraction(thumb) < 0.10 and _green_fraction(thumb) < 0.30


def test_build_thumbs_rebuilds_when_the_thumbnail_size_changed(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,))
    build_thumbs(cache)
    _age(Path(thumb_path(cache, FrameKey(7, 1, "scan"))), +60)

    assert build_thumbs(cache, max_side=64)["written"] == 1
    assert max(_imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2]) == 64


# --------------------------------------------------------------------------
# build_thumbs: failures
# --------------------------------------------------------------------------
def _no_files_under(cache: str) -> bool:
    """No thumbnail and no temp file was left behind (the plan record may exist)."""
    return not any(p.is_file() and p.name != ct.SIDECAR_NAME
                   for p in (Path(cache) / "thumbs").rglob("*"))


def test_build_thumbs_leaves_no_partial_file_when_the_encoder_fails(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))
    monkeypatch.setattr(ct.cv2, "imencode", lambda *a, **k: (False, None))

    stats = build_thumbs(cache, auto_roi=False)

    assert (stats["written"], stats["failed"]) == (0, 1)
    assert stats["failures"][0]["path"].endswith("s001.png")
    assert stats["failures"][0]["error"]
    assert _no_files_under(cache)


def test_build_thumbs_leaves_no_partial_file_when_the_encoder_raises(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))

    def boom(*args, **kwargs):
        raise cv2.error("simulated encoder crash")

    monkeypatch.setattr(ct.cv2, "imencode", boom)
    stats = build_thumbs(cache, auto_roi=False)

    assert (stats["written"], stats["failed"]) == (0, 1)
    assert _no_files_under(cache)


def test_build_thumbs_removes_its_temp_file_when_the_replace_fails(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))
    real = os.replace

    def picky(src, dst, *args, **kwargs):
        if str(dst).endswith(".jpg"):
            raise OSError("simulated replace failure")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(ct.os, "replace", picky)
    stats = build_thumbs(cache, auto_roi=False)

    assert (stats["written"], stats["failed"]) == (0, 1)
    assert _no_files_under(cache)


def test_build_thumbs_leaves_no_partial_sidecar_when_its_replace_fails(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1,))
    real = os.replace

    def picky(src, dst, *args, **kwargs):
        if str(dst).endswith(ct.SIDECAR_NAME):
            raise OSError("simulated replace failure")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(ct.os, "replace", picky)
    with pytest.raises(OSError):
        build_thumbs(cache, auto_roi=False)

    assert [p.name for p in (Path(cache) / "thumbs" / "scan" / "D07").iterdir()] == ["s001.jpg"]


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
    assert stats["written"] == 0 and stats["failed"] == 0 and stats["rois"] == []


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


def test_db_roi_lookup_honours_the_frame_pose_segment_override(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db))  # segment 0 covers steps 1-50 with RED_BOX
    other = (10, 20, 110, 220)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO pose_segment(desktop, view, seg, start_step, end_step, ref_step,"
                 " roi_json) VALUES(7, 'scan', 2, 60, 90, 60, ?)", (json.dumps(list(other)),))
    conn.execute("INSERT INTO frame(desktop, step, view, pose_segment) VALUES(7, 5, 'scan', 2)")
    conn.execute("INSERT INTO frame(desktop, step, view, pose_segment) VALUES(7, 6, 'scan', NULL)")
    conn.commit()
    conn.close()

    with DbRoiLookup(str(db)) as lookup:
        assert lookup(FrameKey(7, 5, "scan")) == other     # the frame's own segment wins
        assert lookup(FrameKey(7, 6, "scan")) == RED_BOX   # NULL: the step range decides
        assert lookup(FrameKey(7, 7, "scan")) == RED_BOX   # no frame row at all


def test_db_roi_lookup_leaves_the_database_alone_and_never_migrates_it(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db), version=99)  # written by a build this code does not know
    before, stamp = db.read_bytes(), db.stat().st_mtime_ns
    tables = _tables(db)
    cache = _make_cache(tmp_path, steps=(1,), images={1: _frame(1000, 1000, box=RED_BOX)})

    with DbRoiLookup(str(db)) as lookup:
        stats = build_thumbs(cache, roi_lookup=lookup)

    # Opening a WAL database read-only may create -wal/-shm side files; what must
    # not change is the database itself, its version and its set of tables.
    assert stats["written"] == 1
    assert _red_fraction(_imread(thumb_path(cache, FrameKey(7, 1, "scan")))) > 0.70
    assert db.read_bytes() == before
    assert db.stat().st_mtime_ns == stamp
    assert _schema_version(db) == "99"
    assert _tables(db) == tables
    with pytest.raises(sqlite3.OperationalError):
        DbRoiLookup(str(db)).conn.execute("INSERT INTO meta(key, value) VALUES('x', 'y')")


def test_db_roi_lookup_notes_a_lock_file_and_reads_anyway(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db))
    (tmp_path / "tda.sqlite.lock").write_text("annotator", encoding="utf-8")
    said: list[str] = []

    with DbRoiLookup(str(db), notify=said.append) as lookup:
        assert lookup(FrameKey(7, 1, "scan")) == RED_BOX

    assert len(said) == 1 and ".lock" in said[0]


class _PickyConn:
    """A connection whose ``frame`` query fails; everything else works."""

    def __init__(self, conn, message: str) -> None:
        self._conn, self._message = conn, message

    def execute(self, sql, *args):
        if "FROM frame" in sql:
            raise sqlite3.OperationalError(self._message)
        return self._conn.execute(sql, *args)

    def close(self) -> None:
        self._conn.close()


def _break_frame_query(lookup: DbRoiLookup, message: str) -> None:
    """Make only the per-frame override query fail, with ``message``."""
    lookup.conn = _PickyConn(lookup.conn, message)


def test_db_roi_lookup_survives_a_database_without_a_frame_table(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db))

    with DbRoiLookup(str(db)) as lookup:
        _break_frame_query(lookup, "no such table: frame")
        assert lookup(FrameKey(7, 1, "scan")) == RED_BOX  # the step ranges still decide


def test_db_roi_lookup_re_raises_an_error_that_is_not_a_missing_table(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db))

    with DbRoiLookup(str(db)) as lookup:
        _break_frame_query(lookup, "database is locked")
        with pytest.raises(sqlite3.OperationalError):
            lookup(FrameKey(7, 1, "scan"))


def test_db_roi_lookup_reports_the_segment_a_frame_belongs_to(tmp_path):
    db = tmp_path / "tda.sqlite"
    _make_db(str(db), segments=[(0, 1, 2, RED_BOX), (1, 3, 4, GREEN_BOX)])

    with DbRoiLookup(str(db)) as lookup:
        assert lookup.segment_of(FrameKey(7, 1, "scan")) == 0
        assert lookup.segment_of(FrameKey(7, 4, "scan")) == 1
        assert lookup.segment_of(FrameKey(7, 9, "scan")) is None


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
    assert max(_imread(thumb_path(cache, FrameKey(7, 1, "scan"))).shape[:2]) == 64
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


def test_main_no_auto_roi_switches_the_automatic_crop_off(tmp_path, capsys):
    cache = _make_cache(tmp_path, steps=(1,), images={1: _scan_frame(CHASSIS)})
    argv = ["--cache", cache, "--thumbs-only", "--first", "7", "--last", "7"]

    assert main(argv + ["--no-auto-roi"]) == 0
    assert "roi=none" in capsys.readouterr().out
    assert [g["source"] for g in _sidecar(cache)["groups"]] == ["none"]
    assert main(argv) == 0  # the plan changed, so the tier is rebuilt
    assert "written=1" in capsys.readouterr().out
    assert [g["source"] for g in _sidecar(cache)["groups"]] == ["auto"]


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


def test_main_without_the_thumb_flags_builds_no_thumbnails(tmp_path, capsys):
    index_path = _oak_index(tmp_path)
    cache = str(tmp_path / "cache")

    assert main(["--index", index_path, "--cache", cache, "--views", "oak1",
                 "--first", "7", "--last", "7"]) == 0

    assert not (Path(cache) / "thumbs").exists()
    assert "thumbs written" not in capsys.readouterr().out


def test_main_thumbs_only_returns_non_zero_when_a_thumbnail_fails(tmp_path, capsys):
    cache = _make_cache(tmp_path, steps=(1,))
    (Path(cache) / "scan" / "D07" / "s001.png").write_bytes(b"broken")

    code = main(["--cache", cache, "--thumbs-only", "--first", "7", "--last", "7"])

    assert code != 0
    assert "failed=1" in capsys.readouterr().out
