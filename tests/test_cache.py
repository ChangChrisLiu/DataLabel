"""Tests for the local image cache, burst selection and ROI suggestion.

Synthetic bursts are written under ``tmp_path`` with the real ``P_k.png``
naming; the only real images used are the checked-in scanner fixtures.  No test
touches the read-only F: drive.
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.core.cache import (
    build_cache,
    burst_metrics,
    cache_path,
    choose_scan_image,
    suggest_roi,
)
from tda.core.index import DesktopIndex, FrameFile
from tda.core.model import FrameKey

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNER_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "images" / "scanner.png"
CROP_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "images" / "scanner_crop_native.png"
# Ground truth measured on scanner.png: bbox of the largest dark component.
CHASSIS = (349, 143, 790, 639)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _texture(seed: int = 0, size: int = 256, block: int = 16, amp: int = 30,
             base: int = 120, noise: float = 4.0) -> np.ndarray:
    """A checkerboard-textured BGR frame with sensor-like noise (so that a burst
    has non-degenerate ``dist_to_median`` values, like the real scanner)."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size]
    img = np.full((size, size), float(base), np.float32)
    img += amp * (((xs // block) + (ys // block)) % 2)
    img += rng.normal(0.0, noise, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _write_burst(dirpath: Path, imgs: list[np.ndarray], start: int = 0) -> list[str]:
    """Write ``imgs`` as ``P_<start+i>.png`` and return their paths."""
    dirpath.mkdir(parents=True, exist_ok=True)
    out = []
    for i, img in enumerate(imgs):
        p = dirpath / f"P_{start + i}.png"
        assert cv2.imwrite(str(p), img)
        out.append(str(p).replace("\\", "/"))
    return out


def _central_box_of(width: int, height: int) -> tuple[int, int, int, int]:
    """The central 70% box the ROI suggestion falls back to."""
    bw, bh = int(round(width * 0.7)), int(round(height * 0.7))
    return ((width - bw) // 2, (height - bh) // 2,
            (width - bw) // 2 + bw, (height - bh) // 2 + bh)


def _m(p_index: int, mean: float = 120.0, sat: float = 0.0,
       lap: float = 1000.0, dist: float = 0.5) -> dict:
    """One hand-built metrics record, so selection can be tested as a pure rule."""
    return {
        "path": f"P_{p_index}.png",
        "p_index": p_index,
        "mean": mean,
        "sat_frac": sat,
        "lap_var": lap,
        "dist_to_median": dist,
    }


def _scan_index(tmp_path: Path, desktop: int = 7, steps=(1, 2),
                burst_len: int = 10, start: int = 0) -> dict[int, DesktopIndex]:
    """A one-desktop index of real synthetic scanner bursts on disk."""
    frames: dict[FrameKey, FrameFile] = {}
    for s in steps:
        imgs = [_texture(seed=s * 100 + i) for i in range(burst_len)]
        paths = _write_burst(tmp_path / "src" / f"D{desktop}" / f"RGB{s}1", imgs, start=start)
        key = FrameKey(desktop, s, "scan")
        frames[key] = FrameFile(
            key=key, path=paths[0], aux={"burst": paths}, src_step_dir=f"RGB{s}1"
        )
    return {desktop: DesktopIndex(desktop=desktop, n_steps=max(steps), frames=frames)}


# --------------------------------------------------------------------------
# cache_path
# --------------------------------------------------------------------------
def test_cache_path_uses_view_desktop_and_zero_padded_step():
    assert (cache_path("D:/DataSet/cache", FrameKey(13, 42, "scan"), "png")
            == "D:/DataSet/cache/scan/D13/s042.png")
    assert cache_path("D:/c", FrameKey(1, 5, "oak1"), "jpg") == "D:/c/oak1/D01/s005.jpg"
    assert cache_path("D:/c", FrameKey(66, 123, "rs"), "png") == "D:/c/rs/D66/s123.png"


# --------------------------------------------------------------------------
# burst_metrics
# --------------------------------------------------------------------------
def test_burst_metrics_reports_one_record_per_image(tmp_path):
    imgs = [_texture(seed=i) for i in range(10)]
    imgs[0] = np.full_like(imgs[0], 9)               # unexposed frame
    imgs[3][0:120, :] = 255                          # blown-out band, ~47% of the frame
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs))

    assert len(metrics) == 10
    assert {"mean", "sat_frac", "lap_var", "dist_to_median"} <= set(metrics[0])
    assert [m["p_index"] for m in metrics] == list(range(10))
    assert metrics[0]["mean"] < 20
    assert metrics[3]["sat_frac"] > 0.2
    assert all(m["sat_frac"] < 0.01 for i, m in enumerate(metrics) if i != 3)
    assert metrics[0]["lap_var"] == 0.0                  # a flat frame has no detail
    assert all(m["lap_var"] > 0 for m in metrics[1:])


def test_burst_metrics_keeps_the_burst_order_and_p_index_of_a_partial_burst(tmp_path):
    imgs = [_texture(seed=i) for i in range(5)]
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs, start=2))
    assert [m["p_index"] for m in metrics] == [2, 3, 4, 5, 6]
    assert metrics[0]["path"].endswith("P_2.png")


def test_burst_metrics_dist_to_median_spikes_for_a_hand_in_frame(tmp_path):
    imgs = [_texture(seed=i) for i in range(10)]
    imgs[6][60:200, 40:210] = 200                    # an arm reaching over the board
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs))
    dists = [m["dist_to_median"] for m in metrics]
    assert dists[6] > 5 * statistics.median(dists)


def test_burst_metrics_lap_var_drops_for_a_blurred_image(tmp_path):
    imgs = [_texture(seed=i) for i in range(4)]
    imgs[2] = cv2.GaussianBlur(imgs[2], (0, 0), 6)
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs))
    assert metrics[2]["lap_var"] < 0.5 * max(m["lap_var"] for m in metrics)


def test_burst_metrics_rejects_an_empty_burst():
    with pytest.raises(ValueError):
        burst_metrics([])


def test_burst_metrics_raises_on_an_unreadable_file(tmp_path):
    with pytest.raises(OSError):
        burst_metrics([str(tmp_path / "nope.png")])


# --------------------------------------------------------------------------
# choose_scan_image - the rule, on hand-built metrics
# --------------------------------------------------------------------------
def test_choose_scan_image_defaults_to_p0():
    assert choose_scan_image([_m(i) for i in range(10)]) == (0, "p0")


def test_choose_scan_image_skips_a_dark_p0_and_takes_the_sharpest():
    metrics = [_m(i) for i in range(10)]
    metrics[0]["mean"] = 8.0
    metrics[7]["lap_var"] = 2500.0
    assert choose_scan_image(metrics) == (7, "p0_dark")


def test_choose_scan_image_skips_a_blown_out_p0():
    metrics = [_m(i) for i in range(10)]
    metrics[0]["sat_frac"] = 0.35
    metrics[4]["lap_var"] = 3000.0
    assert choose_scan_image(metrics) == (4, "p0_saturated")


def test_choose_scan_image_skips_a_p0_that_deviates_from_the_burst_median():
    metrics = [_m(i) for i in range(10)]
    metrics[0]["dist_to_median"] = 20.0
    metrics[6]["lap_var"] = 2000.0
    assert choose_scan_image(metrics) == (6, "p0_outlier")


def test_choose_scan_image_tolerates_burst_noise_near_twice_the_median():
    """A burst whose median dist is ~0 must not reject every image (synthetic
    and near-identical real bursts); only real deviations count."""
    metrics = [_m(i, dist=0.0) for i in range(10)]
    metrics[0]["dist_to_median"] = 0.4
    assert choose_scan_image(metrics) == (0, "p0")


def test_choose_scan_image_ignores_alternatives_that_fail_the_check():
    metrics = [_m(i) for i in range(10)]
    metrics[0]["mean"] = 5.0
    metrics[9].update(lap_var=9999.0, sat_frac=0.5)   # sharpest but blown out
    metrics[3]["lap_var"] = 3000.0
    assert choose_scan_image(metrics) == (3, "p0_dark")


def test_choose_scan_image_reports_p0_missing_for_a_burst_starting_at_p2():
    # D49 / RGB261: the burst on disk starts at P_2 and holds 5 images.
    metrics = [_m(i) for i in (2, 3, 4, 5, 6)]
    assert choose_scan_image(metrics) == (0, "p0_missing")


def test_choose_scan_image_p0_missing_skips_a_dark_first_image():
    metrics = [_m(i) for i in (2, 3, 4, 5, 6)]
    metrics[0]["mean"] = 6.0
    assert choose_scan_image(metrics) == (1, "p0_missing")


def test_choose_scan_image_falls_back_when_no_image_passes():
    metrics = [_m(i, mean=5.0) for i in range(10)]
    metrics[4]["lap_var"] = 5000.0
    idx, reason = choose_scan_image(metrics)
    assert idx == 4
    assert reason == "p0_dark_fallback"


def test_choose_scan_image_keeps_a_single_image_burst():
    idx, reason = choose_scan_image([_m(0, mean=5.0)])
    assert idx == 0
    assert reason.startswith("p0_dark")


def test_choose_scan_image_rejects_empty_metrics():
    with pytest.raises(ValueError):
        choose_scan_image([])


# --------------------------------------------------------------------------
# choose_scan_image - end to end over real image files
# --------------------------------------------------------------------------
def test_choose_scan_image_on_a_normal_synthetic_burst(tmp_path):
    imgs = [_texture(seed=i) for i in range(10)]
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs))
    assert choose_scan_image(metrics) == (0, "p0")


def test_choose_scan_image_on_a_burst_whose_p0_is_dark(tmp_path):
    imgs = [_texture(seed=i) for i in range(10)]
    imgs[0] = np.full_like(imgs[0], 9)
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs))
    idx, reason = choose_scan_image(metrics)
    assert reason == "p0_dark"
    assert idx != 0
    assert metrics[idx]["mean"] > 20


def test_choose_scan_image_on_a_burst_with_no_p0(tmp_path):
    imgs = [_texture(seed=i) for i in range(5)]
    metrics = burst_metrics(_write_burst(tmp_path / "b", imgs, start=2))
    assert choose_scan_image(metrics) == (0, "p0_missing")


# --------------------------------------------------------------------------
# suggest_roi
# --------------------------------------------------------------------------
def test_suggest_roi_finds_the_chassis_in_a_real_scanner_frame():
    img = cv2.imread(str(SCANNER_FIXTURE))
    x0, y0, x1, y1 = suggest_roi(img, "scan")
    cx0, cy0, cx1, cy1 = CHASSIS

    assert all(isinstance(v, int) for v in (x0, y0, x1, y1))
    assert (x0, y0) <= (cx0, cy0) and x1 >= cx1 and y1 >= cy1   # contains the chassis
    assert x0 >= 60 and y0 >= 35 and x1 <= 845 and y1 <= 795    # inside the tape square
    assert (x1 - x0) * (y1 - y0) <= 1.6 * (cx1 - cx0) * (cy1 - cy0)


def test_suggest_roi_on_a_synthetic_tape_square():
    img = np.full((400, 400, 3), 245, np.uint8)
    cv2.rectangle(img, (40, 40), (360, 360), (40, 190, 230), 14)   # yellow tape band
    cv2.rectangle(img, (140, 120), (280, 300), (60, 60, 60), -1)   # dark chassis
    x0, y0, x1, y1 = suggest_roi(img, "scan")
    assert x0 <= 140 and y0 <= 120 and x1 >= 280 and y1 >= 300
    assert x0 >= 45 and y0 >= 45 and x1 <= 355 and y1 <= 355


def test_suggest_roi_ignores_a_yellow_blob_that_does_not_frame_the_board():
    """A label or cable inside the chassis is yellow too; only a square that
    spans the frame is the tape (otherwise the box would be junk)."""
    img = np.full((400, 400, 3), 245, np.uint8)
    cv2.rectangle(img, (150, 150), (260, 250), (40, 190, 230), -1)   # yellow patch
    cv2.rectangle(img, (170, 170), (240, 230), (60, 60, 60), -1)     # dark blob in it
    assert suggest_roi(img, "scan") == _central_box_of(400, 400)


def test_suggest_roi_falls_back_to_the_central_box_without_tape():
    img = cv2.imread(str(CROP_FIXTURE))
    assert suggest_roi(img, "scan") == (135, 135, 765, 765)


def test_suggest_roi_uses_the_central_box_for_other_views():
    img = np.zeros((720, 1280, 3), np.uint8)
    assert suggest_roi(img, "oak1") == (192, 108, 1088, 612)
    assert suggest_roi(img, "rs") == (192, 108, 1088, 612)


def test_suggest_roi_accepts_a_grayscale_image():
    img = cv2.imread(str(SCANNER_FIXTURE), cv2.IMREAD_GRAYSCALE)
    x0, y0, x1, y1 = suggest_roi(img, "scan")
    assert 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]


# --------------------------------------------------------------------------
# build_cache
# --------------------------------------------------------------------------
def test_build_cache_copies_the_chosen_image_and_writes_a_manifest(tmp_path):
    index = _scan_index(tmp_path)
    seen: list[tuple] = []
    stats = build_cache(index, str(tmp_path / "cache"), progress=lambda d, s, v: seen.append((d, s, v)))

    assert stats["copied"] == 2
    assert stats["skipped"] == 0
    assert stats["failures"] == []
    assert seen == [(7, 1, "scan"), (7, 2, "scan")]

    for step in (1, 2):
        dest = tmp_path / "cache" / "scan" / "D07" / f"s{step:03d}.png"
        assert dest.is_file()
        src = index[7].frames[FrameKey(7, step, "scan")].aux["burst"][0]
        assert dest.read_bytes() == Path(src).read_bytes()

    manifest = json.loads((tmp_path / "cache" / "scan" / "D07" / "manifest.json").read_text("utf-8"))
    assert sorted(manifest) == ["1", "2"]
    rec = manifest["1"]
    assert rec["chosen"] == 0
    assert rec["reason"] == "p0"
    assert len(rec["metrics"]) == 10
    assert rec["src"].endswith("P_0.png")
    assert {"mean", "sat_frac", "lap_var", "dist_to_median"} <= set(rec["metrics"][0])


def test_build_cache_copies_the_chosen_non_p0_image(tmp_path):
    index = _scan_index(tmp_path, steps=(4,))
    burst = index[7].frames[FrameKey(7, 4, "scan")].aux["burst"]
    dark = _texture(seed=1) * 0 + 9
    assert cv2.imwrite(burst[0], dark)                     # blank out P_0 on disk

    stats = build_cache(index, str(tmp_path / "cache"))
    manifest = json.loads((tmp_path / "cache" / "scan" / "D07" / "manifest.json").read_text("utf-8"))
    rec = manifest["4"]
    assert rec["chosen"] != 0
    assert rec["reason"] == "p0_dark"
    assert rec["src"] == burst[rec["chosen"]]
    dest = tmp_path / "cache" / "scan" / "D07" / "s004.png"
    assert dest.read_bytes() == Path(rec["src"]).read_bytes()
    assert len(stats["non_p0"]) == 1
    assert stats["non_p0"][0]["desktop"] == 7


def test_build_cache_is_idempotent(tmp_path):
    index = _scan_index(tmp_path)
    cache = str(tmp_path / "cache")
    build_cache(index, cache)
    dest = tmp_path / "cache" / "scan" / "D07" / "s001.png"
    stamp = dest.stat().st_mtime_ns

    stats = build_cache(index, cache)
    assert stats["copied"] == 0
    assert stats["skipped"] == 2
    assert dest.stat().st_mtime_ns == stamp
    manifest = json.loads((tmp_path / "cache" / "scan" / "D07" / "manifest.json").read_text("utf-8"))
    assert sorted(manifest) == ["1", "2"]


def test_build_cache_recopies_a_truncated_file(tmp_path):
    index = _scan_index(tmp_path, steps=(1,))
    cache = str(tmp_path / "cache")
    build_cache(index, cache)
    dest = tmp_path / "cache" / "scan" / "D07" / "s001.png"
    dest.write_bytes(b"truncated")

    stats = build_cache(index, cache)
    assert stats["copied"] == 1
    assert dest.stat().st_size > 100


def test_build_cache_records_failures_and_keeps_going(tmp_path):
    index = _scan_index(tmp_path, steps=(1, 2))
    bad = index[7].frames[FrameKey(7, 1, "scan")]
    bad.aux["burst"] = [str(tmp_path / "missing" / "P_0.png")]

    stats = build_cache(index, str(tmp_path / "cache"))
    assert stats["copied"] == 1
    assert len(stats["failures"]) == 1
    assert stats["failures"][0]["desktop"] == 7 and stats["failures"][0]["step"] == 1
    assert not (tmp_path / "cache" / "scan" / "D07" / "s001.png").exists()
    assert (tmp_path / "cache" / "scan" / "D07" / "s002.png").is_file()
    manifest = json.loads((tmp_path / "cache" / "scan" / "D07" / "manifest.json").read_text("utf-8"))
    assert sorted(manifest) == ["2"]


def test_build_cache_uses_the_right_extension_per_view(tmp_path):
    oak = tmp_path / "src" / "oak.jpg"
    rs = tmp_path / "src" / "rs.png"
    oak.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(oak), _texture(seed=1))
    assert cv2.imwrite(str(rs), _texture(seed=2))
    frames = {
        FrameKey(3, 1, "oak1"): FrameFile(key=FrameKey(3, 1, "oak1"), path=str(oak)),
        FrameKey(3, 1, "rs"): FrameFile(key=FrameKey(3, 1, "rs"), path=str(rs)),
    }
    index = {3: DesktopIndex(desktop=3, n_steps=1, frames=frames)}

    stats = build_cache(index, str(tmp_path / "cache"), views=("oak1", "rs"))
    assert stats["copied"] == 2
    assert (tmp_path / "cache" / "oak1" / "D03" / "s001.jpg").is_file()
    assert (tmp_path / "cache" / "rs" / "D03" / "s001.png").is_file()
    manifest = json.loads((tmp_path / "cache" / "oak1" / "D03" / "manifest.json").read_text("utf-8"))
    assert manifest["1"]["chosen"] == 0
    assert manifest["1"]["metrics"] == []
    assert manifest["1"]["src"] == str(oak).replace("\\", "/")


def test_build_cache_only_touches_the_requested_views_and_desktops(tmp_path):
    index = _scan_index(tmp_path, desktop=7, steps=(1,))
    index.update(_scan_index(tmp_path, desktop=8, steps=(1,)))
    oak_key = FrameKey(7, 1, "oak1")
    index[7].frames[oak_key] = FrameFile(key=oak_key, path=str(tmp_path / "src" / "nope.jpg"))

    stats = build_cache(index, str(tmp_path / "cache"), desktops=[7])
    assert stats["copied"] == 1
    assert stats["failures"] == []
    assert (tmp_path / "cache" / "scan" / "D07" / "s001.png").is_file()
    assert not (tmp_path / "cache" / "scan" / "D08").exists()
    assert not (tmp_path / "cache" / "oak1").exists()


def test_build_cache_never_writes_outside_the_cache_dir(tmp_path):
    index = _scan_index(tmp_path, steps=(1,))
    src_dir = tmp_path / "src" / "D7" / "RGB11"
    before = {p.name: p.stat().st_mtime_ns for p in src_dir.iterdir()}
    build_cache(index, str(tmp_path / "cache"))
    after = {p.name: p.stat().st_mtime_ns for p in src_dir.iterdir()}
    assert before == after
    assert sorted(os.listdir(tmp_path)) == ["cache", "src"]
