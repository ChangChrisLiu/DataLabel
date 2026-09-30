"""U3: the small-part detector without a window -- its file, its tiles, its cache.

No GPU, no ultralytics: the model is a stub (:class:`StubModel`) and the one
test of :class:`~tda.models.detector.YoloTileDetector` stops it before the
import.  The window's half is ``tests/test_app_u3.py``.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tda.models import detector as DM
from tda.models.det_engine import (
    CACHE_VERSION,
    DetCache,
    DetectionEngine,
    PlanItem,
    file_stamp,
)
from tda.models.detector import Det, DetectorConfig, DetectorUnavailable

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# a stub model and a scene of two frames on disk
# --------------------------------------------------------------------------- #
class StubModel:
    """Answers ``detect`` from a table ``{step: [Det]}``; counts its calls."""

    names = {0: "screw", 1: "connector"}

    def __init__(self, table=None, identity: str = "stub-model-1") -> None:
        self.table = dict(table or {})
        self.identity = identity
        self.calls: list = []

    def detect(self, img, crop, view, step=None):
        self.calls.append((step, tuple(crop)))
        return list(self.table.get(step, []))

    def describe(self) -> str:
        return f"stub {self.identity}"


def config(tmp_path: Path, **over) -> DetectorConfig:
    base = dict(source=tmp_path / "detector.yaml", model=tmp_path / "stub.pt",
                classes=("screw",), views=("scan", "oak1"), conf=0.10, roi_crop=True,
                work_scale=(("oak1", 1.0), ("scan", 1.0)), tile=640, stride=512,
                yield_ms=0.0, cache_root=tmp_path / "det")
    base.update(over)
    return DetectorConfig(**base)


def write_frame(path: Path, value: int = 90, square_at: int = 10) -> str:
    img = np.full((64, 64, 3), value, np.uint8)
    img[20:32, square_at:square_at + 10] = 230
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return str(path)


A = Det((10.0, 20.0, 20.0, 32.0), "screw", 0.9)       # the square: it moves
B = Det((40.0, 40.0, 50.0, 50.0), "screw", 0.8)       # the board: it stays
LOW = Det((2.0, 2.0, 8.0, 8.0), "screw", 0.05)        # under conf
CONN = Det((30.0, 40.0, 38.0, 48.0), "connector", 0.95)


@pytest.fixture
def frames(tmp_path):
    j = write_frame(tmp_path / "img" / "s012.png", square_at=10)
    k = write_frame(tmp_path / "img" / "s013.png", square_at=30)
    return j, k


def engine_for(tmp_path, model, **over) -> DetectionEngine:
    eng = DetectionEngine(config(tmp_path, **over), tmp_path / "det",
                          factory=lambda _c: model)
    eng.load()
    eng.open_view(13, "scan")
    return eng


def item(frames, crop=(0, 0, 64, 64), step=12, neighbour=13) -> PlanItem:
    return PlanItem(step=step, candidates=(frames[0],), crop=crop, neighbour=neighbour,
                    neighbour_candidates=(frames[1],))


# --------------------------------------------------------------------------- #
# the file
# --------------------------------------------------------------------------- #
class Keep(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.records: list = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def lines():
    keep = Keep()
    logger = logging.getLogger("tda.models.detector")
    logger.addHandler(keep)
    old = logger.level
    logger.setLevel(logging.DEBUG)
    yield keep.records
    logger.removeHandler(keep)
    logger.setLevel(old)


def good_yaml(tmp_path: Path, **over) -> Path:
    model = tmp_path / "weights.pt"
    model.write_bytes(b"not really a model")
    data = {"enabled": True, "model": str(model), "classes": ["screw"],
            "views": ["scan", "oak1"], "conf": 0.1,
            "tiling": {"roi_crop": True, "work_scale": {"scan": 1.0, "oak1": 1.0},
                       "tile": 640, "stride": 512},
            "device": 0, "half": True, "yield_ms": 20,
            "cache_root": str(tmp_path / "det"),
            "yolo_config_dir": str(tmp_path / "ycfg")}
    for key, value in over.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path = tmp_path / "detector.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_shipped_file_is_scan_and_oak1_screws_with_l2s_tiles(monkeypatch):
    data = yaml.safe_load(DM.DEFAULT_FILE.read_text(encoding="utf-8"))
    parsed = DM._parse(data, DM.DEFAULT_FILE)
    assert parsed.enabled is True
    assert parsed.views == ("scan", "oak1") and parsed.classes == ("screw",)
    assert parsed.conf == pytest.approx(0.10)
    assert parsed.roi_crop is True and parsed.tile == 640 and parsed.stride == 512
    assert parsed.scale("scan") == parsed.scale("oak1") == 1.0
    assert parsed.model.name == "tda_det_yolo26n_l2_hold13.pt"
    assert str(parsed.cache_root).replace("\\", "/") == "D:/DataSet/cache/det"
    assert str(parsed.yolo_config_dir).replace("\\", "/").startswith("D:/")
    assert parsed.half is True and parsed.device == 0


def test_a_good_file_loads_quietly_the_on_line_comes_with_the_model(tmp_path, lines):
    path = good_yaml(tmp_path)
    got = DM.load_detector_config(str(path))
    assert got is not None and got.views == ("scan", "oak1")
    # "on" is said by the worker once the model has loaded, not here
    assert [r for r in lines if r.levelno >= logging.INFO] == []


def test_off_by_the_environment_is_one_info_line(monkeypatch, lines):
    monkeypatch.setenv(DM.ENV_VAR, "off")
    assert DM.load_detector_config() is None
    assert [(r.levelname, r.getMessage()) for r in lines] == [
        ("INFO", "small-part detector off: TDA_DETECTOR=off")]


def test_the_environment_can_name_another_file(monkeypatch, tmp_path):
    monkeypatch.setenv(DM.ENV_VAR, str(good_yaml(tmp_path, conf=0.25)))
    assert DM.load_detector_config().conf == pytest.approx(0.25)


def test_enabled_false_is_one_info_line(tmp_path, lines):
    assert DM.load_detector_config(str(good_yaml(tmp_path, enabled=False))) is None
    assert [r.levelname for r in lines] == ["INFO"]
    assert "enabled is false" in lines[0].getMessage()


BROKEN = {
    "classes empty": {"classes": []},
    "classes not a list": {"classes": "screw"},
    "an unknown view": {"views": ["scan", "oak9"]},
    "conf zero": {"conf": 0},
    "conf above one": {"conf": 1.5},
    "conf nan": {"conf": float("nan")},
    "conf a word": {"conf": "high"},
    "conf a bool": {"conf": True},
    "no tiling": {"tiling": None},
    "no scale for a view": {"tiling": {"roi_crop": True, "work_scale": {"scan": 1.0},
                                       "tile": 640, "stride": 512}},
    "stride above tile": {"tiling": {"roi_crop": True, "work_scale": {"scan": 1, "oak1": 1},
                                     "tile": 640, "stride": 700}},
    "tile not whole": {"tiling": {"roi_crop": True, "work_scale": {"scan": 1, "oak1": 1},
                                  "tile": 640.5, "stride": 512}},
    "a zero scale": {"tiling": {"roi_crop": True, "work_scale": {"scan": 0, "oak1": 1},
                                "tile": 640, "stride": 512}},
    "enabled a word": {"enabled": "yes"},
    "half a word": {"half": "yes"},
    "device a list": {"device": [0]},
    "no model": {"model": None},
}


@pytest.mark.parametrize("what", sorted(BROKEN), ids=sorted(BROKEN))
def test_a_malformed_file_is_a_warning_and_off(tmp_path, lines, what):
    assert DM.load_detector_config(str(good_yaml(tmp_path, **BROKEN[what]))) is None
    assert [r.levelname for r in lines] == ["WARNING"], what
    assert lines[0].getMessage().startswith("small-part detector OFF")


@pytest.mark.parametrize("body", ["", "[1, 2]", "views: [scan\n", "\x00\x01"])
def test_an_unreadable_file_is_a_warning_and_off(tmp_path, lines, body):
    path = tmp_path / "detector.yaml"
    path.write_text(body, encoding="utf-8")
    assert DM.load_detector_config(str(path)) is None
    assert [r.levelname for r in lines] == ["WARNING"]


def test_a_missing_file_is_a_warning_and_off(tmp_path, lines):
    assert DM.load_detector_config(str(tmp_path / "nowhere.yaml")) is None
    assert "does not exist" in lines[0].getMessage() and lines[0].levelname == "WARNING"


def test_a_missing_model_file_is_a_warning_and_off(tmp_path, lines):
    path = good_yaml(tmp_path)
    (tmp_path / "weights.pt").unlink()
    assert DM.load_detector_config(str(path)) is None
    assert "the model file" in lines[0].getMessage()


def test_no_ultralytics_is_a_warning_and_off(tmp_path, lines, monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "ultralytics" else real(name, *a))
    assert DM.load_detector_config(str(good_yaml(tmp_path))) is None
    assert "ultralytics is not installed" in lines[0].getMessage()


def test_no_cuda_stops_the_model_before_ultralytics_is_imported(tmp_path, monkeypatch):
    """The two checks the file cannot make: they end in DetectorUnavailable."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", None)   # would raise if reached
    threads = cv2.getNumThreads()
    cfg = config(tmp_path, model=tmp_path / "w.pt",
                 yolo_config_dir=tmp_path / "ycfg")
    cfg.model.write_bytes(b"x")
    with pytest.raises(DetectorUnavailable, match="CUDA is not available"):
        DM.YoloTileDetector(cfg)
    assert cv2.getNumThreads() == threads
    # ultralytics' settings go to D:, never %APPDATA%; nothing is downloaded
    assert os.environ["YOLO_CONFIG_DIR"] == str(tmp_path / "ycfg")
    assert os.environ["YOLO_OFFLINE"] == "1"


def test_ultralytics_that_does_not_import_is_unavailable_and_opencv_keeps_its_threads(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    threads = cv2.getNumThreads()
    cfg = config(tmp_path, model=tmp_path / "w.pt", yolo_config_dir=tmp_path / "ycfg")
    cfg.model.write_bytes(b"x")
    with pytest.raises(DetectorUnavailable, match="ultralytics does not import"):
        DM.YoloTileDetector(cfg)
    assert cv2.getNumThreads() == threads
    assert "OMP_NUM_THREADS" not in os.environ


# --------------------------------------------------------------------------- #
# tiles and merge (L2's)
# --------------------------------------------------------------------------- #
def test_tiles_are_l2s_640_at_512_flush_with_the_edge_and_grey_padded():
    crop = np.zeros((1026, 1362, 3), np.uint8)
    tiles = DM.make_tiles(crop, 640, 512)
    assert sorted({t.ox for t in tiles}) == [0, 512, 722]
    assert sorted({t.oy for t in tiles}) == [0, 386]
    assert len(tiles) == 6                          # L2: 6 tiles on a scan ROI
    small = DM.make_tiles(np.zeros((300, 200, 3), np.uint8), 640, 512)
    assert len(small) == 1 and (small[0].w, small[0].h) == (200, 300)
    assert small[0].rgb.shape == (640, 640, 3) and small[0].rgb[400, 400, 0] == 114


def test_l2s_tiling_code_and_this_one_cut_the_same_tiles():
    l2 = pytest.importorskip("experiments.l2_detector.tiles")
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (1600, 1600, 3), np.uint8)
    roi = (627, 294, 1385, 1144)
    mine = DM.make_tiles(DM.work_crop(img, roi, 1.0), 640, 512)
    theirs = l2.make_tiles(l2.work_crop(img, roi, "scan"))
    assert [(t.ox, t.oy, t.w, t.h) for t in mine] == [(t.ox, t.oy, t.w, t.h) for t in theirs]
    assert all(np.array_equal(a.rgb, b.rgb) for a, b in zip(mine, theirs))


def test_merge_drops_a_cut_off_box_contained_in_a_whole_one_then_nms():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 6, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    confs = np.array([0.5, 0.9, 0.4, 0.3])
    clss = np.array([0, 0, 0, 1])
    edge = np.array([False, True, False, False])
    kept = DM.merge(boxes, confs, clss, edge)
    # the cut-off 0.9 lies 100 % in the whole 0.5; 0.4 is NMS'd by 0.5
    # (experiments.l2_detector.infer.merge gives the same; importing it here
    # would re-point this process's TMP and thread pools, see its env.py)
    assert list(kept) == [0, 3]


def test_tile_answers_come_back_in_native_pixels_merged():
    crop = (100, 50, 900, 700)
    work = np.zeros((650, 800, 3), np.uint8)
    tiles = DM.make_tiles(work, 640, 512)
    # a screw at crop (600, 300)-(620, 320) seen by two tiles, whole in both
    results = []
    for t in tiles:
        x0, y0 = 600 - t.ox, 300 - t.oy
        if 0 <= x0 and x0 + 20 <= t.w and 0 <= y0 and y0 + 20 <= t.h:
            results.append((np.array([[x0, y0, x0 + 20, y0 + 20]], float),
                            np.array([0.8 if t.ox == 0 else 0.7]), np.array([0])))
        else:
            results.append(None)
    dets = DM.tiles_to_dets(tiles, results, crop, 1.0, work.shape[:2], {0: "screw"})
    assert len(dets) == 1
    assert dets[0].box == (700.0, 350.0, 720.0, 370.0) and dets[0].cls == "screw"
    assert dets[0].conf == pytest.approx(0.8)


def test_box_change_is_l1s_and_large_only_where_the_part_left(frames):
    j = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
    k = cv2.cvtColor(cv2.imread(frames[1]), cv2.COLOR_BGR2RGB)
    moved = DM.box_change(j, k, A.int_box)
    still = DM.box_change(j, k, B.int_box)
    assert moved > 20.0 and still < 1.0
    l1 = pytest.importorskip("experiments.l1_localise.methods")
    for box in (A.int_box, B.int_box, (0, 0, 64, 64), (60, 60, 64, 64)):
        assert DM.box_change(j, k, box) == pytest.approx(l1.box_change(j, k, (0, 0), box))


# --------------------------------------------------------------------------- #
# the engine and its cache
# --------------------------------------------------------------------------- #
def test_a_frame_is_detected_once_and_then_read_from_the_cache(tmp_path, frames):
    model = StubModel({12: [A, B, LOW, CONN]})
    eng = engine_for(tmp_path, model)
    first = eng.process(item(frames))
    assert first.detected and first.measured and model.calls == [(12, (0, 0, 64, 64))]
    # the store floor keeps the 0.05 one, the candidates are the screws >= 0.10
    assert [d.cls for d in first.dets] == ["screw", "screw", "screw", "connector"]
    assert sorted(first.change) == [0, 1]
    assert first.change[0] > 20.0 and first.change[1] < 1.0
    path = eng.flush()
    assert path == tmp_path / "det" / "stub-model-1"[:12] / "13_scan.json"
    assert not list(path.parent.glob("*.tmp"))
    # a new engine, a new process: the same answer, no model call, no dE
    again = StubModel({12: [A]})
    eng2 = engine_for(tmp_path, again)
    second = eng2.process(item(frames))
    assert again.calls == [] and not second.detected and not second.measured
    assert second.dets == first.dets and second.change == first.change


def test_the_cache_is_keyed_by_the_files_size_and_mtime(tmp_path, frames):
    model = StubModel({12: [A]})
    eng = engine_for(tmp_path, model)
    eng.process(item(frames))
    stamp = file_stamp(frames[0])
    os.utime(frames[0], ns=(stamp["mtime_ns"] + 10**9, stamp["mtime_ns"] + 10**9))
    eng.process(item(frames))
    assert len(model.calls) == 2, "a touched file was served from the cache"


def test_a_different_roi_is_a_miss_and_a_different_neighbour_only_remeasures(
        tmp_path, frames):
    model = StubModel({12: [A, B]})
    calls = []

    def change(j, k, box):
        calls.append(tuple(box))
        return 1.0

    eng = DetectionEngine(config(tmp_path), tmp_path / "det", factory=lambda _c: model,
                          change_fn=change)
    eng.load()
    eng.open_view(13, "scan")
    eng.process(item(frames))
    eng.process(item(frames, crop=(0, 0, 60, 60)))
    assert len(model.calls) == 2
    other = write_frame(Path(frames[1]).with_name("s014.png"), square_at=40)
    calls.clear()
    got = eng.process(PlanItem(step=12, candidates=(frames[0],), crop=(0, 0, 60, 60),
                               neighbour=14, neighbour_candidates=(other,)))
    assert len(model.calls) == 2 and not got.detected, "the detections were still good"
    assert got.measured and len(calls) == 2 and got.neighbour == 14


def test_another_model_never_reads_this_models_cache(tmp_path, frames):
    eng = engine_for(tmp_path, StubModel({12: [A]}, identity="aaaaaaaaaaaa-1"))
    eng.process(item(frames))
    eng.flush()
    other = StubModel({12: [B]}, identity="bbbbbbbbbbbb-1")
    eng2 = engine_for(tmp_path, other)
    got = eng2.process(item(frames))
    assert other.calls and got.dets == (Det.from_row(B.row()),)
    eng2.flush()
    assert {p.parent.name for p in (tmp_path / "det").rglob("*.json")} == {
        "aaaaaaaaaaaa", "bbbbbbbbbbbb"}


@pytest.mark.parametrize("damage", ["garbage", "version", "model", "recipe"])
def test_an_unusable_cache_file_is_ignored_not_trusted(tmp_path, frames, damage):
    eng = engine_for(tmp_path, StubModel({12: [A]}))
    eng.process(item(frames))
    path = eng.flush()
    body = json.loads(path.read_text(encoding="utf-8"))
    if damage == "garbage":
        path.write_text("{not json", encoding="utf-8")
    else:
        body[{"version": "version", "model": "model_sha1", "recipe": "recipe"}[damage]] = (
            CACHE_VERSION + 1 if damage == "version" else "other")
        path.write_text(json.dumps(body), encoding="utf-8")
    model = StubModel({12: [A]})
    eng2 = engine_for(tmp_path, model)
    eng2.process(item(frames))
    assert model.calls, f"a cache with another {damage} was served"


def test_a_changed_recipe_empties_the_file_it_would_have_read(tmp_path, frames):
    eng = engine_for(tmp_path, StubModel({12: [A]}))
    eng.process(item(frames))
    eng.flush()
    model = StubModel({12: [A]})
    eng2 = engine_for(tmp_path, model, stride=500)
    eng2.process(item(frames))
    assert model.calls


def test_the_given_pixels_are_used_and_no_neighbour_means_no_de(tmp_path, frames):
    model = StubModel({12: [A]})
    eng = engine_for(tmp_path, model)
    pixels = cv2.cvtColor(cv2.imread(frames[0]), cv2.COLOR_BGR2RGB)
    got = eng.process(PlanItem(step=12, candidates=(), crop=(0, 0, 64, 64),
                               pixels=pixels))
    assert got.detected and got.change == {} and got.neighbour is None
    assert eng.cache.get(12) is None, "no file, no key: nothing may be cached"


def test_a_frame_with_no_pixels_is_no_answer(tmp_path):
    eng = engine_for(tmp_path, StubModel({12: [A]}))
    assert eng.process(PlanItem(step=12, candidates=(str(tmp_path / "gone.png"),),
                                crop=(0, 0, 64, 64))) is None


def test_the_cache_file_is_one_per_desktop_and_view(tmp_path, frames):
    eng = engine_for(tmp_path, StubModel({12: [A]}))
    eng.process(item(frames))
    eng.open_view(13, "oak1")          # flushes scan
    eng.process(item(frames))
    eng.flush()
    names = sorted(p.name for p in (tmp_path / "det").rglob("*.json"))
    assert names == ["13_oak1.json", "13_scan.json"]
    body = json.loads((tmp_path / "det" / "stub-model-1"[:12] / "13_scan.json")
                      .read_text(encoding="utf-8"))
    entry = body["steps"]["12"]
    assert entry["path"] == frames[0].replace("\\", "/") and entry["crop"] == [0, 0, 64, 64]
    assert {"size", "mtime_ns", "dets", "change"} <= set(entry)
    assert entry["change"]["neighbour"] == 13 and entry["change"]["classes"] == ["screw"]


def test_detcache_flush_writes_only_when_something_changed(tmp_path):
    cache = DetCache(tmp_path, "0123456789abcdef", {"a": 1})
    cache.open(13, "scan")
    assert cache.flush() is None
    cache.put(3, {"path": "x"})
    written = cache.flush()
    assert written is not None and written.parent.name == "0123456789ab"
    before = written.stat().st_mtime_ns
    time.sleep(0.01)
    assert cache.flush() is None and written.stat().st_mtime_ns == before
