"""U3/U4: the small-part detector without a window -- its file, its tiles, its cache.

No GPU, no ultralytics, no weights: the model is a stub (:class:`StubModel`),
the tests of :class:`~tda.models.detector.YoloTileDetector` stop it before the
import or hand it a fake ``ultralytics``, and :class:`RFDetrTileDetector`
(U4) runs on a fake ``rfdetr`` (``tests/fake_rfdetr.py``) that changes the
process the way the real one was measured to.  Two slow tests import the real
libraries in a subprocess.  The window's half is ``tests/test_app_u3.py``.
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


def test_the_shipped_file_is_rfdetr_on_scan_oak1_and_oak2_screws_with_l2s_tiles(monkeypatch):
    """U4: L3's RF-DETR small, with the 0.10 floor; U3's YOLO stays, commented, as the fallback."""
    text = DM.DEFAULT_FILE.read_text(encoding="utf-8")
    parsed = DM._parse(yaml.safe_load(text), DM.DEFAULT_FILE)
    assert parsed.enabled is True and parsed.backend == "rfdetr"
    assert parsed.model.name == "tda_det_rfdetr_small_all.pt"
    assert parsed.views == ("scan", "oak1", "oak2") and parsed.classes == ("screw",)
    assert parsed.conf == pytest.approx(0.10) and parsed.merge_floor == pytest.approx(0.10)
    assert parsed.roi_crop is True and parsed.tile == 640 and parsed.stride == 512
    assert parsed.scale("scan") == parsed.scale("oak1") == parsed.scale("oak2") == 1.0
    assert str(parsed.cache_root).replace("\\", "/") == "D:/DataSet/cache/det"
    assert str(parsed.yolo_config_dir).replace("\\", "/").startswith("D:/")
    assert parsed.half is True and parsed.device == 0
    # the fallback, one swap away
    assert "# backend: yolo" in text
    assert "# model: D:/DataSet/models/weights/tda_det_yolo26n_l2_hold13.pt" in text
    fallback = DM._parse(dict(yaml.safe_load(text), backend="yolo",
                              model="D:/DataSet/models/weights/tda_det_yolo26n_l2_hold13.pt"),
                         DM.DEFAULT_FILE)
    assert fallback.backend == "yolo" and fallback.merge_floor == pytest.approx(0.10)


def test_the_shipped_models_manifest_has_its_tiles_and_classes():
    """The manifest L3 wrote beside the weights: the tiling the file asks for, screw first."""
    parsed = DM._parse(yaml.safe_load(DM.DEFAULT_FILE.read_text(encoding="utf-8")),
                       DM.DEFAULT_FILE)
    manifest = DM.read_manifest(parsed.model)
    if manifest is None:
        pytest.skip(f"{parsed.model.with_suffix('.json')} is not on this machine")
    assert manifest["model"] == "rfdetr_small"
    assert DM._manifest_names(manifest)[0] == "screw"
    inference = manifest["inference"]
    assert (inference["tile"], inference["stride"]) == (parsed.tile, parsed.stride)
    assert inference["pad_value"] == DM.PAD_VALUE
    assert all(inference["work_scale"][v] == parsed.scale(v) for v in parsed.views)
    assert inference["precision"] == ("fp16" if parsed.half else "fp32")


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
    # U4
    "an unknown backend": {"backend": "detectron"},
    "backend a list": {"backend": ["rfdetr"]},
    "merge_floor above conf": {"merge_floor": 0.2},
    "merge_floor negative": {"merge_floor": -0.1},
    "merge_floor a word": {"merge_floor": "low"},
    "merge_floor nan": {"merge_floor": float("nan")},
    "merge_floor a bool": {"merge_floor": True},
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
    # ... which an rfdetr file does not need
    lines.clear()
    got = DM.load_detector_config(str(good_yaml(tmp_path, backend="rfdetr")))
    if importlib.util.find_spec("rfdetr") is not None:
        assert got is not None and got.backend == "rfdetr" and lines == []


def test_no_rfdetr_is_a_warning_and_off(tmp_path, lines, monkeypatch):
    """U4: the package the backend imports is looked for without importing it."""
    import importlib.util

    real = importlib.util.find_spec
    asked = []

    def find_spec(name, *a):
        asked.append(name)
        return None if name == "rfdetr" else real(name, *a)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert DM.load_detector_config(str(good_yaml(tmp_path, backend="rfdetr"))) is None
    assert [r.levelname for r in lines] == ["WARNING"]
    assert "rfdetr is not installed" in lines[0].getMessage()
    assert "rfdetr" in asked and "ultralytics" not in asked, asked


def test_the_backend_and_the_floor_are_read_with_their_defaults(tmp_path):
    """No ``backend`` is U3's yolo; no ``merge_floor`` is 0.10, or ``conf`` when that is lower."""
    got = DM.load_detector_config(str(good_yaml(tmp_path)))
    assert got.backend == "yolo" and got.merge_floor == pytest.approx(0.10)
    got = DM.load_detector_config(str(good_yaml(tmp_path, conf=0.05)))
    assert got.merge_floor == pytest.approx(0.05)
    got = DM.load_detector_config(str(good_yaml(tmp_path, merge_floor=0, backend=" yolo ")))
    assert got.merge_floor == 0.0 and got.backend == "yolo"
    got = DM.load_detector_config(str(good_yaml(tmp_path, merge_floor=0.1, conf=0.1,
                                                backend="rfdetr")))
    if got is not None:          # None only where rfdetr is not installed
        assert got.backend == "rfdetr" and got.merge_floor == pytest.approx(0.1)
    # both are in the recipe: a cache written without the floor is not read with it
    base = config(tmp_path)
    assert base.recipe()["merge_floor"] == pytest.approx(0.10)
    assert base.recipe() != config(tmp_path, merge_floor=0.0).recipe()
    assert base.recipe() != config(tmp_path, backend="rfdetr").recipe()


def test_no_cuda_stops_the_model_before_ultralytics_is_imported(tmp_path, monkeypatch):
    """The two checks the file cannot make: they end in DetectorUnavailable."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "ultralytics", None)   # would raise if reached
    for name in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE"):        # put back after the test
        monkeypatch.setenv(name, "unset by the test")
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
    for name in ("YOLO_CONFIG_DIR", "YOLO_OFFLINE"):        # put back after the test
        monkeypatch.setenv(name, "unset by the test")
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


class _MergeSpy:
    """Stands in for :func:`merge` and remembers what reached it."""

    def __init__(self, monkeypatch):
        self.confs: list = []
        real = DM.merge

        def spy(boxes, confs, clss, edge):
            self.confs.append([round(float(c), 3) for c in confs])
            return real(boxes, confs, clss, edge)

        monkeypatch.setattr(DM, "merge", spy)


def test_a_box_under_the_merge_floor_never_reaches_the_merge(monkeypatch):
    """U4 (L3 §5): 0.09 is dropped before the merge, 0.11 goes in; floor 0 is L2's protocol."""
    spy = _MergeSpy(monkeypatch)
    work = np.zeros((300, 300, 3), np.uint8)
    tiles = DM.make_tiles(work, 640, 512)
    results = [(np.array([[10, 10, 30, 30], [100, 100, 120, 120], [200, 200, 220, 220]], float),
                np.array([0.09, 0.11, 0.10]), np.array([0, 0, 0]))]
    dets = DM.tiles_to_dets(tiles, results, (0, 0, 300, 300), 1.0, work.shape[:2],
                            {0: "screw"}, floor=0.10)
    assert spy.confs == [[0.11, 0.10]], "a box under the floor reached the merge"
    assert sorted(round(d.conf, 2) for d in dets) == [0.10, 0.11]
    DM.tiles_to_dets(tiles, results, (0, 0, 300, 300), 1.0, work.shape[:2], {0: "screw"})
    assert spy.confs[-1] == [0.09, 0.11, 0.10]


def test_the_floor_comes_before_the_containment_rule_not_after_it(monkeypatch):
    """A cut-off 0.5 box inside a whole 0.05 one: dropped by L2's merge, kept once 0.05 is floored.

    This is the one way the floor can change what comes out above it -- and
    why it has to be in the merge and not a filter on its result.
    """
    work = np.zeros((650, 800, 3), np.uint8)
    tiles = DM.make_tiles(work, 640, 512)
    assert [(t.ox, t.oy) for t in tiles][:2] == [(0, 0), (160, 0)]
    results = []
    for t in tiles:
        if (t.ox, t.oy) == (0, 0):       # cut off by this tile's right edge (640)
            results.append((np.array([[600, 300, 640, 320]], float), np.array([0.5]),
                            np.array([0])))
        elif (t.ox, t.oy) == (160, 0):   # the whole part, from the next tile, at 0.05
            results.append((np.array([[440, 300, 490, 320]], float), np.array([0.05]),
                            np.array([0])))
        else:
            results.append(None)
    l2 = DM.tiles_to_dets(tiles, results, (0, 0, 800, 650), 1.0, work.shape[:2], {0: "screw"})
    assert [round(d.conf, 2) for d in l2] == [0.05]
    u4 = DM.tiles_to_dets(tiles, results, (0, 0, 800, 650), 1.0, work.shape[:2], {0: "screw"},
                          floor=0.10)
    assert [(d.box, round(d.conf, 2)) for d in u4] == [((600.0, 300.0, 640.0, 320.0), 0.5)]


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
    assert {"size", "mtime_ns", "dets", "changes"} <= set(entry)
    assert list(entry["changes"]) == ["13"] and entry["changes"]["13"]["classes"] == ["screw"]


# --------------------------------------------------------------------------- #
# round 3
# --------------------------------------------------------------------------- #
def test_de_is_kept_per_neighbour_so_browsing_both_ways_measures_once(tmp_path, frames):
    """Forward browsing compares with the step below: two slots, no overwriting."""
    calls = []

    def change(j, k, box):
        calls.append(tuple(box))
        return 1.0

    below = write_frame(Path(frames[0]).with_name("s011.png"), square_at=20)
    eng = DetectionEngine(config(tmp_path), tmp_path / "det",
                          factory=lambda _c: StubModel({12: [A]}), change_fn=change)
    eng.load()
    eng.open_view(13, "scan")
    down = PlanItem(step=12, candidates=(frames[0],), crop=(0, 0, 64, 64), neighbour=11,
                    neighbour_candidates=(below,))
    for walk in (item(frames), down, item(frames), down):
        eng.process(walk)
    assert len(calls) == 2, "a visit from the other side measured again"
    assert sorted(eng.cache.get(12)["changes"]) == ["11", "13"]


def test_decoded_frames_are_keyed_by_the_files_stamp_and_let_go_when_idle(tmp_path, frames):
    released = []

    class Releasing(StubModel):
        def release(self):
            released.append(True)

    eng = engine_for(tmp_path, Releasing({12: [A]}))
    eng.process(item(frames))
    keys = list(eng._decoded)
    assert keys and all(len(k) == 3 and isinstance(k[1], int) for k in keys)
    stamp = file_stamp(frames[1])
    assert (stamp["path"], stamp["size"], stamp["mtime_ns"]) in keys
    eng.release()
    assert eng._decoded == {} and released == [True]


def test_the_cache_answers_before_the_model_exists(tmp_path, frames):
    """Round 3: the cache key needs no model, so a warm start needs no model either."""
    first = DetectionEngine(config(tmp_path), tmp_path / "det",
                            factory=lambda _c: StubModel({12: [A, B]}, identity="warm-id"))
    first.load()
    first.open_view(13, "scan")
    served = first.process(item(frames))
    first.flush()

    built = []

    def never(_config):
        built.append(True)
        raise AssertionError("the model was built to read the cache")

    eng = DetectionEngine(config(tmp_path), tmp_path / "det", factory=never,
                          identity="warm-id")
    eng.prepare()
    eng.open_view(13, "scan")
    got = eng.answer_from_cache(item(frames))
    assert built == [] and eng.model is None
    assert got is not None and got.dets == served.dets and got.change == served.change
    assert not got.detected and not got.measured
    # a frame the cache does not hold is no answer (the worker keeps it for later)
    assert eng.answer_from_cache(item(frames, crop=(0, 0, 60, 60))) is None
    assert eng.answer_from_cache(PlanItem(step=12, candidates=(frames[0],),
                                          crop=(0, 0, 64, 64), neighbour=14,
                                          neighbour_candidates=(frames[1],))) is None


def test_the_real_models_cache_key_is_its_files_sha1_without_building_it(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    eng = DetectionEngine(config(tmp_path, model=weights), tmp_path / "det")
    eng.prepare()
    assert eng.model is None and eng.identity == DM.file_sha1(weights)
    assert eng.cache.dir == tmp_path / "det" / DM.file_sha1(weights)[:12]


#: The fake's code runs *as* the ``ultralytics`` package (its globals'
#: ``__name__``), which is how keep_process_state tells its warning filters
#: from everyone else's.
_FAKE_ULTRALYTICS = '''
import os, warnings
import cv2, numpy as np, PIL.Image, torch


def patch_everything():
    cv2.setNumThreads(1)
    cv2.imread = lambda *a, **k: (_ for _ in ()).throw(cv2.error("patched imread"))
    cv2.imwrite = lambda *a, **k: True
    PIL.Image.open = lambda *a, **k: None
    torch.save = lambda *a, **k: None
    os.environ["NUMEXPR_MAX_THREADS"] = "8"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.set_printoptions(linewidth=320)
    warnings.filterwarnings("ignore", message="u3 fake")
    warnings.simplefilter("ignore", category=BytesWarning)
    for other in DURING:
        other()


class YOLO:
    names = {0: "screw", 1: "connector"}

    def __init__(self, path):
        patch_everything()

    def predict(self, batch, **kw):
        return []
'''


def _fake_ultralytics(monkeypatch, zero: Path, during=()):
    """An ``ultralytics`` whose ``YOLO()`` patches the process like the real one.

    ``during`` are called inside ``YOLO()``, i.e. inside the second snapshot's
    span -- where SAM's loader thread really does set things of its own.
    """
    import types

    torch = pytest.importorskip("torch")
    fake = types.ModuleType("ultralytics")
    fake.DURING = list(during)
    exec(compile(_FAKE_ULTRALYTICS, "<fake ultralytics>", "exec"), fake.__dict__)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return fake


@pytest.fixture
def guard_process(monkeypatch):
    """Whatever the code under test fails to put back, the suite gets back."""
    import warnings

    import PIL.Image
    torch = pytest.importorskip("torch")
    for owner, name in ((cv2, "imread"), (cv2, "imwrite"), (cv2, "imshow"),
                        (PIL.Image, "open"), (torch, "save")):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    threads, printopts, filters = cv2.getNumThreads(), np.get_printoptions(), list(warnings.filters)
    env = dict(os.environ)
    yield
    cv2.setNumThreads(threads)
    np.set_printoptions(**printopts)
    warnings.filters[:] = filters
    for key in [k for k in os.environ if k not in env]:
        os.environ.pop(key, None)
    os.environ.update(env)


def test_after_the_model_has_loaded_the_process_is_as_it_was(tmp_path, monkeypatch,
                                                             guard_process):
    """Round 3, review item 2: every ultralytics patch undone once the engine has loaded."""
    import warnings

    import PIL.Image
    torch = pytest.importorskip("torch")
    zero = tmp_path / "empty.png"
    zero.write_bytes(b"")
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ycfg"))
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    inside: list = []
    filters_before = list(warnings.filters)

    def look():
        inside.append(([f for f in warnings.filters if f not in filters_before],
                       os.environ.get("NUMEXPR_MAX_THREADS"), cv2.imread))

    _fake_ultralytics(monkeypatch, zero, during=[look])
    before = {"env": dict(os.environ), "threads": cv2.getNumThreads(),
              "imread": cv2.imread, "imwrite": cv2.imwrite, "imshow": cv2.imshow,
              "pil": PIL.Image.open, "save": torch.save,
              "np": np.get_printoptions(), "filters": list(warnings.filters)}
    eng = DetectionEngine(config(tmp_path, model=weights,
                                 yolo_config_dir=tmp_path / "ycfg"), tmp_path / "det")
    eng.load()
    assert eng.model is not None and eng.model.names[0] == "screw"
    # the fake really did patch the process (two filters of its own, the env, imread) ...
    (added, numexpr, imread), = inside
    assert len(added) == 2 and numexpr == "8" and imread is not before["imread"]
    # ... and none of it is left
    assert cv2.imread(str(zero)) is None
    assert cv2.imread is before["imread"] and cv2.imwrite is before["imwrite"]
    assert cv2.imshow is before["imshow"] and cv2.getNumThreads() == before["threads"]
    assert PIL.Image.open is before["pil"] and torch.save is before["save"]
    assert dict(os.environ) == before["env"]
    assert np.get_printoptions() == before["np"] and warnings.filters == before["filters"]
    assert warnings.filterwarnings.__name__ == "filterwarnings"     # unwrapped again
    assert not hasattr(warnings.simplefilter, "__wrapped__")


class _OtherThreadsWarning(DeprecationWarning):
    """Stands for sympy's ``SymPyDeprecationWarning``, filtered on SAM's thread."""


def test_what_another_thread_sets_while_the_model_loads_stays(tmp_path, monkeypatch,
                                                              guard_process):
    """Round 4: SAM's loader sets an env variable and sympy a filter meanwhile; both stay.

    Only what ultralytics' own code changed is put back: its variables in
    ``ULTRALYTICS_ENV`` and the filters its own calls added.
    """
    import threading
    import warnings

    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ycfg"))
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    monkeypatch.setenv("TDA_U3_SHARED", "before")
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "6")
    monkeypatch.setenv("NUMEXPR_MAX_THREADS", "3")
    inductor = str(tmp_path / "inductor")
    filters_before = list(warnings.filters)

    def sam_loader():
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor     # torch/_inductor does this
        os.environ["TDA_U3_SHARED"] = "after"
        warnings.simplefilter("once", _OtherThreadsWarning)  # sympy does this

    def meanwhile():
        other = threading.Thread(target=sam_loader, name="tda-sam-load-u3")
        other.start()
        other.join()

    _fake_ultralytics(monkeypatch, tmp_path / "empty.png", during=[meanwhile])
    eng = DetectionEngine(config(tmp_path, model=weights,
                                 yolo_config_dir=tmp_path / "ycfg"), tmp_path / "det")
    eng.load()
    # the other thread's changes stay ...
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == inductor
    assert os.environ["TDA_U3_SHARED"] == "after"
    new = [f for f in warnings.filters if f not in filters_before]
    assert new == [("once", None, _OtherThreadsWarning, None, 0)]
    # ... ultralytics' do not: the variables it changed, the one it added, its two filters
    assert os.environ["OMP_NUM_THREADS"] == "6"
    assert os.environ["NUMEXPR_MAX_THREADS"] == "3"
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    assert not any(f[0] == "ignore" and f[2] is BytesWarning for f in warnings.filters)
    assert not any(f[1] is not None and f[1].pattern == "u3 fake" for f in warnings.filters)


def test_a_filter_ultralytics_repeats_is_not_taken_away(guard_process):
    """A filter that was there before the span stays, even if ultralytics adds it again."""
    import types
    import warnings

    warnings.filterwarnings("ignore", message="u3 already there")
    had = list(warnings.filters)
    fake = types.ModuleType("ultralytics.u3_fake")
    exec("import warnings\n"
         "def again():\n"
         "    warnings.filterwarnings('ignore', message='u3 already there')\n"
         "    warnings.filterwarnings('ignore', message='u3 new')\n", fake.__dict__)
    with DM.keep_process_state():
        fake.again()
        assert any(f[1] is not None and f[1].pattern == "u3 new" for f in warnings.filters)
    assert sorted(map(repr, warnings.filters)) == sorted(map(repr, had))


@pytest.mark.slow
def test_importing_the_real_ultralytics_inside_keep_process_state_changes_nothing(tmp_path):
    """The real import (no model, no GPU), in a fresh process: it patches, and it is undone."""
    import importlib.util
    import subprocess

    # find_spec, not importorskip: importing it here would patch *this* process
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("ultralytics is not installed")
    zero = tmp_path / "empty.png"
    zero.write_bytes(b"")
    code = f"""
import json, os, sys, threading, warnings
sys.path.insert(0, {str(REPO)!r})
import cv2, numpy as np, torch, PIL.Image
from tda.models.detector import keep_process_state
zero = {str(zero)!r}
before = dict(os.environ)
filters = list(warnings.filters)
fns = (cv2.imread, cv2.imwrite, cv2.imshow, PIL.Image.open, torch.save)
threads = cv2.getNumThreads()
printopts = np.get_printoptions()
class Other(DeprecationWarning):
    pass
def sam_loader():
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "u3-inductor"
    warnings.simplefilter("once", Other)
with keep_process_state(torch):
    import ultralytics
    from ultralytics import YOLO
    other = threading.Thread(target=sam_loader)
    other.start()
    other.join()
    inside = [cv2.imread is not fns[0], PIL.Image.open is not fns[3],
              torch.save is not fns[4], os.environ.get("NUMEXPR_MAX_THREADS"),
              cv2.getNumThreads(),
              sum(f not in filters for f in warnings.filters)]
after = dict(os.environ)
inductor = after.pop("TORCHINDUCTOR_CACHE_DIR", None)
print(json.dumps({{
    "inside": inside,
    "imread_none": cv2.imread(zero) is None,
    "same_fns": [a is b for a, b in zip((cv2.imread, cv2.imwrite, cv2.imshow,
                                         PIL.Image.open, torch.save), fns)],
    "env_same": after == before,
    "inductor": inductor,
    "new_filters": [repr(f) for f in warnings.filters if f not in filters],
    "numexpr": os.environ.get("NUMEXPR_MAX_THREADS"),
    "threads_same": cv2.getNumThreads() == threads,
    "np_same": np.get_printoptions() == printopts,
}}))
"""
    env = dict(os.environ, YOLO_CONFIG_DIR=str(tmp_path / "ycfg"), YOLO_OFFLINE="1",
               TMP=str(tmp_path), TEMP=str(tmp_path))
    env.pop("NUMEXPR_MAX_THREADS", None)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    # ultralytics really did patch the process ...
    assert out["inside"][:3] == [True, True, True] and out["inside"][3] is not None
    # (its six filters and the other thread's one)
    assert out["inside"][5] >= 7, out["inside"]
    # ... and none of it survived
    assert out["imread_none"] and all(out["same_fns"]) and out["env_same"]
    assert out["numexpr"] is None and out["threads_same"] and out["np_same"]
    # while what another thread set meanwhile (round 4) did
    assert out["inductor"] == "u3-inductor"
    assert out["new_filters"] == ["('once', None, <class '__main__.Other'>, None, 0)"]


# --------------------------------------------------------------------------- #
# task U4: the RF-DETR backend (a fake rfdetr -- no weights, no GPU)
# --------------------------------------------------------------------------- #
import fake_rfdetr as FR  # noqa: E402  (tests/ is on the path, like app_scene)


@pytest.fixture
def guard_rfdetr(guard_process):
    """U4's items too: whatever the backend fails to put back, the suite gets back."""
    import PIL.Image
    torch = pytest.importorskip("torch")
    precision = torch.get_float32_matmul_precision()
    pil_max = PIL.Image.MAX_IMAGE_PIXELS
    had = vars(np).get("complex_", FR)
    yield
    if torch.get_float32_matmul_precision() != precision:
        torch.set_float32_matmul_precision(precision)
    PIL.Image.MAX_IMAGE_PIXELS = pil_max
    if had is FR:
        vars(np).pop("complex_", None)


def _process_now() -> dict:
    """Everything rfdetr (and ultralytics) was measured to change, as it is now."""
    import warnings

    import PIL.Image
    torch = pytest.importorskip("torch")
    return {"precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "pil_max": PIL.Image.MAX_IMAGE_PIXELS, "complex_": "complex_" in vars(np),
            "env": dict(os.environ), "filters": list(warnings.filters),
            "threads": cv2.getNumThreads(), "imread": cv2.imread,
            "pil_open": PIL.Image.open, "save": torch.save,
            "np_print": np.get_printoptions()}


def _rf_config(tmp_path: Path, **over) -> DetectorConfig:
    weights = tmp_path / "rf.pt"
    if not weights.exists():
        weights.write_bytes(b"rf-detr weights, or so the fake says")
    return config(tmp_path, backend="rfdetr", model=weights, **over)


def _rf_ready(monkeypatch):
    """CUDA 'available' (nothing touches it) and the variables the load writes put back."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")          # set on purpose by the backend
    for name in DM.RFDETR_ENV:
        monkeypatch.delenv(name, raising=False)


def test_the_backend_is_chosen_by_the_file(tmp_path, monkeypatch):
    built = []

    class Built(StubModel):
        def __init__(self, cfg, kind):
            super().__init__(identity=DM.file_sha1(cfg.model))
            built.append((kind, cfg.backend))

    monkeypatch.setattr(DM, "YoloTileDetector", lambda cfg: Built(cfg, "yolo"))
    monkeypatch.setattr(DM, "RFDetrTileDetector", lambda cfg: Built(cfg, "rfdetr"))
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"w")
    DM.make_detector(config(tmp_path, model=weights))
    DM.make_detector(config(tmp_path, model=weights, backend="rfdetr"))
    assert built == [("yolo", "yolo"), ("rfdetr", "rfdetr")]
    # the engine builds through it when no stub is given
    eng = DetectionEngine(config(tmp_path, model=weights, backend="rfdetr"), tmp_path / "det")
    assert eng.factory is DM.make_detector
    eng.load()
    assert built[-1] == ("rfdetr", "rfdetr") and eng.identity == DM.file_sha1(weights)


def test_rfdetr_loads_and_the_process_is_as_it_was(tmp_path, monkeypatch, guard_rfdetr):
    """U4: what loading rfdetr changes process-wide is put back -- and it did change it."""
    _rf_ready(monkeypatch)
    seen: dict = {}
    fake = FR.install(monkeypatch, after_import=[lambda: seen.update(import_=_process_now())])
    before = _process_now()
    import warnings
    eng = DetectionEngine(_rf_config(tmp_path), tmp_path / "det")
    text = eng.load()
    assert isinstance(eng.model, DM.RFDetrTileDetector)
    # the fake really did what rfdetr does ...
    inside = seen["import_"]
    assert inside["precision"] == "high" and inside["tf32"] is True
    assert inside["pil_max"] == 2_000_000_000 and inside["complex_"] is True
    assert inside["env"]["KMP_DUPLICATE_LIB_OK"] == "True"
    assert inside["env"]["KMP_INIT_AT_FORK"] == "FALSE"
    assert len([f for f in inside["filters"] if f not in before["filters"]]) == 4
    # ... and none of it is left
    after = _process_now()
    for key in ("precision", "tf32", "pil_max", "complex_", "env", "filters", "threads",
                "np_print"):
        assert after[key] == before[key], key
    for key in ("imread", "pil_open", "save"):
        assert after[key] is before[key], key
    assert not hasattr(warnings.filterwarnings, "__wrapped__")
    assert not hasattr(np, "complex_")
    # built as L3 ran it: fp16, a warm-up of two batches of four, at the floor
    assert ("inference", False, "torch.float16") in fake.log
    assert [e for e in fake.log if e[0] == "predict"] == [("predict", 4, 0.10, False)] * 2
    assert fake.log[0] == ("from_checkpoint", str(tmp_path / "rf.pt"), {"device": "cuda:0"})
    assert text.startswith("rfdetr model") and "merge floor 0.1" in text
    assert "no manifest" in text and "at 640 px" in text


class _OtherThreadsWarning(DeprecationWarning):
    """Stands for sympy's filter, added on SAM's loader thread meanwhile."""


def test_rfdetr_what_another_thread_sets_while_the_model_loads_stays(tmp_path, monkeypatch,
                                                                     guard_rfdetr):
    import threading
    import warnings

    _rf_ready(monkeypatch)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.setenv("TDA_U4_SHARED", "before")
    inductor = str(tmp_path / "inductor")
    filters_before = list(warnings.filters)

    def sam_loader():
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor
        os.environ["TDA_U4_SHARED"] = "after"
        warnings.simplefilter("once", _OtherThreadsWarning)

    def meanwhile():
        other = threading.Thread(target=sam_loader, name="tda-sam-load-u4")
        other.start()
        other.join()

    FR.install(monkeypatch, during=[meanwhile])
    DetectionEngine(_rf_config(tmp_path), tmp_path / "det").load()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == inductor
    assert os.environ["TDA_U4_SHARED"] == "after"
    new = [f for f in warnings.filters if f not in filters_before]
    assert new == [("once", None, _OtherThreadsWarning, None, 0)]


def test_an_rfdetr_that_patches_and_then_fails_to_import_is_off_and_put_back(
        tmp_path, monkeypatch, guard_rfdetr):
    _rf_ready(monkeypatch)
    FR.install(monkeypatch, fail_import=True)
    before = _process_now()
    with pytest.raises(DetectorUnavailable, match="rfdetr does not import"):
        DM.RFDetrTileDetector(_rf_config(tmp_path))
    after = _process_now()
    for key in ("precision", "tf32", "pil_max", "complex_", "env", "filters"):
        assert after[key] == before[key], key


def test_no_cuda_stops_rfdetr_before_it_is_imported(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "rfdetr", None)       # would raise if reached
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")              # put back after the test
    with pytest.raises(DetectorUnavailable, match="CUDA is not available"):
        DM.RFDetrTileDetector(_rf_config(tmp_path))
    assert os.environ["HF_HUB_OFFLINE"] == "1", "downloads were not forbidden first"


@pytest.mark.parametrize("what", ["unreadable", "another resolution", "no screw class"])
def test_an_rfdetr_model_that_cannot_run_here_is_off(tmp_path, monkeypatch, guard_rfdetr, what):
    _rf_ready(monkeypatch)
    kwargs, message = {
        "unreadable": ({"fail_load": RuntimeError("invalid load key")}, "rfdetr cannot load"),
        "another resolution": ({"resolution": 560}, "runs at 560 px, the tiles are 640"),
        "no screw class": ({"class_names": ("connector", "cpu")}, r"no class \['screw'\]"),
    }[what]
    FR.install(monkeypatch, **kwargs)
    before = _process_now()
    with pytest.raises(DetectorUnavailable, match=message):
        DM.RFDetrTileDetector(_rf_config(tmp_path))
    assert _process_now()["precision"] == before["precision"]
    assert _process_now()["env"] == before["env"]


def _manifest(tmp_path: Path, **over) -> Path:
    weights = tmp_path / "rf.pt"
    body = {"model": "rfdetr_small", "classes": list(FR.L3_CLASSES),
            "bytes": weights.stat().st_size,
            "sha256": DM.file_digests(weights, ["sha256"])["sha256"]}
    for key, value in over.items():
        if value is None:
            body.pop(key, None)
        else:
            body[key] = value
    path = tmp_path / "rf.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.parametrize("what", ["agrees", "yolo's mapping", "classes reordered",
                                  "a class short", "bytes", "sha256", "sha1", "not json",
                                  "no classes"])
def test_the_manifest_must_agree_with_the_file_and_its_classes(tmp_path, monkeypatch,
                                                               guard_rfdetr, what):
    """U4: read beside the weights, checked against the model like the configured classes."""
    _rf_ready(monkeypatch)
    FR.install(monkeypatch)
    cfg = _rf_config(tmp_path)
    over = {
        "agrees": {},
        "yolo's mapping": {"classes": {str(i): c for i, c in enumerate(FR.L3_CLASSES)},
                           "sha1": DM.file_sha1(cfg.model), "sha256": None},
        "classes reordered": {"classes": list(reversed(FR.L3_CLASSES))},
        "a class short": {"classes": list(FR.L3_CLASSES[:-1])},
        "bytes": {"bytes": 1},
        "sha256": {"sha256": "0" * 64},
        "sha1": {"sha1": "0" * 40},
        "not json": None,
        "no classes": {"classes": None},
    }[what]
    if over is None:
        (tmp_path / "rf.json").write_text("{not json", encoding="utf-8")
    else:
        _manifest(tmp_path, **over)
    if what in ("agrees", "yolo's mapping"):
        det = DM.RFDetrTileDetector(cfg)
        assert "manifest rf.json agrees" in det.describe()
        return
    message = {"classes reordered": "lists the classes", "a class short": "lists the classes",
               "bytes": "is for a file of 1 bytes", "sha256": "is for another file: its sha256",
               "sha1": "is for another file: its sha1", "not json": "cannot be read",
               "no classes": "has no list of classes"}[what]
    with pytest.raises(DetectorUnavailable, match=message):
        DM.RFDetrTileDetector(cfg)


def test_rfdetr_detections_come_back_native_floored_and_without_the_extra_slot(
        tmp_path, monkeypatch, guard_rfdetr):
    """L3's forward: RGB tiles in, the (num_classes + 1)th slot and the sub-floor boxes out."""
    _rf_ready(monkeypatch)
    fake = FR.install(monkeypatch)
    det = DM.RFDetrTileDetector(_rf_config(tmp_path))
    spy = _MergeSpy(monkeypatch)
    img = np.zeros((800, 1000, 3), np.uint8)
    img[350:370, 700:720] = (230, 0, 0)            # red: found only if the tiles are RGB
    dets = det.detect(img, (100, 50, 900, 700), "scan", step=7)
    assert [(d.box, d.cls, round(d.conf, 2)) for d in dets] == [
        ((700.0, 350.0, 720.0, 370.0), "screw", 0.9)]
    # four tiles each saw the square whole; neither 0.09 nor the extra slot got in
    assert spy.confs == [[0.9, 0.9, 0.9, 0.9]]
    assert fake.log[-1] == ("predict", 4, 0.10, False)


@pytest.mark.slow
def test_importing_the_real_rfdetr_inside_keep_process_state_changes_nothing(tmp_path):
    """The real import (no model, no GPU), in a fresh process: it patches, and it is undone."""
    import importlib.util
    import subprocess

    if importlib.util.find_spec("rfdetr") is None:
        pytest.skip("rfdetr is not installed")
    code = f"""
import json, os, sys, threading, warnings
sys.path.insert(0, {str(REPO)!r})
import cv2, numpy as np, torch, PIL.Image
from tda.models.detector import RFDETR_PATCHES, keep_process_state
before = dict(os.environ)
filters = list(warnings.filters)
precision = torch.get_float32_matmul_precision()
pil_max = PIL.Image.MAX_IMAGE_PIXELS
class Other(DeprecationWarning):
    pass
def sam_loader():
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "u4-inductor"
    warnings.simplefilter("once", Other)
with keep_process_state(torch, RFDETR_PATCHES):
    from rfdetr import RFDETR
    other = threading.Thread(target=sam_loader)
    other.start()
    other.join()
    inside = {{"precision": torch.get_float32_matmul_precision(),
              "pil_max": PIL.Image.MAX_IMAGE_PIXELS, "complex_": "complex_" in vars(np),
              "kmp": [os.environ.get(k) for k in ("KMP_DUPLICATE_LIB_OK", "KMP_INIT_AT_FORK")],
              "new_filters": [repr(f) for f in warnings.filters if f not in filters]}}
after = dict(os.environ)
inductor = after.pop("TORCHINDUCTOR_CACHE_DIR", None)
print(json.dumps({{
    "inside": inside,
    "precision": torch.get_float32_matmul_precision(),
    "tf32": torch.backends.cuda.matmul.allow_tf32,
    "pil_max_same": PIL.Image.MAX_IMAGE_PIXELS == pil_max,
    "complex_": "complex_" in vars(np),
    "env_same": after == before, "inductor": inductor,
    "new_filters": [repr(f) for f in warnings.filters if f not in filters],
}}))
"""
    env = dict(os.environ, TMP=str(tmp_path), TEMP=str(tmp_path), HF_HUB_OFFLINE="1")
    for name in DM.RFDETR_ENV + ("TORCHINDUCTOR_CACHE_DIR",):
        env.pop(name, None)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    inside = out["inside"]
    # rfdetr really did change the process ...
    assert inside["precision"] == "high" and inside["pil_max"] == 2_000_000_000
    assert inside["complex_"] and inside["kmp"] == ["True", "FALSE"]
    theirs = [f for f in inside["new_filters"] if any(
        p in f for p in ("scipy", "urllib3", "requests", "matrix subclass"))]
    assert len(theirs) >= 5, inside["new_filters"]
    # ... and none of that survived
    assert out["precision"] == "highest" and out["tf32"] is False
    assert out["pil_max_same"] and not out["complex_"] and out["env_same"]
    assert not [f for f in out["new_filters"] if any(
        p in f for p in ("scipy", "urllib3", "requests", "matrix subclass"))], out["new_filters"]
    # while what another thread set meanwhile, and torch's own, did
    assert out["inductor"] == "u4-inductor"
    assert "('once', None, <class '__main__.Other'>, None, 0)" in out["new_filters"]


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
