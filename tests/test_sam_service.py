"""Tests for the SAM 2.1 service and its background queue.

GPU-backed tests skip cleanly when ``SamService.available()`` is False (no CUDA
or no checkpoint). The queue's scheduling behaviour is tested against a stub
service, so it runs everywhere.

Prompt coordinates refer to ``tests/fixtures/images/scanner_crop_native.png``,
a 900x900 native-resolution scanner crop of a desktop chassis. The CPU cooler
sits in the top-right corner: ``FAN_POINT`` lands on a fan blade and ``FAN_BOX``
encloses the whole cooler.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tda.models import SamQueue, SamRequest, SamResult, SamService
from tda.models import sam_service
from tda.models.sam_service import (
    REFINE_RADIUS_PX,
    blend_local,
    default_checkpoint,
    mask_to_low_res_logits,
    points_within_radius,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "images" / "scanner_crop_native.png"

FAN_POINT = (760.0, 270.0)  # on the CPU cooler fan
FAN_BOX = (660.0, 95.0, 899.0, 335.0)  # encloses the cooler
FAN_NEG_POINT = (760.0, 160.0)  # inside the cooler mask, used to carve it back
CHASSIS_POINT = (450.0, 450.0)  # drive-cage / PSU panel in the middle of the crop

requires_sam = pytest.mark.skipif(
    not SamService.available(),
    reason="SAM 2.1 unavailable (needs CUDA torch, sam2 and the checkpoint)",
)


@pytest.fixture(scope="module")
def crop() -> np.ndarray:
    bgr = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    assert bgr is not None, f"fixture image missing: {FIXTURE}"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="module")
def service(crop: np.ndarray) -> SamService:
    if not SamService.available():
        pytest.skip("SAM 2.1 unavailable (needs CUDA torch, sam2 and the checkpoint)")
    svc = SamService()
    # Warm up CUDA kernels once so per-test latencies are representative.
    svc.predict(SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)]))
    return svc


@pytest.fixture(scope="module")
def fan_mask(service: SamService, crop: np.ndarray) -> np.ndarray:
    """Whole-cooler mask from the box prompt, used as a refinement prior."""
    return service.predict(SamRequest(image_crop=crop, box=FAN_BOX)).mask


# ---------------------------------------------------------------------------
# Pure helpers (no GPU)
# ---------------------------------------------------------------------------
def test_low_res_logits_shape_and_sign():
    mask = np.zeros((900, 900), dtype=bool)
    mask[200:400, 300:500] = True
    logits = mask_to_low_res_logits(mask)
    assert logits.shape == (1, 256, 256)
    assert logits.dtype == np.float32
    # Interior of the box is positive, far outside is negative.
    assert logits[0, 85, 114] > 0
    assert logits[0, 10, 10] < 0


def test_points_within_radius_is_a_disc():
    near = points_within_radius((200, 200), [(100.0, 100.0, 1)], radius=48.0)
    assert near[100, 100]
    assert near[100, 100 + 48]
    assert not near[100, 100 + 49]
    assert not near[0, 0]
    # Area of a discrete disc of r=48 is within 1% of pi*r^2.
    assert abs(near.sum() - np.pi * 48**2) / (np.pi * 48**2) < 0.01


def test_blend_local_keeps_prior_outside_radius():
    prior = np.zeros((300, 300), dtype=bool)
    prior[50:250, 50:250] = True
    proposal = np.zeros((300, 300), dtype=bool)  # proposal deletes everything
    out = blend_local(prior, proposal, [(150.0, 150.0, 0)], radius=48.0)
    near = points_within_radius(prior.shape, [(150.0, 150.0, 0)], radius=48.0)
    assert not out[near].any()
    assert np.array_equal(out[~near], prior[~near])
    # Without points nothing may change.
    assert np.array_equal(blend_local(prior, proposal, [], radius=48.0), prior)


def test_sam_result_defaults_candidates_to_the_single_mask():
    """Old two-field construction keeps working and still exposes a candidate."""
    mask = np.zeros((4, 4), dtype=bool)
    result = SamResult(mask=mask, score=0.5, ms=1.0)
    assert len(result.candidates) == 1
    assert result.candidates[0] is mask
    assert result.scores == [0.5]


def test_available_is_a_bool():
    assert isinstance(SamService.available(), bool)
    assert SamService.available(checkpoint="Z:/definitely/missing.pt") is False


@pytest.mark.parametrize(
    "bad_yaml",
    [
        "weights_dir: [unclosed\n",  # YAML syntax error
        "- just\n- a list\n",  # top-level sequence
        "a plain string\n",  # top-level scalar
        "weights_dir:\n  nested: 1\n",  # weights_dir is not a path
        "weights_dir: 17\n",  # weights_dir is not a path either
    ],
    ids=["syntax", "list", "scalar", "mapping", "int"],
)
def test_malformed_paths_yaml_degrades_to_unavailable(monkeypatch, tmp_path, bad_yaml):
    """A config typo must not turn the collection-time skip into an error."""
    cfg = tmp_path / "paths.yaml"
    cfg.write_text(bad_yaml, encoding="utf-8")
    monkeypatch.setattr(sam_service, "_PATHS_YAML", cfg)
    assert default_checkpoint() is None
    assert SamService.available() is False
    with pytest.raises(FileNotFoundError):
        SamService()


def test_missing_paths_yaml_uses_the_in_repo_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(sam_service, "_PATHS_YAML", tmp_path / "does_not_exist.yaml")
    fallback = default_checkpoint()
    assert fallback is not None
    assert fallback.name == sam_service.CHECKPOINT_NAME
    assert isinstance(SamService.available(), bool)


def test_empty_paths_yaml_uses_the_in_repo_fallback(monkeypatch, tmp_path):
    cfg = tmp_path / "paths.yaml"
    cfg.write_text("# nothing here\n", encoding="utf-8")
    monkeypatch.setattr(sam_service, "_PATHS_YAML", cfg)
    assert default_checkpoint() is not None


def test_default_checkpoint_reads_weights_dir(monkeypatch, tmp_path):
    cfg = tmp_path / "paths.yaml"
    cfg.write_text(f'weights_dir: "{tmp_path.as_posix()}/w"\n', encoding="utf-8")
    monkeypatch.setattr(sam_service, "_PATHS_YAML", cfg)
    assert default_checkpoint() == tmp_path / "w" / sam_service.CHECKPOINT_NAME


# ---------------------------------------------------------------------------
# Service (GPU)
# ---------------------------------------------------------------------------
@requires_sam
def test_set_image_caches_embedding(service: SamService, crop: np.ndarray):
    first = service.set_image(crop)
    assert isinstance(first, str) and first

    # Same object -> identity fast path, no recompute.
    t0 = time.perf_counter()
    again = service.set_image(crop)
    identity_ms = (time.perf_counter() - t0) * 1000.0
    assert again == first
    assert identity_ms < 1.0

    # Equal copy -> same id via the content hash, still no recompute.
    assert service.set_image(crop.copy()) == first

    # Different content -> different id.
    other = crop.copy()
    other[:100, :100] = 0
    assert service.set_image(other) != first
    assert service.set_image(crop) == first


@requires_sam
@pytest.mark.parametrize("point", [CHASSIS_POINT, FAN_POINT])
def test_point_prompt_returns_a_plausible_mask(service, crop, point):
    result = service.predict(SamRequest(image_crop=crop, points=[(*point, 1)]))
    assert isinstance(result, SamResult)
    assert result.mask.dtype == np.bool_
    assert result.mask.shape == crop.shape[:2]
    assert result.mask.sum() > 500
    assert result.score > 0.5
    assert result.mask[int(point[1]), int(point[0])]
    assert result.ms > 0


@requires_sam
def test_box_prompt_agrees_with_point_prompt(service, crop, fan_mask):
    point = service.predict(SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)])).mask
    iou = (point & fan_mask).sum() / max(1, (point | fan_mask).sum())
    assert iou > 0.5, f"box/point IoU too low: {iou:.3f}"


@requires_sam
def test_multimask_picks_the_best_candidate(service, crop):
    result = service.predict(
        SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)], multimask=True)
    )
    assert result.mask.sum() > 500
    assert result.score > 0.5


@requires_sam
def test_multimask_returns_three_candidates_sorted_by_score(service, crop):
    """One point is ambiguous (part vs assembly), so the annotator gets 3 offers."""
    result = service.predict(
        SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)], multimask=True)
    )
    assert len(result.candidates) == 3
    assert len(result.scores) == 3
    assert result.scores == sorted(result.scores, reverse=True)
    # The legacy fields still name the best candidate.
    assert result.scores[0] == result.score
    assert np.array_equal(result.candidates[0], result.mask)
    for cand in result.candidates:
        assert cand.dtype == np.bool_
        assert cand.shape == crop.shape[:2]
    # Cycling would be pointless if SAM returned the same mask three times.
    assert not np.array_equal(result.candidates[0], result.candidates[1])
    assert not np.array_equal(result.candidates[1], result.candidates[2])


@requires_sam
def test_single_mask_prediction_has_exactly_one_candidate(service, crop):
    result = service.predict(SamRequest(image_crop=crop, box=FAN_BOX))
    assert len(result.candidates) == 1
    assert result.scores == [result.score]
    assert np.array_equal(result.candidates[0], result.mask)


@requires_sam
def test_local_refinement_only_changes_pixels_near_the_new_point(service, crop, fan_mask):
    prior = fan_mask
    assert prior[int(FAN_NEG_POINT[1]), int(FAN_NEG_POINT[0])], "neg point must sit inside the prior"

    refined = service.predict(
        SamRequest(
            image_crop=crop,
            points=[(*FAN_NEG_POINT, 0)],
            mask_input=prior,
        )
    ).mask

    near = points_within_radius(prior.shape, [(*FAN_NEG_POINT, 0)], REFINE_RADIUS_PX)
    far_prior = prior & ~near

    # A negative click removes area ...
    assert refined.sum() < prior.sum()
    assert (prior & near).sum() > (refined & near).sum()

    # ... but only locally: everything farther than 48 px is untouched.
    kept = (refined & far_prior).sum() / far_prior.sum()
    assert kept >= 0.8, f"only {kept:.3f} of far prior pixels survived"
    assert np.array_equal(refined[~near], prior[~near])


@requires_sam
def test_local_refinement_offers_only_the_blended_result(service, crop, fan_mask):
    """Cycling raw proposals after a refinement would undo the blend."""
    result = service.predict(
        SamRequest(
            image_crop=crop,
            points=[(*FAN_NEG_POINT, 0)],
            mask_input=fan_mask,
            multimask=True,
        )
    )
    assert len(result.candidates) == 1
    assert len(result.scores) == 1
    assert np.array_equal(result.candidates[0], result.mask)
    assert not np.array_equal(result.mask, fan_mask), "the blend must have changed it"


@requires_sam
def test_mask_input_without_points_is_not_blended(service, crop, fan_mask):
    """No point means no anchor for a local edit, so SAM's output stands."""
    grown = fan_mask.copy()
    grown[400:500, 400:500] = True  # nonsense region far from the cooler
    out = service.predict(
        SamRequest(image_crop=crop, box=FAN_BOX, mask_input=grown)
    ).mask
    assert not np.array_equal(out, grown)
    assert out[400:500, 400:500].sum() < 100


@requires_sam
def test_predict_without_a_prompt_raises(service, crop):
    with pytest.raises(ValueError):
        service.predict(SamRequest(image_crop=crop))


@requires_sam
def test_mask_input_shape_is_validated(service, crop):
    with pytest.raises(ValueError):
        service.predict(
            SamRequest(
                image_crop=crop,
                points=[(*FAN_POINT, 1)],
                mask_input=np.zeros((10, 10), dtype=bool),
            )
        )


@requires_sam
def test_queue_delivers_a_result_within_5s(service, crop):
    done = threading.Event()
    box: list[SamResult] = []
    q = SamQueue(service)
    try:
        q.submit(
            SamRequest(image_crop=crop, points=[(*FAN_POINT, 1)]),
            lambda res: (box.append(res), done.set()),
        )
        assert done.wait(5.0), "callback did not fire within 5 s"
    finally:
        q.stop()
    assert not q.running
    assert box[0].mask.shape == crop.shape[:2]
    assert box[0].mask.sum() > 500


# ---------------------------------------------------------------------------
# Queue scheduling (stub service, runs without a GPU)
# ---------------------------------------------------------------------------
class _StubService:
    """Duck-typed stand-in for SamService: sleeps, then echoes the point count."""

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.seen: list[int] = []
        self.started = threading.Event()

    def predict(self, req: SamRequest) -> SamResult:
        self.started.set()
        time.sleep(self.delay)
        tag = int(req.points[0][0])
        self.seen.append(tag)
        return SamResult(mask=np.zeros((4, 4), dtype=bool), score=float(tag), ms=0.0)


def _req(tag: int) -> SamRequest:
    return SamRequest(image_crop=np.zeros((4, 4, 3), np.uint8), points=[(float(tag), 0.0, 1)])


def test_queue_latest_wins_drops_stale_requests():
    stub = _StubService(delay=0.25)
    q = SamQueue(stub)  # type: ignore[arg-type]
    got: list[float] = []
    try:
        q.submit(_req(1), got.append)  # starts running immediately
        assert stub.started.wait(2.0)
        q.submit(_req(2), got.append)  # queued
        q.submit(_req(3), got.append)  # replaces 2
        q.submit(_req(4), got.append)  # replaces 3
        deadline = time.time() + 5.0
        while len(got) < 2 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        q.stop()
    assert [r.score for r in got] == [1.0, 4.0]
    assert stub.seen == [1, 4]
    assert q.dropped == 2


def test_queue_survives_a_failing_predict_and_a_failing_callback():
    class Boom(_StubService):
        def predict(self, req: SamRequest) -> SamResult:
            if req.points[0][0] == 0.0:
                raise RuntimeError("bad prompt")
            return super().predict(req)

    stub = Boom(delay=0.0)
    q = SamQueue(stub)  # type: ignore[arg-type]
    done = threading.Event()
    try:
        q.submit(_req(0), lambda res: None)  # predict raises
        time.sleep(0.1)
        q.submit(_req(7), lambda res: (_ for _ in ()).throw(ValueError("bad cb")))
        time.sleep(0.1)
        q.submit(_req(9), lambda res: done.set())
        assert done.wait(5.0), "queue died after an exception"
    finally:
        q.stop()
    assert isinstance(q.last_error, BaseException)


def test_queue_rejects_submit_after_stop():
    q = SamQueue(_StubService(delay=0.0))  # type: ignore[arg-type]
    q.stop()
    assert not q.running
    with pytest.raises(RuntimeError):
        q.submit(_req(1), lambda res: None)
    q.stop()  # idempotent
