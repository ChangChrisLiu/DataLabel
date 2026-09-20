"""The compiler's output, pinned byte for byte against a recorded golden.

Why this file exists
--------------------
:mod:`tda.core.compiler` is the generator behind every stored mask and behind
the ``input_hash`` that decides whether a frozen frame has to be looked at
again (spec 3.4).  Making it *faster* is therefore only allowed if it stays
**bit-identical**: one differing run length moves a row's RLE, one differing
hash queues every verified frame of every desktop for a re-check.

So the answers are recorded once, from the implementation as it stood, into
``tests/fixtures/compiler_golden.json``, and this module recompiles the same
scenes and compares:

* the frame's ``input_hash``, ``problems``, ``painted`` and ``layers``;
* per instance the **sha1 of the visible mask's RLE counts** -- the exact bytes
  that go into the truth table -- plus the amodal RLE, the box, the occlusion
  ratio, the visibility label, the placement and the keyframe id.

The scenes are generated from a fixed seed by :func:`scenes` and deliberately
cover everything a window-bounded compiler could get wrong: shapes that hang
over the canvas edge, shapes far outside the pose segment's ROI, multi-part
shapes, bench boxes, frame-level occluders, frame overrides (pixels and
labels), pairwise overrides and cycles, layers missing from the z-order,
non-identity transforms, missing keyframes and wrong-size RLEs.  They are all
under 96 px on a side, so :func:`big_scenes` adds the two canvases the
annotator really works on -- 1600x1600 and 4032x3040 -- where every windowing
decision was made and where an off-by-one costs a real annotation.

Regenerate the fixture **only** when the compiler's semantics are meant to
change (and then say so in the commit)::

    D:\\Anaconda\\envs\\tda\\python.exe -m tests.test_compiler_golden --write
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from tda.core import masks
from tda.core.compiler import CompiledFrame, compile_frame
from tda.core.model import (
    FrameKey,
    FrameOverride,
    OccluderMask,
    PairOverride,
    Placement,
    ShapeKeyframe,
    ShapePart,
    Similarity,
    Visibility,
    ZOrderRec,
)

GOLDEN = Path(__file__).parent / "fixtures" / "compiler_golden.json"

#: Canvas sizes the scenes are built on. The tall/wide pair matters: a window
#: implementation that swaps x and y passes on a square canvas.
SIZES = ((64, 64), (48, 96), (96, 48), (37, 53))
#: How many random scenes the fixture holds.
N_SCENES = 60
SEED = 20260919

CHASSIS = Placement.IN_CHASSIS.value
BENCH = Placement.ON_BENCH.value


# --------------------------------------------------------------------------- #
# scene generation
# --------------------------------------------------------------------------- #
def _rect(box, hw) -> np.ndarray:
    """A filled rectangle, clipped to the canvas (the box may hang over it)."""
    mask = np.zeros(hw, dtype=bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return mask


def _blob(rng: random.Random, hw, *, outside: bool = False):
    """A random box; ``outside`` lets it hang over an edge of the canvas."""
    height, width = hw
    if outside:
        x0 = rng.randint(-width // 3, width - 2)
        y0 = rng.randint(-height // 3, height - 2)
    else:
        x0 = rng.randint(0, max(0, width - 4))
        y0 = rng.randint(0, max(0, height - 4))
    x1 = min(width, x0 + rng.randint(3, max(4, width // 2)))
    y1 = min(height, y0 + rng.randint(3, max(4, height // 2)))
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def _keyframe(rng: random.Random, instance: str, hw, kf_id: int,
              *, placement: str, anchor: int) -> ShapeKeyframe:
    """One keyframe: mask geometry with 1-3 parts, or a bench box."""
    if placement == BENCH:
        return ShapeKeyframe(
            id=kf_id, instance=instance, desktop=1, view="scan", pose_segment=1,
            anchor_step=anchor, placement=BENCH, geom_type="box",
            parts=[ShapePart("main", None,
                             tuple(float(v) for v in _blob(rng, hw)))],
            version=rng.randint(1, 3),
        )
    parts: list[ShapePart] = []
    for index in range(rng.choice((1, 1, 1, 2, 3))):
        box = _blob(rng, hw, outside=rng.random() < 0.25)
        if rng.random() < 0.2:
            # a part of a *mask* keyframe that carries only a rectangle: it is
            # rasterised rather than decoded, which is a second code path
            parts.append(ShapePart(f"p{index}", None, box))
        else:
            parts.append(ShapePart(f"p{index}", masks.encode_rle(_rect(box, hw)),
                                   box))
        if rng.random() < 0.06 and index:
            # the same part name twice: one layer, the union of the two
            parts.append(ShapePart(f"p{index - 1}",
                                   masks.encode_rle(_rect(_blob(rng, hw), hw))))
    return ShapeKeyframe(
        id=kf_id, instance=instance, desktop=1, view="scan", pose_segment=1,
        anchor_step=anchor, placement=CHASSIS, geom_type="mask", parts=parts,
        version=rng.randint(1, 3),
    )


def scenes() -> Iterator[tuple[str, dict]]:
    """``(name, compile_frame kwargs)`` for every golden scene, deterministically."""
    rng = random.Random(SEED)
    for index in range(N_SCENES):
        hw = SIZES[index % len(SIZES)]
        key = FrameKey(1, rng.randint(1, 9), "scan")
        count = rng.randint(1, 7)
        names = [f"i{n:02d}" for n in range(count)]
        needs: dict[str, str] = {}
        placements: dict[str, str] = {}
        keyframes: dict[str, list[ShapeKeyframe]] = {}
        order: list[tuple[str, str]] = []
        for n, name in enumerate(names):
            on_bench = rng.random() < 0.2
            placements[name] = BENCH if on_bench else CHASSIS
            needs[name] = "box" if on_bench else "mask"
            if rng.random() < 0.12:
                continue                      # no keyframe at all: missing_shape
            anchor = key.step + rng.randint(0, 4)
            shape = _keyframe(rng, name, hw, 100 + index * 10 + n,
                              placement=placements[name], anchor=anchor)
            keyframes[name] = [shape]
            if shape.geom_type == "mask" and rng.random() > 0.15:
                order.extend((name, part.name) for part in shape.parts)
        rng.shuffle(order)

        overrides = []
        if count >= 2 and rng.random() < 0.35:
            a, b = rng.sample(names, 2)
            overrides.append(PairOverride(1, "scan", 1, above=a, below=b))
            if rng.random() < 0.3:            # the contradiction: a cycle
                overrides.append(PairOverride(1, "scan", 1, above=b, below=a))

        occluders = []
        if rng.random() < 0.35:
            occluders.append(OccluderMask(
                frame=key, occluder_type=rng.choice(("hand", "tool", "cable")),
                rle=masks.encode_rle(_rect(_blob(rng, hw), hw)),
            ))
        if rng.random() < 0.1:                # an occluder of another frame
            occluders.append(OccluderMask(
                frame=FrameKey(1, key.step + 1, "scan"), occluder_type="arm",
                rle=masks.encode_rle(_rect(_blob(rng, hw), hw)),
            ))

        frame_overrides: dict[str, FrameOverride] = {}
        if rng.random() < 0.3:
            who = rng.choice(names)
            frame_overrides[who] = FrameOverride(
                frame=key, instance=who,
                visible_rle=(masks.encode_rle(_rect(_blob(rng, hw), hw))
                             if rng.random() < 0.7 else None),
                visibility=(rng.choice([v.value for v in Visibility])
                            if rng.random() < 0.5 else None),
            )
        if rng.random() < 0.08:               # a mask in the wrong coordinates
            who = rng.choice(names)
            if who in keyframes and keyframes[who][0].geom_type == "mask":
                other = (hw[0] + 7, hw[1] + 5)
                keyframes[who][0].parts[0] = ShapePart(
                    keyframes[who][0].parts[0].name,
                    masks.encode_rle(_rect(_blob(rng, other), other)),
                )

        transform = Similarity()
        if rng.random() < 0.25:
            transform = Similarity(
                scale=round(rng.uniform(0.8, 1.2), 3),
                theta=round(rng.uniform(-0.2, 0.2), 3),
                tx=float(rng.randint(-6, 6)), ty=float(rng.randint(-6, 6)),
            )

        yield f"{index:02d}-{hw[0]}x{hw[1]}", {
            "key": key,
            "hw": hw,
            "needs": needs,
            "keyframes": keyframes,
            "zorder": ZOrderRec(1, "scan", 1, order, version=rng.randint(1, 4)),
            "overrides": overrides,
            "occluders": occluders,
            "frame_overrides": frame_overrides,
            "transform": transform,
            "placements": placements,
            "pose_segment": 1,
            "bench_roi": [0, 0, hw[1] // 2, hw[0] // 2],
        }


#: The two canvases the annotator actually works on, as one scene each.
#:
#: Every generated scene above is at most 96 px on a side, which keeps the
#: fixture quick -- and leaves the big-canvas path unpinned, where every
#: windowed decision was made and where an off-by-one costs a real annotation.
#: These two are built by hand rather than drawn from the seeded stream, so
#: adding them left the sixty random scenes' answers exactly where they were.
BIG_SIZES = {"scanner-1600x1600": (1600, 1600), "oak-3040x4032": (3040, 4032)}


def big_scenes():
    """``(name, spec)`` for a full-size scanner frame and a full-size OAK one."""
    for name, hw in BIG_SIZES.items():
        height, width = hw
        key = FrameKey(1, 4, "scan")
        wide, tall = width // 8, height // 8
        chassis = (wide, tall, width - wide, height - tall)
        # a multi-part instance, one part of which hangs off the canvas
        cover = [ShapeKeyframe(
            id=901, instance="cover", desktop=1, view="scan", pose_segment=1,
            anchor_step=6, placement=CHASSIS, geom_type="mask", version=2,
            parts=[
                ShapePart("a", masks.encode_rle(_rect(chassis, hw)), chassis),
                ShapePart("b", masks.encode_rle(
                    _rect((-wide, -tall, wide * 2, tall * 2), hw)),
                    (-wide, -tall, wide * 2, tall * 2)),
            ],
        )]
        board = [ShapeKeyframe(
            id=902, instance="board", desktop=1, view="scan", pose_segment=1,
            anchor_step=5, placement=CHASSIS, geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(
                _rect((wide * 2, tall * 2, width - wide * 2, height - tall * 2), hw)))],
        )]
        # a bare rectangle, rasterised rather than decoded, and warped
        bracket = [ShapeKeyframe(
            id=903, instance="bracket", desktop=1, view="scan", pose_segment=1,
            anchor_step=9, placement=CHASSIS, geom_type="mask",
            parts=[ShapePart("main", None,
                             (wide * 3.5, tall * 1.25, wide * 4.5, tall * 6.75))],
        )]
        screw = [ShapeKeyframe(
            id=904, instance="screw", desktop=1, view="scan", pose_segment=1,
            anchor_step=4, placement=BENCH, geom_type="box",
            parts=[ShapePart("main", None, (12.0, 9.0, 60.0, 57.0))],
        )]
        yield name, {
            "key": key,
            "hw": hw,
            "needs": {"cover": "mask", "board": "mask", "bracket": "mask",
                      "screw": "box", "gone": "mask"},
            "keyframes": {"cover": cover, "board": board, "bracket": bracket,
                          "screw": screw},
            "zorder": ZOrderRec(1, "scan", 1,
                                [("board", "main"), ("cover", "a"), ("cover", "b")],
                                version=3),
            "overrides": [PairOverride(1, "scan", 1, above="board", below="cover")],
            "occluders": [OccluderMask(
                frame=key, occluder_type="hand",
                rle=masks.encode_rle(
                    _rect((width // 2, 0, width // 2 + wide, height // 3), hw)),
            )],
            "frame_overrides": {"board": FrameOverride(
                frame=key, instance="board",
                visible_rle=masks.encode_rle(
                    _rect((wide * 3, tall * 3, wide * 5, tall * 5), hw)),
                visibility=None,
            )},
            "transform": Similarity(scale=1.0, theta=0.02, tx=3.0, ty=-5.0),
            "placements": {"cover": CHASSIS, "board": CHASSIS,
                           "bracket": CHASSIS, "screw": BENCH, "gone": CHASSIS},
            "pose_segment": 1,
            "bench_roi": [0, 0, width // 2, height // 2],
        }


def compile_scene(spec: dict) -> CompiledFrame:
    return compile_frame(
        spec["key"], spec["hw"], spec["needs"], spec["keyframes"], spec["zorder"],
        spec["overrides"], spec["occluders"], spec["frame_overrides"],
        spec["transform"], "1",
        placements=spec["placements"], pose_segment=spec["pose_segment"],
        bench_roi=spec["bench_roi"],
    )


# --------------------------------------------------------------------------- #
# what is recorded
# --------------------------------------------------------------------------- #
def _mask_digest(mask) -> object:
    """sha1 of the mask's **RLE counts** -- the bytes the truth table stores."""
    if mask is None:
        return None
    counts = masks.encode_rle(mask)["counts"]
    return hashlib.sha1(counts.encode("ascii")).hexdigest()


def fingerprint(compiled: CompiledFrame) -> dict:
    """Everything one compilation says, as JSON."""
    return {
        "input_hash": compiled.input_hash,
        "problems": sorted(compiled.problems),
        "painted": {g: list(v) for g, v in sorted(compiled.painted.items())},
        "layers": {g: [list(layer) for layer in v]
                   for g, v in sorted(compiled.layers.items())},
        "instances": {
            name: {
                "visible": _mask_digest(inst.visible),
                "amodal": _mask_digest(inst.amodal),
                "box": None if inst.box is None else [float(v) for v in inst.box],
                "occlusion_ratio": round(float(inst.occlusion_ratio), 12),
                "visibility": inst.visibility,
                "placement": inst.placement,
                "keyframe_id": inst.keyframe_id,
            }
            for name, inst in sorted(compiled.instances.items())
        },
    }


def all_scenes():
    """The sixty generated scenes, then the two full-size ones."""
    yield from scenes()
    yield from big_scenes()


def build() -> dict:
    return {name: fingerprint(compile_scene(spec)) for name, spec in all_scenes()}


# --------------------------------------------------------------------------- #
# the test
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():  # pragma: no cover - only before the first --write
        pytest.fail(f"{GOLDEN} is missing; regenerate it deliberately")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_golden_covers_the_cases_a_window_could_break(golden):
    """The fixture is only a guard if the scenes really exercise the corners."""
    problems = {p.split(":")[0] for fp in golden.values() for p in fp["problems"]}
    assert {"missing_shape", "zorder_missing", "shape_size_mismatch",
            "zorder_cycle", "bench_missing"} <= problems
    labels = {inst["visibility"] for fp in golden.values()
              for inst in fp["instances"].values()}
    assert {"visible", "occluded_partial", "occluded_full", "out_of_view"} <= labels
    assert len(golden) == N_SCENES + len(BIG_SIZES)
    for name in BIG_SIZES:
        assert name in golden, "the full-size canvases are not pinned"

    specs = dict(scenes())
    assert len(specs) == N_SCENES
    kinds = {
        "box geometry": any(kf[0].geom_type == "box" for s in specs.values()
                            for kf in s["keyframes"].values()),
        "a part with no RLE": any(
            part.rle is None and part.box is not None
            for s in specs.values() for kf in s["keyframes"].values()
            for part in kf[0].parts if kf[0].geom_type == "mask"),
        "a repeated part name": any(
            len({p.name for p in kf[0].parts}) != len(kf[0].parts)
            for s in specs.values() for kf in s["keyframes"].values()),
        "a shape over the canvas edge": any(
            part.box is not None and (part.box[0] < 0 or part.box[1] < 0)
            for s in specs.values() for kf in s["keyframes"].values()
            for part in kf[0].parts),
        "a non-identity transform": any(
            not s["transform"].is_identity() for s in specs.values()),
        "a frame override with pixels": any(
            fo.visible_rle is not None for s in specs.values()
            for fo in s["frame_overrides"].values()),
        "an occluder": any(s["occluders"] for s in specs.values()),
        "a multi-part shape": any(len(kf[0].parts) > 1 for s in specs.values()
                                  for kf in s["keyframes"].values()),
    }
    assert all(kinds.values()), f"the scenes are missing: {sorted(k for k, v in kinds.items() if not v)}"


@pytest.mark.parametrize("name", [name for name, _spec in scenes()])
def test_compilation_is_byte_identical_to_the_golden(golden, name: str):
    spec = dict(scenes())[name]
    assert fingerprint(compile_scene(spec)) == golden[name]


@pytest.mark.parametrize("name", list(BIG_SIZES))
def test_a_full_size_canvas_is_byte_identical_to_the_golden(golden, name: str):
    """1600x1600 and 4032x3040: the canvases every window decision was made on."""
    spec = dict(big_scenes())[name]
    assert spec["hw"] == BIG_SIZES[name]
    assert fingerprint(compile_scene(spec)) == golden[name]


def test_every_scene_is_reproduced_from_the_seed():
    """Two generations of the scene list give the same scenes, in order."""
    first = [name for name, _ in scenes()]
    assert first == [name for name, _ in scenes()]


if __name__ == "__main__":  # pragma: no cover - the regeneration entry point
    import sys

    if "--write" not in sys.argv:
        raise SystemExit("pass --write to overwrite the golden fixture")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {GOLDEN} ({N_SCENES} generated scenes "
          f"+ {len(BIG_SIZES)} full-size ones)")
