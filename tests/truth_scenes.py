"""Builders for the truth-table tests (not a test module itself).

One synthetic scene backs every test in ``test_truth.py`` and
``test_truth_conflicts.py``: desktop 1, view ``scan``, a 64x64 image, three
logical steps and two instances -- a PSU and the screw that fastens it, which is
unscrewed and taken out at step 3 (so it lies on the bench there, tracked by a
box). Shapes are rectangles encoded as COCO RLE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.model import (
    ActionRec,
    FrameKey,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy
from tda.core.truth import TruthService

DESKTOP = 1
VIEW = "scan"
HW = (64, 64)
PSU = "psu.01"
SCREW = "screw.psu.01"
FAN = "case_fan.01"

PSU_RECT = (10, 10, 50, 50)
SCREW_RECT = (14, 14, 22, 22)
FAN_RECT = (52, 52, 62, 62)
BENCH_BOX = (2.0, 2.0, 12.0, 12.0)


# --------------------------------------------------------------------------- #
# scene
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """A filled rectangle ``[x0, x1) x [y0, y1)`` as a 64x64 bool mask."""
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def mask_kf(instance: str, box, anchor: int, placement: str = "in_chassis") -> ShapeKeyframe:
    return ShapeKeyframe(
        id=None,
        instance=instance,
        desktop=DESKTOP,
        view=VIEW,
        pose_segment=1,
        anchor_step=anchor,
        placement=placement,
        geom_type="mask",
        parts=[ShapePart("main", masks.encode_rle(rect(*box)))],
    )


def box_kf(instance: str, box, anchor: int, placement: str = "on_bench") -> ShapeKeyframe:
    return ShapeKeyframe(
        id=None,
        instance=instance,
        desktop=DESKTOP,
        view=VIEW,
        pose_segment=1,
        anchor_step=anchor,
        placement=placement,
        geom_type="box",
        parts=[ShapePart("main", None, tuple(float(v) for v in box))],
    )


@dataclass
class Scene:
    """The seeded database plus the service and the keyframes under test."""

    db: Db
    tax: Taxonomy
    svc: TruthService
    psu_kf: ShapeKeyframe
    screw_kf: ShapeKeyframe
    bench_kf: ShapeKeyframe

    def key(self, step: int) -> FrameKey:
        return FrameKey(DESKTOP, step, VIEW)

    def refresh_all(self) -> dict:
        return self.svc.refresh_range(DESKTOP, VIEW, [1, 2, 3])

    def rows(self, step: int) -> dict[str, dict]:
        return self.db.compiled(self.key(step))

    def row(self, step: int, instance: str) -> dict:
        return self.rows(step)[instance]

    def counts(self, step: int, instance: str) -> str | None:
        rle = self.row(step, instance)["visible_rle"]
        return None if rle is None else rle["counts"]

    def review_status(self, step: int) -> str | None:
        return self.db.get_frame(self.key(step))["review_status"]

    def add_frame(self, step: int, hw=HW) -> None:
        aux = {} if hw is None else {"hw": [int(hw[0]), int(hw[1])]}
        self.db.upsert_frame(self.key(step), f"s{step:03d}.jpg", aux, None)


def build_scene(db: Db) -> Scene:
    """Seed ``db`` with the whole scene and return it wrapped in a :class:`Scene`."""
    tax = load_taxonomy()
    db.upsert_instance(InstanceRec(key=PSU, desktop=DESKTOP, cls="psu"))
    db.upsert_instance(
        InstanceRec(
            key=SCREW,
            desktop=DESKTOP,
            cls="screw",
            attrs={"role": "psu", "head": "PH2", "captive": False},
            parent=PSU,
            attached=True,  # it leaves the chassis with the PSU it fastens
            fastens=PSU,
        )
    )
    db.replace_steps(
        DESKTOP,
        [
            StepRec(DESKTOP, 1, "initial", "initial state"),
            StepRec(DESKTOP, 2, "normal", "loosen psu screw"),
            StepRec(DESKTOP, 3, "normal", "psu screw"),
        ],
        [ActionRec(DESKTOP, 3, 0, SCREW, "remove", tool="PH2")],
    )
    scene = Scene(
        db=db,
        tax=tax,
        svc=TruthService(db, tax),
        psu_kf=mask_kf(PSU, PSU_RECT, anchor=3),
        screw_kf=mask_kf(SCREW, SCREW_RECT, anchor=2),
        bench_kf=box_kf(SCREW, BENCH_BOX, anchor=3),
    )
    for step in (1, 2, 3):
        scene.add_frame(step)
    for kf in (scene.psu_kf, scene.screw_kf, scene.bench_kf):
        db.add_keyframe(kf)
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, [(PSU, "main"), (SCREW, "main")]))
    return scene


def replace_parts(scene: Scene, kf: ShapeKeyframe, parts: list[ShapePart]) -> None:
    """Redraw one keyframe (which bumps its stored version)."""
    kf.parts = parts
    scene.db.update_keyframe(kf)
