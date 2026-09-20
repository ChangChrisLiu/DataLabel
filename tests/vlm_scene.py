"""The synthetic teardown the VLM task tests are generated from (not a test).

One desktop, eight logical steps, two views, and every corner the P0 task set of
spec 8.2 has to handle:

* a rule-derived constraint graph -- ``fastened_by``, ``connected_to`` (both the
  socket host and the PSU harness) and ``locked_by`` -- so V4/V5/V6/V16 have
  something to be right or wrong about;
* a ``dupli`` step (V12's "no change" positive) and a ``failed`` attempt the
  graph explains (V4's explained negative);
* a latch that goes out inside the motherboard (``gone_with_parent``), an
  implied instance, and a Label Studio draft -- none of which may be asked about;
* per-frame overrides forcing ``occluded_full`` / ``out_of_view`` in one view and
  ``visible`` in the other, which is all V14 and V15 are allowed to read.

The frames are marked ``review_status="verified"`` so the perception tasks are
emitted at all; their compiled rows stay ``auto``, which is what the truth
service produces for a frame nobody has frozen row by row.
"""
from __future__ import annotations

import numpy as np

from tda.core import masks
from tda.core.db import Db
from tda.core.graph import edges_to_db, propose_edges
from tda.core.model import (
    ActionRec,
    FrameKey,
    FrameOverride,
    InstanceRec,
    ShapeKeyframe,
    ShapePart,
    StepRec,
    ZOrderRec,
)
from tda.core.taxonomy import Taxonomy, load_taxonomy

DESKTOP = 7
VIEW = "scan"
OTHER = "oak1"
VIEWS = (VIEW, OTHER)
HW = (64, 64)
LAST_STEP = 8
STEPS = tuple(range(1, LAST_STEP + 1))

CHASSIS = "chassis.01"
BOARD = "motherboard.01"
PSU = "psu.01"
SCREW = "screw.psu.01"
PLUG = "connector.01"
RAM = "ram_module.01"
LATCH = "ram_latch.01"
DRAFT = "ls:Screw#3"

#: instance -> its rectangle (x0, y0, x1, y1) in every frame of every view
RECTS = {
    CHASSIS: (0, 0, 64, 64),
    BOARD: (4, 4, 44, 44),
    PSU: (46, 4, 62, 30),
    SCREW: (47, 5, 51, 9),
    PLUG: (40, 8, 46, 14),
    RAM: (8, 8, 14, 36),
    LATCH: (8, 36, 14, 40),
}

#: The staging area both views can see, so a removed part keeps a box row.
BENCH_ROI = (0, 44, 64, 64)
BENCH_BOXES = {PSU: (2.0, 46.0, 20.0, 60.0), SCREW: (24.0, 46.0, 28.0, 50.0),
               RAM: (32.0, 46.0, 38.0, 60.0)}


def rect(box) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = np.zeros(HW, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _steps() -> list[StepRec]:
    return [
        StepRec(DESKTOP, 1, "initial", "initial state"),
        StepRec(DESKTOP, 2, "normal", "psu screw"),
        StepRec(DESKTOP, 3, "dupli", "psu screw (repeat)", dupli=True),
        StepRec(DESKTOP, 4, "normal", "atx connector"),
        StepRec(DESKTOP, 5, "normal", "psu"),
        StepRec(DESKTOP, 6, "failed", "ram (stuck)"),
        StepRec(DESKTOP, 7, "normal", "ram clip"),
        StepRec(DESKTOP, 8, "normal", "ram"),
    ]


def _actions() -> list[ActionRec]:
    return [
        ActionRec(DESKTOP, 2, 0, SCREW, "unscrew", tool="PH2", direction="+Z"),
        ActionRec(DESKTOP, 4, 0, PLUG, "disconnect", tool="hand", direction="+Z"),
        ActionRec(DESKTOP, 5, 0, PSU, "remove", tool="hand", direction="+Z"),
        ActionRec(DESKTOP, 6, 0, RAM, "remove", tool="hand", direction="+Z",
                  result="failed", failure_reason="blocked_by_part"),
        ActionRec(DESKTOP, 7, 0, LATCH, "open", tool="hand"),
        ActionRec(DESKTOP, 8, 0, RAM, "remove", tool="hand", direction="+Z"),
    ]


def _instances() -> list[InstanceRec]:
    return [
        InstanceRec(key=CHASSIS, desktop=DESKTOP, cls="chassis"),
        # implied: the sheet never operated the board, but every frame shows it
        InstanceRec(key=BOARD, desktop=DESKTOP, cls="motherboard",
                    attrs={"implied": True}),
        InstanceRec(key=PSU, desktop=DESKTOP, cls="psu", mounted_on=CHASSIS),
        InstanceRec(key=SCREW, desktop=DESKTOP, cls="screw",
                    attrs={"role": "psu", "head": "PH2", "captive": False},
                    fastens=PSU),
        InstanceRec(key=PLUG, desktop=DESKTOP, cls="connector",
                    attrs={"kind": "atx_24pin"}, socket_host=BOARD,
                    cable="cable:psu"),
        InstanceRec(key=RAM, desktop=DESKTOP, cls="ram_module", mounted_on=BOARD),
        # moulded into the board: it leaves the chassis inside it (spec 3.3)
        InstanceRec(key=LATCH, desktop=DESKTOP, cls="ram_latch",
                    attrs={"of": RAM}, parent=BOARD, attached=True),
        # draft material: never asked about, never counted (spec 3.2)
        InstanceRec(key=DRAFT, desktop=DESKTOP, cls="screw",
                    attrs={"role": "psu", "head": "PH2", "captive": False},
                    raw_names=["PSU Screw"]),
    ]


def _keyframes(view: str) -> list[ShapeKeyframe]:
    """One in-chassis shape per instance, plus a bench box for what comes out."""
    out = [
        ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=1,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", rle=masks.encode_rle(rect(box)))],
            amodal_complete=True,
        )
        for key, box in RECTS.items()
    ]
    out.extend(
        ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=1,
            anchor_step=LAST_STEP, placement="on_bench", geom_type="box",
            parts=[ShapePart("main", box=box)], amodal_complete=True,
        )
        for key, box in BENCH_BOXES.items()
    )
    return out


#: (view, step, instance) -> the visibility the frame is forced to report.
FORCED = {
    (VIEW, 7, LATCH): "occluded_full",
    (VIEW, 4, RAM): "occluded_full",
    (VIEW, 2, PLUG): "out_of_view",
    (OTHER, 7, LATCH): "visible",
    (OTHER, 4, RAM): "visible",
}


def build(db: Db, *, views=VIEWS, verified_steps=STEPS) -> Taxonomy:
    """Seed ``db`` with the whole scene and return the taxonomy it was built on."""
    tax = load_taxonomy()
    db.upsert_desktop(DESKTOP, {"brand": "HP", "model_family": "EliteDesk 800 G2 TWR",
                                "chassis_type": "twr"})
    for rec in _instances():
        db.upsert_instance(rec)
    db.replace_steps(DESKTOP, _steps(), _actions())
    for view in views:
        for step in STEPS:
            db.upsert_frame(
                FrameKey(DESKTOP, step, view), f"F:/{view}/007/{step:03d}/P_0.png",
                {"hw": [HW[0], HW[1]]}, "2025-06-01T09:00:00",
                flags={"review_status": ("verified" if step in verified_steps
                                         else "unlabeled")},
            )
        for kf in _keyframes(view):
            db.add_keyframe(kf)
        db.set_zorder(ZOrderRec(DESKTOP, view, 1, [
            (LATCH, "main"), (RAM, "main"), (SCREW, "main"), (PLUG, "main"),
            (PSU, "main"), (BOARD, "main"), (CHASSIS, "main"),
        ]))
        db.set_pose_segment(DESKTOP, view, 1, 1, LAST_STEP, 1, None, None)
        db.set_pose_segment_bench_roi(DESKTOP, view, 1, BENCH_ROI)
    for (view, step, instance), visibility in FORCED.items():
        if view in views:
            db.set_frame_override(
                FrameOverride(FrameKey(DESKTOP, step, view), instance, None, visibility)
            )
    edges_to_db(db, DESKTOP, propose_edges(db.instances(DESKTOP), tax))
    return tax
