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
from tda.core.truth import TruthService

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
    SCREW: (48, 6, 62, 20),
    PLUG: (34, 8, 40, 14),   # 6 px: `visible_tiny`, so V8 has a point case
    RAM: (8, 6, 20, 28),
    LATCH: (6, 30, 20, 44),
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


#: The PSU's loom, released at step 4 when ``cable=True``. It is a virtual node
#: (spec 3.1): no instance row, no mask, its own small state machine -- and a
#: `blocked_by` edge may name it, which is the case V16 has to catch.
CABLE = "cable:psu"


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


def _keyframes(view: str, tiny_screw: bool = False) -> list[ShapeKeyframe]:
    """One in-chassis shape per instance, plus a bench box for what comes out."""
    rects = dict(RECTS)
    if tiny_screw:
        rects[TINY_SCREW] = TINY_RECT  # 3 px: `too_small`, so it cannot be pointed at
    out = [
        ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=view, pose_segment=1,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", rle=masks.encode_rle(rect(box)))],
            amodal_complete=True,
        )
        for key, box in rects.items()
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


#: A second PSU-role screw nobody drew large enough to point at. It fastens
#: nothing, so it adds no constraint edge and the teardown is unchanged; what it
#: adds is a screw the count has to include and the frame cannot observe.
TINY_SCREW = "screw.psu.02"
TINY_RECT = (60, 60, 63, 63)


def build(db: Db, *, views=VIEWS, verified_steps=STEPS, skip_actions=(),
          tiny_screw=False, cable=False, edges=()) -> Taxonomy:
    """Seed ``db`` with the whole scene and return the taxonomy it was built on.

    ``skip_actions`` drops the action of those steps from the log without
    dropping the step, which is what a real log gap looks like: with ``(2,)``
    the PSU comes out at step 5 with its screw still fastened, and the graph
    says that was impossible -- one of the 18 lines of
    ``reports/constraints_report.md``, reproduced in miniature.
    """
    tax = load_taxonomy()
    db.upsert_desktop(DESKTOP, {"brand": "HP", "model_family": "EliteDesk 800 G2 TWR",
                                "chassis_type": "twr"})
    for rec in _instances():
        db.upsert_instance(rec)
    if tiny_screw:
        db.upsert_instance(InstanceRec(
            key=TINY_SCREW, desktop=DESKTOP, cls="screw",
            attrs={"role": "psu", "head": "PH2", "captive": False},
        ))
    steps, actions = _steps(), [a for a in _actions()
                                if a.step not in set(skip_actions)]
    if cable:
        # step 4 becomes compound: the plug comes out, then the loom is freed
        for rec in steps:
            if rec.step == 4:
                rec.step_type = "compound"
        actions.append(ActionRec(DESKTOP, 4, 1, CABLE, "release", tool="hand"))
    db.replace_steps(DESKTOP, steps, actions)
    for view in views:
        for step in STEPS:
            db.upsert_frame(
                FrameKey(DESKTOP, step, view), f"F:/{view}/007/{step:03d}/P_0.png",
                {"hw": [HW[0], HW[1]]}, "2025-06-01T09:00:00",
                flags={"review_status": ("verified" if step in verified_steps
                                         else "unlabeled")},
            )
        for kf in _keyframes(view, tiny_screw):
            db.add_keyframe(kf)
        # bottom-up: the chassis is behind everything, the small parts in front
        order = [(CHASSIS, "main"), (BOARD, "main"), (PSU, "main"), (PLUG, "main"),
                 (RAM, "main"), (LATCH, "main"), (SCREW, "main")]
        if tiny_screw:
            order.append((TINY_SCREW, "main"))
        db.set_zorder(ZOrderRec(DESKTOP, view, 1, order))
        db.set_pose_segment(DESKTOP, view, 1, 1, LAST_STEP, 1, None, None)
        db.set_pose_segment_bench_roi(DESKTOP, view, 1, BENCH_ROI)
    for (view, step, instance), visibility in FORCED.items():
        if view in views:
            db.set_frame_override(
                FrameOverride(FrameKey(DESKTOP, step, view), instance, None, visibility)
            )
    edges_to_db(db, DESKTOP, propose_edges(db.instances(DESKTOP), tax))
    for spec in edges:
        # `{"type", "target", "blocker", "mode"?, "status"?, "source"?, "necessity"?}`
        db.add_relation(DESKTOP, spec["type"], spec["target"], spec["blocker"],
                        necessity=spec.get("necessity", "required"),
                        mode=spec.get("mode"), source=spec.get("source", "manual"),
                        status=spec.get("status", "accepted"))
    freeze(db, tax, views=views, steps=verified_steps)
    return tax


# --------------------------------------------------------------------------- #
# a corpus shaped like the real database
# --------------------------------------------------------------------------- #
#: How many of each operated class a wide desktop carries. The small scene above
#: has one screw and one connector, which is fine for "is this rule obeyed?" and
#: useless for "can the wording answer the question?": with one instance of a
#: class there is never a second one in another state to ask about. These are
#: the proportions of a real teardown -- six board screws, four plugs, three
#: sticks of RAM and their clips.
WIDE = {"screw": 6, "connector": 4, "ram": 3}
WIDE_DESKTOPS = (21, 22, 23, 24)


def _wide_log() -> tuple[list[StepRec], list[ActionRec], int]:
    """An interleaved teardown, so each verb comes round again and again.

    Interleaved on purpose: the *second* time a verb is used, an instance it was
    already used on is in the other state, which is what a matched negative
    needs. A log that did all six screws first would give V4 nothing to pair its
    first five positives with.
    """
    order: list[tuple[str, str]] = []
    for i in range(1, WIDE["ram"] + 1):
        order.append(("unscrew", f"screw.motherboard.{i:02d}"))
        order.append(("disconnect", f"connector.{i:02d}"))
        order.append(("open", f"ram_latch.{i:02d}"))
        order.append(("remove", f"ram_module.{i:02d}"))
    order.append(("unscrew", "screw.motherboard.04"))
    order.append(("disconnect", "connector.04"))
    order.append(("unscrew", "screw.motherboard.05"))
    order.append(("unscrew", "screw.motherboard.06"))
    order.append(("remove", "motherboard.01"))
    steps = [StepRec(0, 1, "initial", "initial state")]
    actions: list[ActionRec] = []
    for k, (verb, target) in enumerate(order, start=2):
        steps.append(StepRec(0, k, "normal", f"{verb} {target}"))
        actions.append(ActionRec(0, k, 0, target, verb, tool="hand"))
    return steps, actions, len(order) + 1


def build_wide(db: Db, desktops=WIDE_DESKTOPS) -> Taxonomy:
    """Several desktops with enough parts for a shortcut to be measurable.

    Planning only: no frame is confirmed, which is the state the real database
    is in, so what comes out is V3/V4/V5/V6/V10/V16 -- exactly the tasks the
    text-only shortcut is measured on.
    """
    tax = load_taxonomy()
    steps, actions, n_steps = _wide_log()
    for desktop in desktops:
        db.upsert_desktop(desktop, {"brand": "HP", "model_family": "Wide",
                                    "chassis_type": "sff"})
        db.upsert_instance(InstanceRec(key=CHASSIS, desktop=desktop, cls="chassis"))
        db.upsert_instance(InstanceRec(key=BOARD, desktop=desktop, cls="motherboard"))
        for i in range(1, WIDE["screw"] + 1):
            db.upsert_instance(InstanceRec(
                key=f"screw.motherboard.{i:02d}", desktop=desktop, cls="screw",
                attrs={"role": "motherboard", "head": "PH2", "captive": False},
                fastens=BOARD))
        for i in range(1, WIDE["connector"] + 1):
            db.upsert_instance(InstanceRec(
                key=f"connector.{i:02d}", desktop=desktop, cls="connector",
                attrs={"kind": "front_panel"}, socket_host=BOARD))
        for i in range(1, WIDE["ram"] + 1):
            db.upsert_instance(InstanceRec(key=f"ram_module.{i:02d}",
                                           desktop=desktop, cls="ram_module",
                                           mounted_on=BOARD))
            db.upsert_instance(InstanceRec(
                key=f"ram_latch.{i:02d}", desktop=desktop, cls="ram_latch",
                attrs={"of": f"ram_module.{i:02d}"}, parent=BOARD, attached=True))
        db.replace_steps(
            desktop,
            [StepRec(desktop, s.step, s.step_type, s.raw_name) for s in steps],
            [ActionRec(desktop, a.step, a.idx, a.target, a.verb, tool=a.tool)
             for a in actions],
        )
        for step in range(1, n_steps + 1):
            db.upsert_frame(FrameKey(desktop, step, VIEW),
                            f"F:/{VIEW}/{desktop:03d}/{step:03d}/P_0.png",
                            {"hw": [HW[0], HW[1]]}, "2025-06-01T09:00:00",
                            flags={"review_status": "unlabeled"})
        db.set_pose_segment(desktop, VIEW, 1, 1, n_steps, 1, None, None)
        edges_to_db(db, desktop, propose_edges(db.instances(desktop), tax))
    return tax


def set_status(db: Db, rel_type: str, target: str, blocker: str, status: str) -> None:
    """Decide an existing edge the way the Relations tab does (B5)."""
    with db.conn:
        db.conn.execute(
            'UPDATE relation SET status=? WHERE desktop=? AND "type"=? '
            "AND target=? AND blocker=?",
            (status, DESKTOP, rel_type, target, blocker),
        )


def freeze(db: Db, tax: Taxonomy, *, views=VIEWS, steps=STEPS) -> None:
    """Compile and confirm the wanted frames, the way pressing Space does.

    Setting ``review_status`` by hand is not enough and must not be: the truth
    service demotes a confirmed frame the moment its instance set changes, so a
    frame "verified" with nothing compiled under it comes back as
    ``needs_review``. The perception tasks then correctly emit nothing, which is
    a fine rule and a useless fixture.
    """
    service = TruthService(db, tax)
    for view in views:
        service.refresh_range(DESKTOP, view, STEPS)
        for step in steps:
            service.verify_frame(FrameKey(DESKTOP, step, view), "tester")
