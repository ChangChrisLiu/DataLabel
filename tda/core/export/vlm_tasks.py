"""One generator per VLM task of spec 8.2, P0 set (task B6).

``V1 V2 V3 V4 V5 V6 V8 V10 V12 V14 V15 V16`` -- twelve questions, each with a
structured answer and an ``answer_check`` block saying how a checker re-derives
it. **A question whose answer cannot be checked by program is not emitted.**

Two sources of truth, and they are not interchangeable:

* **perception** (V1, V2, V8, V12, V14, V15) reads the compiled truth table and
  is emitted **only from VERIFIED frames**, only about instances that have a
  compiled row in *this* view. Nothing here is inferred: what the frame does not
  carry, the frame is not asked about.
* **planning / history** (V3, V4, V5, V6, V10, V16) reads the step log, the
  state machine and the constraint graph, so it exists for every desktop the
  moment the step table is imported -- which is all 66 of them, today, with no
  pixels annotated at all.

Legality is the graph's, never the log's order (spec 8.2 principle 6): V5 is the
full legal-action set at the state, V6's answer is checked as *membership* of
that set with the logged action as the reference, and V16 asks whether a
candidate sequence is legal. Where the two disagree -- a logged action the graph
says was impossible, the 18 violations of ``reports/constraints_report.md`` --
nothing is emitted: a ground truth that contradicts itself is worse than a gap,
and :func:`illegal_steps` hands the list to the caller to report.

Everything is pure and Qt-free; :mod:`tda.core.export.vlm` owns the loop, the
file and the refusals.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Sequence

from tda.core.export import vlm_reasoning as R
from tda.core.export.coco import (
    ANSWERABLE,
    VERIFIED,
    DesktopCtx,
    row_bbox_xywh,
)
from tda.core.graph import (
    Edge,
    GATES,
    REQUIRED_STATES,
    applicable_preconditions,
    legal_actions,
    unmet,
)
from tda.core.graph_rules import CABLE_PREFIX, active_edges, cable_nodes, verb_applies
from tda.core.implied import is_implied
from tda.core.model import (
    NO_CHANGE_STEP_TYPES,
    ActionRec,
    FrameKey,
    InstanceRec,
    StepType,
)
from tda.core.states import FrameState, events_from_actions, state_at
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CHECK_TYPES",
    "GENERATORS",
    "LAYERS",
    "PERCEPTION_TASKS",
    "PLANNING_TASKS",
    "TASKS",
    "FrameData",
    "TaskCtx",
    "class_label",
    "illegal_steps",
    "instance_label",
    "record",
]

#: The P0 task set of spec 8.2, in the order records are written per frame.
TASKS = ("V1", "V2", "V3", "V4", "V5", "V6", "V8", "V10", "V12", "V14", "V15", "V16")

#: Tasks whose answer is read off the compiled truth table of one view.
PERCEPTION_TASKS = frozenset({"V1", "V2", "V8", "V12", "V14", "V15"})
#: Tasks whose answer is the step log, the state machine and the graph.
PLANNING_TASKS = frozenset({"V3", "V4", "V5", "V6", "V10", "V16"})

#: Spec 8.2's capability layers. V3/V12 are single- and two-frame perception, so
#: L1; V10 is history and progress, which the spec files under planning, so L3.
LAYERS = {
    "V1": "L1", "V2": "L1", "V3": "L1", "V12": "L1", "V8": "L1",
    "V4": "L2", "V5": "L2",
    "V6": "L3", "V10": "L3", "V16": "L3",
    "V14": "L4", "V15": "L4",
}

#: The closed set of ``answer_check.type`` values. A record carrying anything
#: else is a record no checker can execute, and :func:`record` refuses it.
CHECK_TYPES = frozenset({
    "boxes",        # a set of instances, each localised: set F1 + box IoU
    "exact",        # named fields must match exactly
    "feasibility",  # V4: a yes/no plus the blocker set
    "set",          # V5: the whole legal-action set, scored as a set
    "member_of",    # V6: any member of the legal set is right; one is the log's
    "history",      # V10: the done set, the remaining count and the bin
    "plan",         # V16: validity, the first bad index and the violated edge
})

#: V10's progress bins, quarters of the desktop's own action count.
PROGRESS_BINS = ("0-25", "25-50", "50-75", "75-100")

CABLE_CLASS = "cable"
GEOM_BOX = "box"
STATE_ATTR = "state"


# --------------------------------------------------------------------------- #
# wording
# --------------------------------------------------------------------------- #
#: Class-id and attribute words that are initialisms rather than words. A
#: question reading "what state is the psu 1 in?" is a question about a typo;
#: ``str.title()`` would make it "Psu", which is worse.
INITIALISMS = frozenset({
    "psu", "cpu", "gpu", "ram", "ssd", "hdd", "io", "atx", "sata", "usb",
    "wlan", "pc", "pcb", "24pin", "ph1", "ph2", "ph3",
})


def _words(text: str) -> list[str]:
    """``"cpu_cooler"`` -> ``["CPU", "cooler"]``: initialisms up, rest as written."""
    return [
        word.upper() if word.lower() in INITIALISMS else word
        for word in str(text).replace("_", " ").split()
    ]


def class_label(cls: str, tax: Optional[Taxonomy] = None) -> str:
    """How a person says a taxonomy class: ``ram_module`` -> "RAM module"."""
    defn = (tax.classes.get(cls) if tax is not None else None) or {}
    given = defn.get("label") or defn.get("name")
    return " ".join(_words(given if isinstance(given, str) and given else cls))


def instance_label(instance: str, rec: Optional[InstanceRec],
                   tax: Optional[Taxonomy] = None) -> str:
    """Human wording of an instance key: ``screw.motherboard.03`` -> "Motherboard screw 3"."""
    if rec is None:
        return instance
    disc = rec.attrs.get("role") or rec.attrs.get("kind") or ""
    words = _words(disc) if disc else []
    words.extend(_words(class_label(str(rec.cls), tax)))
    ordinal = instance.rsplit(".", 1)[-1]
    if ordinal.isdigit():
        words.append(str(int(ordinal)))
    words = [w for w in words if w]
    if words and not words[0].isupper():
        words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words)


def _verb_phrase(verb: str, label: str) -> str:
    """"unscrew" + "Motherboard screw 3" -> "unscrew Motherboard screw 3"."""
    return f"{verb} {label}"


def pick(templates: Sequence[str], seed: str) -> tuple[int, str]:
    """Choose one wording deterministically: same record id, same question."""
    index = zlib.crc32(seed.encode("utf-8")) % len(templates)
    return index, templates[index]


def stable_order(items: Sequence[Any], seed: str,
                 key: Callable[[Any], str] = str) -> list[Any]:
    """A deterministic pseudo-random permutation of ``items``.

    Seeded by the caller's ``seed`` (desktop, step and task), so the hard
    negatives of one frame are the same in every export of the same database and
    are not the same ones on the next frame -- which is what keeps a sampled
    task from always asking about the first screw in the table.
    """
    return sorted(items, key=lambda item: (zlib.crc32(f"{seed}|{key(item)}"
                                                      .encode("utf-8")), key(item)))


# --------------------------------------------------------------------------- #
# per-frame and per-desktop context
# --------------------------------------------------------------------------- #
@dataclass
class FrameData:
    """What one frame of one view offers a task generator.

    ``rows`` are the compiled rows with their masks dropped: every task here
    needs a box, a visibility, a placement and a status, and keeping several
    hundred 12 MP RLEs alive for the length of a desktop is the one thing that
    would make this export expensive.
    """

    step: int
    image: str
    verified: bool
    rows: dict[str, dict] = field(default_factory=dict)
    pointable: dict[str, tuple[dict, list]] = field(default_factory=dict)


def slim_row(row: dict) -> dict:
    """The part of a compiled row a question can be asked about."""
    return {
        "visibility": row.get("visibility"),
        "placement": row.get("placement"),
        "occlusion_ratio": row.get("occlusion_ratio"),
        "status": row.get("status"),
        "geom_type": row.get("geom_type"),
        "box": row_bbox_xywh(row),
    }


def pointable_rows(ctx: DesktopCtx, rows: dict[str, dict],
                   only_verified: bool) -> dict[str, tuple[dict, list]]:
    """``instance -> (row, bbox)`` for the rows this view can be asked about.

    Both geometries qualify -- a visible mask and a bench box -- as long as the
    row's visibility is answerable and its instance carries a taxonomy class. An
    instance whose visibility is ``occluded_full`` or ``out_of_view`` is exactly
    the one V14 must abstain on, so it is deliberately not here.
    """
    out: dict[str, tuple[dict, list]] = {}
    for instance in sorted(rows):
        row = rows[instance]
        if row.get("visibility") not in ANSWERABLE:
            continue
        if only_verified and row.get("status") != VERIFIED:
            continue
        if ctx.cls_of(instance) is None:
            continue
        box = row.get("box") if "box" in row else row_bbox_xywh(row)
        if box is not None:
            out[instance] = (row, list(box))
    return out


@dataclass
class TaskCtx:
    """One desktop in one view, with everything the twelve generators read."""

    ctx: DesktopCtx
    view: str
    tier: Optional[str]
    graph_version: Optional[str]
    meta: dict
    edges: list[Edge]
    frames: dict[int, FrameData]
    steps: list[int]
    #: view -> {step -> FrameData}; only filled when V15 is wanted.
    others: dict[str, dict[int, FrameData]] = field(default_factory=dict)
    #: step -> the edges a logged action of that step broke (spec 7.4).
    illegal: dict[int, list[str]] = field(default_factory=dict)
    #: how many hard negatives / positives one frame may contribute per task.
    budget: int = 2

    # -- shorthand ---------------------------------------------------------- #
    @property
    def tax(self) -> Taxonomy:
        return self.ctx.tax

    @property
    def desktop(self) -> int:
        return self.ctx.desktop

    def frame(self, step: int) -> Optional[FrameData]:
        return self.frames.get(step)

    def state(self, step: int) -> FrameState:
        return self.ctx.state_at(step)

    def label(self, instance: str) -> str:
        return instance_label(instance, self.ctx.instances.get(instance), self.tax)

    def key(self, step: int) -> FrameKey:
        return FrameKey(self.desktop, step, self.view)

    def seed(self, step: int, task: str) -> str:
        return f"D{self.desktop}|s{step}|{self.view}|{task}"

    def next_action_step(self, step: int) -> Optional[int]:
        """The next step after ``step`` that carries a named successful action."""
        for candidate in self.steps:
            if candidate > step and self.named_actions(candidate):
                return candidate
        return None

    def named_actions(self, step: int) -> list[ActionRec]:
        """Successful actions of ``step`` whose target this export can name."""
        return [a for a in self.ctx.actions_at(step)
                if target_class(self.ctx, a.target) is not None]

    def failed_actions(self, step: int) -> list[ActionRec]:
        return sorted(
            (a for a in self.ctx.actions
             if a.step == step and a.result != "success"
             and target_class(self.ctx, a.target) is not None),
            key=lambda a: a.idx,
        )

    def legal_here(self, step: int) -> list[tuple[str, str]]:
        """The required-only legal-action set at the state after ``step``."""
        return legal_actions(self.ctx.instances, self.edges, self.state(step),
                             self.tax, strict=False)

    def legal_strict(self, step: int) -> list[tuple[str, str]]:
        return legal_actions(self.ctx.instances, self.edges, self.state(step),
                             self.tax, strict=True)


def target_class(ctx: DesktopCtx, target: str) -> Optional[str]:
    """Class of an action target: ``cable`` for a virtual node, else the taxonomy's."""
    if target.startswith(CABLE_PREFIX):
        return CABLE_CLASS
    return ctx.cls_of(target)


# --------------------------------------------------------------------------- #
# the graph, asked the three ways the tasks need
# --------------------------------------------------------------------------- #
def blocked_actions(tc: TaskCtx, step: int) -> list[tuple[str, str, list[Edge]]]:
    """``(verb, target, unmet edges)`` for everything the graph forbids right now.

    The complement of :func:`~tda.core.graph.legal_actions` inside the same
    candidate set: the verb applies to the class in this state (spec 6.3), so
    the question "can you do this?" is a sensible one, and at least one hard
    precondition is not satisfied, so the answer is no and the graph says why.
    """
    state = tc.state(step)
    active = active_edges(tc.edges)
    out: list[tuple[str, str, list[Edge]]] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        inst = state.get(key)
        if inst is None or inst.state == "removed":
            continue
        for verb in tc.tax.verbs:
            if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            bad = unmet(applicable_preconditions(active, (verb, key)), state, "required")
            if bad:
                out.append((verb, key, bad))
    for key, current in sorted(cable_nodes(active, state).items()):
        if not verb_applies(tc.tax, CABLE_CLASS, {}, "release", current):
            continue
        bad = unmet(applicable_preconditions(active, ("release", key)), state, "required")
        if bad:
            out.append(("release", key, bad))
    return out


def unmet_for(tc: TaskCtx, verb: str, target: str, state: FrameState) -> list[Edge]:
    """Which required preconditions of ``(verb, target)`` are not satisfied."""
    return unmet(applicable_preconditions(active_edges(tc.edges), (verb, target)),
                 state, "required")


def illegal_steps(instances: dict[str, InstanceRec], edges: list[Edge],
                  actions: list[ActionRec], tax: Taxonomy) -> dict[int, list[str]]:
    """``step -> the edges its logged action broke`` (spec 7.4, first check).

    The same replay ``constraints --validate`` prints, kept as data rather than
    as prose so the export can act on it: a step listed here is one where the
    log and the graph contradict each other, and V5/V6/V16 refuse to speak about
    the moment before it. On the real database this is the 15 "likely log gap"
    lines of ``reports/constraints_report.md``; the three failed attempts the
    graph cannot explain are *not* here -- they break no constraint, they only
    fail to be explained by one, and V4 drops them instead (see :func:`gen_v4`).
    """
    ordered = sorted(actions, key=lambda a: (a.step, a.idx))
    active = active_edges(edges)
    out: dict[int, list[str]] = {}
    for i, action in enumerate(ordered):
        if action.result != "success":
            continue
        state = state_at(instances, events_from_actions(instances, ordered[:i], tax),
                         10**9, tax)
        bad = unmet(applicable_preconditions(active, action), state, "required")
        if bad:
            out.setdefault(action.step, []).extend(e.label() for e in bad)
    return out


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def evidence(bboxes: dict[str, list], tc: Optional[TaskCtx] = None,
             attrs: Optional[dict] = None, **extra) -> dict:
    """The instance keys, boxes and per-instance attributes an answer rests on.

    ``attributes.implied`` mirrors the COCO export's field (spec 8.1): the
    instance was created by :mod:`tda.core.implied` because the desktop plainly
    has one and its log never touched it, not because a step named it. A
    consumer that must not train on inferred identities can drop those rows in
    either format with the same test.
    """
    names = sorted(set(bboxes) | set(attrs or {}))
    table: dict[str, dict] = {}
    for name in names:
        entry = dict((attrs or {}).get(name) or {})
        if tc is not None:
            rec = tc.ctx.instances.get(name)
            entry.setdefault("implied", bool(rec is not None and is_implied(rec)))
        table[name] = entry
    out = {"instances": names, "bboxes": bboxes}
    if table:
        out["attributes"] = table
    out.update(extra)
    return out


def record(tc: TaskCtx, task: str, rec_id: str, step: int, images: Sequence[str],
           question: str, template_index: int, answer: dict, check: dict,
           evid: dict, rationale: dict, verified: bool, *,
           views: Optional[Sequence[str]] = None,
           negative: Optional[str] = None) -> dict:
    """One JSONL record, with the fields every task shares.

    ``answer_check`` is not decoration: it is the promise that an independent
    checker can re-derive this answer, and a type outside :data:`CHECK_TYPES`
    raises here rather than shipping a question nobody can grade.

    ``model_family`` travels with every record so a platform-disjoint split can
    be made later (spec 8.1). Nothing else in a record may depend on a split:
    there is none yet, and inventing one now would bake it into the corpus.
    """
    if check.get("type") not in CHECK_TYPES:
        raise ValueError(f"{rec_id}: answer_check type {check.get('type')!r} "
                         f"is not one a checker can execute")
    out = {
        "id": rec_id,
        "task": task,
        "layer": LAYERS[task],
        "template_id": f"{task}.{template_index + 1}",
        "desktop": tc.desktop,
        "model_family": tc.meta.get("model_family"),
        "chassis_type": tc.meta.get("chassis_type"),
        "step": int(step),
        "view": tc.view,
        "images": list(images),
        "question": question,
        "answer": answer,
        "answer_check": check,
        "evidence": evid,
        "rationale": rationale,
        "tier": tc.tier,
        "verified": bool(verified),
        "graph_version": tc.graph_version,
    }
    if views:
        out["views"] = list(views)
    if negative:
        out["negative"] = negative
    return out


def frame_id(desktop: int, view: str, step: int) -> str:
    return f"D{desktop:02d}-{view}-s{step:03d}"


def row_verified(rows: Sequence[dict], frame_verified: bool) -> bool:
    """Has a human confirmed the frame *and* every row this answer was read off?"""
    return bool(frame_verified) and all(r.get("status") == VERIFIED for r in rows)


def observed(instance: str, value: Any, tc: TaskCtx,
             seen: dict[str, tuple[dict, list]], step: Optional[int] = None) -> dict:
    """``observe`` when the frame can point at the instance, else ``propagate_state``."""
    if instance not in seen:
        return R.propagate_state(instance, value)
    row, box = seen[instance]
    return R.observe(instance, value, tc.view, box, row.get("visibility"), step=step)


def state_is_askable(tax: Taxonomy, cls: str) -> bool:
    """Is "what state is it in?" a real question for this class?

    Only when the class has at least two states the annotation actually tracks
    (spec 6.2's ``needs_mask`` table). A ``connector`` has one -- ``plugged`` --
    because an unplugged connector is not tracked at all, so every frame that
    carries one answers "plugged" and the question teaches nothing; a
    ``chassis`` is always ``present``. Both are dropped rather than asked.
    """
    try:
        states = tax.states_of(cls)
    except KeyError:
        return False
    return sum(1 for s in states if tax.needs_mask(cls, s, "in_chassis")) >= 2


# --------------------------------------------------------------------------- #
# V1 -- what is visible, and where
# --------------------------------------------------------------------------- #
V1_QUESTIONS = (
    "List every component visible in this image with its bounding box.",
    "Which components can you see in this image? Give each one a bounding box.",
    "Name the parts visible in this photo of the desktop PC and localise them.",
    "Enumerate the visible components and their boxes.",
)


def gen_v1(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified or not frame.pointable:
        return
    components, bboxes, steps, used, attrs = [], {}, [], [], {}
    for instance, (row, box) in frame.pointable.items():
        cls = tc.ctx.cls_of(instance)
        if cls is None:
            continue
        components.append({"class": cls, "instance": instance, "bbox": box,
                           "placement": row.get("placement")})
        bboxes[instance] = box
        attrs[instance] = {"visibility": row.get("visibility")}
        used.append(row)
        steps.append(R.observe(instance, cls, tc.view, box, row.get("visibility")))
    if not components:
        return
    rec_id = f"V1-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V1_QUESTIONS, rec_id)
    yield record(
        tc, "V1", rec_id, step, [frame.image], question, index,
        {"components": components},
        {"type": "boxes", "field": "components", "id_field": "instance",
         "box_field": "bbox", "exhaustive": True, "metric": "set_f1+box_iou",
         "iou_threshold": 0.5},
        evidence(bboxes, tc, attrs), R.rationale(steps),
        row_verified(used, frame.verified),
    )


# --------------------------------------------------------------------------- #
# V2 -- per-instance state and fastener counts
# --------------------------------------------------------------------------- #
V2_STATE_QUESTIONS = (
    "What is the state of {label}?",
    "In this image, what state is {label} in?",
    "Report the current state of {label}.",
)
V2_COUNT_QUESTIONS = (
    "How many {role} screws are still fastened?",
    "Count the {role} screws that are still fastened in this image.",
)


def gen_v2(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    yield from _v2_states(tc, frame)
    yield from _v2_counts(tc, frame)


def _v2_states(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    state = tc.state(frame.step)
    for instance, (row, box) in frame.pointable.items():
        cls = tc.ctx.cls_of(instance)
        inst = state.get(instance)
        if cls is None or inst is None or not state_is_askable(tc.tax, cls):
            continue
        rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-state-{instance}"
        index, template = pick(V2_STATE_QUESTIONS, rec_id)
        yield record(
            tc, "V2", rec_id, frame.step, [frame.image],
            template.format(label=tc.label(instance)), index,
            {"state": inst.state},
            {"type": "exact", "derive": "instance_state", "instance": instance,
             "fields": ["state"], "allowed": sorted(tc.tax.states_of(cls))},
            evidence({instance: box}, tc, {instance: {"visibility": row.get("visibility")}}),
            R.rationale([R.observe(instance, inst.state, tc.view, box,
                                   row.get("visibility"))]),
            row_verified([row], frame.verified),
        )


def _v2_counts(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    """"How many <role> screws are still fastened?" -- counted on the state machine.

    The count covers every screw of that role on the machine, not only the ones
    this view can see, because that is what the question asks; the evidence
    lists the boxes of those that are visible here, and the ones that are not
    enter the chain as ``propagate_state``.
    """
    state = tc.state(frame.step)
    roles = sorted({
        str(rec.attrs.get("role") or "other")
        for key, rec in tc.ctx.instances.items()
        if rec.cls == "screw" and key in frame.pointable
    })
    for role in roles:
        fastened = [
            key for key, rec in sorted(tc.ctx.instances.items())
            if rec.cls == "screw" and str(rec.attrs.get("role") or "other") == role
            and (state.get(key).state if state.get(key) else None) == "fastened"
        ]
        bboxes = {k: frame.pointable[k][1] for k in fastened if k in frame.pointable}
        rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-count-{role}"
        index, template = pick(V2_COUNT_QUESTIONS, rec_id)
        yield record(
            tc, "V2", rec_id, frame.step, [frame.image],
            template.format(role=role.replace("_", " ")), index,
            {"count": len(fastened)},
            {"type": "exact", "derive": "screw_count", "role": role,
             "state": "fastened", "fields": ["count"]},
            evidence(bboxes, tc),
            R.rationale([observed(k, "fastened", tc, frame.pointable) for k in fastened]),
            row_verified([frame.pointable[k][0] for k in fastened
                          if k in frame.pointable], frame.verified),
        )


# --------------------------------------------------------------------------- #
# V3 -- what happened between two frames
# --------------------------------------------------------------------------- #
V3_QUESTIONS = (
    "What action was just performed between these two images?",
    "Compare the two images: which action was carried out?",
    "What did the operator just do between the first and the second image?",
)


def _slots(tc: TaskCtx, action: ActionRec) -> dict:
    return {"verb": action.verb, "target_class": target_class(tc.ctx, action.target),
            "target_instance": action.target, "tool": action.tool}


def gen_v3(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame, before = tc.frame(step), tc.frame(step - 1)
    if frame is None or before is None:
        return
    if tc.ctx.step_type(step) in NO_CHANGE_STEP_TYPES:
        return
    named = tc.named_actions(step)
    if not named:
        return
    answer = _slots(tc, named[0])
    if len(named) > 1:  # a compound step: keep every slot set, first one on top
        answer["actions"] = [_slots(tc, a) for a in named]

    target = named[0].target
    source_step, source = step - 1, before.pointable
    if target not in before.pointable and target in frame.pointable:
        source_step, source = step, frame.pointable
    bboxes = {target: source[target][1]} if target in source else {}

    rec_id = f"V3-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V3_QUESTIONS, rec_id)
    chain = [R.compare_frames([step - 1, step], tc.view),
             observed(target, answer["verb"], tc, source, step=source_step)]
    yield record(
        tc, "V3", rec_id, step, [before.image, frame.image], question, index, answer,
        {"type": "exact", "derive": "action_slots",
         "fields": ["verb", "target_class", "target_instance", "tool"]},
        evidence(bboxes, tc, from_step=source_step), R.rationale(chain),
        # the grade follows the frame the answer was actually read off, which
        # for a V3 pair is usually the earlier one: the part is gone from the later
        row_verified([source[target][0]] if target in source else [],
                     (before if source_step == step - 1 else frame).verified),
    )


# --------------------------------------------------------------------------- #
# V4 -- can this be done now, and what is in the way
# --------------------------------------------------------------------------- #
V4_QUESTIONS = (
    "Can you {phrase} right now? If not, what has to be dealt with first?",
    "Is it possible to {phrase} in this state? Name whatever blocks it.",
    "Right now, would {phrase} work? List the parts standing in the way.",
)
V4_FAILED_QUESTIONS = (
    "The operator tried to {phrase} here and could not. Why not?",
    "This attempt to {phrase} failed. What was in the way?",
)


def _v4_record(tc: TaskCtx, frame: FrameData, verb: str, target: str,
               bad: Sequence[Edge], templates: Sequence[str], tag: str) -> dict:
    state = tc.state(frame.step)
    blockers = sorted({e.blocker for e in bad})
    rec_id = f"V4-{frame_id(tc.desktop, tc.view, frame.step)}-{tag}-{verb}-{target}"
    index, template = pick(templates, rec_id)
    chain: list[dict] = [observed(target, state[target].state, tc, frame.pointable)]
    for edge in sorted(bad, key=lambda e: (e.type, e.blocker)):
        current = state[edge.blocker].state if edge.blocker in state else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          REQUIRED_STATES.get(edge.type, frozenset()),
                                          current, False))
    bboxes = {k: frame.pointable[k][1]
              for k in [target, *blockers] if k in frame.pointable}
    return record(
        tc, "V4", rec_id, frame.step, [frame.image],
        template.format(phrase=_verb_phrase(verb, tc.label(target))), index,
        {"feasible": not bad, "blockers": blockers},
        {"type": "feasibility", "verb": verb, "target": target,
         "state_after": frame.step,
         "edges": [{"type": e.type, "target": e.target, "blocker": e.blocker,
                    "necessity": e.necessity, "mode": e.mode,
                    "required": sorted(REQUIRED_STATES.get(e.type, frozenset()))}
                   for e in sorted(bad, key=lambda e: (e.type, e.blocker))]},
        evidence(bboxes, tc), R.rationale(chain),
        row_verified([frame.pointable[k][0] for k in bboxes], frame.verified),
        negative=None if not bad else tag,
    )


def gen_v4(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Balanced feasibility questions, plus the failed attempts the graph explains.

    A failed attempt (spec 6.3 ``result = failed``) is the best hard negative in
    the dataset -- a human really could not do it -- but only when the graph
    says why. The three attempts of ``reports/constraints_report.md`` that break
    no constraint are exactly the case where the dataset does not yet know the
    answer, so they are not emitted as explained negatives; the constraint
    editor (task B5) is where that gap is filled.
    """
    frame = tc.frame(step)
    if frame is None:
        return
    state = tc.state(step)
    seed = tc.seed(step, "V4")
    blocked = [(v, t, bad) for v, t, bad in blocked_actions(tc, step)
               if t in tc.ctx.instances]
    legal = [(v, t) for v, t in tc.legal_here(step) if t in tc.ctx.instances]

    for verb, target, bad in stable_order(blocked, seed, lambda it: f"{it[0]}|{it[1]}")[
            :tc.budget]:
        yield _v4_record(tc, frame, verb, target, bad, V4_QUESTIONS, "blocked")
    for verb, target in stable_order(legal, seed + "|pos", lambda it: f"{it[0]}|{it[1]}")[
            :tc.budget]:
        if target in state:
            yield _v4_record(tc, frame, verb, target, (), V4_QUESTIONS, "legal")

    # the attempt the operator actually made and could not finish
    nxt = step + 1
    for action in tc.failed_actions(nxt):
        if action.target not in tc.ctx.instances:
            continue
        if not verb_applies(tc.tax, tc.ctx.instances[action.target].cls,
                            tc.ctx.instances[action.target].attrs, action.verb,
                            state[action.target].state):
            continue
        bad = unmet_for(tc, action.verb, action.target, state)
        if not bad:
            continue  # the graph cannot explain it: not a ground truth yet
        yield _v4_record(tc, frame, action.verb, action.target, bad,
                         V4_FAILED_QUESTIONS, "failed_attempt")


# --------------------------------------------------------------------------- #
# V5 -- the whole legal-action set
# --------------------------------------------------------------------------- #
V5_QUESTIONS = (
    "Which disassembly actions are possible right now? List every one.",
    "Given this state, what can be done next? Give the complete set of legal actions.",
    "Enumerate every action that is physically possible on this machine now.",
)


def _actions_json(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    return [{"verb": verb, "target": target} for verb, target in sorted(pairs)]


def gen_v5(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None:
        return
    nxt = tc.next_action_step(step)
    if nxt is not None and nxt in tc.illegal:
        return  # the log and the graph disagree here: no ground truth to ship
    loose = tc.legal_here(step)
    if not loose:
        return
    logged = None
    if nxt is not None:
        action = tc.named_actions(nxt)[0]
        logged = [action.verb, action.target]
        if (action.verb, action.target) not in set(loose):
            return
    rec_id = f"V5-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V5_QUESTIONS, rec_id)
    state = tc.state(step)
    chain = [observed(target, state[target].state if target in state else None,
                      tc, frame.pointable)
             for _, target in sorted(loose)[:tc.budget]]
    yield record(
        tc, "V5", rec_id, step, [frame.image], question, index,
        {"actions": _actions_json(loose)},
        {"type": "set", "field": "actions", "metric": "set_f1",
         "necessity": "required", "logged_next": logged,
         "strict_actions": _actions_json(tc.legal_strict(step))},
        evidence({k: v[1] for k, v in frame.pointable.items()
                  if k in {t for _, t in loose}}, tc),
        R.rationale(chain), frame.verified,
    )


# --------------------------------------------------------------------------- #
# V6 -- the next step
# --------------------------------------------------------------------------- #
V6_QUESTIONS = (
    "What is the next disassembly step? Answer with one verb and one target.",
    "Looking at this frame, which action would you perform next?",
    "Name the single action to carry out next on this machine.",
)
#: How many legal-but-not-taken actions ride along as V6 distractors (spec 8.2's
#: hard-negative list), plus one blocked action so "any option is legal" is false.
V6_DISTRACTORS = 3


def gen_v6(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    nxt = tc.next_action_step(step)
    if frame is None or nxt is None or nxt in tc.illegal:
        return
    action = tc.named_actions(nxt)[0]
    legal = set(tc.legal_here(step))
    if (action.verb, action.target) not in legal:
        return  # an action the graph says was impossible is nobody's reference

    seed = tc.seed(step, "V6")
    others = stable_order(sorted(legal - {(action.verb, action.target)}), seed,
                          lambda it: f"{it[0]}|{it[1]}")[:V6_DISTRACTORS]
    options = [{"verb": action.verb, "target": action.target, "legal": True}]
    options += [{"verb": v, "target": t, "legal": True} for v, t in others]
    blocked = stable_order([(v, t) for v, t, _ in blocked_actions(tc, step)],
                           seed + "|neg", lambda it: f"{it[0]}|{it[1]}")[:1]
    options += [{"verb": v, "target": t, "legal": False} for v, t in blocked]
    options = stable_order(options, seed + "|order",
                           lambda o: f"{o['verb']}|{o['target']}")

    rec_id = f"V6-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V6_QUESTIONS, rec_id)
    state = tc.state(step)
    chain = [observed(action.target,
                      state[action.target].state if action.target in state else None,
                      tc, frame.pointable),
             R.check_precondition(action.target, ["legal"], "legal", True)]
    yield record(
        tc, "V6", rec_id, step, [frame.image], question, index,
        {"verb": action.verb, "target": action.target,
         "target_class": target_class(tc.ctx, action.target), "tool": action.tool},
        {"type": "member_of", "reference": [action.verb, action.target],
         "reference_step": nxt, "options": options, "metric": "legal_rate+match_rate",
         "legal_actions": _actions_json(legal)},
        evidence({action.target: frame.pointable[action.target][1]}
                 if action.target in frame.pointable else {}, tc),
        R.rationale(chain), frame.verified,
    )


# --------------------------------------------------------------------------- #
# V8 -- referring-expression grounding
# --------------------------------------------------------------------------- #
V8_QUESTIONS = (
    "Where is {phrase}? Give its bounding box.",
    "Localise {phrase} in this image.",
    "Point at {phrase}: answer with one bounding box.",
)
#: How many instances one frame is asked to ground.
V8_BUDGET = 3


def _referring(tc: TaskCtx, instance: str) -> tuple[str, str]:
    """``(phrase, family)``: a relational phrase when it is unambiguous, else the label.

    A relational expression is the interesting half of V8 -- it forces the model
    to use the assembly structure rather than a class name -- but only while it
    names exactly one part. ``the screw that fastens the PSU`` is a referring
    expression; with a second such screw it is a trap.
    """
    rec = tc.ctx.instances.get(instance)
    label = tc.label(instance)
    if rec is None:
        return label, "label"
    if rec.cls == "screw" and rec.fastens in tc.ctx.instances:
        peers = [k for k, r in tc.ctx.instances.items()
                 if r.cls == "screw" and r.fastens == rec.fastens]
        if len(peers) == 1:
            return f"the screw that fastens {tc.label(rec.fastens)}", "fastens"
    of = rec.attrs.get("of")
    if of in tc.ctx.instances:
        peers = [k for k, r in tc.ctx.instances.items()
                 if r.cls == rec.cls and r.attrs.get("of") == of]
        if len(peers) == 1:
            verb = "covers" if rec.cls == "cover" else "locks"
            return (f"the {class_label(rec.cls, tc.tax)} that {verb} "
                    f"{tc.label(of)}"), "of"
    return label, "label"


def gen_v8(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    seed = tc.seed(step, "V8")
    for instance in stable_order(sorted(frame.pointable), seed)[:V8_BUDGET]:
        row, box = frame.pointable[instance]
        phrase, family = _referring(tc, instance)
        rec_id = f"V8-{frame_id(tc.desktop, tc.view, step)}-{instance}"
        index, template = pick(V8_QUESTIONS, rec_id)
        tiny = row.get("visibility") == "visible_tiny"
        yield record(
            tc, "V8", rec_id, step, [frame.image],
            template.format(phrase=phrase), index,
            {"instance": instance, "bbox": box},
            {"type": "boxes", "field": None, "id_field": "instance",
             "box_field": "bbox", "iou_threshold": 0.5,
             "metric": "point_in_box" if tiny else "box_iou",
             "expression": family},
            evidence({instance: box}, tc,
                     {instance: {"visibility": row.get("visibility")}}),
            R.rationale([R.observe(instance, tc.ctx.cls_of(instance), tc.view, box,
                                   row.get("visibility"))]),
            row_verified([row], frame.verified),
        )


# --------------------------------------------------------------------------- #
# V10 -- what has already been done, and how far along this is
# --------------------------------------------------------------------------- #
V10_QUESTIONS = (
    "Which disassembly actions have already been carried out on this machine?",
    "From this single frame, list the actions that have happened so far and say "
    "how far the teardown has got.",
    "What has been done to this machine up to now, and what fraction of the "
    "teardown is finished?",
)


def gen_v10(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None:
        return
    done = [a for s in tc.steps if s <= step for a in tc.named_actions(s)]
    total = sum(len(tc.named_actions(s)) for s in tc.steps)
    if not total:
        return
    share = len(done) / total
    bin_name = PROGRESS_BINS[min(int(share * len(PROGRESS_BINS)), len(PROGRESS_BINS) - 1)]
    state = tc.state(step)
    rec_id = f"V10-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V10_QUESTIONS, rec_id)
    chain = [observed(a.target,
                      state[a.target].state if a.target in state else None,
                      tc, frame.pointable, step=a.step)
             for a in done[-tc.budget:]]
    yield record(
        tc, "V10", rec_id, step, [frame.image], question, index,
        {"done": [{"verb": a.verb, "target": a.target} for a in done],
         "remaining_actions": total - len(done), "progress_bin": bin_name},
        {"type": "history", "bins": list(PROGRESS_BINS), "total_actions": total,
         "metric": "set_f1+bin_accuracy"},
        evidence({a.target: frame.pointable[a.target][1] for a in done
                  if a.target in frame.pointable}, tc),
        R.rationale(chain), frame.verified,
    )


# --------------------------------------------------------------------------- #
# V12 -- did anything change between these two frames
# --------------------------------------------------------------------------- #
V12_QUESTIONS = (
    "Did anything change between these two images? If so, what?",
    "Compare the two frames: is the machine in a different state?",
    "Are these two photos of the same machine state, or did something change?",
)


def gen_v12(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Change / no-change over a consecutive pair, both frames confirmed.

    The "no change" half comes from ``dupli`` rows -- a genuine near-duplicate
    the operator shot twice -- which is spec 8.2's own answer to how to build
    that negative. A ``failed`` step is deliberately **not** used: no state
    changed, but a hand or a tool usually did, and a question whose answer is
    "nothing changed" about a picture where something visibly moved is a
    question about the annotation, not about the machine.
    """
    frame, before = tc.frame(step), tc.frame(step - 1)
    if frame is None or before is None:
        return
    if not (frame.verified and before.verified):
        return
    kind = tc.ctx.step_type(step)
    if kind == StepType.DUPLI.value:
        changed, events = False, []
    elif kind in NO_CHANGE_STEP_TYPES or kind == StepType.FAILED.value:
        return
    else:
        old, new = tc.state(step - 1), tc.state(step)
        events = [{"target": key, "old": old[key].state, "new": new[key].state}
                  for key in sorted(old.keys() & new.keys())
                  if old[key].state != new[key].state]
        if not events:
            return
        changed = True
    rec_id = f"V12-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V12_QUESTIONS, rec_id)
    touched = [e["target"] for e in events]
    yield record(
        tc, "V12", rec_id, step, [before.image, frame.image], question, index,
        {"changed": changed, "events": events},
        {"type": "exact", "derive": "changed", "fields": ["changed", "events"],
         "step_type": kind},
        evidence({k: before.pointable[k][1] for k in touched if k in before.pointable},
                 tc),
        R.rationale([R.compare_frames([step - 1, step], tc.view),
                     *(observed(k, "changed", tc, before.pointable, step=step - 1)
                       for k in touched[:tc.budget])]),
        frame.verified and before.verified,
        negative=None if changed else "dupli_no_change",
    )


# --------------------------------------------------------------------------- #
# V14 -- can this view decide at all
# --------------------------------------------------------------------------- #
V14_QUESTIONS = (
    "From this view alone, can you tell what state {label} is in? "
    "Answer with the state, or say you cannot tell.",
    "Is {label} judgeable in this image? If it is, give its state.",
    "Can this single view decide the state of {label}?",
)


def gen_v14(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The abstain task: an instance this view cannot see has no state to report.

    Spec 8.2 principle 5. The truth is the compiled ``visibility`` and nothing
    else: ``occluded_full`` and ``out_of_view`` -- and ``too_small`` /
    ``motion_blur``, which are the same admission in other words -- mean the
    honest answer is "not from here". Both halves are emitted from the same
    frame, balanced, so the task cannot be won by always abstaining.
    """
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    state = tc.state(step)
    seed = tc.seed(step, "V14")
    askable = [(key, row) for key, row in sorted(frame.rows.items())
               if tc.ctx.cls_of(key) is not None
               and state_is_askable(tc.tax, tc.ctx.cls_of(key))
               and key in state]
    blind = [(k, r) for k, r in askable if r.get("visibility") not in ANSWERABLE]
    seen = [(k, r) for k, r in askable if r.get("visibility") in ANSWERABLE]
    chosen = (stable_order(blind, seed, lambda it: it[0])[:tc.budget]
              + stable_order(seen, seed + "|pos", lambda it: it[0])[:tc.budget])
    for instance, row in chosen:
        answerable = row.get("visibility") in ANSWERABLE
        rec_id = f"V14-{frame_id(tc.desktop, tc.view, step)}-{instance}"
        index, template = pick(V14_QUESTIONS, rec_id)
        if answerable:
            box = frame.pointable[instance][1]
            chain = R.rationale([R.observe(instance, state[instance].state, tc.view,
                                           box, row.get("visibility"))])
            bboxes = {instance: box}
        else:
            chain = R.rationale(
                [R.propagate_state(instance, None)],
                R.abstain(f"{instance} is {row.get('visibility')} in view {tc.view}"),
            )
            bboxes = {}
        yield record(
            tc, "V14", rec_id, step, [frame.image],
            template.format(label=tc.label(instance)), index,
            {"answerable": answerable,
             "state": state[instance].state if answerable else None},
            {"type": "exact", "derive": "answerable", "instance": instance,
             "fields": ["answerable", "state"], "metric": "risk_coverage"},
            evidence(bboxes, tc, {instance: {"visibility": row.get("visibility")}}),
            chain, row_verified([row], frame.verified),
            negative=None if answerable else "unanswerable",
        )


# --------------------------------------------------------------------------- #
# V15 -- two views of the same moment
# --------------------------------------------------------------------------- #
V15_CROSS_QUESTIONS = (
    "{label} cannot be judged in the first image. Using the second view, what "
    "state is it in, and which view answers better?",
    "The first view cannot see {label}. Read its state off the second view and "
    "say which view is the right one to ask.",
)
V15_MOMENT_QUESTIONS = (
    "Are these two images the same machine at the same moment of the teardown?",
    "Do these two views show the same disassembly state of the same machine?",
)
#: How far away the negative half of the same-moment question is taken from.
V15_MOMENT_OFFSET = 2


def gen_v15(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Cross-view consistency, built from per-view ``visibility`` and nothing else.

    Spec v1.5 removed cross-view geometry from the design: there is no homography
    and no projection, so this task may not use one. What is left is exactly what
    the annotation does carry -- the *same instance*, at the *same logical step*,
    with a visibility of its own in each view -- and that is enough for both
    halves of the spec's V15: which view can answer, and whether two frames are
    the same moment at all.
    """
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    state = tc.state(step)
    for other, frames in sorted(tc.others.items()):
        mate = frames.get(step)
        if mate is None or not mate.verified:
            continue
        yield from _v15_cross(tc, step, frame, other, mate, state)
        yield from _v15_moment(tc, step, frame, other, frames)


def _v15_cross(tc: TaskCtx, step: int, frame: FrameData, other: str,
               mate: FrameData, state: FrameState) -> Iterator[dict]:
    seed = tc.seed(step, f"V15|{other}")
    blind = [key for key, row in sorted(frame.rows.items())
             if row.get("visibility") not in ANSWERABLE
             and key in mate.pointable and key in state
             and tc.ctx.cls_of(key) is not None
             and state_is_askable(tc.tax, tc.ctx.cls_of(key))]
    for instance in stable_order(blind, seed)[:tc.budget]:
        row, box = mate.pointable[instance]
        here = frame.rows[instance]
        rec_id = f"V15-{frame_id(tc.desktop, tc.view, step)}-{other}-{instance}"
        index, template = pick(V15_CROSS_QUESTIONS, rec_id)
        yield record(
            tc, "V15", rec_id, step, [frame.image, mate.image],
            template.format(label=tc.label(instance)), index,
            {"state": state[instance].state, "best_view": other},
            {"type": "exact", "derive": "cross_view", "instance": instance,
             "blind_view": tc.view, "answer_view": other,
             "fields": ["state", "best_view"]},
            evidence({instance: box}, tc,
                     {instance: {"visibility": here.get("visibility"),
                                 f"visibility_{other}": row.get("visibility")}}),
            R.rationale([
                R.propagate_state(instance, None),
                R.observe(instance, state[instance].state, other, box,
                          row.get("visibility"), step=step),
            ]),
            row_verified([row], frame.verified and mate.verified),
            views=[tc.view, other],
        )


def _v15_moment(tc: TaskCtx, step: int, frame: FrameData, other: str,
                frames: dict[int, FrameData]) -> Iterator[dict]:
    """Same moment or not -- the negative is the *same* view a few steps away."""
    shared = sorted(set(frame.rows) & set(frames[step].rows))
    if not shared:
        return
    seed = tc.seed(step, f"V15moment|{other}")
    candidates: list[tuple[bool, FrameData]] = [(True, frames[step])]
    for delta in (V15_MOMENT_OFFSET, -V15_MOMENT_OFFSET):
        mate = frames.get(step + delta)
        if mate is not None and mate.verified and mate.step != step:
            candidates.append((False, mate))
            break
    for same, mate in candidates:
        rec_id = (f"V15m-{frame_id(tc.desktop, tc.view, step)}-{other}"
                  f"-s{mate.step:03d}")
        index, question = pick(V15_MOMENT_QUESTIONS, rec_id)
        yield record(
            tc, "V15", rec_id, step, [frame.image, mate.image], question, index,
            {"same_moment": same},
            {"type": "exact", "derive": "same_moment", "same_moment": same,
             "fields": ["same_moment"], "other_view": other,
             "other_step": mate.step},
            evidence({}, tc, {k: {} for k in shared[:tc.budget]}),
            R.rationale([R.compare_frames([step, mate.step], other)]),
            frame.verified and mate.verified, views=[tc.view, other],
            negative=None if same else "different_moment",
        )
    if len(candidates) == 1:
        return


# --------------------------------------------------------------------------- #
# V16 -- is this plan legal, and where does it first go wrong
# --------------------------------------------------------------------------- #
V16_QUESTIONS = (
    "Here is a proposed continuation of the teardown. Is it legal? If not, "
    "which step is the first that cannot be carried out?",
    "Check this plan against the machine's current state: is every step "
    "possible in this order, and where does it first break?",
    "Verify the following sequence. Say whether it is valid and, if not, the "
    "index of the first impossible action.",
)
#: How many logged actions a candidate plan is made of.
V16_PLAN_LEN = 4


def _simulate(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]]
              ) -> tuple[Optional[int], Optional[Edge]]:
    """Replay ``plan`` from the state after ``step``; ``(first bad index, edge)``."""
    sim: dict[str, str] = {k: v.state for k, v in tc.state(step).items()}
    instances = tc.ctx.instances
    children = {key: [k for k, r in instances.items()
                      if r.attached and r.parent == key] for key in instances}
    active = active_edges(tc.edges)
    for index, (verb, target) in enumerate(plan):
        rec = instances.get(target)
        if rec is None:
            return index, None
        bad = [e for e in active
               if e.target == target and verb in GATES.get(e.type, frozenset())
               and e.necessity == "required"
               and (sim.get(e.blocker) is not None and sim[e.blocker] != "removed")
               and sim[e.blocker] not in REQUIRED_STATES.get(e.type, frozenset())]
        if bad:
            return index, sorted(bad, key=lambda e: (e.type, e.blocker))[0]
        if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, sim.get(target, "")):
            return index, None
        effect = tc.tax.apply_verb(rec.cls, rec.attrs, verb)
        if effect is None or effect[0] != STATE_ATTR:
            continue
        sim[target] = effect[1]
        if effect[1] == "removed" and rec.cls != "connector":
            for child in children.get(target, ()):
                sim[child] = "removed"
    return None, None


def _plan_record(tc: TaskCtx, frame: FrameData, plan: Sequence[tuple[str, str]],
                 first_bad: Optional[int], edge: Optional[Edge], tag: str) -> dict:
    step = frame.step
    rec_id = f"V16-{frame_id(tc.desktop, tc.view, step)}-{tag}"
    index, question = pick(V16_QUESTIONS, rec_id)
    listed = [{"verb": v, "target": t} for v, t in plan]
    chain: list[dict] = []
    if edge is not None:
        current = tc.state(step)[edge.blocker].state if edge.blocker in tc.state(step) \
            else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          REQUIRED_STATES.get(edge.type, frozenset()),
                                          current, False))
    else:
        state = tc.state(step)
        chain.extend(observed(t, state[t].state if t in state else None,
                              tc, frame.pointable) for _, t in plan[:tc.budget])
    return record(
        tc, "V16", rec_id, step, [frame.image],
        question + "\n" + "\n".join(f"{i + 1}. {v} {tc.label(t)}"
                                    for i, (v, t) in enumerate(plan)),
        index,
        {"valid": first_bad is None, "first_error_index": first_bad,
         "violated_edge": None if edge is None else edge.label()},
        {"type": "plan", "plan": listed, "metric": "accuracy+localisation"},
        evidence({t: frame.pointable[t][1] for _, t in plan if t in frame.pointable},
                 tc, plan_length=len(listed)),
        R.rationale(chain), frame.verified,
        negative=None if first_bad is None else tag,
    )


def gen_v16(tc: TaskCtx, step: int) -> Iterator[dict]:
    """A true continuation and, where one exists, a corruption the graph catches.

    The negative is built by swapping two adjacent actions of the *real* plan,
    which is the corruption the spec asks for and the only one that is certain
    to be wrong for a reason the data can name. A swap that violates nothing --
    two screws of the same group, say -- is **not** a negative: it is a
    different, equally legal plan, and calling it an error would teach exactly
    the wrong lesson (spec 8.2 principle 6, legality not uniqueness).
    """
    frame = tc.frame(step)
    if frame is None:
        return
    plan: list[tuple[str, str]] = []
    for candidate in tc.steps:
        if candidate <= step:
            continue
        if candidate in tc.illegal:
            return  # a suffix built on a contradiction is not a plan
        for action in tc.named_actions(candidate):
            if action.target in tc.ctx.instances:
                plan.append((action.verb, action.target))
        if len(plan) >= V16_PLAN_LEN:
            break
    plan = plan[:V16_PLAN_LEN]
    if len(plan) < 2:
        return
    first_bad, edge = _simulate(tc, step, plan)
    if first_bad is not None:
        return  # the logged suffix does not replay cleanly: say nothing
    yield _plan_record(tc, frame, plan, None, None, "true")

    seed = tc.seed(step, "V16")
    for i in stable_order(list(range(len(plan) - 1)), seed, str):
        swapped = list(plan)
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        bad_at, broken = _simulate(tc, step, swapped)
        if bad_at is None or broken is None:
            continue  # a swap that violates nothing is a different legal plan
        yield _plan_record(tc, frame, swapped, bad_at, broken, "swapped")
        return


#: ``task id -> generator``. The order is the order records are written.
GENERATORS: dict[str, Callable[[TaskCtx, int], Iterator[dict]]] = {
    "V1": gen_v1, "V2": gen_v2, "V3": gen_v3, "V4": gen_v4, "V5": gen_v5,
    "V6": gen_v6, "V8": gen_v8, "V10": gen_v10, "V12": gen_v12, "V14": gen_v14,
    "V15": gen_v15, "V16": gen_v16,
}
