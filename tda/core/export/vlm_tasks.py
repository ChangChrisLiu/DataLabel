"""One generator per VLM task of spec 8.2, P0 set (task B6).

``V1 V2 V3 V4 V5 V6 V8 V10 V12 V14 V15 V16`` -- twelve questions, each with a
structured answer and an ``answer_check`` block saying how a checker re-derives
it. **A question whose answer cannot be checked by program is not emitted.**

What the graph knows, and what it does not
------------------------------------------
The constraint graph records physical *necessity* and it is incomplete: there
are no ``blocked_by`` edges in the database at all today, so a cooler screw
under a fan shroud has no edge saying the shroud is in the way. Three truths
follow, and every affordance and planning task here is built on them:

* what the graph **forbids** is certainly impossible -- a required precondition
  is a physical fact, and an unmet one cannot be worked around;
* what the **state** forbids is certainly impossible too, and for a different
  reason: an already-unplugged connector cannot be disconnected and a part that
  has left the machine cannot be removed. No edge is involved, and the answer is
  in the picture rather than in the graph;
* what the graph **permits** is only an **upper bound** -- it is the set of
  actions no *recorded* constraint rules out, not the set of actions that work;
* what the log **demonstrates** -- a successful action recorded at that state --
  is certainly possible.

The third one is there because the first two were not enough. Built from graph
negatives alone, V4 was 83 % solvable from the question text: the verbs barely
overlapped (`disconnect` was positive 987 times and negative never, because no
edge gates a connector), so "which verb is it?" answered the question without
looking at the image. A state negative is drawn to match its positive's verb and
class, which is what makes the two indistinguishable on paper.

So V4's positives are demonstrated actions and never merely permitted ones;
V5 answers with a bounded set (what must be in, what must not be in, and the
permitted upper bound named as such); V6 is graded on the demonstrated
reference and on how often an answer is certainly blocked; V16's valid plans are
logged suffixes only. Nothing here ever calls the permitted set "every action
that is physically possible".

Two sources of truth
--------------------
* **perception** (V1, V2, V8, V12, V14, V15) reads the compiled truth table and
  is emitted **only from VERIFIED frames**, only about instances that have a
  compiled row in *this* view;
* **planning / history** (V3, V4, V5, V6, V10, V16) reads the step log, the
  state machine and the constraint graph, so it exists for every desktop whose
  steps have been imported.

Where the log and the graph contradict each other -- the 18 violations of
``reports/constraints_report.md`` -- nothing is emitted: a ground truth that
argues with itself is worse than a gap, and :func:`illegal_steps` hands the list
to the caller to report.

Everything is pure and Qt-free; :mod:`tda.core.export.vlm` owns the loop, the
files and the refusals.
"""
from __future__ import annotations

import hashlib
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
    REQUIRED_STATES,
    applicable_preconditions,
    find_deadlocks,
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
from tda.core.model import is_provisional
from tda.core.states import FrameState, InstState, gone_with_parent
from tda.core.taxonomy import Taxonomy

__all__ = [
    "CHASSIS_CLASS",
    "CHECK_TYPES",
    "GENERATORS",
    "IMAGE_ID_SALT",
    "LAYERS",
    "PERCEPTION_TASKS",
    "PLANNING_TASKS",
    "TASKS",
    "TRUTH_SOURCES",
    "FrameData",
    "TaskCtx",
    "class_label",
    "exclusion_reason",
    "illegal_steps",
    "instance_label",
    "opaque_image_id",
    "record",
    "row_verified",
    "unresolved_action_targets",
]

#: The P0 task set of spec 8.2, in the order records are written per frame.
TASKS = ("V1", "V2", "V3", "V4", "V5", "V6", "V8", "V10", "V12", "V14", "V15", "V16")

#: Tasks whose answer is read off the compiled truth table of one view.
PERCEPTION_TASKS = frozenset({"V1", "V2", "V8", "V12", "V14", "V15"})
#: Tasks whose answer is the step log, the state machine and the graph.
PLANNING_TASKS = frozenset({"V3", "V4", "V5", "V6", "V10", "V16"})
#: Tasks that cannot be derived without a trustworthy constraint graph.
GRAPH_TASKS = frozenset({"V4", "V5", "V6", "V16"})

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
    "boxes",         # a set of instances, each localised: set F1 + box IoU
    "exact",         # named fields must match exactly
    "feasibility",   # V4: a yes/no plus the blocker set
    "bounded_set",   # V5: must-include, must-not-include, permitted upper bound
    "next_action",   # V6: the demonstrated next action, plus the blocked set
    "history",       # V10: the done set, the remaining count and the bin
    "plan",          # V16: validity, the first bad index and the violated edge
})

#: Where a V4 label's certainty comes from (the governing principle, above).
TRUTH_SOURCES = ("demonstrated", "graph_blocked", "state_inapplicable",
                 "failed_attempt")
DEMONSTRATED, GRAPH_BLOCKED, STATE_INAPPLICABLE, FAILED_ATTEMPT = TRUTH_SOURCES

#: The three corruptions V16 builds a certainly-wrong plan with.
CORRUPTIONS = ("swap", "move", "substitute")

#: V10's progress bins, quarters of the desktop's own action count.
PROGRESS_BINS = ("0-25", "25-50", "50-75", "75-100")

#: The salt the opaque image ids are hashed with. It is a constant, not a
#: secret: the point is that a *prompt* carries no desktop, view or step, while
#: two exports of the same database still name the same frame the same way. The
#: manifest records it, so a reader can recompute the mapping.
IMAGE_ID_SALT = "tda-vlm-images-v1"

CABLE_CLASS = "cable"
CHASSIS_CLASS = "chassis"
REMOVED = "removed"
STATE_ATTR = "state"
#: A tool value that is not a fact about the machine and is never graded (I3).
UNGRADED_TOOLS = frozenset({None, "", "unknown"})


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

#: The present participle of every verb of spec 6.3, so a template can say
#: "Would removing the PSU work?" instead of "would remove the PSU work".
GERUNDS = {
    "unscrew": "unscrewing", "disconnect": "disconnecting", "open": "opening",
    "release": "releasing", "remove": "removing", "displace": "displacing",
    "reorient": "reorienting",
}


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


def verb_phrase(verb: str, label: str) -> str:
    """``("unscrew", "PSU screw 1")`` -> ``"unscrew PSU screw 1"`` (imperative)."""
    return f"{verb} {label}"


def verb_gerund(verb: str, label: str) -> str:
    """``("unscrew", "PSU screw 1")`` -> ``"unscrewing PSU screw 1"``."""
    return f"{GERUNDS.get(verb, verb + 'ing')} {label}"


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
    task from always asking about the first screw in the table. CRC32 rather
    than ``hash()``: the built-in is salted per process.
    """
    return sorted(items, key=lambda item: (zlib.crc32(f"{seed}|{key(item)}"
                                                      .encode("utf-8")), key(item)))


def opaque_image_id(desktop: int, view: str, step: int,
                    salt: str = IMAGE_ID_SALT) -> str:
    """The prompt-side name of one frame: no desktop, no view, no step.

    A prompt that says ``scan/D13/s017.png`` has answered "how far along is this
    teardown?" before the model has looked at the picture, and half of the P0
    set asks exactly that. The label side keeps the path, through the manifest
    :mod:`tda.core.export.vlm` writes beside the JSONL.
    """
    body = f"{salt}|{int(desktop)}|{view}|{int(step)}".encode("utf-8")
    return "img_" + hashlib.sha256(body).hexdigest()[:16]


def opaque_record_id(readable: str, salt: str = IMAGE_ID_SALT) -> str:
    """The record's own id, with the teardown hashed out of it.

    ``V4-D07-scan-s001-demonstrated-remove-psu.01`` names the desktop, the view,
    the step and the answer's own target. It is not on the prompt side, but a
    harness that prints record ids -- and they all do, in logs and in error
    messages -- would hand a model everything the opaque image ids were there to
    withhold. The readable form stays on the label side as ``readable_id``.
    """
    return "rec_" + hashlib.sha256(f"{salt}|{readable}".encode("utf-8")
                                   ).hexdigest()[:16]


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
    image_id: str
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


@dataclass(frozen=True)
class Candidate:
    """One action that is **certainly not possible** right now, and why.

    ``graph_blocked`` carries the unmet edges; ``state_inapplicable`` carries a
    reason read off the state -- the plug is already out, the part is already
    gone. Both are certainties; only the second needs the picture to settle.
    """

    verb: str
    target: str
    kind: str
    edges: tuple = ()
    reason: Optional[str] = None


def inapplicable_reason(state: FrameState, key: str) -> str:
    """Why this instance cannot be acted on: it is elsewhere, or already there."""
    inst = state.get(key)
    if inst is None:
        return "not_present"
    if inst.state == REMOVED or gone_with_parent(state, key):
        return "not_present"
    return f"already_{inst.state}"


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
    #: why this desktop's graph-derived tasks are refused, or ``None`` (C2).
    excluded: Optional[str] = None
    #: how many hard negatives one frame may contribute per task.
    budget: int = 2
    _legal: dict[tuple[int, bool], list] = field(default_factory=dict, repr=False)
    _blocked: dict[int, list] = field(default_factory=dict, repr=False)
    _inapplicable: dict[int, list] = field(default_factory=dict, repr=False)
    _by_target: Optional[dict[str, list[Edge]]] = field(default=None, repr=False)
    _v12: Optional[set[int]] = field(default=None, repr=False)
    _first_app: Optional[dict] = field(default=None, repr=False)
    _verbs: Optional[frozenset] = field(default=None, repr=False)

    # -- shorthand ---------------------------------------------------------- #
    @property
    def tax(self) -> Taxonomy:
        return self.ctx.tax

    @property
    def desktop(self) -> int:
        return self.ctx.desktop

    @property
    def log_steps(self) -> list[int]:
        """The desktop's own annotatable steps, frame or no frame.

        Not the same list as :attr:`steps`, which is the frames *this view* has.
        A scanner frame may be missing (spec 4.2, 缺帧处理) and the action still
        happened, so "what is the next step" and "how much is left" are counted
        on the log; only the questions that need a picture are counted on frames.
        """
        return sorted(s for s in self.ctx.steps if self.ctx.exportable(s))

    @property
    def by_target(self) -> dict[str, list[Edge]]:
        """``target -> its active edges``, built once per desktop.

        :func:`~tda.core.graph.applicable_preconditions` scans the whole edge
        list, which is nothing for one call and 76 million comparisons for a
        66-desktop export: every frame asks for the permitted set and the
        blocked set, and every candidate action asks for its own preconditions.
        """
        if self._by_target is None:
            index: dict[str, list[Edge]] = {}
            for edge in active_edges(self.edges):
                index.setdefault(edge.target, []).append(edge)
            self._by_target = index
        return self._by_target

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
        for candidate in self.log_steps:
            if candidate > step and self.named_actions(candidate):
                return candidate
        return None

    def demonstrated(self, step: int) -> Optional[tuple[int, ActionRec]]:
        """``(step, action)`` the log shows being carried out from this state.

        The *first* action of the next step that has one. A compound step's
        later actions happen after its first one and are not demonstrated at
        *this* state, so they are not claimed here.

        ``None`` for three reasons, and all three mean "no certainty here":
        nothing follows, the log and the graph contradict each other at the next
        step, or the next action changes no state at all. The last is
        ``reorient``: spec 6.3 gives it no effect and nothing gates it, so
        "could you do it?" has no answer the constraint graph can grade -- and
        answering with the step *after* it would be answering a different
        question.
        """
        nxt = self.next_action_step(step)
        if nxt is None or nxt in self.illegal:
            return None
        action = self.named_actions(nxt)[0]
        return (nxt, action) if self.changes_state(action) else None

    def changes_state(self, action: ActionRec) -> bool:
        """Does this action move the state machine at all? (``reorient`` does not.)"""
        rec = self.ctx.instances.get(action.target)
        cls = rec.cls if rec is not None else (
            CABLE_CLASS if action.target.startswith(CABLE_PREFIX) else None)
        if cls is None:
            return False
        effect = self.tax.apply_verb(cls, rec.attrs if rec is not None else {},
                                     action.verb)
        return effect is not None and effect[0] == STATE_ATTR

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

    def permitted(self, step: int) -> list[tuple[str, str]]:
        """The graph-permitted **upper bound** at the state after ``step``.

        Required edges only, which is the standard
        :func:`~tda.core.graph.validate_sequence` judges the log by. This is not
        "what is possible": it is what no recorded constraint forbids.
        """
        return self._permitted(step, False)

    def permitted_strict(self, step: int) -> list[tuple[str, str]]:
        return self._permitted(step, True)

    def _permitted(self, step: int, strict: bool) -> list[tuple[str, str]]:
        hit = self._legal.get((step, strict))
        if hit is None:
            hit = legal_actions(self.ctx.instances, self.edges, self.state(step),
                                self.tax, strict=strict)
            self._legal[(step, strict)] = hit
        return hit

    def blocked(self, step: int) -> list[tuple[str, str, list[Edge]]]:
        """``(verb, target, unmet edges)`` -- what the graph *certainly* forbids."""
        hit = self._blocked.get(step)
        if hit is None:
            hit = blocked_actions(self, step)
            self._blocked[step] = hit
        return hit

    @property
    def first_applicable(self) -> dict[tuple[str, str], int]:
        """``(verb, instance) -> the earliest step the verb could be performed``.

        What makes "can you disconnect this connector?" a fair question at step
        40 is that it *was* a fair question at step 12: the plug was in, and the
        model has to see that it is not any more. An action that was never
        applicable at any point -- ``open`` on a PSU, ``unscrew`` on a cable --
        is not a question about this machine at all.
        """
        if self._first_app is None:
            out: dict[tuple[str, str], int] = {}
            for j in [0, *self.log_steps]:
                state = self.state(j)
                for key, rec in self.ctx.instances.items():
                    inst = state.get(key)
                    if inst is None:
                        continue
                    for verb in self.tax.verbs:
                        if (verb, key) in out:
                            continue
                        if verb_applies(self.tax, rec.cls, rec.attrs, verb,
                                        inst.state):
                            out[(verb, key)] = j
            self._first_app = out
        return self._first_app

    def askable(self, key: str) -> bool:
        """Is this instance one a question may be asked about at all?

        Real, not a Label Studio draft, and not an identity
        :mod:`tda.core.implied` invented because the log referenced a class
        nobody operated -- an implied motherboard has no history of its own, so
        "is it still there?" is a question about the inference, not the machine.
        """
        rec = self.ctx.instances.get(key)
        return (rec is not None and not is_provisional(key)
                and not is_implied(rec))

    def inapplicable(self, step: int) -> list[Candidate]:
        """Actions the *state* rules out right now (cached per frame)."""
        hit = self._inapplicable.get(step)
        if hit is None:
            hit = inapplicable_in(self, self.state(step), step)
            self._inapplicable[step] = hit
        return hit

    def certainly_wrong(self, step: int) -> list[Candidate]:
        """Everything that certainly cannot be done at ``step``, both kinds."""
        return [*(Candidate(v, t, GRAPH_BLOCKED, tuple(e))
                  for v, t, e in self.blocked(step) if self.askable(t)),
                *self.inapplicable(step)]

    def matched(self, candidates: Sequence[Candidate], verb: str,
                cls: Optional[str], seed: str) -> list[Candidate]:
        """``candidates`` ordered so the closest match to ``(verb, cls)`` is first.

        Same verb and same class beats same verb, which beats same class, which
        beats anything -- and inside a tier a graph-blocked action beats a
        state-inapplicable one, because the graph's "no" needs no picture to be
        certain. The tail is a seeded permutation, so the choice is stable and
        is not always the first instance in the table.
        """
        used = self.demonstrated_verbs

        def rank(c: Candidate) -> tuple:
            same_verb = c.verb == verb
            same_cls = self.ctx.cls_of(c.target) == cls
            tier = (0 if same_verb and same_cls else
                    1 if same_verb else 2 if same_cls else 3)
            # when the verb cannot be matched, at least prefer one this teardown
            # actually uses: a verb that is only ever a negative -- `displace`,
            # which no sheet records -- is a giveaway of its own
            spoken = 0 if c.verb in used else 1
            kind = 0 if c.kind == GRAPH_BLOCKED else 1
            return (tier, spoken, kind,
                    zlib.crc32(f"{seed}|{c.verb}|{c.target}".encode("utf-8")),
                    c.verb, c.target)
        return sorted(candidates, key=rank)

    @property
    def demonstrated_verbs(self) -> frozenset[str]:
        """The verbs this desktop's log actually uses."""
        if self._verbs is None:
            self._verbs = frozenset(a.verb for s in self.log_steps
                                    for a in self.named_actions(s))
        return self._verbs

    @property
    def v12_steps(self) -> set[int]:
        """The steps V12 may ask about, balanced across this desktop.

        Its negatives can only come from ``dupli`` rows (spec 8.2), and there
        are sixteen of those in the whole dataset against two and a half
        thousand ordinary steps. Publishing every change would make "yes,
        something changed" right 99 % of the time, so the positives are
        subsampled to the number of negatives the desktop actually has, by a
        seed of the desktop. A desktop with no ``dupli`` row asks nothing.
        """
        if self._v12 is None:
            negative, positive = [], []
            for step in sorted(self.frames):
                before = self.frame(step - 1)
                if before is None or not (before.verified
                                          and self.frames[step].verified):
                    continue
                rec = self.ctx.steps.get(step)
                kind = self.ctx.step_type(step)
                if kind == StepType.DUPLI.value or (rec is not None and rec.dupli):
                    negative.append(step)
                elif kind not in NO_CHANGE_STEP_TYPES and kind != StepType.FAILED.value:
                    if self.state(step - 1) != self.state(step):
                        positive.append(step)
            chosen = stable_order(positive, f"D{self.desktop}|{self.view}|V12",
                                  str)[:len(negative)]
            self._v12 = set(negative) | set(chosen)
        return self._v12


def target_class(ctx: DesktopCtx, target: str) -> Optional[str]:
    """Class of an action target: ``cable`` for a virtual node, else the taxonomy's."""
    if target.startswith(CABLE_PREFIX):
        return CABLE_CLASS
    return ctx.cls_of(target)


def graded_tool(tool: Optional[str]) -> Optional[str]:
    """The tool if it is a fact, else ``None`` -- ``unknown`` is never graded (I3)."""
    return None if tool in UNGRADED_TOOLS else tool


# --------------------------------------------------------------------------- #
# the graph, asked the three ways the tasks need
# --------------------------------------------------------------------------- #
def blocked_actions(tc: TaskCtx, step: int) -> list[tuple[str, str, list[Edge]]]:
    """``(verb, target, unmet edges)`` for everything the graph forbids right now.

    The verb applies to the class in this state (spec 6.3), so "can you do this?"
    is a sensible question, and at least one hard precondition is not satisfied,
    so the answer is certainly no and the graph says why. This is the half of
    the graph that is a fact rather than an upper bound.
    """
    state = tc.state(step)
    out: list[tuple[str, str, list[Edge]]] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        inst = state.get(key)
        if inst is None or inst.state == REMOVED or key not in tc.by_target:
            continue  # nothing points at it: nothing can be blocking it
        for verb in tc.tax.verbs:
            if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            bad = unmet_for(tc, verb, key, state)
            if bad:
                out.append((verb, key, bad))
    for key, current in sorted(cable_nodes(active_edges(tc.edges), state).items()):
        if not verb_applies(tc.tax, CABLE_CLASS, {}, "release", current):
            continue
        bad = unmet_for(tc, "release", key, state)
        if bad:
            out.append(("release", key, bad))
    return out


def blocked_in(tc: TaskCtx, state: FrameState) -> list[Candidate]:
    """:func:`blocked_actions`, asked of an arbitrary state (a plan's mid-point)."""
    out: list[Candidate] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        inst = state.get(key)
        if inst is None or inst.state == REMOVED or key not in tc.by_target:
            continue
        for verb in tc.tax.verbs:
            if not verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            bad = unmet_for(tc, verb, key, state)
            if bad:
                out.append(Candidate(verb, key, GRAPH_BLOCKED, tuple(bad)))
    return out


def inapplicable_in(tc: TaskCtx, state: FrameState, step: int) -> list[Candidate]:
    """Actions the state rules out: the verb no longer starts from where it is.

    Only verbs that belong to the class (spec 6.3) and that **were** applicable
    to this very instance at some earlier step: the question is "has this
    already happened?", which is answerable from the picture, and not "does this
    verb exist?", which is answerable from the vocabulary.
    """
    first = tc.first_applicable
    out: list[Candidate] = []
    for key, rec in sorted(tc.ctx.instances.items()):
        if not tc.askable(key):
            continue
        inst = state.get(key)
        if inst is None:
            continue
        reason = inapplicable_reason(state, key)
        for verb in tc.tax.verbs:
            if verb_applies(tc.tax, rec.cls, rec.attrs, verb, inst.state):
                continue
            began = first.get((verb, key))
            if began is None or began >= step:
                continue
            out.append(Candidate(verb, key, STATE_INAPPLICABLE, (), reason))
    return out


def unmet_for(tc: TaskCtx, verb: str, target: str, state: FrameState) -> list[Edge]:
    """Which required preconditions of ``(verb, target)`` are not satisfied.

    The **one** way anything in this module asks that question -- V4, V5, V6 and
    V16 all come through here, so a change to
    :func:`~tda.core.graph.applicable_preconditions` (B5 narrows the gate by
    ``blocked_by.mode``) reaches all four at once. A private re-implementation
    in the plan simulator once made V16 disagree with V4 about a ``cable:*``
    blocker that is not in the snapshot.
    """
    return unmet(applicable_preconditions(tc.by_target.get(target, []), (verb, target)),
                 state, "required")


def _copy_state(state: FrameState) -> FrameState:
    return {key: InstState(state=inst.state, placement=inst.placement)
            for key, inst in state.items()}


def _apply_effect(tax: Taxonomy, instances: dict[str, InstanceRec],
                  sim: FrameState, target: str, verb: str) -> None:
    """Apply one verb to a simulated state, with the spec-3.3 attached cascade."""
    rec = instances.get(target)
    cls = rec.cls if rec is not None else (
        CABLE_CLASS if target.startswith(CABLE_PREFIX) else None)
    if cls is None:
        return
    effect = tax.apply_verb(cls, rec.attrs if rec is not None else {}, verb)
    if effect is None or effect[0] != STATE_ATTR:
        return
    inst = sim.get(target)
    if inst is None:
        inst = sim[target] = InstState(state=tax.default_state(cls),
                                       placement="in_chassis")
    inst.state = effect[1]
    if effect[1] != REMOVED or (rec is not None and rec.cls == "connector"):
        return
    for key, child in instances.items():
        if child.attached and child.parent == target and key in sim:
            sim[key].state = REMOVED


def illegal_steps(ctx: DesktopCtx, edges: list[Edge],
                  tax: Taxonomy) -> dict[int, list[str]]:
    """``step -> the edges its logged action broke`` (spec 7.4, first check).

    The same replay ``constraints --validate`` prints, kept as data rather than
    as prose so the export can act on it: a step listed here is one where the
    log and the graph contradict each other, and V4/V5/V6/V16 refuse to speak
    about the moment before it.

    The state it folds is :meth:`~tda.core.export.coco.DesktopCtx.state_at` --
    **the same one every task reads**, hand-written events included. Folding the
    derived events only, as an independent replay would, is how a desktop whose
    annotator corrected the log by hand came to be judged against a history
    nobody had.
    """
    by_target: dict[str, list[Edge]] = {}
    for edge in active_edges(edges):
        by_target.setdefault(edge.target, []).append(edge)
    out: dict[int, list[str]] = {}
    for step in sorted({a.step for a in ctx.actions}):
        sim = _copy_state(ctx.state_at(step - 1))
        for action in sorted((a for a in ctx.actions if a.step == step),
                             key=lambda a: a.idx):
            if action.result != "success":
                continue
            bad = unmet(
                applicable_preconditions(by_target.get(action.target, []), action),
                sim, "required")
            if bad:
                out.setdefault(step, []).extend(e.label() for e in bad)
            _apply_effect(tax, ctx.instances, sim, action.target, action.verb)
    return out


def unresolved_action_targets(ctx: DesktopCtx) -> list[str]:
    """Targets of successful actions this export cannot name (C3).

    ``?``, ``connector.?`` and friends: a step whose sheet did not say which part
    was operated. They are not instances, so the history V10 reports is
    incomplete by exactly those actions -- and V10's metric is exhaustive, so an
    incomplete history is a wrong answer rather than a partial one.
    """
    return sorted({a.target for a in ctx.actions
                   if a.result == "success" and ctx.exportable(a.step)
                   and target_class(ctx, a.target) is None})


def exclusion_reason(ctx: DesktopCtx, edges: list[Edge],
                     graph_version: Optional[str]) -> Optional[str]:
    """Why this desktop's graph-derived tasks are refused, or ``None`` (C2).

    Three reasons, and all three are the same reason: the blocked and permitted
    sets would not be trustworthy.

    * ``no_graph_version`` / ``no_constraint_edges`` -- a desktop with no graph
      is not one whose affordance questions are easy, it is one whose every
      answer would be "nothing is blocked", the most confidently wrong ground
      truth this export could ship;
    * ``deadlock`` -- :func:`~tda.core.graph_plan.find_deadlocks` finds a ring of
      actions each of which waits on the next (spec 7.4). Nothing in that ring
      can ever be done, so the graph contradicts a teardown that demonstrably
      happened, and "what can be done now" has no honest answer while it stands.
      Required edges only: a ``recommended`` edge is a preference and never
      makes anything impossible.

    A desktop carrying more Label Studio drafts than resolved instances used to
    be refused here too. It is not a reason: ``ls:*`` rows are reference
    geometry somebody traced in a parallel table, they are already out of
    :attr:`~tda.core.export.coco.DesktopCtx.instances`, and the five desktops it
    caught (D13, D18, D19, D24, D33) carry 41-49 hard edges each and a complete
    instance table.
    """
    if graph_version is None:
        return "no_graph_version"
    if not active_edges(edges):
        return "no_constraint_edges"
    if find_deadlocks(edges, ctx.instances, ctx.tax, necessity="required"):
        return "deadlock"
    return None


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def evidence(bboxes: dict[str, list], tc: Optional[TaskCtx] = None,
             attrs: Optional[dict] = None, **extra) -> dict:
    """The instance keys, boxes and per-instance attributes an answer rests on.

    Boxes are the **visible**-mask box of the compiled row (spec 3.4); the COCO
    export additionally ships the amodal shape, which is where a consumer that
    wants the whole part should look.

    ``attributes.implied`` mirrors the COCO export's field (spec 8.1): the
    instance was created by :mod:`tda.core.implied` because the desktop plainly
    has one and its log never touched it, not because a step named it.
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


def record(tc: TaskCtx, task: str, rec_id: str, step: int,
           frames: Sequence[FrameData], question: str, template_index: int,
           answer: dict, check: dict, evid: dict, rationale: dict, verified: bool,
           *, views: Optional[Sequence[str]] = None,
           options: Optional[Sequence[dict]] = None,
           negative: Optional[str] = None,
           truth_source: Optional[str] = None) -> dict:
    """One JSONL record, split into what the model sees and what grades it.

    ``prompt`` is everything a model may be shown: the task id, the **opaque**
    image ids, the question and, where the task has them, the options. It
    carries no step number, no view, no desktop, no instance key and no field of
    the answer. ``label`` is everything else.

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
    if truth_source is not None and truth_source not in TRUTH_SOURCES:
        raise ValueError(f"{rec_id}: truth_source {truth_source!r} is not one of "
                         f"{', '.join(TRUTH_SOURCES)}")
    prompt: dict[str, Any] = {
        "task": task,
        "images": [f.image_id for f in frames],
        "question": question,
    }
    if options is not None:
        prompt["options"] = [{"verb": o["verb"], "target_label": o["target_label"]}
                             for o in options]
    label: dict[str, Any] = {
        "readable_id": rec_id,
        "answer": answer,
        "answer_check": check,
        "evidence": evid,
        "rationale": rationale,
        "desktop": tc.desktop,
        "step": int(step),
        "view": tc.view,
        "layer": LAYERS[task],
        "template_id": f"{task}.{template_index + 1}",
        "tier": tc.tier,
        "verified": bool(verified),
        "graph_version": tc.graph_version,
        "model_family": tc.meta.get("model_family"),
        "chassis_type": tc.meta.get("chassis_type"),
    }
    if views:
        label["views"] = list(views)
    if options is not None:
        label["options"] = list(options)
    if negative:
        label["negative"] = negative
    if truth_source:
        label["truth_source"] = truth_source
    return {"id": opaque_record_id(rec_id), "prompt": prompt, "label": label}


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


def question_label(tc: TaskCtx, instance: str,
                   frame: Optional[FrameData] = None) -> str:
    """How a question names an instance, without giving another one away.

    "Motherboard screw 3" says this machine has at least three of them, and on a
    frame where V2 asks "how many motherboard screws are still fastened?" that
    is part of the other answer. Where an unambiguous relational phrase exists
    it is used instead (V8's rule, extended to V4, V6 and V16).

    Where one does not, the ordinal stays: V4, V6 and V16 are *about* that one
    action, and dropping the question would cost the only affordance label on
    that part to protect a much weaker inference than V8's -- the ordinal bounds
    how many screws exist, while the count asks how many are still in. Today the
    two never meet on the same frame anyway: V2 needs a verified frame and none
    is.
    """
    if frame is None:
        return tc.label(instance)
    rec = tc.ctx.instances.get(instance)
    if rec is None or rec.cls != "screw":
        return tc.label(instance)
    if str(rec.attrs.get("role") or "other") not in set(counted_roles(tc, frame)):
        return tc.label(instance)
    relational = _referring(tc, instance, avoid_ordinal=True)
    return relational[0] if relational else tc.label(instance)


def state_is_askable(tax: Taxonomy, cls: str) -> bool:
    """Is "what state is it in?" a real question for this class?

    Only when the class has at least two states the annotation actually tracks
    (spec 6.2's ``needs_mask`` table). A ``connector`` has one -- ``plugged`` --
    because an unplugged connector is not tracked at all, so every frame that
    carries one answers "plugged" and the question teaches nothing; a
    ``chassis`` is always ``present``.
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
    "List every component visible in this image, with a bounding box for each.",
    "Which components can you see in this image? Give each one a bounding box.",
    "Name the parts visible in this photo of the desktop PC and localise them.",
    "Enumerate the visible components and give their boxes.",
)


def gen_v1(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Every pointable instance of a verified frame, except the chassis.

    The chassis is in every frame, fills most of it, and is named by the task --
    a model that answers "a chassis, somewhere near the middle" scores on it
    without looking. It is excluded from the exhaustive set rather than left in
    to inflate the F1 (the COCO export still carries it).
    """
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    components, bboxes, steps, used, attrs = [], {}, [], [], {}
    for instance, (row, box) in frame.pointable.items():
        cls = tc.ctx.cls_of(instance)
        if cls is None or cls == CHASSIS_CLASS:
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
        tc, "V1", rec_id, step, [frame], question, index,
        {"components": components},
        {"type": "boxes", "field": "components", "id_field": "instance",
         "box_field": "bbox", "exhaustive": True, "exclude_classes": [CHASSIS_CLASS],
         "geometry": "visible_mask_bbox", "metric": "set_f1+box_iou",
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
            tc, "V2", rec_id, frame.step, [frame],
            template.format(label=tc.label(instance)), index,
            {"state": inst.state},
            {"type": "exact", "derive": "instance_state", "instance": instance,
             "class": cls, "fields": ["state"],
             "allowed": sorted(tc.tax.states_of(cls))},
            evidence({instance: box}, tc, {instance: {"visibility": row.get("visibility")}}),
            R.rationale([R.observe(instance, inst.state, tc.view, box,
                                   row.get("visibility"))]),
            row_verified([row], frame.verified),
        )


def counted_roles(tc: TaskCtx, frame: FrameData) -> list[str]:
    """Screw roles this frame asks a V2 counting question about."""
    return sorted({
        str(rec.attrs.get("role") or "other")
        for key, rec in tc.ctx.instances.items()
        if rec.cls == "screw" and key in frame.pointable
    })


def _v2_counts(tc: TaskCtx, frame: FrameData) -> Iterator[dict]:
    """"How many <role> screws are still fastened?" -- counted on the state machine.

    The count covers every screw of that role on the machine, not only the ones
    this view can see, because that is what the question asks; the evidence
    lists the boxes of those that are visible here, and the ones that are not
    enter the chain as ``propagate_state``.
    """
    state = tc.state(frame.step)
    for role in counted_roles(tc, frame):
        fastened = [
            key for key, rec in sorted(tc.ctx.instances.items())
            if rec.cls == "screw" and str(rec.attrs.get("role") or "other") == role
            and (state.get(key).state if state.get(key) else None) == "fastened"
        ]
        bboxes = {k: frame.pointable[k][1] for k in fastened if k in frame.pointable}
        rec_id = f"V2-{frame_id(tc.desktop, tc.view, frame.step)}-count-{role}"
        index, template = pick(V2_COUNT_QUESTIONS, rec_id)
        yield record(
            tc, "V2", rec_id, frame.step, [frame],
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
    "What did the operator just do between the first image and the second?",
)


def _slots(tc: TaskCtx, action: ActionRec) -> dict:
    return {"verb": action.verb, "target_class": target_class(tc.ctx, action.target),
            "target_instance": action.target, "tool": graded_tool(action.tool)}


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
    # `unknown` is not a fact about the machine and is never graded (I3)
    fields = ["verb", "target_class", "target_instance"]
    if answer["tool"] is not None:
        fields.append("tool")

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
        tc, "V3", rec_id, step, [before, frame], question, index, answer,
        {"type": "exact", "derive": "action_slots", "fields": fields},
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
    "Is it possible to {phrase} in this state? Name anything that blocks it.",
    "Would {gerund} work right now? If not, list what is in the way.",
)
V4_FAILED_QUESTIONS = (
    "The operator tried to {phrase} here and could not. Why not?",
    "This attempt at {gerund} failed. What was in the way?",
)


def _v4_record(tc: TaskCtx, frame: FrameData, verb: str, target: str,
               bad: Sequence[Edge], templates: Sequence[str],
               truth_source: str, reason: Optional[str] = None) -> dict:
    state = tc.state(frame.step)
    blockers = sorted({e.blocker for e in bad})
    rec_id = (f"V4-{frame_id(tc.desktop, tc.view, frame.step)}"
              f"-{truth_source}-{verb}-{target}")
    index, template = pick(templates, rec_id)
    label = question_label(tc, target, frame)
    chain: list[dict] = [observed(target, state[target].state, tc, frame.pointable)]
    for edge in sorted(bad, key=lambda e: (e.type, e.blocker)):
        current = state[edge.blocker].state if edge.blocker in state else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          REQUIRED_STATES.get(edge.type, frozenset()),
                                          current, False))
    if truth_source == STATE_INAPPLICABLE:
        chain.append(R.check_precondition(target, ["a state the verb starts from"],
                                          state[target].state, False))
    feasible = truth_source == DEMONSTRATED
    # one field, one meaning: why this answer is what it is. An edge said no, or
    # the state did, or nothing did and the operator went on to do it.
    reason = None if feasible else (reason or ("violates_edge" if bad else None))
    bboxes = {k: frame.pointable[k][1]
              for k in [target, *blockers] if k in frame.pointable}
    return record(
        tc, "V4", rec_id, frame.step, [frame],
        template.format(phrase=verb_phrase(verb, label),
                        gerund=verb_gerund(verb, label)), index,
        {"feasible": feasible, "blockers": blockers, "reason": reason},
        {"type": "feasibility", "verb": verb, "target": target,
         "target_class": tc.ctx.cls_of(target),
         "state_after": frame.step, "truth_source": truth_source,
         "metric": "per_verb_accuracy_macro", "secondary_metric": "accuracy",
         "edges": [{"type": e.type, "target": e.target, "blocker": e.blocker,
                    "necessity": e.necessity, "mode": e.mode,
                    "required": sorted(REQUIRED_STATES.get(e.type, frozenset()))}
                   for e in sorted(bad, key=lambda e: (e.type, e.blocker))]},
        evidence(bboxes, tc), R.rationale(chain),
        row_verified([frame.pointable[k][0] for k in bboxes], frame.verified),
        negative=None if feasible else truth_source,
        truth_source=truth_source,
    )


def gen_v4(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Demonstrated positives against graph-blocked negatives, one for one.

    A "yes" here is never merely permitted. The graph is incomplete -- there is
    not one ``blocked_by`` edge in the database -- so "no recorded constraint
    forbids it" is not "it works", and a benchmark whose positive class is that
    guess teaches a model to trust an upper bound. The only certain yes is the
    one the operator went on to do, so that is the only yes this task ships.

    The negatives are the certain noes: a graph-blocked action, and the failed
    attempts the graph explains. The three attempts of
    ``reports/constraints_report.md`` that break no constraint are exactly the
    case where the dataset does not yet know the answer; the constraint editor
    (task B5) is where that gap is filled.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    shown = tc.demonstrated(step)
    if shown is None:
        return  # no demonstrated action here: nothing certain to be positive about
    _, action = shown
    state = tc.state(step)
    if action.target not in state:
        return
    yield _v4_record(tc, frame, action.verb, action.target, (), V4_QUESTIONS,
                     DEMONSTRATED)

    # one certain "no" for each certain "yes", matched to the positive's verb
    # and class so the pair cannot be told apart without looking at the picture
    matched = tc.matched(tc.certainly_wrong(step), action.verb,
                         target_class(tc.ctx, action.target), tc.seed(step, "V4"))
    for candidate in matched[:1]:
        yield _v4_record(tc, frame, candidate.verb, candidate.target,
                         candidate.edges, V4_QUESTIONS, candidate.kind,
                         candidate.reason)

    # the attempt the operator actually made and could not finish
    for attempt in tc.failed_actions(step + 1):
        rec = tc.ctx.instances.get(attempt.target)
        if rec is None or attempt.target not in state:
            continue
        if not verb_applies(tc.tax, rec.cls, rec.attrs, attempt.verb,
                            state[attempt.target].state):
            continue
        bad = unmet_for(tc, attempt.verb, attempt.target, state)
        if not bad:
            continue  # the graph cannot explain it: not a ground truth yet
        yield _v4_record(tc, frame, attempt.verb, attempt.target, bad,
                         V4_FAILED_QUESTIONS, FAILED_ATTEMPT, "violates_edge")


# --------------------------------------------------------------------------- #
# V5 -- the bounded set
# --------------------------------------------------------------------------- #
V5_QUESTIONS = (
    "What can be done to this machine right now? List the actions you are "
    "confident are possible.",
    "Given this state, which disassembly actions could be carried out next? "
    "List them.",
    "Name the actions that could be performed on this machine as it stands.",
)


def _actions_json(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    return [{"verb": verb, "target": target} for verb, target in sorted(pairs)]


def gen_v5(tc: TaskCtx, step: int) -> Iterator[dict]:
    """A bounded set, because the exact set is not knowable from this data.

    ``must_include`` is what the log went on to do, so it is certainly possible;
    ``must_not_include`` is what the graph forbids, so it is certainly
    impossible; ``permitted_upper_bound`` is everything no recorded constraint
    rules out, and is labelled as an upper bound because that is all it is. An
    answer is right when it contains every certainty and none of the
    impossibilities -- not when it reproduces the upper bound.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    shown = tc.demonstrated(step)
    if shown is None:
        return
    nxt, action = shown
    permitted = tc.permitted(step)
    must_include = [(action.verb, action.target)]
    if must_include[0] not in set(permitted):
        return  # the graph would forbid what the log did: say nothing
    must_not = [(v, t) for v, t, _ in tc.blocked(step)]

    rec_id = f"V5-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V5_QUESTIONS, rec_id)
    state = tc.state(step)
    chain = [observed(target, state[target].state if target in state else None,
                      tc, frame.pointable)
             for _, target in sorted(permitted)[:tc.budget]]
    yield record(
        tc, "V5", rec_id, step, [frame], question, index,
        {"must_include": _actions_json(must_include),
         "must_not_include": _actions_json(must_not),
         "permitted_upper_bound": _actions_json(permitted)},
        {"type": "bounded_set", "reference_step": nxt,
         "metric": "demonstrated_recall+blocked_rate", "necessity": "required",
         "strict_upper_bound": _actions_json(tc.permitted_strict(step))},
        evidence({k: v[1] for k, v in frame.pointable.items()
                  if k in {t for _, t in permitted}}, tc),
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
#: How many graph-permitted "unknown" distractors ride along, and how many
#: certainly-wrong ones the options list needs before it is offered at all.
V6_UNKNOWN_OPTIONS = 2
V6_WRONG_OPTIONS = 2


def gen_v6(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The demonstrated next action, graded on match and on blocked rate.

    "Legal rate" was the wrong metric: the permitted set is an upper bound, so
    an answer inside it has only failed to be *provably* wrong. What can be
    measured exactly is the opposite -- how often an answer is one the graph
    certainly forbids -- and that is what ``blocked_rate`` is.

    The options, when they are offered, are one demonstrated action, at least
    two certainly-blocked ones and a couple of merely-permitted distractors. The
    prompt does not say which is which; the label does, so an evaluator can
    score the blocked ones as wrong and, if it chooses, ignore the unknowns.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    shown = tc.demonstrated(step)
    if shown is None:
        return
    nxt, action = shown
    permitted = set(tc.permitted(step))
    if (action.verb, action.target) not in permitted:
        return  # an action the graph says was impossible is nobody's reference

    seed = tc.seed(step, "V6")
    blocked = [(v, t) for v, t, _ in tc.blocked(step)]
    # the wrong options match the demonstrated verb wherever they can: "pick the
    # option whose verb is never blocked" scored 45 % against 20 % random while
    # the only certainly-wrong options were graph-blocked ones, and no edge
    # gates a connector or a latch
    wrong = tc.matched(tc.certainly_wrong(step), action.verb, None,
                       seed + "|wrong")[:V6_WRONG_OPTIONS]
    # the distractors match the demonstrated verb too, or "which option has the
    # verb that is usually right?" answers the question without the picture
    unknown = tc.matched([Candidate(v, t, "permitted_unknown")
                          for v, t in sorted(permitted
                                             - {(action.verb, action.target)})],
                         action.verb, None, seed + "|unknown")[:V6_UNKNOWN_OPTIONS]
    # a frame with only one certainly-wrong action left is no multiple choice --
    # so the question is asked open-ended instead of being dropped, and says so
    options: Optional[list[dict]] = None
    if len(wrong) >= V6_WRONG_OPTIONS:
        listed = [{"verb": action.verb, "target": action.target, "kind": DEMONSTRATED}]
        listed += [{"verb": c.verb, "target": c.target, "kind": c.kind}
                   for c in wrong]
        listed += [{"verb": c.verb, "target": c.target, "kind": "permitted_unknown"}
                   for c in unknown]
        for option in listed:
            option["target_label"] = question_label(tc, option["target"], frame)
        options = stable_order(listed, seed + "|order",
                               lambda o: f"{o['verb']}|{o['target']}")

    rec_id = f"V6-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V6_QUESTIONS, rec_id)
    state = tc.state(step)
    chain = [observed(action.target,
                      state[action.target].state if action.target in state else None,
                      tc, frame.pointable, step=step)]
    answer = {"verb": action.verb, "target": action.target,
              "target_class": target_class(tc.ctx, action.target),
              "tool": graded_tool(action.tool)}
    fields = ["verb", "target"] + (["tool"] if answer["tool"] is not None else [])
    yield record(
        tc, "V6", rec_id, step, [frame], question, index, answer,
        {"type": "next_action", "reference": [action.verb, action.target],
         "reference_step": nxt, "fields": fields,
         "metric": "match_rate+blocked_rate",
         "options_kind": "listed" if options is not None else "open",
         # `blocked_rate` is the share of answers the graph certainly forbids;
         # on a frame where it forbids nothing the rate is undefined, not zero
         "blocked_rate_defined": bool(blocked),
         "blocked_actions": _actions_json(blocked),
         "permitted_upper_bound": _actions_json(sorted(permitted))},
        evidence({action.target: frame.pointable[action.target][1]}
                 if action.target in frame.pointable else {}, tc),
        R.rationale(chain), frame.verified, options=options,
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


def _referring(tc: TaskCtx, instance: str,
               avoid_ordinal: bool) -> Optional[tuple[str, str]]:
    """``(phrase, family)``, or ``None`` when this frame may not name it.

    A relational expression is the interesting half of V8 -- it forces the model
    to use the assembly structure rather than a class name -- but only while it
    names exactly one part. ``the screw that fastens the PSU`` is a referring
    expression; with a second such screw it is a trap.

    ``avoid_ordinal`` is the leak rule: "Motherboard screw 6" tells a model that
    this machine has at least six motherboard screws, and if the same frame asks
    "how many motherboard screws are still fastened?" that is a large part of
    the other answer. Then only a relational phrase will do, and if there is
    none, the instance is not asked about here.
    """
    rec = tc.ctx.instances.get(instance)
    label = tc.label(instance)
    if rec is None:
        return (label, "label") if not avoid_ordinal else None
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
    return (label, "label") if not avoid_ordinal else None


def gen_v8(tc: TaskCtx, step: int) -> Iterator[dict]:
    frame = tc.frame(step)
    if frame is None or not frame.verified:
        return
    counted = set(counted_roles(tc, frame))
    seed = tc.seed(step, "V8")
    candidates = [k for k in sorted(frame.pointable)
                  if tc.ctx.cls_of(k) != CHASSIS_CLASS]
    for instance in stable_order(candidates, seed)[:V8_BUDGET]:
        rec = tc.ctx.instances.get(instance)
        avoid = bool(rec is not None and rec.cls == "screw"
                     and str(rec.attrs.get("role") or "other") in counted)
        phrase_family = _referring(tc, instance, avoid)
        if phrase_family is None:
            continue
        phrase, family = phrase_family
        row, box = frame.pointable[instance]
        rec_id = f"V8-{frame_id(tc.desktop, tc.view, step)}-{instance}"
        index, template = pick(V8_QUESTIONS, rec_id)
        tiny = row.get("visibility") == "visible_tiny"
        yield record(
            tc, "V8", rec_id, step, [frame],
            template.format(phrase=phrase), index,
            {"instance": instance, "bbox": box},
            {"type": "boxes", "field": None, "id_field": "instance",
             "box_field": "bbox", "iou_threshold": 0.5,
             "geometry": "visible_mask_bbox",
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
    if unresolved_action_targets(tc.ctx):
        return  # an exhaustive metric cannot be asked of an incomplete log (C3)
    # counted on the *log*, not on this view's frames: a missing scanner frame
    # (spec 4.2) does not mean the action never happened, and "how much is
    # left" must be the same number in all four views
    done = [a for s in tc.log_steps if s <= step for a in tc.named_actions(s)]
    total = sum(len(tc.named_actions(s)) for s in tc.log_steps)
    if not total:
        return
    # `done` is a *set* (spec 8.2's metric for V10 is set F1), so a chassis
    # reoriented seven times is one thing that has happened -- while the
    # remainder and the progress bin count the log's actions, because "how many
    # steps are left" is not a question about distinct pairs
    pairs = sorted({(a.verb, a.target) for a in done})
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
        tc, "V10", rec_id, step, [frame], question, index,
        {"done": [{"verb": verb, "target": target} for verb, target in pairs],
         "remaining_actions": total - len(done), "progress_bin": bin_name},
        {"type": "history", "bins": list(PROGRESS_BINS), "total_actions": total,
         "done_actions": len(done), "metric": "set_f1+bin_accuracy"},
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
    if not (frame.verified and before.verified) or step not in tc.v12_steps:
        return
    kind = tc.ctx.step_type(step)
    rec = tc.ctx.steps.get(step)
    if kind == StepType.DUPLI.value or (rec is not None and rec.dupli):
        changed, events = False, []
    elif kind in NO_CHANGE_STEP_TYPES or kind == StepType.FAILED.value:
        return
    else:
        old, new = tc.state(step - 1), tc.state(step)
        events = [{"target": key, "old": old[key].state, "new": new[key].state}
                  for key in sorted(old.keys() & new.keys())
                  if old[key].state != new[key].state]
        if not events:
            # a `reorient` or an `auxiliary` step: the picture changed a great
            # deal and the machine did not, which is neither answer
            return
        changed = True
    rec_id = f"V12-{frame_id(tc.desktop, tc.view, step)}"
    index, question = pick(V12_QUESTIONS, rec_id)
    touched = [e["target"] for e in events]
    yield record(
        tc, "V12", rec_id, step, [before, frame], question, index,
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
    "Can this image decide the state of {label}? If it can, give the state.",
    "Does this single view settle the state of {label}? Say so, or give it.",
)


def gen_v14(tc: TaskCtx, step: int) -> Iterator[dict]:
    """The abstain task: an instance this view cannot see has no state to report.

    Spec 8.2 principle 5. The truth is the compiled ``visibility`` and nothing
    else: ``occluded_full`` and ``out_of_view`` -- and ``too_small`` /
    ``motion_blur``, which are the same admission in other words -- mean the
    honest answer is "not from here".

    Both halves come off the same frame in equal numbers, drawn by the frame's
    own seed, so the majority-class baseline is 50 % by construction and a frame
    with nothing unanswerable contributes nothing rather than a run of easy
    yeses.
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
    take = min(len(blind), len(seen), tc.budget)
    if not take:
        return
    chosen = (stable_order(blind, seed, lambda it: it[0])[:take]
              + stable_order(seen, seed + "|pos", lambda it: it[0])[:take])
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
            tc, "V14", rec_id, step, [frame],
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
#: How far away the negative half of the same-moment question may be taken from,
#: nearest first: a neighbouring step is the hardest honest negative there is.
V15_MOMENT_OFFSETS = (2, -2, 1, -1, 3, -3)


def gen_v15(tc: TaskCtx, step: int) -> Iterator[dict]:
    """Cross-view consistency, built from per-view ``visibility`` and nothing else.

    Spec v1.5 removed cross-view geometry from the design: there is no
    homography and no projection, so this task may not use one. What is left is
    exactly what the annotation does carry -- the *same instance*, at the *same
    logical step*, with a visibility of its own in each view -- and that is
    enough for both halves of the spec's V15: which view can answer, and whether
    two frames are the same moment at all.
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
            tc, "V15", rec_id, step, [frame, mate],
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
    """Same moment or not, one of each.

    The negative is a nearby frame of the other view whose folded state really
    **differs** from this one. Two steps apart is not enough on its own: a
    ``dupli`` pair or a pair of ``reorient`` steps leaves the machine in exactly
    the same state, and "no" would then be the wrong answer to "do these show
    the same disassembly state?".
    """
    shared = sorted(set(frame.rows) & set(frames[step].rows))
    if not shared:
        return
    here = tc.state(step)
    candidates: list[tuple[bool, FrameData]] = [(True, frames[step])]
    for delta in V15_MOMENT_OFFSETS:
        mate = frames.get(step + delta)
        if mate is None or not mate.verified or mate.step == step:
            continue
        if tc.state(mate.step) == here:
            continue  # the same state at another step: "no" would be a lie
        candidates.append((False, mate))
        break
    if len(candidates) == 1:
        return  # no honest negative near this frame: skip the pair entirely
    for same, mate in candidates:
        rec_id = (f"V15m-{frame_id(tc.desktop, tc.view, step)}-{other}"
                  f"-s{mate.step:03d}")
        index, question = pick(V15_MOMENT_QUESTIONS, rec_id)
        yield record(
            tc, "V15", rec_id, step, [frame, mate], question, index,
            {"same_moment": same},
            {"type": "exact", "derive": "same_moment", "same_moment": same,
             "fields": ["same_moment"], "other_view": other,
             "other_step": mate.step},
            evidence({}, tc, {k: {} for k in shared[:tc.budget]}),
            R.rationale([R.compare_frames([step, mate.step], other)]),
            frame.verified and mate.verified, views=[tc.view, other],
            negative=None if same else "different_moment",
        )


# --------------------------------------------------------------------------- #
# V16 -- is this plan legal, and where does it first go wrong
# --------------------------------------------------------------------------- #
V16_QUESTIONS = (
    "Here is a proposed continuation of the teardown. Can it be carried out as "
    "written? If not, which step is the first that cannot be?",
    "Check this plan against the machine's current state: is every step "
    "possible in this order, and where does it first break?",
    "Verify the following sequence. Say whether it can be carried out and, if "
    "not, the index of the first impossible action.",
)
#: How many logged actions a candidate plan is made of.
#:
#: Five, so that the first error can land on index 0, 1, 2 or 3. It can never
#: land on the **last** index of a rearranged plan, and that is a fact about
#: disassembly rather than a gap in the corruption search: taking a part out
#: only ever *clears* preconditions, so an action moved later is never more
#: blocked than it was, and the step that breaks is always one that has been
#: pulled forward. The reachable range is therefore ``0 .. len(plan) - 2``.
V16_PLAN_LEN = 5


def _simulate(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]]
              ) -> tuple[Optional[int], list[Edge], Optional[str]]:
    """Replay ``plan`` from the state after ``step``; ``(first bad index, edge)``.

    Through :func:`unmet_for`, so this asks the constraint graph exactly the
    question V4, V5 and V6 ask it. A private re-implementation used to let V16
    disagree with them about a ``cable:*`` blocker that is not in the snapshot
    (satisfied here, unmet there) and about an edge type nobody recognises
    (gating nothing here, everything there).
    """
    sim = _copy_state(tc.state(step))
    for index, (verb, target) in enumerate(plan):
        rec = tc.ctx.instances.get(target)
        # a `cable:*` node is a real step of the teardown with no instance row:
        # releasing a loom is what makes the part behind it reachable, and a
        # plan that leaves it out is a plan that cannot be carried out
        cls = rec.cls if rec is not None else (
            CABLE_CLASS if target.startswith(CABLE_PREFIX) else None)
        if cls is None:
            return index, [], "not_present"
        attrs = rec.attrs if rec is not None else {}
        bad = unmet_for(tc, verb, target, sim)
        if bad:
            # every edge this step breaks, not the first one the list happened
            # to hold: "which constraint stopped it" has no single answer when
            # a part is both screwed down and still plugged in
            return (index, sorted(bad, key=lambda e: (e.type, e.target, e.blocker)),
                    "violates_edge")
        current = (sim[target].state if target in sim
                   else tc.tax.default_state(cls))
        if not verb_applies(tc.tax, cls, attrs, verb, current):
            return index, [], inapplicable_reason(sim, target)
        _apply_effect(tc.tax, tc.ctx.instances, sim, target, verb)
    return None, [], None


def _state_before(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]],
                  index: int) -> FrameState:
    """The simulated state a plan reaches just before its ``index``-th action."""
    sim = _copy_state(tc.state(step))
    for verb, target in plan[:index]:
        _apply_effect(tc.tax, tc.ctx.instances, sim, target, verb)
    return sim


def _corruptions(plan: Sequence[tuple[str, str]], seed: str) -> list[tuple[str, str]]:
    """Every corruption of ``plan`` worth trying, as ``(family, tag)`` descriptors.

    Three families. Adjacent **swaps** put the first error at ``i``; **moves**
    put it where the moved action lands. Neither can ever break the *last* step
    of a valid plan -- disassembly only clears preconditions, so an action
    pushed later is never more blocked -- which is why the third family exists:
    a **substitute** replaces one step with an action that is certainly wrong at
    exactly that point, so the error can land anywhere, the last index included.

    The descriptors are materialised one at a time (a substitute has to look at
    the simulated state), in a seeded order, so the three families are mixed.
    """
    out: list[tuple[str, str]] = []
    for i in range(len(plan) - 1):
        out.append(("swap", f"swap{i}"))
    for j in range(len(plan)):
        for i in range(len(plan)):
            if abs(i - j) < 2:
                continue  # that is the adjacent swap, already listed
            out.append(("move", f"move{j}to{i}"))
    for i in range(len(plan)):
        out.append(("substitute", f"substitute{i}"))
    return stable_order(out, seed, lambda it: it[1])


def _rearranged(plan: Sequence[tuple[str, str]], tag: str) -> list[tuple[str, str]]:
    """Apply a ``swap``/``move`` tag to a plan."""
    out = list(plan)
    if tag.startswith("swap"):
        i = int(tag[len("swap"):])
        out[i], out[i + 1] = out[i + 1], out[i]
        return out
    j, i = (int(part) for part in tag[len("move"):].split("to"))
    out.insert(i, out.pop(j))
    return out


def _substituted(tc: TaskCtx, step: int, plan: Sequence[tuple[str, str]],
                 index: int, seed: str
                 ) -> Optional[list[tuple[str, str]]]:
    """Replace ``plan[index]`` with an action that cannot be done at that point.

    Same verb wherever one is available, so the corrupted plan still reads like
    a plan and the difference is a fact about the machine rather than about the
    wording.
    """
    state = _state_before(tc, step, plan, index)
    verb, target = plan[index]
    candidates = [c for c in [*blocked_in(tc, state),
                              *inapplicable_in(tc, state, step)]
                  if (c.verb, c.target) not in set(plan)]
    ordered = tc.matched(candidates, verb, tc.ctx.cls_of(target),
                         f"{seed}|substitute{index}")
    if not ordered:
        return None
    chosen = ordered[0]
    out = list(plan)
    out[index] = (chosen.verb, chosen.target)
    return out


def _plan_record(tc: TaskCtx, frame: FrameData, plan: Sequence[tuple[str, str]],
                 first_bad: Optional[int], edges: Sequence[Edge], tag: str,
                 family: Optional[str] = None,
                 reason: Optional[str] = None) -> dict:
    step = frame.step
    rec_id = f"V16-{frame_id(tc.desktop, tc.view, step)}-{tag}"
    index, question = pick(V16_QUESTIONS, rec_id)
    listed = [{"verb": v, "target": t} for v, t in plan]
    state = tc.state(step)
    chain: list[dict] = []
    for edge in edges:
        current = state[edge.blocker].state if edge.blocker in state else None
        chain.append(R.recall_relation(edge.type, edge.target, edge.blocker,
                                       edge.necessity, edge.mode))
        chain.append(observed(edge.blocker, current, tc, frame.pointable))
        chain.append(R.check_precondition(edge.blocker,
                                          REQUIRED_STATES.get(edge.type, frozenset()),
                                          current, False))
    if not edges:
        chain.extend(observed(t, state[t].state if t in state else None,
                              tc, frame.pointable) for _, t in plan[:tc.budget])
    labels = [e.label() for e in edges]
    source = (DEMONSTRATED if first_bad is None
              else GRAPH_BLOCKED if edges else STATE_INAPPLICABLE)
    check = {"type": "plan", "plan": listed, "metric": "accuracy+localisation",
             "truth_source": source}
    if family is not None:
        check["corruption"] = family
        check["corruption_tag"] = tag
    return record(
        tc, "V16", rec_id, step, [frame],
        question + "\n" + "\n".join(
            f"{i + 1}. {verb_phrase(v, question_label(tc, t, frame))}"
            for i, (v, t) in enumerate(plan)),
        index,
        {"valid": first_bad is None, "first_error_index": first_bad,
         "violated_edge": labels[0] if labels else None,
         "violated_edges": labels, "reason": reason},
        check,
        evidence({t: frame.pointable[t][1] for _, t in plan if t in frame.pointable},
                 tc, plan_length=len(listed)),
        R.rationale(chain), frame.verified,
        negative=None if first_bad is None else family,
        truth_source=source,
    )


def gen_v16(tc: TaskCtx, step: int) -> Iterator[dict]:
    """A demonstrated continuation and, where one exists, a corruption of it.

    The valid plan is the logged suffix and nothing else: a rearrangement the
    graph happens to permit is not known to work, because the graph is an upper
    bound. The invalid one is a rearrangement that violates a required edge --
    certainly impossible, and the edge says why. A corruption that violates
    nothing is **not** emitted: it is a different plan of unknown validity, and
    calling it an error would teach exactly the wrong lesson.
    """
    frame = tc.frame(step)
    if frame is None or tc.excluded:
        return
    plan: list[tuple[str, str]] = []
    for candidate in tc.log_steps:
        if candidate <= step:
            continue
        if candidate in tc.illegal:
            return  # a suffix built on a contradiction is not a plan
        for action in tc.named_actions(candidate):
            # a plan is a sequence of disassembly actions; `reorient` is a
            # capture action with no state effect and nothing gating it. A
            # `cable:*` node has no instance row and belongs in the plan all the
            # same -- releasing a loom is what frees the part behind it.
            known = (action.target in tc.ctx.instances
                     or action.target.startswith(CABLE_PREFIX))
            if known and tc.changes_state(action):
                plan.append((action.verb, action.target))
        if len(plan) >= V16_PLAN_LEN:
            break
    plan = plan[:V16_PLAN_LEN]
    if len(plan) < 2:
        return
    first_bad, _, _ = _simulate(tc, step, plan)
    if first_bad is not None:
        return  # the logged suffix does not replay cleanly: say nothing

    seed = tc.seed(step, "V16")
    corruption = None
    for family, tag in _corruptions(plan, seed):
        if family == "substitute":
            corrupted = _substituted(tc, step, plan, int(tag[len("substitute"):]),
                                     seed)
            if corrupted is None:
                continue
        else:
            corrupted = _rearranged(plan, tag)
        bad_at, broken, reason = _simulate(tc, step, corrupted)
        if bad_at is None:
            continue
        if family != "substitute" and not broken:
            # a rearrangement that violates nothing is a different plan of
            # unknown validity, not a wrong one (spec 8.2 principle 6)
            continue
        corruption = (family, tag, corrupted, bad_at, broken, reason)
        break
    if corruption is None:
        return  # nothing about this suffix is certainly wrong: say nothing
    yield _plan_record(tc, frame, plan, None, (), "true")
    family, tag, corrupted, bad_at, broken, reason = corruption
    yield _plan_record(tc, frame, corrupted, bad_at, broken, tag, family, reason)


#: ``task id -> generator``. The order is the order records are written.
GENERATORS: dict[str, Callable[[TaskCtx, int], Iterator[dict]]] = {
    "V1": gen_v1, "V2": gen_v2, "V3": gen_v3, "V4": gen_v4, "V5": gen_v5,
    "V6": gen_v6, "V8": gen_v8, "V10": gen_v10, "V12": gen_v12, "V14": gen_v14,
    "V15": gen_v15, "V16": gen_v16,
}
