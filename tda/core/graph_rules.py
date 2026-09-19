"""The constraint graph's vocabulary, and the rules that derive its edges.

The spec 7.2 semantics, the :class:`Edge` type, the verb-applicability tables
and the spec 7.3 attribute rules. :mod:`tda.core.graph` (the reasoning layer),
:mod:`tda.core.graph_plan` (the planner) and :mod:`tda.core.graph_templates`
(the family templates) all import from here, which is what keeps the modules
acyclic; :mod:`tda.core.graph` re-exports everything a caller needs.

One function matters here: :func:`propose_edges`, the spec 7.3 attribute rules
(~90% coverage). Everything in this module is pure.

Below it sits :mod:`tda.core.graph_infer`: which instance a name means, and the
heuristics that fill the relational fields those rules read
(:func:`infer_relational_fields`, :func:`unresolved_relations`). They are
re-exported here so the import path callers already use keeps working, and the
dependency runs one way only -- rules import inference, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from tda.core.graph_infer import (
    AMBIGUOUS,
    LS_PREFIX,
    NO_CANDIDATE,
    SCREW_ROLE_CLASSES,
    UNRESOLVED,
    infer_relational_fields,
    is_provisional,
    real_instances as _real_instances,
    resolve_ref as _resolve,
    unique_of_class as _unique_of_class,
    unresolved_kind,
    unresolved_relations,
)
from tda.core.model import InstanceRec
from tda.core.states import FrameState
from tda.core.taxonomy import Taxonomy

__all__ = [
    "AMBIGUOUS",
    "CABLE_DEFAULT_STATE",
    "CABLE_PREFIX",
    "LS_PREFIX",
    "NO_CANDIDATE",
    "SCREW_ROLE_CLASSES",
    "UNRESOLVED",
    "Edge",
    "GATED_VERBS",
    "GATES",
    "HARD_TYPES",
    "REQUIRED_STATES",
    "VerbTarget",
    "gated_verbs",
    "active_edges",
    "blocker_state",
    "cable_nodes",
    "cable_owner",
    "connector_owner",
    "infer_relational_fields",
    "is_provisional",
    "propose_edges",
    "unresolved_fan_owners",
    "unresolved_kind",
    "unresolved_relations",
    "verb_applies",
    "verb_effect",
]

#: The five hard-constraint edge types of spec 7.2, most-specific first.
HARD_TYPES = ("fastened_by", "connected_to", "locked_by", "covered_by", "blocked_by")

#: Spec 7.2: the blocker must be in one of these states before the target of an
#: edge of this type may be acted on.
REQUIRED_STATES: dict[str, frozenset[str]] = {
    "fastened_by": frozenset({"loosened", "removed"}),
    "connected_to": frozenset({"unplugged", "removed"}),
    "locked_by": frozenset({"open"}),
    "covered_by": frozenset({"open", "removed", "displaced"}),
    "blocked_by": frozenset({"removed", "displaced", "released"}),
}

#: ``blocked_by`` is the only type that carries a mode (spec 7.1).
BLOCKED_MODES = ("physical_path", "tool_access", "cable_tension")

#: Verbs a hard constraint can block at all (spec 7.2). ``reorient`` is a
#: capture action on the chassis, not a disassembly step, so nothing gates it.
GATED_VERBS = frozenset({"remove", "displace", "open", "unscrew", "disconnect", "release"})

#: Which of those verbs each edge type actually gates. One table, read by
#: :func:`tda.core.graph.applicable_preconditions` (and so by ``legal_actions``
#: and ``validate_sequence``) *and* by :mod:`tda.core.graph_plan`, so the
#: checker and the planner cannot drift apart.
#:
#: Every type used to gate every verb, which said a slim-case PSU could not be
#: swung out of the way until its whole harness was unplugged -- backwards, as
#: swinging it out is how you reach the plugs. Across the 66 sheets that alone
#: produced 34 ``displace psu.01 violates connected_to(...)`` lines, every one
#: of them describing correct work. The distinction is *moving the part* versus
#: *reaching it*:
#:
#: * ``fastened_by`` / ``locked_by`` -- a screwed-down or latched part does not
#:   move, but you can still work on what is plugged into it;
#: * ``covered_by`` / ``blocked_by`` -- no access at all, so every verb waits
#:   (``blocked_by`` by its ``mode``, as before);
#: * ``connected_to`` -- you may displace, open or unscrew a part whose cables
#:   are still plugged; you may not take it away.
GATES: dict[str, frozenset[str]] = {
    "fastened_by": frozenset({"remove", "displace", "open"}),
    "locked_by": frozenset({"remove", "displace", "open"}),
    "covered_by": GATED_VERBS,
    "blocked_by": GATED_VERBS,
    "connected_to": frozenset({"remove"}),
}


def gated_verbs(edge_type: str) -> frozenset[str]:
    """Which verbs an edge of this type gates; an unknown type gates them all.

    Gating too much is noisy and visible; gating too little silently drops a
    constraint out of the ground truth, so an unrecognised type errs the loud way.
    """
    return GATES.get(edge_type, GATED_VERBS)

CABLE_PREFIX = "cable:"
REMOVED = "removed"
REJECTED = "rejected"

#: The default state of a ``cable:*`` node, mirroring ``taxonomy.yaml``. A
#: cable enters the frame state only once an event names it (see
#: :func:`tda.core.states.state_at`), so an edge may well point at one that is
#: not in the snapshot yet; it is still routed until something releases it.
CABLE_DEFAULT_STATE = "routed"

#: Which states a verb may be applied *from*, where the taxonomy's effect table
#: alone is too permissive. ``remove`` is handled by :data:`REMOVE_FROM_STATES`.
VERB_FROM_STATES: dict[str, frozenset[str]] = {
    "unscrew": frozenset({"fastened"}),
    "disconnect": frozenset({"plugged"}),
    "open": frozenset({"closed"}),
    "release": frozenset({"closed", "routed"}),
    "displace": frozenset({"installed"}),
}

#: Classes that need an intermediate step before ``remove``: a screw has to come
#: loose first. Every other class can be taken straight out of whatever state it
#: is in -- including a ``connector``, which spec 6.3 lets you remove while it
#: is still plugged, meaning the whole cable was pulled out in one go.
REMOVE_FROM_STATES: dict[str, frozenset[str]] = {
    "screw": frozenset({"loosened"}),
}

#: One action as the graph talks about it: ``(verb, target)``.
VerbTarget = tuple[str, str]


@dataclass(frozen=True)
class Edge:
    """One hard constraint: ``blocker`` gates every action on ``target``.

    ``necessity`` is ``required`` or ``recommended``; ``mode`` is only set on
    ``blocked_by``; ``source`` is ``rule`` / ``template`` / ``manual``;
    ``status`` is ``proposed`` / ``accepted`` / ``rejected`` (spec 7.3).
    """

    type: str
    target: str
    blocker: str
    necessity: str = "required"
    mode: Optional[str] = None
    reason: str = ""
    source: str = "rule"
    evidence_step: Optional[int] = None
    status: str = "proposed"

    def label(self) -> str:
        """``fastened_by(motherboard.01, screw.motherboard.03)``."""
        return f"{self.type}({self.target}, {self.blocker})"


# --------------------------------------------------------------------------- #
# shared helpers: edge status, verb applicability, cable nodes
# --------------------------------------------------------------------------- #
def active_edges(edges: Iterable["Edge"]) -> list["Edge"]:
    """Edges a human has not rejected (spec 7.3: proposed / accepted / rejected)."""
    return [e for e in edges if e.status != REJECTED]


def verb_effect(tax: Taxonomy, cls: str, attrs: dict, verb: str) -> Optional[str]:
    """The state this verb would put the class in, or ``None`` for no effect."""
    effect = tax.apply_verb(cls, attrs, verb)
    if effect is None or effect[0] != "state":
        return None
    return effect[1]


def verb_applies(tax: Taxonomy, cls: str, attrs: dict, verb: str, current: str) -> bool:
    """Can this verb be performed on an instance of ``cls`` in state ``current``?

    Independent of the constraint graph: it only asks whether the verb belongs
    to the class (spec 6.3), does something, and starts from a sane state.
    """
    new = verb_effect(tax, cls, attrs, verb)
    if new is None or current == REMOVED or new == current:
        return False
    allowed = REMOVE_FROM_STATES.get(cls) if verb == "remove" else VERB_FROM_STATES.get(verb)
    return allowed is None or current in allowed


def blocker_state(state: FrameState, blocker: str) -> Optional[str]:
    """A node's state, or ``None`` when it is not a node of this desktop."""
    inst = state.get(blocker)
    if inst is not None:
        return inst.state
    if blocker.startswith(CABLE_PREFIX):
        return CABLE_DEFAULT_STATE
    return None


def cable_nodes(edges: list["Edge"], state: FrameState) -> dict[str, str]:
    """Every virtual cable node this graph mentions, with its current state."""
    keys = {e.blocker for e in edges if e.blocker.startswith(CABLE_PREFIX)}
    keys |= {k for k in state if k.startswith(CABLE_PREFIX)}
    return {k: blocker_state(state, k) or CABLE_DEFAULT_STATE for k in keys}


# --------------------------------------------------------------------------- #
# cables
# --------------------------------------------------------------------------- #
#: ``cable:<owner>`` tokens the logs use that are not taxonomy class names.
CABLE_OWNER_ALIASES = {
    "cpu_fan": "cpu_cooler",
    "cooler_fan": "cpu_cooler",
    "hdd": "storage_drive",
    "ssd": "storage_drive",
    "drive": "storage_drive",
    "optical": "optical_drive",
}

#: Classes whose cables are *not* captive to them, however ``logs.py`` groups
#: them. A drive's SATA leads unplug at both ends, so spec 7.2 says each plug
#: gates only the part it sits on -- unlike a fan lead or a front-panel loom,
#: which are moulded into their part and gate it wherever they plug in.
DETACHABLE_CABLE_OWNERS = frozenset({"storage_drive", "optical_drive"})

#: Connector kinds that are power leads: whatever ``taxonomy_map`` tagged them
#: with, the far end of one is moulded into the PSU, so the PSU harness owns it
#: (spec 7.2: every plug of the harness must be out before the PSU can go).
POWER_LEAD_KINDS = frozenset({"atx_24pin", "cpu_power", "sata_power", "molex"})

#: Connector kinds with a detachable plug at *both* ends, owned by nobody.
TWO_ENDED_KINDS = frozenset({"sata_data"})

#: The class a fan lead can be captive to.
COOLER_CLASS = "cpu_cooler"

#: ``cpu_cooler.kind`` values that actually carry a fan. ``taxonomy.yaml`` lists
#: ``fan`` / ``heatsink`` / ``heatsink_fan``, and a bare ``heatsink`` is a block
#: of metal: nothing is moulded into it, so no fan lead can belong to it.
FAN_CAPABLE_COOLER_KINDS = frozenset({"fan", "heatsink_fan"})


def cable_owner(
    cable_id: str,
    instances: Optional[dict[str, InstanceRec]] = None,
) -> Optional[str]:
    """The part a virtual cable node belongs to (spec 3.1 ``cable.owner``).

    Without ``instances`` this answers the owner *class* (``"cable:psu"`` ->
    ``"psu"``); with them it resolves that class to its instance key, but only
    when the desktop has exactly one instance of it -- an ambiguous or absent
    owner yields ``None`` rather than a guess. A key that is not a cable node
    yields ``None`` either way.

    Note this is the *cable's* owner, in the spec-7.1 sense of "the cable is
    permanently attached to this part", which is narrower than the grouping key
    ``logs.py`` writes: see :data:`DETACHABLE_CABLE_OWNERS`.
    """
    if not cable_id.startswith(CABLE_PREFIX):
        return None
    owner = cable_id[len(CABLE_PREFIX):].strip()
    if not owner:
        return None
    owner = CABLE_OWNER_ALIASES.get(owner, owner)
    if owner in DETACHABLE_CABLE_OWNERS:
        return None
    if instances is None:
        return owner
    return _unique_of_class(instances, owner)


def connector_owner(
    rec: InstanceRec,
    instances: dict[str, InstanceRec],
) -> Optional[str]:
    """The part one connector's cable is captive to, or ``None``.

    The connector's ``kind`` decides first, because it is the physical fact and
    ``logs.py``'s ``cable_owner`` is only a grouping key:

    * a power lead (:data:`POWER_LEAD_KINDS`) belongs to the PSU harness, even
      when the sheet filed it under the drive it feeds -- so the PSU cannot come
      out until every one of its plugs is pulled;
    * a ``sata_data`` lead has a detachable plug at both ends and belongs to
      nobody, so each end gates only its own socket host (spec 7.2);
    * anything else falls back to the tagged cable owner, which still drops the
      drive classes of :data:`DETACHABLE_CABLE_OWNERS` -- an untyped connector
      filed under a drive is one of its SATA leads -- and, for a fan lead, a
      cooler that cannot carry a fan (see :func:`owns_fan_lead`).
    """
    kind = str(rec.attrs.get("kind") or "")
    if kind in POWER_LEAD_KINDS:
        return _unique_of_class(instances, "psu")
    if kind in TWO_ENDED_KINDS:
        return None
    owner = cable_owner(rec.cable or "", instances)
    if owner and not owns_fan_lead(instances.get(owner)):
        return None
    return owner


def owns_fan_lead(owner: Optional[InstanceRec]) -> bool:
    """Can this instance be the far end of a cooler fan lead?

    Anything that is not a cooler at all is none of this rule's business and
    answers ``True``; a cooler answers whether its ``kind`` has a fan
    (:data:`FAN_CAPABLE_COOLER_KINDS`).

    ``cable:cpu_fan`` used to resolve onto "the one ``cpu_cooler`` of this
    desktop" and stop there. On twelve machines that one instance is a bare
    ``kind=heatsink`` -- the sheet operated the heatsink and never the fan, so
    the fan was never instantiated -- and the rule bound the fan lead to a part
    with no fan, producing fifteen ``connected_to`` edges that said the heatsink
    could not come out until a connector that is not on it was unplugged, and
    eleven of the thirty sequence violations across the dataset. A missing
    instance is a question for stage S1 (:func:`unresolved_fan_owners`), not
    something to answer with the nearest part of roughly the right class.
    """
    if owner is None or owner.cls != COOLER_CLASS:
        return True
    return str(owner.attrs.get("kind") or "") in FAN_CAPABLE_COOLER_KINDS


def unresolved_fan_owners(instances: dict[str, InstanceRec]) -> list[str]:
    """One line per fan lead whose cooler cannot carry a fan, for stage S1.

    The connector keeps its ``socket_host`` edge -- the plug really is in that
    board -- so this is not a lost constraint but a missing *instance*: the
    desktop has a fan, the log never named it, and only a human can say whether
    to create one. Sorted, so a re-run of the same desktop reports the same
    thing in the same order.
    """
    out: list[str] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector" or is_provisional(key):
            continue
        kind = str(rec.attrs.get("kind") or "")
        if kind in POWER_LEAD_KINDS or kind in TWO_ENDED_KINDS:
            continue
        owner_key = cable_owner(rec.cable or "", instances)
        owner = instances.get(owner_key or "")
        if owner_key and not owns_fan_lead(owner):
            out.append(
                f"unresolved fan owner: {key} is a cooler fan lead, but the only "
                f"cpu_cooler on this desktop is {owner_key} "
                f"(kind={str(owner.attrs.get('kind') or '')!r}), which has no fan. "
                f"No owner edge was proposed; create the fan instance if the "
                f"machine has one."
            )
    return out


# --------------------------------------------------------------------------- #
# propose_edges (spec 7.3 item 1)
# --------------------------------------------------------------------------- #
#: Latch classes whose ``of`` attribute names what they lock.
LATCH_OF_CLASSES = ("ram_latch", "drive_latch", "card_latch", "cooler_latch")

#: Latch classes that lock a fixed part class (the spec-7.3 template edges).
LATCH_TEMPLATE_HOSTS = {"cpu_socket_lever": "cpu", "psu_latch": "psu"}

#: ``cover.of`` -> what the cover hides. ``("cls", ...)`` matches instance
#: classes, ``("role", ...)`` matches ``screw.role``. ``front_bezel`` and
#: ``other`` name no part, so they are left to the manual panel.
COVER_TARGETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "cpu_cooler": ("cls", ("cpu_cooler",)),
    "motherboard_screws": ("role", ("motherboard",)),
    "ram": ("cls", ("ram_module",)),
    "expansion_slot": ("cls", ("expansion_card",)),
    "drive": ("cls", ("storage_drive", "optical_drive")),
}

#: Fixed template edges of spec 7.3: the cooler always sits on the CPU.
FIXED_COVERED_BY = (("cpu", "cpu_cooler"),)


def propose_edges(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """Derive every hard edge the recorded attributes imply (spec 7.3 item 1).

    Rules, in order: ``screw.fastens`` -> ``fastened_by``; the connector rule of
    spec 7.2 -> ``connected_to``; ``latch.of`` plus the fixed CPU-lever and
    PSU-latch edges -> ``locked_by``; ``cover.of`` plus the fixed cooler-over-CPU
    edge -> ``covered_by``. ``cable_clip.holds`` yields no edge on its own -- a
    clipped cable only blocks a target once a human says which one, so those
    ``blocked_by`` edges stay manual (spec 7.3 item 3).

    Every edge comes back ``necessity="required"``, ``source="rule"``,
    ``status="proposed"``; duplicates are collapsed, keeping the first reason.
    Instances whose relational fields are still empty simply derive nothing --
    run :func:`infer_relational_fields` first to fill the obvious ones.
    """
    out: list[Edge] = []
    for rule in (_fastened_by, _connected_to, _locked_by, _covered_by):
        out.extend(rule(instances, tax))
    return _dedupe(out)


def _dedupe(edges: list[Edge]) -> list[Edge]:
    """Drop repeated ``(type, target, blocker)`` triples, keeping the first."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Edge] = []
    for edge in edges:
        key = (edge.type, edge.target, edge.blocker)
        if key in seen or edge.target == edge.blocker:
            continue
        seen.add(key)
        out.append(edge)
    return out


def _edge(etype: str, target: str, blocker: str, reason: str) -> Edge:
    return Edge(type=etype, target=target, blocker=blocker, reason=reason, source="rule")


def _fastened_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``screw.fastens = X`` -> ``fastened_by(X, screw)`` (spec 7.1)."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "screw":
            continue
        target = _resolve(instances, rec.fastens)
        if target:
            out.append(_edge("fastened_by", target, key, f"{key} fastens {target}"))
    return out


def _connected_to(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """The connector rule of spec 7.2.

    For a part X, every connector whose ``socket_host`` is X *or* whose cable is
    owned by X must be unplugged before X may be removed. The two ends of a SATA
    data cable carry no owner, so each gates only the part it plugs into; every
    plug of the PSU harness gates the PSU as well as its own socket host. See
    :func:`connector_owner` for how the owner is decided.
    """
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "connector":
            continue
        host = _resolve(instances, rec.socket_host)
        if host:
            out.append(_edge("connected_to", host, key, f"{key} plugs into {host}"))
        owner = connector_owner(rec, instances)
        if owner and owner != host:
            out.append(
                _edge("connected_to", owner, key, f"{key} is on the {owner} cable")
            )
    return out


def _locked_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``latch.of`` -> ``locked_by``, plus the fixed CPU-lever / PSU-latch edges."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        target = None
        if rec.cls in LATCH_OF_CLASSES:
            target = _resolve(instances, rec.attrs.get("of"))
        elif rec.cls in LATCH_TEMPLATE_HOSTS:
            target = _unique_of_class(instances, LATCH_TEMPLATE_HOSTS[rec.cls])
        if target:
            out.append(_edge("locked_by", target, key, f"{key} locks {target}"))
    return out


def _covered_by(instances: dict[str, InstanceRec], tax: Taxonomy) -> list[Edge]:
    """``cover.of`` -> ``covered_by``, plus the fixed cooler-over-CPU edge."""
    out: list[Edge] = []
    for key, rec in sorted(instances.items()):
        if rec.cls != "cover":
            continue
        for target in _cover_targets(instances, rec):
            out.append(_edge("covered_by", target, key, f"{key} covers {target}"))
    for under_cls, over_cls in FIXED_COVERED_BY:
        under = _unique_of_class(instances, under_cls)
        over = _unique_of_class(instances, over_cls)
        if under and over:
            out.append(_edge("covered_by", under, over, f"{over} sits on {under}"))
    return out


def _cover_targets(instances: dict[str, InstanceRec], cover: InstanceRec) -> list[str]:
    """Which instances this cover hides, from its ``of`` attribute."""
    of = str(cover.attrs.get("of") or "")
    if not of:
        return []
    if of in instances:  # `of` already names one instance
        return [of]
    rule = COVER_TARGETS.get(of)
    if rule is None:
        return []
    kind, wanted = rule

    def hit(rec: InstanceRec) -> bool:
        if kind == "cls":
            return rec.cls in wanted
        return rec.cls == "screw" and rec.attrs.get("role") in wanted

    return [key for key, rec in sorted(instances.items()) if hit(rec)]
