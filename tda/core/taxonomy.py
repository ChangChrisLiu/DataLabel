"""Vocabulary layer: the taxonomy (spec 6.1-6.5) and raw step-name parsing (6.6).

Two configs back this module:

* ``configs/taxonomy.yaml``      -- classes, states, mask policy, verbs, tools.
* ``configs/taxonomy_map.yaml``  -- ordered regex rules mapping the annotators'
  free-text step names onto a class + attributes + the 53-group canonical label
  used for review.

Both are loaded lazily and cached, so repeated ``parse_raw_name`` calls over the
2,830-row step table stay cheap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"
TAXONOMY_MAP_PATH = REPO_ROOT / "configs" / "taxonomy_map.yaml"

FALLBACK_RULE = "fallback"


# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
@dataclass
class Taxonomy:
    """The unified vocabulary of spec section 6."""

    classes: dict[str, dict]
    verbs: dict[str, dict]
    tools: list[str]
    virtual_nodes: dict[str, dict] = field(default_factory=dict)
    on_bench_needs_geom: bool = True
    directions: list[str] = field(default_factory=list)

    # -- class queries ------------------------------------------------------
    def _defn(self, cls: str) -> dict:
        try:
            return self.classes[cls]
        except KeyError:
            pass
        try:
            return self.virtual_nodes[cls]
        except KeyError:
            raise KeyError(f"unknown taxonomy class: {cls!r}") from None

    def states_of(self, cls: str) -> list[str]:
        return list(self._defn(cls)["states"])

    def group_of(self, cls: str) -> str:
        """The class's group of spec 6.1 (``structure``, ``part``, ...), or ``""``.

        A rough statement of how deep in the machine a class sits, which is what
        lets a caller order instances the way they are physically stacked; an
        unknown class simply has no group rather than raising, since this is
        only ever used to sort.
        """
        try:
            return str(self._defn(cls).get("group") or "")
        except KeyError:
            return ""

    def default_state(self, cls: str) -> str:
        return self._defn(cls)["default_state"]

    def needs_mask(self, cls: str, state: str, placement: str) -> bool:
        """Does this instance need its own geometry in the given placement?

        ``on_bench``  -> yes for every class but ``chassis`` (which never goes
        on the bench) and the virtual nodes (spec 6.2: no mask, ever);
        ``elsewhere`` -> no (it is out of every view);
        ``in_chassis`` -> the per-class state table from ``taxonomy.yaml``.
        """
        if placement == "on_bench":
            if cls == "chassis" or cls in self.virtual_nodes:
                return False
            return bool(self.on_bench_needs_geom)
        if placement == "elsewhere":
            return False
        return bool(self._defn(cls)["needs_mask"].get(state, False))

    # -- verbs --------------------------------------------------------------
    def apply_verb(self, cls: str, attrs: dict, verb: str) -> Optional[tuple[str, str]]:
        """Return ``(attr, new_value)`` or ``None`` when nothing changes.

        ``None`` covers both "this verb does not apply to this class" and
        "this verb changes no state" (``reorient`` only breaks the pose).
        """
        spec = self.verbs.get(verb)
        if spec is None or cls not in spec["applies_to"]:
            return None
        effect = spec.get("effect")
        if not effect:
            return None
        attr = effect["attr"]
        if "to" in effect:
            return (attr, effect["to"])
        if "by_class" in effect:
            value = effect["by_class"].get(cls)
            return (attr, value) if value else None
        cond = effect["conditional"]
        chosen = cond["if_true"] if attrs.get(cond["if_attr"]) else cond["if_false"]
        return (attr, chosen)


@lru_cache(maxsize=None)
def _load_taxonomy_cached(path: str) -> Taxonomy:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return Taxonomy(
        classes=cfg["classes"],
        verbs=cfg["verbs"],
        tools=list(cfg["tools"]),
        virtual_nodes=cfg.get("virtual_nodes", {}),
        on_bench_needs_geom=bool(cfg.get("on_bench_needs_geom", True)),
        directions=list(cfg.get("directions", [])),
    )


def load_taxonomy(path: str | Path = TAXONOMY_PATH) -> Taxonomy:
    """Load (and cache) ``configs/taxonomy.yaml``."""
    return _load_taxonomy_cached(str(Path(path).resolve()))


# --------------------------------------------------------------------------- #
# Raw-name parsing
# --------------------------------------------------------------------------- #
@dataclass
class ParsedTarget:
    """What one raw step name says about its target."""

    cls: str = ""                         # taxonomy class, "" when none is named
    attrs: dict = field(default_factory=dict)
    instance_no: Optional[int] = None
    virtual: Optional[str] = None         # e.g. "cable" for cable-routing steps
    verb: Optional[str] = None
    multi: bool = False                   # names several targets at once
    attempt: bool = False                 # "try to ..." / "failed"
    step_type_hint: Optional[str] = None
    canon_group: str = ""
    matched_rule: str = FALLBACK_RULE


@dataclass(frozen=True)
class _RuleSet:
    rules: tuple[tuple[re.Pattern, dict], ...]
    typos: tuple[tuple[re.Pattern, str], ...]
    verb_prefixes: tuple[tuple[re.Pattern, str], ...]
    verb_class_overrides: dict


@lru_cache(maxsize=None)
def _load_rules_cached(path: str) -> _RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _RuleSet(
        rules=tuple(
            (re.compile(r["pattern"], re.IGNORECASE), r) for r in cfg["rules"]
        ),
        typos=tuple(
            (re.compile(re.escape(bad), re.IGNORECASE), good)
            for bad, good in (cfg.get("typos") or {}).items()
        ),
        verb_prefixes=tuple(
            (re.compile(p["pattern"], re.IGNORECASE), p["verb"])
            for p in (cfg.get("verb_prefixes") or [])
        ),
        verb_class_overrides=cfg.get("verb_class_overrides") or {},
    )


_WS = re.compile(r"\s+")
_DASH = re.compile(r"\s*-\s*")
_PAREN = re.compile(r"\(([^()]*)\)")
_BRACKET = re.compile(r"\[([^\[\]]*)\]")
_OPEN_BRACKET_TAIL = re.compile(r"[(\[](.*)$", re.DOTALL)
_CLOSERS = re.compile(r"[)\]]")
_ATTEMPT = re.compile(r"\btry to\b|\bfailed\b|\bfailure\b", re.IGNORECASE)
# "1&2&3", "1/2/3", "1, 2 and 3"
_NUM_RUN_SEP = re.compile(r"(\d+(?:\s*(?:[&/,]|and)\s*\d+)+)\s*$")
# "connector 1 2 3 4 5"
_NUM_RUN_SPACE = re.compile(r"(\d+(?:\s+\d+)+)\s*$")
_TRAILING_NUM = re.compile(r"(\d+)\s*$")
_ANY_NUM = re.compile(r"\d+")
# "2 Case - motherboard connector" = two connectors, not connector #2.
_LEADING_COUNT = re.compile(r"^\d+\s+\D")
_MULTI_WORDS = re.compile(r"\ball\b|\bhalf\b", re.IGNORECASE)
_MARKER_HINTS = frozenset({"initial", "dupli", "ignore"})


def _normalize(raw: str, typos: tuple[tuple[re.Pattern, str], ...]) -> str:
    s = raw.replace(" ", " ").strip().lower()
    for pattern, good in typos:
        s = pattern.sub(good, s)
    s = _DASH.sub(" - ", s)
    return _WS.sub(" ", s).strip()


def _split_qualifier(s: str) -> tuple[str, list[str]]:
    """Return (text without bracketed parts, the bracketed parts)."""
    quals: list[str] = []

    def _take(m: re.Match) -> str:
        text = m.group(1).strip()
        if text:
            quals.append(text)
        return " "

    stripped = _PAREN.sub(_take, s)
    stripped = _BRACKET.sub(_take, stripped)
    # An unbalanced opening bracket: everything after it is a qualifier tail.
    tail = _OPEN_BRACKET_TAIL.search(stripped)
    if tail:
        text = tail.group(1).strip()
        if text:
            quals.append(text)
        stripped = stripped[: tail.start()]
    stripped = _CLOSERS.sub(" ", stripped)
    return _WS.sub(" ", stripped).strip(), quals


def _read_instances(text: str) -> tuple[Optional[int], list[int], bool]:
    """Read the instance number(s) off the bracket-free, nest-free text."""
    run = _NUM_RUN_SEP.search(text) or _NUM_RUN_SPACE.search(text)
    if run:
        numbers = [int(n) for n in _ANY_NUM.findall(run.group(1))]
        return numbers[0], numbers, True
    if _LEADING_COUNT.match(text) and not _TRAILING_NUM.search(text):
        return None, [], True
    single = _TRAILING_NUM.search(text)
    if single:
        value = int(single.group(1))
        return value, [value], False
    # Not trailing, but unambiguous: "2 Case - motherboard connector",
    # "Connector 1 in optical drive", "SSD - motherboard connector 1 SATA".
    numbers = _ANY_NUM.findall(text)
    if len(numbers) == 1:
        value = int(numbers[0])
        return value, [value], False
    return None, [], False


def _first_matching_rule(rules: _RuleSet, candidates: list[str]) -> Optional[dict]:
    """First rule matching the earliest candidate text that matches at all."""
    for text in candidates:
        for pattern, rule in rules.rules:
            if pattern.search(text):
                return rule
    return None


def _resolve_verb(rules: _RuleSet, text: str, cls: str, rule_verb: Optional[str]) -> Optional[str]:
    if rule_verb:
        return rule_verb
    for pattern, verb in rules.verb_prefixes:
        if pattern.search(text):
            return rules.verb_class_overrides.get(cls, {}).get(verb, verb)
    return None


def parse_raw_name(
    raw: str,
    nest_group: str = "",
    rules_path: str | Path = TAXONOMY_MAP_PATH,
) -> ParsedTarget:
    """Map one raw step name (plus its optional nest group) onto a target."""
    rules = _load_rules_cached(str(Path(rules_path).resolve()))
    normalized = _normalize(raw, rules.typos)
    stripped, quals = _split_qualifier(normalized)

    match_text = stripped
    if quals:
        match_text = f"{match_text} {' '.join(quals)}".strip()
    match_text = _WS.sub(" ", match_text)

    # The nest group is only a tie-breaker for a name that matches nothing on
    # its own ("Connector 3" under "Power Supply Unit (PSU)"). Trying the bare
    # name first also keeps anchored patterns like `^case fan$` working for rows
    # that happen to carry a nest group.
    candidates = [match_text]
    if nest_group.strip():
        nest, _ = _split_qualifier(_normalize(nest_group, rules.typos))
        candidates.append(_WS.sub(" ", f"{match_text} nest: {nest}").strip())

    instance_no, instance_nos, multi = _read_instances(stripped)
    multi = multi or bool(_MULTI_WORDS.search(stripped))

    parsed = ParsedTarget(instance_no=instance_no, multi=multi)
    if quals:
        # Keep the annotator's original casing in the qualifier.
        raw_quals = _split_qualifier(raw.strip())[1] or quals
        parsed.attrs["qualifier"] = " | ".join(raw_quals)
    if len(instance_nos) > 1:
        parsed.attrs["instance_nos"] = instance_nos
    parsed.attempt = bool(_ATTEMPT.search(match_text))

    rule = _first_matching_rule(rules, candidates)
    if rule is not None:
        parsed.cls = rule.get("cls") or ""
        parsed.attrs.update(rule.get("attrs") or {})
        parsed.virtual = rule.get("virtual")
        parsed.multi = parsed.multi or bool(rule.get("multi"))
        parsed.step_type_hint = rule.get("step_type_hint")
        parsed.canon_group = rule["canon_group"]
        parsed.matched_rule = rule["name"]
        parsed.verb = _resolve_verb(rules, stripped, parsed.cls, rule.get("verb"))

    if parsed.step_type_hint in _MARKER_HINTS:
        # A capture marker names no target, so any number in it is not an index.
        parsed.instance_no = None
        parsed.attrs.pop("instance_nos", None)
    return parsed


# --------------------------------------------------------------------------- #
# Tools (spec 6.5)
# --------------------------------------------------------------------------- #
_PHILLIPS = re.compile(r"ph\s*([123])\b", re.IGNORECASE)
_TORX = re.compile(r"\bt\s*(15|20)\b", re.IGNORECASE)
_HAND = re.compile(r"\bhand\b", re.IGNORECASE)


def map_tool(raw: str) -> str:
    """Normalize a raw tool string onto the spec 6.5 vocabulary."""
    s = _WS.sub(" ", (raw or "").strip())
    if not s:
        return "none"
    m = _PHILLIPS.search(s)
    if m:
        return f"PH{m.group(1)}"
    m = _TORX.search(s)
    if m:
        return f"T{m.group(1)}"
    if _HAND.search(s):
        return "hand"
    return "none"
