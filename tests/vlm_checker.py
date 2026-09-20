"""An independent re-derivation of every VLM answer (not a test module).

The point of this file is that it shares **no code** with the generator. It
reads the database and the taxonomy -- the data -- and re-implements the spec
from scratch: its own fold of the action log into a state, its own transcription
of the precondition table, its own notion of which verb may be applied to which
class in which state. Nothing from ``tda.core.graph*``, ``tda.core.states`` or
``tda.core.export.vlm*`` is imported, so a bug that lives in one of those
modules cannot hide by being used on both sides of the assertion.

:class:`Checker` dispatches on the record's ``answer_check`` block, which is the
contract: a question whose ``answer_check`` this file cannot execute is a
question the export may not emit, and :meth:`Checker.check` says so. Every
derivation is a real one -- V12's events come from a state diff and V15's
same-moment answer from comparing two folded states -- so corrupting a record
makes this file fail, which ``test_vlm_tasks.py`` asserts directly.

**The two tables below are transcribed, and one of them is not the spec's
literal text.** :data:`REQUIRED_STATES` is spec 7.2 verbatim. :data:`GATES` is
spec 7.2 **as amended by the implementation** (``tda.core.graph_rules.GATES``):
the spec says every hard edge gates every verb, which across the 66 sheets
produced 34 "displace psu.01 violates connected_to(...)" lines, every one of
them describing correct work -- swinging a PSU aside is how you reach its plugs.
The amendment distinguishes *moving the part* from *reaching it*, and the
controller is amending the spec to match. Transcribing it here rather than
importing it means a change to that table has to be a deliberate act in two
files, which is what an independent checker is for.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

#: spec 7.2: the blocker must be in one of these states.
REQUIRED_STATES = {
    "fastened_by": {"loosened", "removed"},
    "connected_to": {"unplugged", "removed"},
    "locked_by": {"open"},
    "covered_by": {"open", "removed", "displaced"},
    "blocked_by": {"removed", "displaced", "released"},
}

#: Every verb a hard constraint can stand in the way of (spec 7.2).
GATED_VERBS = {"remove", "displace", "open", "unscrew", "disconnect", "release"}

#: Which of those verbs each edge type actually gates (spec 7.2 as amended --
#: see the module docstring).
GATES = {
    "fastened_by": {"remove", "displace", "open"},
    "locked_by": {"remove", "displace", "open"},
    "covered_by": set(GATED_VERBS),
    "blocked_by": set(GATED_VERBS),
    "connected_to": {"remove"},
}

HARD_TYPES = tuple(REQUIRED_STATES)

#: Which states a verb may be applied *from* (spec 6.2/6.3).
VERB_FROM_STATES = {
    "unscrew": {"fastened"},
    "disconnect": {"plugged"},
    "open": {"closed"},
    "release": {"closed", "routed"},
    "displace": {"installed"},
    "remove": None,  # any state but `removed`, except a screw (below)
}
REMOVE_FROM_STATES = {"screw": {"loosened"}}

REMOVED = "removed"
IN_CHASSIS = "in_chassis"
ON_BENCH = "on_bench"
CABLE_PREFIX = "cable:"
LS_PREFIX = "ls:"
SKIP_STEP_TYPES = {"ignore"}
ANSWERABLE = {"visible", "occluded_partial", "visible_tiny"}
CHASSIS_CLASS = "chassis"
UNGRADED_TOOLS = {None, "", "unknown"}


class CheckFailure(AssertionError):
    """One record whose stated answer is not what the data says."""


def _fail(record: dict, message: str) -> None:
    named = (record.get("label") or {}).get("readable_id") or record.get("id")
    raise CheckFailure(f"{named}: {message}")


def _key(action) -> tuple:
    return (action.step, action.idx)


class Checker:
    """Re-derives the answers of one desktop from the database alone."""

    def __init__(self, db, tax, desktop: int):
        self.db = db
        self.tax = tax
        self.desktop = int(desktop)
        self.instances = {
            key: rec for key, rec in db.instances(desktop).items()
            if not key.startswith(LS_PREFIX)
        }
        self.actions = sorted(db.actions(desktop), key=_key)
        self.steps = {s.step: s for s in db.steps(desktop)}
        self.edges = [
            row for row in db.relations(desktop)
            if row["type"] in HARD_TYPES and row["status"] != "rejected"
            and not str(row["target"]).startswith(LS_PREFIX)
            and not str(row["blocker"]).startswith(LS_PREFIX)
        ]
        self.manual = [e for e in db.events(desktop) if not e.auto]
        self._rows: dict[tuple[str, int], dict] = {}
        self._states: dict[int, dict] = {}
        self.checked = 0
        self.by_type: dict[str, int] = {}

    # -- the data ---------------------------------------------------------- #
    def rows(self, view: str, step: int) -> dict[str, dict]:
        from tda.core.model import FrameKey

        hit = self._rows.get((view, step))
        if hit is None:
            hit = self.db.compiled(FrameKey(self.desktop, step, view))
            self._rows[(view, step)] = hit
        return hit

    def step_type(self, step: int) -> str:
        rec = self.steps.get(step)
        return "normal" if rec is None else rec.step_type

    def exportable(self, step: int) -> bool:
        return self.step_type(step) not in SKIP_STEP_TYPES

    def log_steps(self) -> list[int]:
        return [s for s in sorted(self.steps) if self.exportable(s)]

    # -- the state machine, folded by hand --------------------------------- #
    def _effect(self, target: str, verb: str) -> Optional[tuple[str, str]]:
        rec = self.instances.get(target)
        if rec is None:
            cls = "cable" if target.startswith(CABLE_PREFIX) else None
            if cls is None:
                return None
            return self.tax.apply_verb(cls, {}, verb)
        return self.tax.apply_verb(rec.cls, rec.attrs, verb)

    def state_at(self, step: int) -> dict[str, dict]:
        """``key -> {"state", "placement", "left_with"}`` after logical step ``step``."""
        if step in self._states:
            return self._states[step]
        frame: dict[str, dict] = {
            key: {"state": self.tax.default_state(rec.cls),
                  "placement": IN_CHASSIS, "left_with": None}
            for key, rec in self.instances.items()
        }
        removed_at: dict[str, int] = {}
        for action in self.actions:
            if action.step > step or action.result != "success":
                continue
            effect = self._effect(action.target, action.verb)
            if effect is None:
                continue
            attr, value = effect
            slot = frame.get(action.target)
            if slot is None:
                if not action.target.startswith(CABLE_PREFIX):
                    continue
                slot = frame.setdefault(
                    action.target,
                    {"state": self.tax.default_state("cable"),
                     "placement": IN_CHASSIS, "left_with": None},
                )
            if attr == "state" and value == REMOVED and slot["state"] != REMOVED:
                removed_at[action.target] = action.step
            slot[attr] = value
            if attr == "state" and value == REMOVED and slot["placement"] == IN_CHASSIS:
                slot["placement"] = ON_BENCH
        # a hand-written event corrects the derived log; it never replaces it
        for event in sorted((e for e in self.manual if e.step <= step),
                            key=lambda e: e.step):
            slot = frame.get(event.target)
            if slot is not None and event.attr in ("state", "placement"):
                slot[event.attr] = event.new
        self._cascade(frame, removed_at)
        self._states[step] = frame
        return frame

    def _cascade(self, frame: dict[str, dict], removed_at: dict[str, int]) -> None:
        """Spec 3.3: an ``attached`` child leaves inside its parent."""
        children: dict[str, list[str]] = {}
        for key, rec in self.instances.items():
            if rec.attached and rec.parent:
                children.setdefault(rec.parent, []).append(key)
        for parent in sorted(children):
            rec = self.instances.get(parent)
            slot = frame.get(parent)
            if rec is None or slot is None or slot["state"] != REMOVED:
                continue
            if rec.cls == "connector":  # removing one means the cable went out
                continue
            stack, seen = [parent], {parent}
            while stack:
                node = stack.pop()
                for child in sorted(children.get(node, ())):
                    if child in seen:
                        continue
                    seen.add(child)
                    own, theirs = removed_at.get(child), removed_at.get(node)
                    if own is not None and theirs is not None and own < theirs:
                        continue  # it had already left on its own
                    slot = frame[child]
                    slot["state"] = REMOVED
                    if slot["placement"] == IN_CHASSIS:
                        slot["placement"] = ON_BENCH
                    slot["left_with"] = node
                    stack.append(child)

    def bare_state(self, step: int) -> dict[str, str]:
        """The folded state as ``key -> state``, for comparing two moments."""
        return {k: v["state"] for k, v in self.state_at(step).items()}

    # -- the constraint graph ---------------------------------------------- #
    def _blocker_state(self, frame: dict[str, dict], blocker: str) -> Optional[str]:
        slot = frame.get(blocker)
        if slot is not None:
            return slot["state"]
        if blocker.startswith(CABLE_PREFIX):
            return "routed"
        return None

    def unmet(self, verb: str, target: str, frame: dict[str, dict]) -> list[dict]:
        """Every required edge of spec 7.2 this action would break."""
        if verb not in GATED_VERBS:
            return []
        out = []
        for edge in self.edges:
            if edge["target"] != target or edge["necessity"] != "required":
                continue
            if verb not in GATES.get(edge["type"], GATED_VERBS):
                continue
            current = self._blocker_state(frame, edge["blocker"])
            if current is None or current == REMOVED:
                continue
            if current not in REQUIRED_STATES.get(edge["type"], set()):
                out.append(dict(edge))
        return out

    def applies(self, key: str, verb: str, frame: dict[str, dict]) -> bool:
        slot = frame.get(key)
        if slot is None or slot["state"] == REMOVED:
            return False
        effect = self._effect(key, verb)
        if effect is None or effect[0] != "state" or effect[1] == slot["state"]:
            return False
        rec = self.instances.get(key)
        cls = rec.cls if rec is not None else "cable"
        allowed = (REMOVE_FROM_STATES.get(cls) if verb == "remove"
                   else VERB_FROM_STATES.get(verb))
        return allowed is None or slot["state"] in allowed

    def _candidates(self, frame: dict[str, dict]) -> list[tuple[str, str]]:
        out = [(verb, key) for key in self.instances for verb in self.tax.verbs
               if self.applies(key, verb, frame)]
        cables = {e["blocker"] for e in self.edges
                  if str(e["blocker"]).startswith(CABLE_PREFIX)}
        cables |= {k for k in frame if k.startswith(CABLE_PREFIX)}
        for key in cables:
            current = self._blocker_state(frame, key) or "routed"
            if current in VERB_FROM_STATES["release"]:
                out.append(("release", key))
        return out

    def permitted(self, step: int) -> set[tuple[str, str]]:
        """What no recorded constraint forbids -- an upper bound, never a truth."""
        frame = self.state_at(step)
        return {(v, t) for v, t in self._candidates(frame)
                if not self.unmet(v, t, frame)}

    def blocked(self, step: int) -> set[tuple[str, str]]:
        """What the graph certainly forbids at this state."""
        frame = self.state_at(step)
        return {(v, t) for v, t in self._candidates(frame)
                if self.unmet(v, t, frame)}

    def successful_actions(self, step: int) -> list:
        return [a for a in self.actions
                if a.step == step and a.result == "success"
                and self.exportable(a.step)]

    def named(self, step: int) -> list:
        return [a for a in self.successful_actions(step)
                if _cls_of(self.instances, a.target) is not None]

    def changes_state(self, action) -> bool:
        """Does this action move the state machine? ``reorient`` does not."""
        effect = self._effect(action.target, action.verb)
        return effect is not None and effect[0] == "state"

    def planning_actions(self, step: int) -> list:
        return [a for a in self.named(step) if self.changes_state(a)]

    def demonstrated(self, after: int) -> Optional[tuple[int, Any]]:
        for step in self.log_steps():
            if step > after and self.named(step):
                action = self.named(step)[0]
                return (step, action) if self.changes_state(action) else None
        return None

    # -- dispatch ----------------------------------------------------------- #
    def check(self, record: dict) -> None:
        self._check_common(record)
        spec = record["label"].get("answer_check")
        if not isinstance(spec, dict) or "type" not in spec:
            _fail(record, "carries no answer_check block")
        handler = getattr(self, f"_check_{spec['type']}", None)
        if handler is None:
            _fail(record, f"answer_check type {spec['type']!r} has no checker")
        handler(record, spec)
        self.checked += 1
        self.by_type[spec["type"]] = self.by_type.get(spec["type"], 0) + 1

    def check_all(self, records: Iterable[dict]) -> int:
        for record in records:
            self.check(record)
        return self.checked

    # -- the checks --------------------------------------------------------- #
    def _check_common(self, record: dict) -> None:
        if set(record) != {"id", "prompt", "label"}:
            _fail(record, f"a record is id/prompt/label, not {sorted(record)}")
        prompt, label = record["prompt"], record["label"]
        if set(prompt) - {"task", "images", "question", "options"}:
            _fail(record, f"the prompt side carries {sorted(prompt)}")
        for field in ("readable_id", "answer", "answer_check", "evidence",
                      "rationale", "desktop", "step", "view", "layer",
                      "template_id", "tier", "verified", "graph_version",
                      "model_family"):
            if field not in label:
                _fail(record, f"missing label field {field!r}")
        if not re.fullmatch(r"rec_[0-9a-f]{16}", str(record["id"])):
            _fail(record, f"the record id {record['id']!r} is not opaque")
        if label["desktop"] != self.desktop:
            _fail(record, "wrong desktop")
        if LS_PREFIX in repr(record):
            _fail(record, "a Label Studio draft reached the record")
        if not self.exportable(int(label["step"])):
            _fail(record, "emitted on a step nobody annotates")
        for image in prompt["images"]:
            if not re.fullmatch(r"img_[0-9a-f]{16}", str(image)):
                _fail(record, f"image {image!r} is not an opaque id")

    @staticmethod
    def _answer(record: dict) -> dict:
        return record["label"]["answer"]

    # V1, V8 -- a set of instances, each with a box
    def _check_boxes(self, record: dict, spec: dict) -> None:
        label = record["label"]
        rows = self.rows(label["view"], label["step"])
        answer = self._answer(record)
        items = answer[spec["field"]] if spec.get("field") else [answer]
        seen = set()
        for item in items:
            key = item[spec["id_field"]]
            seen.add(key)
            if key not in rows:
                _fail(record, f"{key} has no compiled row on this frame")
            if rows[key].get("visibility") not in ANSWERABLE:
                _fail(record, f"{key} cannot be pointed at in this view")
            want = _row_box(rows[key])
            if _iou(want, item[spec["box_field"]]) < 0.999:
                _fail(record, f"{key}: box {item[spec['box_field']]} != {want}")
        if spec.get("exhaustive"):
            dropped = set(spec.get("exclude_classes") or ())
            expect = {k for k, r in rows.items()
                      if r.get("visibility") in ANSWERABLE and _row_box(r) is not None
                      and k in self.instances
                      and self.instances[k].cls not in dropped}
            if seen != expect:
                _fail(record, f"component set {sorted(seen)} != {sorted(expect)}")

    # V2 states and counts, V3, V12, V14, V15
    def _check_exact(self, record: dict, spec: dict) -> None:
        answer = self._answer(record)
        for field, expected in self._expected(record, spec).items():
            if field not in spec.get("fields", [field]):
                continue
            if answer.get(field) != expected:
                _fail(record, f"{field}: {answer.get(field)!r} != {expected!r}")

    def _expected(self, record: dict, spec: dict) -> dict:
        kind = spec.get("derive")
        label = record["label"]
        step, view = int(label["step"]), label["view"]
        if kind == "instance_state":
            return {"state": self.state_at(step)[spec["instance"]]["state"]}
        if kind == "screw_count":
            role, state = spec["role"], spec["state"]
            frame = self.state_at(step)
            n = sum(1 for key, rec in self.instances.items()
                    if rec.cls == "screw"
                    and str(rec.attrs.get("role") or "other") == role
                    and frame[key]["state"] == state)
            return {"count": n}
        if kind == "action_slots":
            actions = self.named(step)
            if not actions:
                _fail(record, "a change question on a step with no action")
            first = actions[0]
            out = {"verb": first.verb, "target_instance": first.target,
                   "target_class": _cls_of(self.instances, first.target)}
            if first.tool not in UNGRADED_TOOLS:
                out["tool"] = first.tool
            elif "tool" in spec.get("fields", ()):
                _fail(record, "an unknown tool must not be a graded field")
            return out
        if kind == "changed":
            old, new = self.state_at(step - 1), self.state_at(step)
            events = [{"target": key, "old": old[key]["state"], "new": new[key]["state"]}
                      for key in sorted(old.keys() & new.keys())
                      if old[key]["state"] != new[key]["state"]]
            if spec.get("step_type") == "dupli" or (
                    self.steps.get(step) is not None and self.steps[step].dupli):
                if events:
                    _fail(record, "a dupli step whose state actually moved")
                return {"changed": False, "events": []}
            return {"changed": bool(events), "events": events}
        if kind == "answerable":
            row = self.rows(view, step).get(spec["instance"])
            if row is None:
                _fail(record, f"{spec['instance']} has no compiled row here")
            ok = row.get("visibility") in ANSWERABLE
            return {"answerable": ok,
                    "state": self.state_at(step)[spec["instance"]]["state"] if ok else None}
        if kind == "cross_view":
            row = self.rows(spec["answer_view"], step).get(spec["instance"])
            if row is None or row.get("visibility") not in ANSWERABLE:
                _fail(record, "the answering view cannot see it either")
            blind = self.rows(spec["blind_view"], step).get(spec["instance"])
            if blind is not None and blind.get("visibility") in ANSWERABLE:
                _fail(record, "the blind view could answer after all")
            return {"state": self.state_at(step)[spec["instance"]]["state"],
                    "best_view": spec["answer_view"]}
        if kind == "same_moment":
            # derived, not copied: two folded states, compared
            other = int(spec["other_step"])
            return {"same_moment": self.bare_state(step) == self.bare_state(other)}
        _fail(record, f"unknown derivation {kind!r}")
        return {}

    # V4
    def _check_feasibility(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(spec.get("state_after", label["step"]))
        frame = self.state_at(step)
        verb, target = spec["verb"], spec["target"]
        answer = self._answer(record)
        if not self.applies(target, verb, frame):
            _fail(record, f"{verb} {target} does not apply in this state at all")
        bad = self.unmet(verb, target, frame)
        blockers = sorted({e["blocker"] for e in bad})
        if sorted(answer.get("blockers") or []) != blockers:
            _fail(record, f"blockers {answer.get('blockers')} != {blockers}")

        source = spec.get("truth_source")
        if source == "demonstrated":
            # a "yes" is only ever the action the log went on to perform
            shown = self.demonstrated(step)
            if shown is None or (shown[1].verb, shown[1].target) != (verb, target):
                _fail(record, f"{verb} {target} is not what the log demonstrates here")
            if answer["feasible"] is not True:
                _fail(record, "a demonstrated action must be feasible")
            if bad:
                _fail(record, "the graph forbids what the log demonstrates")
        elif source in ("graph_blocked", "failed_attempt"):
            if not bad:
                _fail(record, "a negative whose blocker set is empty")
            if answer["feasible"] is not False:
                _fail(record, "a blocked action must not be feasible")
            if source == "failed_attempt":
                attempts = [a for a in self.actions
                            if a.step == step + 1 and a.result != "success"
                            and (a.verb, a.target) == (verb, target)]
                if not attempts:
                    _fail(record, "no failed attempt at the next step says this")
        else:
            _fail(record, f"unknown truth_source {source!r}")

    # V5
    def _check_bounded_set(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        shown = self.demonstrated(step)
        if shown is None:
            _fail(record, "a bounded set with nothing demonstrated after it")
        nxt, action = shown
        if int(spec["reference_step"]) != nxt:
            _fail(record, f"reference step {spec['reference_step']} != {nxt}")
        must = _pairs(answer["must_include"])
        must_not = _pairs(answer["must_not_include"])
        upper = _pairs(answer["permitted_upper_bound"])
        if must != {(action.verb, action.target)}:
            _fail(record, f"must_include {sorted(must)} is not the demonstrated action")
        if must_not != self.blocked(step):
            _fail(record, f"must_not_include differs: "
                          f"missing {sorted(self.blocked(step) - must_not)}, "
                          f"extra {sorted(must_not - self.blocked(step))}")
        if upper != self.permitted(step):
            _fail(record, f"upper bound differs: "
                          f"missing {sorted(self.permitted(step) - upper)}, "
                          f"extra {sorted(upper - self.permitted(step))}")
        if must & must_not:
            _fail(record, f"{sorted(must & must_not)} is both required and forbidden")
        if not must <= upper:
            _fail(record, "the demonstrated action is outside the upper bound")

    # V6
    def _check_next_action(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        shown = self.demonstrated(step)
        if shown is None:
            _fail(record, "a next-step question with nothing demonstrated after it")
        nxt, action = shown
        if int(spec["reference_step"]) != nxt:
            _fail(record, f"reference step {spec['reference_step']} != {nxt}")
        if (answer["verb"], answer["target"]) != (action.verb, action.target):
            _fail(record, "the answer is not the action the log performed")
        if tuple(spec["reference"]) != (action.verb, action.target):
            _fail(record, "the reference and the answer disagree")
        if answer.get("tool") in UNGRADED_TOOLS and "tool" in spec.get("fields", ()):
            _fail(record, "an unknown tool must not be a graded field")
        blocked, permitted = self.blocked(step), self.permitted(step)
        if _pairs(spec["blocked_actions"]) != blocked:
            _fail(record, "the materialised blocked set is not the graph's")
        if _pairs(spec["permitted_upper_bound"]) != permitted:
            _fail(record, "the materialised upper bound is not the graph's")
        options = label.get("options")
        if options is None:
            return
        kinds: dict[str, int] = {}
        for option in options:
            pair = (option["verb"], option["target"])
            kind = option["kind"]
            kinds[kind] = kinds.get(kind, 0) + 1
            if kind == "demonstrated" and pair != (action.verb, action.target):
                _fail(record, f"option {pair} is not the demonstrated one")
            if kind == "graph_blocked" and pair not in blocked:
                _fail(record, f"option {pair} is not blocked")
            if kind == "permitted_unknown" and (pair not in permitted
                                                or pair == (action.verb, action.target)):
                _fail(record, f"option {pair} is not a permitted distractor")
        if kinds.get("demonstrated") != 1:
            _fail(record, "an options list needs exactly one demonstrated action")
        if kinds.get("graph_blocked", 0) < 2:
            _fail(record, "an options list needs at least two blocked actions")
        shown_options = record["prompt"].get("options") or []
        if len(shown_options) != len(options):
            _fail(record, "the prompt and the label list different options")
        for prompt_option, option in zip(shown_options, options):
            if prompt_option["verb"] != option["verb"]:
                _fail(record, "the prompt options are in another order")
            if "kind" in prompt_option or "target" in prompt_option:
                _fail(record, "the prompt options give away which is which")

    # V10
    def _check_history(self, record: dict, spec: dict) -> None:
        label = record["label"]
        step = int(label["step"])
        answer = self._answer(record)
        log = self.log_steps()
        if any(_cls_of(self.instances, a.target) is None
               for s in log for a in self.successful_actions(s)):
            _fail(record, "an exhaustive history on a log with unresolved targets")
        happened = [a for s in log if s <= step for a in self.named(s)]
        done = {(a.verb, a.target) for a in happened}
        got = _pairs(answer["done"])
        if got != done:
            _fail(record, f"history differs: missing {sorted(done - got)}, "
                          f"extra {sorted(got - done)}")
        total = sum(len(self.named(s)) for s in log)
        remaining = total - len(happened)
        if answer["remaining_actions"] != remaining:
            _fail(record, f"remaining {answer['remaining_actions']} != {remaining}")
        share = 0.0 if not total else len(happened) / total
        want = spec["bins"][min(int(share * 4), 3)]
        if answer["progress_bin"] != want:
            _fail(record, f"bin {answer['progress_bin']} != {want}")

    # V16
    def _check_plan(self, record: dict, spec: dict) -> None:
        label = record["label"]
        answer = self._answer(record)
        frame = {k: dict(v) for k, v in self.state_at(int(label["step"])).items()}
        first_bad, violated = None, []
        for i, item in enumerate(spec["plan"]):
            verb, target = item["verb"], item["target"]
            bad = self.unmet(verb, target, frame)
            if bad or not self.applies(target, verb, frame):
                first_bad = i
                # every edge the step breaks: which one "stopped" it has no
                # single answer when a part is both screwed down and plugged in
                violated = sorted(_edge_label(e) for e in bad)
                break
            self._apply(frame, target, verb)
        valid = first_bad is None
        if bool(answer["valid"]) != valid:
            _fail(record, f"valid {answer['valid']} != {valid}")
        if answer.get("first_error_index") != first_bad:
            _fail(record, f"first error {answer.get('first_error_index')} != {first_bad}")
        if valid:
            # a valid plan must be one the log actually performed (demonstrated)
            logged = [(a.verb, a.target) for s in self.log_steps()
                      if s > int(label["step"]) for a in self.planning_actions(s)]
            wanted = [(i["verb"], i["target"]) for i in spec["plan"]]
            if logged[:len(wanted)] != wanted:
                _fail(record, "a valid plan that the log never carried out")
            if answer.get("violated_edges"):
                _fail(record, "a valid plan that names a violated edge")
            return
        if sorted(answer.get("violated_edges") or []) != violated:
            _fail(record, f"violated edges {answer.get('violated_edges')} != {violated}")
        if not violated:
            _fail(record, "an invalid plan that violates no edge")
        if answer.get("violated_edge") != violated[0]:
            _fail(record, "violated_edge is not the first of violated_edges")

    def _apply(self, frame: dict[str, dict], target: str, verb: str) -> None:
        effect = self._effect(target, verb)
        if effect is None or effect[0] != "state":
            return
        frame[target]["state"] = effect[1]
        if effect[1] != REMOVED:
            return
        rec = self.instances.get(target)
        if rec is not None and rec.cls == "connector":
            return
        for key, child in self.instances.items():
            if child.attached and child.parent == target:
                frame[key]["state"] = REMOVED


# --------------------------------------------------------------------------- #
# the prompt-side leak scan
# --------------------------------------------------------------------------- #
#: Words that only ever appear on the label side of a record.
LEAK_WORDS = ("feasible", "blocker", "must_include", "must_not_include",
              "permitted_upper_bound", "graph_version", "verified", "truth_source",
              "answer_check", "occluded_full", "out_of_view", "in_chassis",
              "on_bench", "graph_blocked", "demonstrated", LS_PREFIX)
#: A frame's position in the teardown, which several tasks ask the model to
#: infer, and the machine it belongs to, which tells it which teardown to recall.
STEP_PATTERNS = (re.compile(r"\bs\d{3}\b"), re.compile(r"\bstep\s+\d+", re.I),
                 re.compile(r"\bD\d+\b"),
                 re.compile(r"\b(scan|oak1|oak2|rs)\b", re.I))
#: An instance key -- `psu.01`, `screw.motherboard.03` -- rather than its wording.
INSTANCE_KEY = re.compile(r"\b[a-z_]+(?:\.[a-z_0-9]+)*\.\d{2,}\b")


def prompt_strings(record: dict) -> list[str]:
    """Every string a model would be shown -- and the id a harness would print.

    The record id is not on the prompt side, but every harness logs it, and
    ``V4-D07-scan-s001-demonstrated-remove-psu.01`` would hand over the desktop,
    the view, the step and the answer's own target. It is hashed for that
    reason, and it is scanned here for the same one.
    """
    out = [str(record.get("id") or ""), str(record["prompt"].get("question") or "")]
    out.extend(str(i) for i in record["prompt"].get("images") or [])
    for option in record["prompt"].get("options") or []:
        out.extend(str(v) for v in option.values())
    return out


def scan_prompt_leaks(records: Iterable[dict]) -> list[str]:
    """One line per prompt that gives away part of its own -- or a sibling's -- label.

    Three families, and every one of them was reachable before the prompt/label
    split: a path or an id that spells the step, the desktop or the view; a
    label-side word; and a raw instance key (which carries an ordinal, and an
    ordinal is most of the answer to "how many of these are there?").
    """
    problems: list[str] = []
    for record in records:
        answer = json_dumps(record["label"]["answer"])
        named = record["label"].get("readable_id") or record.get("id")
        for text in prompt_strings(record):
            for pattern in STEP_PATTERNS:
                if pattern.search(text):
                    problems.append(f"{named}: prompt names a step/desktop/view: "
                                    f"{text[:80]!r}")
            for word in LEAK_WORDS:
                if word in text:
                    problems.append(f"{named}: prompt uses the label word {word!r}")
            for hit in INSTANCE_KEY.findall(text):
                if hit in answer:
                    problems.append(f"{named}: prompt names {hit!r}, which is in "
                                    f"its own answer")
    return problems


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _pairs(items: Iterable[dict]) -> set[tuple[str, str]]:
    return {(i["verb"], i["target"]) for i in items}


def _cls_of(instances: dict, target: str) -> Optional[str]:
    if target.startswith(CABLE_PREFIX):
        return "cable"
    rec = instances.get(target)
    return None if rec is None else rec.cls


def _edge_label(edge: dict) -> str:
    return f"{edge['type']}({edge['target']}, {edge['blocker']})"


def _row_box(row: dict) -> Optional[list]:
    """``[x, y, w, h]`` of a compiled row, measured without the export's helpers."""
    if row.get("geom_type") == "box" and row.get("box"):
        x0, y0, x1, y1 = row["box"]
        return [x0, y0, x1 - x0, y1 - y0]
    rle = row.get("visible_rle")
    if not rle:
        return None
    import numpy as np
    from pycocotools import mask as mask_utils

    decoded = np.asarray(mask_utils.decode(rle), dtype=bool)
    ys, xs = np.nonzero(decoded)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def _iou(a: Optional[Any], b: Optional[Any]) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, aw, ah = (float(v) for v in a)
    bx0, by0, bw, bh = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / (aw * ah + bw * bh - inter)
