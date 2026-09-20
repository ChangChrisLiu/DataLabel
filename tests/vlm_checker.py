"""An independent re-derivation of every VLM answer (not a test module).

The point of this file is that it shares **no code** with the generator. It
reads the database and the taxonomy -- the data -- and re-implements the spec
from scratch: its own fold of the action log into a state, its own transcription
of the spec-7.2 precondition table, its own notion of which verb may be applied
to which class in which state. Nothing from ``tda.core.graph*``,
``tda.core.states`` or ``tda.core.export.vlm*`` is imported, so a bug that lives
in one of those modules cannot hide by being used on both sides of the
assertion.

:class:`Checker` dispatches on the record's ``answer_check`` block, which is the
contract: a question whose ``answer_check`` this file cannot execute is a
question the export may not emit, and :meth:`Checker.check` says so.

The two tables below are transcribed rather than imported, deliberately:

* :data:`REQUIRED_STATES` is spec 7.2 verbatim;
* :data:`GATES` is spec 7.2 **as amended** by the measurement recorded in
  ``tda.core.graph_rules`` -- a screwed-down part does not move but you can
  still pull the plug out of it. Changing the amendment has to be a deliberate
  act in two files, which is what an independent checker is for.
"""
from __future__ import annotations

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

#: Which of those verbs each edge type actually gates (spec 7.2 as amended).
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


class CheckFailure(AssertionError):
    """One record whose stated answer is not what the data says."""


def _fail(record: dict, message: str) -> None:
    raise CheckFailure(f"{record.get('id')}: {message}")


def _key(action) -> tuple:
    return (action.step, action.idx)


class Checker:
    """Re-derives the answers of one desktop from the database alone."""

    def __init__(self, db, tax, desktop: int, *, views: Iterable[str] = ()):
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
        self.views = tuple(views)
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

    def legal(self, step: int) -> set[tuple[str, str]]:
        """The full legal-action set at the state after ``step``."""
        frame = self.state_at(step)
        out: set[tuple[str, str]] = set()
        for key in self.instances:
            for verb in self.tax.verbs:
                if self.applies(key, verb, frame) and not self.unmet(verb, key, frame):
                    out.add((verb, key))
        cables = {e["blocker"] for e in self.edges
                  if str(e["blocker"]).startswith(CABLE_PREFIX)}
        cables |= {k for k in frame if k.startswith(CABLE_PREFIX)}
        for key in cables:
            current = self._blocker_state(frame, key) or "routed"
            if current in VERB_FROM_STATES["release"] and not self.unmet(
                    "release", key, frame):
                out.add(("release", key))
        return out

    def successful_actions(self, step: int) -> list:
        return [a for a in self.actions
                if a.step == step and a.result == "success"
                and self.exportable(a.step)]

    # -- dispatch ----------------------------------------------------------- #
    def check(self, record: dict) -> None:
        spec = record.get("answer_check")
        if not isinstance(spec, dict) or "type" not in spec:
            _fail(record, "carries no answer_check block")
        handler = getattr(self, f"_check_{spec['type']}", None)
        if handler is None:
            _fail(record, f"answer_check type {spec['type']!r} has no checker")
        self._check_common(record)
        handler(record, spec)
        self.checked += 1
        self.by_type[spec["type"]] = self.by_type.get(spec["type"], 0) + 1

    def check_all(self, records: Iterable[dict]) -> int:
        for record in records:
            self.check(record)
        return self.checked

    # -- the checks --------------------------------------------------------- #
    def _check_common(self, record: dict) -> None:
        for field in ("id", "task", "desktop", "step", "view", "images", "question",
                      "answer", "tier", "verified", "graph_version", "layer",
                      "template_id", "model_family"):
            if field not in record:
                _fail(record, f"missing field {field!r}")
        if record["desktop"] != self.desktop:
            _fail(record, "wrong desktop")
        if LS_PREFIX in repr(record):
            _fail(record, "a Label Studio draft reached the record")
        if not self.exportable(int(record["step"])):
            _fail(record, "emitted on a step nobody annotates")
        for image in record["images"]:
            if ":" in image or image.startswith("/"):
                _fail(record, f"image path {image!r} is not dataset-relative")

    # V1, V8 -- a set of instances, each with a box
    def _check_boxes(self, record: dict, spec: dict) -> None:
        rows = self.rows(record["view"], record["step"])
        items = record["answer"][spec["field"]] if spec.get("field") else [record["answer"]]
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
            expect = {k for k, r in rows.items()
                      if r.get("visibility") in ANSWERABLE and _row_box(r) is not None
                      and k in self.instances}
            if seen != expect:
                _fail(record, f"component set {sorted(seen)} != {sorted(expect)}")

    # V2 states, V12, V14's answerable half, V15
    def _check_exact(self, record: dict, spec: dict) -> None:
        for field, expected in self._expected(record, spec).items():
            if record["answer"].get(field) != expected:
                _fail(record, f"{field}: {record['answer'].get(field)!r} != {expected!r}")

    def _expected(self, record: dict, spec: dict) -> dict:
        kind = spec.get("derive")
        step, view = int(record["step"]), record["view"]
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
            actions = self.successful_actions(step)
            if not actions:
                _fail(record, "a change question on a step with no action")
            first = actions[0]
            return {"verb": first.verb, "target_instance": first.target,
                    "target_class": _cls_of(self.instances, first.target),
                    "tool": first.tool}
        if kind == "changed":
            return {"changed": bool(self.successful_actions(step))}
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
            return {"same_moment": bool(spec["same_moment"])}
        _fail(record, f"unknown derivation {kind!r}")
        return {}

    # V4
    def _check_feasibility(self, record: dict, spec: dict) -> None:
        step = int(record["step"])
        frame = self.state_at(spec.get("state_after", step))
        verb, target = spec["verb"], spec["target"]
        if not self.applies(target, verb, frame):
            _fail(record, f"{verb} {target} does not apply in this state at all")
        bad = self.unmet(verb, target, frame)
        feasible = not bad
        if bool(record["answer"]["feasible"]) != feasible:
            _fail(record, f"feasible {record['answer']['feasible']} != {feasible}")
        blockers = sorted({e["blocker"] for e in bad})
        if sorted(record["answer"].get("blockers") or []) != blockers:
            _fail(record, f"blockers {record['answer'].get('blockers')} != {blockers}")

    # V5
    def _check_set(self, record: dict, spec: dict) -> None:
        got = {(a["verb"], a["target"]) for a in record["answer"][spec["field"]]}
        want = self.legal(int(record["step"]))
        if got != want:
            _fail(record, f"legal set differs: missing {sorted(want - got)}, "
                          f"extra {sorted(got - want)}")
        if spec.get("logged_next") is not None:
            nxt = tuple(spec["logged_next"])
            if nxt not in want:
                _fail(record, f"the logged next action {nxt} is not legal here")

    # V6
    def _check_member_of(self, record: dict, spec: dict) -> None:
        want = self.legal(int(record["step"]))
        answer = (record["answer"]["verb"], record["answer"]["target"])
        if answer not in want:
            _fail(record, f"the reference answer {answer} is not in the legal set")
        if tuple(spec["reference"]) != answer:
            _fail(record, "the reference and the answer disagree")
        for option in spec.get("options", ()):
            pair = (option["verb"], option["target"])
            if bool(option.get("legal")) != (pair in want):
                _fail(record, f"option {pair} is mislabelled")

    # V10
    def _check_history(self, record: dict, spec: dict) -> None:
        step = int(record["step"])
        done = {(a.verb, a.target) for s in range(1, step + 1)
                for a in self.successful_actions(s)
                if _cls_of(self.instances, a.target) is not None}
        got = {(a["verb"], a["target"]) for a in record["answer"]["done"]}
        if got != done:
            _fail(record, f"history differs: missing {sorted(done - got)}, "
                          f"extra {sorted(got - done)}")
        total = sum(1 for s in self.steps if self.exportable(s)
                    for _ in self.successful_actions(s))
        remaining = total - len(done)
        if record["answer"]["remaining_actions"] != remaining:
            _fail(record, f"remaining {record['answer']['remaining_actions']} != {remaining}")
        share = 0.0 if not total else len(done) / total
        want = spec["bins"][min(int(share * 4), 3)]
        if record["answer"]["progress_bin"] != want:
            _fail(record, f"bin {record['answer']['progress_bin']} != {want}")

    # V16
    def _check_plan(self, record: dict, spec: dict) -> None:
        frame = {k: dict(v) for k, v in self.state_at(int(record["step"])).items()}
        first_bad, violated = None, None
        for i, item in enumerate(spec["plan"]):
            verb, target = item["verb"], item["target"]
            bad = self.unmet(verb, target, frame)
            if bad or not self.applies(target, verb, frame):
                first_bad = i
                violated = None if not bad else _edge_label(bad[0])
                break
            self._apply(frame, target, verb)
        valid = first_bad is None
        if bool(record["answer"]["valid"]) != valid:
            _fail(record, f"valid {record['answer']['valid']} != {valid}")
        if record["answer"].get("first_error_index") != first_bad:
            _fail(record, f"first error {record['answer'].get('first_error_index')} "
                          f"!= {first_bad}")
        if valid:
            return
        if record["answer"].get("violated_edge") != violated:
            _fail(record, f"violated edge {record['answer'].get('violated_edge')} "
                          f"!= {violated}")

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
# small helpers
# --------------------------------------------------------------------------- #
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
