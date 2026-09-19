"""Minimal VLM question/answer export -- tasks V1, V2 and V3 of spec 8.2.

One JSON object per line::

    {"id", "task", "desktop", "step", "view", "images", "question",
     "answer", "evidence", "rationale", "tier", "verified", "graph_version"}

``tier`` is the **view's** annotation standard (spec 8.1: scanner and OAK1 gold,
OAK2 silver, RealSense bronze) and is the same for every record of one file;
``verified`` says whether a human confirmed the frame and every row this
particular answer was read off. They are two fields because they are two facts:
one field spelling both -- "gold" when confirmed, "auto" when not -- answered
the review question in the tier's vocabulary.

* **V1** (perception): one record per frame -- every component the view can be
  pointed at, with its box, ``on_bench`` parts included.
* **V2** (state / counting): per-instance state questions for the classes whose
  state is multi-valued, plus one "how many <role> screws are still fastened?"
  per screw role present in the frame.
* **V3** (change): one record per consecutive frame pair whose later step
  carries at least one successful action -- verb, target and tool.

Answers come from the truth table and the state machine, never from the image,
so every record is reproducible from the database: templates are chosen by a
checksum of the record id rather than at random, and the file is written in a
fixed frame/task order.

Grounding (spec 8.2, principle 1 and the rationale format): a record is only
emitted for instances whose compiled row carries geometry -- a visible mask or
a bench box -- *and* an answerable visibility, so every ``observe`` step of the
``rationale`` chain names the box it read the value off. An instance the frame
cannot localise enters the chain as ``propagate_state`` instead, which carries
no evidence box by construction.

Steps typed ``ignore`` produce no records at all, and an ``initial``, ``dupli``
or ``ignore`` step is never the "after" frame of a V3 pair even when the step
table records an action for it.

``graph_version`` is ``None`` until the constraint graph exists (it is a Plan-B
module); the field is written now so the JSONL schema does not change later.

The export **writes**: like the COCO one it calls
:meth:`~tda.core.truth.TruthService.ensure_fresh` per desktop, so the
compiled rows it reads are complete. A caller therefore needs the
single-user lock of spec 3.5.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from tda.core.db import Db
from tda.core.export.coco import (
    ANSWERABLE,
    NO_CHANGE_STEP_TYPES,
    VERIFIED,
    DesktopCtx,
    frame_file_name,
    frame_is_verified,
    load_ctx,
    row_bbox_xywh,
    view_tier,
)
from tda.core.model import ActionRec, FrameKey, InstanceRec
from tda.core.taxonomy import Taxonomy
from tda.core.truth import TruthService

__all__ = ["TASKS", "class_label", "export_vlm", "instance_label"]

TASKS = ("V1", "V2", "V3")

V1_QUESTIONS = (
    "List every component visible in this image with its bounding box.",
    "Which components can you see in this image? Give each one a bounding box.",
    "Name the parts visible in this photo of the desktop PC and localise them.",
    "Enumerate the visible components and their boxes.",
)
V2_STATE_QUESTIONS = (
    "What is the state of {label}?",
    "In this image, what state is {label} in?",
    "Report the current state of {label}.",
)
V2_COUNT_QUESTION = "How many {role} screws are still fastened?"
V3_QUESTIONS = (
    "What action was just performed between these two images?",
    "Compare the two images: which action was carried out?",
    "What did the operator just do between the first and the second image?",
)

CABLE_PREFIX = "cable:"
CABLE_CLASS = "cable"


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
    """How a person says a taxonomy class: ``ram_module`` -> "RAM module".

    A class definition in ``configs/taxonomy.yaml`` may carry a ``label`` (or
    ``name``) and that wins; none does today, so the id is spelled out instead.
    The class *id* is what every structured field keeps -- this is only ever the
    wording of a question or an answer a human reads.
    """
    defn = (tax.classes.get(cls) if tax is not None else None) or {}
    given = defn.get("label") or defn.get("name")
    return " ".join(_words(given if isinstance(given, str) and given else cls))


def instance_label(
    instance: str, rec: Optional[InstanceRec], tax: Optional[Taxonomy] = None
) -> str:
    """Human wording of an instance key: ``screw.motherboard.03`` -> "Motherboard screw 3".

    The discriminator (a screw's ``role``, a card's ``kind``) leads, then the
    class, then the ordinal; the whole label is capitalised once at the front,
    so ``psu.01`` reads "PSU 1", ``cpu_cooler.01`` "CPU cooler 1" and
    ``ram_module.02`` "RAM module 2". An instance with no record at all keeps
    its raw key: an unnamed key is better than a pretty guess.
    """
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


def _pick(templates: Sequence[str], seed: str) -> str:
    """Deterministically choose one template: same record id, same wording."""
    return templates[zlib.crc32(seed.encode("utf-8")) % len(templates)]


# --------------------------------------------------------------------------- #
# record assembly
# --------------------------------------------------------------------------- #
def _evidence(bboxes: dict[str, list], **extra) -> dict:
    """The instance keys and boxes an answer was read off."""
    return {"instances": sorted(bboxes), "bboxes": bboxes, **extra}


def _observe(instance: str, value: Any, view: str, bbox: list,
             visibility: Optional[str], step: Optional[int] = None) -> dict:
    """One ``observe`` step of a rationale chain (spec 8.2 closed op set).

    Every ``observe`` carries the box it read the value off; an instance this
    frame cannot localise gets :func:`_propagate` instead.
    """
    evidence: dict[str, Any] = {"view": view, "bbox": bbox, "visibility": visibility}
    if step is not None:
        evidence["step"] = step
    return {"op": "observe", "target": instance, "value": value, "evidence": evidence}


def _propagate(instance: str, value: Any) -> dict:
    """``propagate_state``: a value carried over from the event log, not seen here."""
    return {"op": "propagate_state", "target": instance, "value": value}


def _step_for(instance: str, value: Any, view: str,
              seen: dict[str, tuple[dict, list]], step: Optional[int] = None) -> dict:
    """``observe`` when the frame can point at the instance, else ``propagate_state``."""
    if instance not in seen:
        return _propagate(instance, value)
    row, box = seen[instance]
    return _observe(instance, value, view, box, row.get("visibility"), step=step)


def _rationale(steps: list[dict]) -> dict:
    """Close a rationale chain with ``conclude`` and record its depth."""
    chain = [*steps, {"op": "conclude"}]
    return {"depth": len(chain), "steps": chain}


def _record(rec_id: str, task: str, key: FrameKey, images: list[str], question: str,
            answer: dict, evidence: dict, rationale: dict, verified: bool) -> dict:
    """One JSONL record. ``tier`` is filled in by :func:`export_vlm`: it is the
    view's, so it is the same for every record of one file."""
    return {
        "id": rec_id, "task": task, "desktop": key.desktop, "step": key.step,
        "view": key.view, "images": images, "question": question, "answer": answer,
        "evidence": evidence, "rationale": rationale, "verified": bool(verified),
        "tier": None, "graph_version": None,
    }


def _frame_id(key: FrameKey) -> str:
    return f"D{key.desktop:02d}-{key.view}-s{key.step:03d}"


def _verified(rows: Iterable[dict], frame_verified: bool) -> bool:
    """Has a human confirmed everything this answer rests on?

    Both halves are needed. Reading it off the cited rows alone was not enough:
    a V2 question about the one screw a human happened to confirm came out
    confirmed on a frame nobody had otherwise looked at, and a V1 record listing
    that screw claimed the same about the whole image. So the *frame* has to be
    verified (:func:`tda.core.export.coco.frame_is_verified`) **and** every row
    the answer was read off with it.

    A record with **no** evidence rows -- "how many screws are still fastened?"
    on a frame where none is visible, answered off the state machine -- follows
    its frame. Requiring a non-empty evidence list here dropped every such fact
    out of a verified export, which is the opposite of what those exports are
    for: an answer nobody could see is exactly the kind a model has to learn to
    reason to.
    """
    return bool(frame_verified) and all(
        row.get("status") == VERIFIED for row in rows
    )


def _pointable(ctx: DesktopCtx, rows: dict[str, dict],
               only_verified: bool) -> dict[str, tuple[dict, list]]:
    """``instance -> (row, bbox)`` for the rows this view can be asked about.

    Both geometries qualify -- a visible mask and a bench box -- as long as the
    row's visibility is answerable and its instance carries a taxonomy class.
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
        box = row_bbox_xywh(row)
        if box is not None:
            out[instance] = (row, box)
    return out


# --------------------------------------------------------------------------- #
# V1 -- what is visible, and where
# --------------------------------------------------------------------------- #
def _v1(ctx: DesktopCtx, key: FrameKey, image: str,
        pointable: dict[str, tuple[dict, list]], verified: bool) -> Optional[dict]:
    components, bboxes, steps, used = [], {}, [], []
    for instance, (row, box) in pointable.items():
        cls = ctx.cls_of(instance)
        if cls is None:
            continue
        components.append({"class": cls, "instance": instance, "bbox": box,
                           "placement": row.get("placement")})
        bboxes[instance] = box
        used.append(row)
        steps.append(_observe(instance, cls, key.view, box, row.get("visibility")))
    if not components:
        return None
    rec_id = f"V1-{_frame_id(key)}"
    return _record(rec_id, "V1", key, [image], _pick(V1_QUESTIONS, rec_id),
                   {"components": components}, _evidence(bboxes), _rationale(steps),
                   _verified(used, verified))


# --------------------------------------------------------------------------- #
# V2 -- per-instance state and fastener counts
# --------------------------------------------------------------------------- #
def _v2_states(ctx: DesktopCtx, key: FrameKey, image: str,
               pointable: dict[str, tuple[dict, list]], verified: bool) -> list[dict]:
    """One state question per instance whose class has more than one state."""
    frame_state = ctx.state_at(key.step)
    records = []
    for instance, (row, box) in pointable.items():
        cls = ctx.cls_of(instance)
        inst_state = frame_state.get(instance)
        if cls is None or inst_state is None or len(ctx.tax.states_of(cls)) < 2:
            continue
        label = instance_label(instance, ctx.instances.get(instance), ctx.tax)
        rec_id = f"V2-{_frame_id(key)}-state-{instance}"
        question = _pick(V2_STATE_QUESTIONS, rec_id).format(label=label)
        records.append(_record(
            rec_id, "V2", key, [image], question, {"state": inst_state.state},
            _evidence({instance: box}),
            _rationale([_observe(instance, inst_state.state, key.view, box,
                                 row.get("visibility"))]),
            _verified([row], verified),
        ))
    return records


def _screw_roles(ctx: DesktopCtx, pointable: dict[str, tuple[dict, list]]) -> list[str]:
    """Screw roles this frame can be asked to count, in a stable order."""
    roles = set()
    for instance in pointable:
        rec = ctx.instances.get(instance)
        if rec is not None and rec.cls == "screw":
            roles.add(str(rec.attrs.get("role") or "other"))
    return sorted(roles)


def _v2_counts(ctx: DesktopCtx, key: FrameKey, image: str,
               pointable: dict[str, tuple[dict, list]], verified: bool) -> list[dict]:
    """"How many <role> screws are still fastened?" -- counted on the state machine.

    The count covers every screw of that role on the machine, not only the ones
    this view can see, because that is what the question asks; the evidence
    lists the boxes of those that are visible here.
    """
    frame_state = ctx.state_at(key.step)
    records = []
    for role in _screw_roles(ctx, pointable):
        fastened = [
            inst for inst, rec in sorted(ctx.instances.items())
            if rec.cls == "screw" and str(rec.attrs.get("role") or "other") == role
            and (frame_state.get(inst).state if frame_state.get(inst) else None) == "fastened"
        ]
        bboxes = {i: pointable[i][1] for i in fastened if i in pointable}
        rec_id = f"V2-{_frame_id(key)}-count-{role}"
        steps = [_step_for(i, "fastened", key.view, pointable) for i in fastened]
        records.append(_record(
            rec_id, "V2", key, [image],
            V2_COUNT_QUESTION.format(role=role.replace("_", " ")),
            {"count": len(fastened)}, _evidence(bboxes), _rationale(steps),
            _verified([pointable[i][0] for i in fastened if i in pointable], verified),
        ))
    return records


# --------------------------------------------------------------------------- #
# V3 -- what happened between two frames
# --------------------------------------------------------------------------- #
def _target_class(ctx: DesktopCtx, target: str) -> Optional[str]:
    """Class of an action target: ``cable`` for a virtual node, else the taxonomy's.

    ``None`` marks a target this export cannot name -- a provisional or deleted
    instance key -- and drops the action from the answer.
    """
    if target.startswith(CABLE_PREFIX):
        return CABLE_CLASS
    return ctx.cls_of(target)


def _slots(ctx: DesktopCtx, action: ActionRec) -> dict:
    return {
        "verb": action.verb,
        "target_class": _target_class(ctx, action.target),
        "target_instance": action.target,
        "tool": action.tool,
    }


def _v3(ctx: DesktopCtx, key: FrameKey, images: list[str], actions: list[ActionRec],
        before: dict[str, tuple[dict, list]], after: dict[str, tuple[dict, list]],
        verified: dict[int, bool]) -> Optional[dict]:
    named = [a for a in actions if _target_class(ctx, a.target) is not None]
    if not named or ctx.step_type(key.step) in NO_CHANGE_STEP_TYPES:
        return None
    answer = _slots(ctx, named[0])
    if len(named) > 1:  # a compound step: keep every slot set, first one on top
        answer["actions"] = [_slots(ctx, a) for a in named]

    target = named[0].target
    source_step, source = key.step - 1, before
    if target not in before and target in after:
        source_step, source = key.step, after
    bboxes = {target: source[target][1]} if target in source else {}

    rec_id = f"V3-{_frame_id(key)}"
    steps = [
        {"op": "compare_frames", "frames": [key.step - 1, key.step], "view": key.view},
        _step_for(target, answer["verb"], key.view, source, step=source_step),
    ]
    # The grade follows the frame the answer was actually read off, which for a
    # V3 pair is usually the *earlier* one: the part is gone from the later.
    return _record(rec_id, "V3", key, images, _pick(V3_QUESTIONS, rec_id), answer,
                   _evidence(bboxes, from_step=source_step), _rationale(steps),
                   _verified([source[target][0]] if target in source else [],
                            verified.get(source_step, False)))


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def export_vlm(
    db: Db,
    tax: Taxonomy,
    desktops: list[int],
    view: str,
    out_jsonl: str,
    tasks: Iterable[str] = TASKS,
    only_verified: bool = False,
    *,
    truth: Optional[TruthService] = None,
) -> dict:
    """Write the V1/V2/V3 question set of ``desktops`` in ``view`` as JSONL.

    Records are grouped by frame in step order and, inside a frame, by task, so
    two exports of the same database are byte-identical. ``only_verified``
    restricts every question to the compiled rows a human confirmed and keeps
    only the records that come out ``verified``.

    Every record carries the view's ``tier`` (spec 8.1, from
    ``configs/taxonomy.yaml``) -- the same value for the whole file, because a
    tier is a property of the camera, not of one answer.

    Returns ``{"path", "records", "by_task", "desktops", "view"}``.
    """
    wanted = [t for t in TASKS if t in set(tasks)]
    records: list[dict] = []
    tier = view_tier(view, tax)

    def emit(record: Optional[dict]) -> None:
        """Keep a record, unless ``only_verified`` and nothing verified backs it."""
        if record is None:
            return
        if only_verified and not record["verified"]:
            return
        record["tier"] = tier
        records.append(record)

    service = truth or TruthService(db, tax)

    for desktop in desktops:
        ctx = load_ctx(db, tax, desktop, view)
        # the compiled rows of an unverified frame are a cache the
        # annotator's commits leave stale (spec 3.4): fill it before
        # reading the view out, or a frame nobody visited is exported
        # as it was several edits ago -- or silently not at all
        service.ensure_fresh(desktop, view, only_verified)
        previous: Optional[tuple[int, dict[str, tuple[dict, list]], str]] = None
        verified: dict[int, bool] = {}
        for frame in db.frames_for(desktop, view):
            key = FrameKey(desktop, frame["step"], view)
            if not ctx.exportable(key.step):
                continue  # an `ignore` step is no moment of the teardown
            image = frame_file_name(frame, key)
            rows = db.compiled(key)
            verified[key.step] = frame_is_verified(db, key, rows)
            pointable = _pointable(ctx, rows, only_verified)

            if "V1" in wanted:
                emit(_v1(ctx, key, image, pointable, verified[key.step]))
            if "V2" in wanted:
                for record in _v2_states(ctx, key, image, pointable, verified[key.step]):
                    emit(record)
                for record in _v2_counts(ctx, key, image, pointable, verified[key.step]):
                    emit(record)
            if "V3" in wanted and previous is not None and previous[0] == key.step - 1:
                emit(_v3(ctx, key, [previous[2], image], ctx.actions_at(key.step),
                         previous[1], pointable, verified))
            previous = (key.step, pointable, image)

    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_task: dict[str, int] = {}
    for record in records:
        by_task[record["task"]] = by_task.get(record["task"], 0) + 1
    return {
        "path": str(out), "records": len(records), "by_task": by_task,
        "desktops": [int(d) for d in desktops], "view": view,
    }
