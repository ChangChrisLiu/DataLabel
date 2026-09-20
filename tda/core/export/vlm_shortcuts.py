"""How much of this benchmark can be answered without looking at the image.

A question-answer corpus generated from structured labels is always at risk of
being solvable from its own wording: if every ``disconnect`` question is a yes
and every ``remove`` question is a no, a model that reads the verb and ignores
the picture scores well and has learned nothing. That is not a hypothetical --
it is what a review found in V4's first two rounds, at 83 %.

So the export measures the shortcut itself, publishes it beside the counts and
is judged on it. Four classifiers, each given the answer key and fitted to the
corpus it is scoring:

* **majority** -- always answer the commonest label;
* **verb** -- the best answer per verb named in the question;
* **class** -- the best answer per target class;
* **verb_class** -- the best answer per ``(verb, class)`` pair;
* **template** -- the best answer per question template.

Each is an *in-sample upper bound*: the majority label of each key is chosen
after seeing the labels, so no real text-only model can beat these numbers. A
task whose shortcut accuracy is near its majority rate is a task where the
wording carries nothing; a task far above it has a leak in its question text.

Everything here is pure and reads only the records, so the numbers in the
summary file are the numbers of the file that was written.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Optional

__all__ = ["KEYS", "LABELLED_TASKS", "option_verb_shortcut", "shortcut_report",
           "task_key", "task_label"]

#: task -> the answer field a text-only classifier would try to guess. Only the
#: tasks with a small closed answer are scored; V1, V5, V8 and V10 answer with
#: sets and boxes, which no wording shortcut can produce.
LABELLED_TASKS = {
    "V4": "feasible", "V12": "changed", "V14": "answerable", "V16": "valid",
    "V15": "same_moment", "V2": "state", "V3": "verb",
}

#: The keys a question-text-only classifier may condition on.
KEYS = ("majority", "verb", "class", "verb_class", "template")


def task_label(record: dict) -> Optional[Any]:
    """The label a text-only classifier would have to produce, or ``None``."""
    task = record["prompt"]["task"]
    field = LABELLED_TASKS.get(task)
    if field is None:
        return None
    answer = record["label"]["answer"]
    return answer.get(field) if field in answer else None


def _v16_first(record: dict) -> tuple[Optional[str], Optional[str]]:
    plan = record["label"]["answer_check"].get("plan") or []
    if not plan:
        return None, None
    return plan[0]["verb"], None


def task_key(record: dict, ctx: Optional[dict] = None
             ) -> tuple[Optional[str], Optional[str]]:
    """``(verb, class)`` **as the prompt gives them away**, or ``(None, None)``.

    Only what a reader of the question can see. V4 names one action, so both;
    V16 renders the whole plan and its first step is what a reader meets first;
    V2 names one instance, so its class but no verb. V3, V12, V14 and V15 name
    neither -- "what action was just performed?" carries no verb and no class --
    so those tasks collapse to a single bucket and their shortcut accuracy is
    their majority rate, which is the right answer rather than a missing one.

    Conditioning on something only the *answer* knows would not measure a
    shortcut, it would measure the label: V3's target class is in its answer and
    maps almost one-to-one onto its verb, which would have read as a 100 % leak
    in a task whose question text is four words long.
    """
    task = record["prompt"]["task"]
    check = record["label"]["answer_check"]
    if task == "V4":
        return check.get("verb"), check.get("target_class")
    if task == "V16":
        return _v16_first(record)
    if task == "V2":
        return None, check.get("class")
    return None, None


def _accuracy(pairs: Iterable[tuple[Any, Any]]) -> float:
    """Best achievable accuracy of "answer the majority label of each key"."""
    buckets: dict[Any, Counter] = defaultdict(Counter)
    total = 0
    for key, label in pairs:
        buckets[key][label] += 1
        total += 1
    if not total:
        return 0.0
    return sum(counts.most_common(1)[0][1] for counts in buckets.values()) / total


def _round(value: float) -> float:
    """Four decimals, so the summary file is byte-stable across platforms."""
    return round(float(value), 4)


def shortcut_report(records: Iterable[dict]) -> dict:
    """``task -> {records, labels, majority, verb, class, verb_class, template}``.

    Only the tasks of :data:`LABELLED_TASKS` appear, and only when every record
    of the task carries its label field.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["prompt"]["task"]].append(record)

    out: dict[str, dict] = {}
    for task in sorted(by_task):
        if task not in LABELLED_TASKS:
            continue
        rows = [(r, task_label(r)) for r in by_task[task]]
        rows = [(r, label) for r, label in rows if label is not None]
        if not rows:
            continue
        keys = {r["id"]: task_key(r) for r, _ in rows}
        templates = {r["id"]: r["label"]["template_id"] for r, _ in rows}
        counts = Counter(label for _, label in rows)
        out[task] = {
            "records": len(rows),
            "labels": {str(k): v for k, v in sorted(counts.items(), key=str)},
            "majority": _round(counts.most_common(1)[0][1] / len(rows)),
            "verb": _round(_accuracy((keys[r["id"]][0], label) for r, label in rows)),
            "class": _round(_accuracy((keys[r["id"]][1], label) for r, label in rows)),
            "verb_class": _round(_accuracy((keys[r["id"]], label) for r, label in rows)),
            "template": _round(_accuracy((templates[r["id"]], label)
                                         for r, label in rows)),
        }
    return out


def option_verb_shortcut(records: Iterable[dict]) -> dict:
    """V6's own shortcut: "pick the option whose verb is usually the right one".

    Fitted on the corpus and then scored on it, like the others, so it is an
    upper bound. ``None`` fields when no V6 record carries options.
    """
    listed = [r for r in records
              if r["prompt"]["task"] == "V6" and (r["label"].get("options"))]
    if not listed:
        return {"records": 0, "shortcut": None, "match_rate_baseline": None}
    wins: Counter = Counter()
    seen: Counter = Counter()
    for record in listed:
        reference = tuple(record["label"]["answer_check"]["reference"])
        for option in record["label"]["options"]:
            seen[option["verb"]] += 1
            if (option["verb"], option["target"]) == reference:
                wins[option["verb"]] += 1
    score: Callable[[str], float] = lambda verb: (
        wins[verb] / seen[verb] if seen[verb] else 0.0)

    correct = 0
    for record in listed:
        reference = tuple(record["label"]["answer_check"]["reference"])
        best = max(record["label"]["options"],
                   key=lambda o: (score(o["verb"]), o["verb"], o["target"]))
        correct += (best["verb"], best["target"]) == reference
    sizes = Counter(len(r["label"]["options"]) for r in listed)
    random_rate = sum(n / size for size, n in sizes.items()) / len(listed)
    return {
        "records": len(listed),
        "shortcut": _round(correct / len(listed)),
        "random": _round(random_rate),
        "by_verb": {verb: _round(score(verb)) for verb in sorted(seen)},
    }
