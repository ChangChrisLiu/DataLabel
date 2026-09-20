"""The full P0 VLM task set of spec 8.2 (task B6).

Every assertion here is made twice: once by naming what the record should say,
and once by :mod:`tests.vlm_checker`, which re-derives the answer from the
database without touching a line of the generator. A question this repository
cannot re-derive is a question the export is not allowed to emit.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import vlm_scene as S
from tda.core.db import Db
from tda.core.export.vlm import TASKS, export_vlm
from vlm_checker import Checker


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def scene(tmp_db_path: str):
    db = Db(tmp_db_path)
    tax = S.build(db)
    yield db, tax
    db.close()


def _run(db, tax, out: Path, **kw) -> list[dict]:
    kw.setdefault("tasks", TASKS)
    kw.setdefault("view", S.VIEW)
    view = kw.pop("view")
    export_vlm(db, tax, [S.DESKTOP], view, str(out), **kw)
    return [json.loads(line) for line in
            out.read_text(encoding="utf-8").splitlines() if line]


def _of(records: list[dict], task: str) -> list[dict]:
    return [r for r in records if r["task"] == task]


# --------------------------------------------------------------------------- #
# the contract
# --------------------------------------------------------------------------- #
def test_every_p0_task_is_emitted_and_re_derivable(scene, tmp_path: Path):
    db, tax = scene
    records = _run(db, tax, tmp_path / "v.jsonl")
    tasks = {r["task"] for r in records}
    assert tasks == set(TASKS), sorted(tasks)

    checker = Checker(db, tax, S.DESKTOP)
    assert checker.check_all(records) == len(records)


def test_every_record_carries_the_contract_fields(scene, tmp_path: Path):
    db, tax = scene
    for record in _run(db, tax, tmp_path / "v.jsonl"):
        assert record["graph_version"] and isinstance(record["graph_version"], str)
        assert record["layer"] in ("L1", "L2", "L3", "L4")
        assert record["tier"] == "gold"
        assert record["model_family"] == "EliteDesk 800 G2 TWR"
        assert record["template_id"].startswith(record["task"] + ".")
        assert record["answer_check"]["type"]
        assert record["images"] and all(
            img.startswith(f"{record['view']}/D") or img.startswith("oak1/D")
            for img in record["images"]
        )
