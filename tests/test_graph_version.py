"""One definition of ``graph_version``, for the accessor and for the stamp.

``constraints`` stamped the digest of the desktop's *constraint graph* -- the
five hard edge types, provisional keys out -- while ``graph.graph_version``
digested every row of the ``relation`` table, including the 90 ``partner_of`` /
``is_pre-request_of`` rows the Label Studio import files there between ``ls:*``
drafts. On the desktops that carry those rows the two answers differed, so the
report said one version, the desktop meta said the same one, and the VLM export
(which reads the accessor) would have published a third that matched neither.

The edge set is the same one everything else in this module reasons about.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tda.core.db import Db
from tda.core.graph import Edge, edge_digest, edges_to_db, graph_version

DESKTOP = 24


@pytest.fixture
def db(tmp_path: Path) -> Db:
    handle = Db(str(tmp_path / "tda.sqlite"))
    handle.upsert_desktop(DESKTOP, {})
    try:
        yield handle
    finally:
        handle.close()


def hard_edges() -> list[Edge]:
    return [
        Edge("fastened_by", "motherboard.01", "screw.motherboard.01", source="rule"),
        Edge("connected_to", "psu.01", "connector.atx_24pin.01", source="rule"),
    ]


def labelstudio_rows() -> list[Edge]:
    """What `import-ls` files in the same table: not constraints, and on drafts."""
    return [
        Edge("partner_of", "ls:Motherboard Screw#2", "ls:Motherboard Screw#1",
             reason="labelstudio:scan", source="labelstudio"),
        Edge("is_pre-request_of", "ls:CPU Chip#1", "ls:CPU Socket#1",
             reason="labelstudio:rs", source="labelstudio"),
    ]


# --------------------------------------------------------------------------- #
# what the digest is taken over
# --------------------------------------------------------------------------- #
def test_a_label_studio_row_does_not_change_the_version(db):
    edges_to_db(db, DESKTOP, hard_edges())
    before = graph_version(db, DESKTOP)
    edges_to_db(db, DESKTOP, labelstudio_rows())
    assert graph_version(db, DESKTOP) == before


def test_a_provisional_key_does_not_change_the_version(db):
    edges_to_db(db, DESKTOP, hard_edges())
    before = graph_version(db, DESKTOP)
    edges_to_db(db, DESKTOP, [
        Edge("blocked_by", "ls:Drive Cage#1", "chassis", source="manual"),
    ])
    assert graph_version(db, DESKTOP) == before


def test_a_real_hard_edge_does_change_it(db):
    edges_to_db(db, DESKTOP, hard_edges())
    before = graph_version(db, DESKTOP)
    edges_to_db(db, DESKTOP, [
        Edge("blocked_by", "psu.01", "chassis", mode="physical_path", source="manual"),
    ])
    assert graph_version(db, DESKTOP) != before


def test_a_desktop_with_only_label_studio_rows_has_no_version(db):
    edges_to_db(db, DESKTOP, labelstudio_rows())
    assert graph_version(db, DESKTOP) is None


def test_the_accessor_agrees_with_the_digest_of_the_hard_edges(db):
    edges_to_db(db, DESKTOP, hard_edges() + labelstudio_rows())
    assert graph_version(db, DESKTOP) == edge_digest(hard_edges())


# --------------------------------------------------------------------------- #
# and the stamp agrees with the accessor
# --------------------------------------------------------------------------- #
def test_constraints_stamps_exactly_what_the_accessor_answers(tmp_path):
    """The case that differed on the real D24 and D33."""
    import yaml
    from graph_scenes import DESKTOP as BENCH, bench_instances, good_sequence

    from tda.cli import main

    db_path = tmp_path / "annotations" / "tda.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump({
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(db_path),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
    }), encoding="utf-8")
    handle = Db(str(db_path))
    try:
        handle.upsert_desktop(BENCH, {})
        for rec in bench_instances().values():
            handle.upsert_instance(rec)
        handle.replace_steps(BENCH, [], good_sequence())
        edges_to_db(handle, BENCH, labelstudio_rows())
    finally:
        handle.close()

    assert main(["--paths", str(tmp_path / "paths.yaml"), "constraints",
                 "--desktops", str(BENCH)]) == 0
    handle = Db(str(db_path))
    try:
        stamped = (handle.get_desktop(BENCH) or {}).get("graph_version")
        assert stamped
        assert graph_version(handle, BENCH) == stamped
    finally:
        handle.close()
