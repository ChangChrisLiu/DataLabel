"""Seeding helper shared by the stage-S1 step-table tests.

Writes one desktop's import drafts the way stage S0 does, so every step-table
test starts from a database that looks like a freshly prepared machine.

* **D13** is the clean fixture: every row parses, no compound rows, no issues.
* **D63** carries the compound rows and the reorient steps.
"""
from __future__ import annotations

from pathlib import Path

from tda.core.db import Db
from tda.core.logs import LogImport, import_log, read_desktop_csv
from tda.core.states import events_from_actions
from tda.core.taxonomy import Taxonomy

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"
#: The desktops every step-table test database is seeded with.
DESKTOPS = (13, 63)


def seed(db: Db, desktop: int, tax: Taxonomy) -> LogImport:
    """Import one log fixture into ``db`` and return what it yielded."""
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{desktop:02d}.csv")
    imp = import_log(desktop, rows, meta, tax)
    db.upsert_desktop(desktop, {"brand": meta.get("brand_model_raw") or ""})
    db.replace_steps(desktop, imp.steps, imp.actions)
    for inst in imp.instances.values():
        db.upsert_instance(inst)
    db.replace_events(desktop, events_from_actions(imp.instances, imp.actions, tax))
    return imp


def seeded_db(path: str | Path, tax: Taxonomy, desktops=DESKTOPS) -> Db:
    """A fresh database holding the drafts of every fixture desktop."""
    db = Db(str(path))
    for desktop in desktops:
        seed(db, desktop, tax)
    return db
