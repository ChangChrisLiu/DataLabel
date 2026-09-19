"""Builders for the session tests (not a test module itself).

The scene is the real D13 step sheet (``tests/fixtures/logs/desktop_13.csv``)
imported with :func:`tda.core.logs.import_log` and truncated to its first
:data:`LAST_STEP` logical steps, plus one synthetic frame per step written into
a temporary cache directory.  The database is therefore exactly what S0/S1
would leave behind, and the session under test is driven through the public
:class:`tda.ui.session_api.SessionLike` surface only.

Two things the importer cannot know are filled in the way stage S0/S1 does
(spec 4.1, :func:`tda.core.graph_infer.infer_relational_fields`): the four
captive CPU-cooler screws are marked ``attached`` to the cooler, so that
removing the cooler at step 13 takes them out with it, and the board-mounted
latches (``taxonomy.yaml``'s ``host_class``) are marked ``attached`` to the
motherboard, so that lifting the board at step 42 takes them out with it.

:func:`make_session` builds a 64x64 scene for the behaviour tests;
``hw=(1600, 1600)`` with ``instances=None`` is what the benchmark uses.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.logs import import_log, read_desktop_csv
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.ui.session import AnnotationSession

FIXTURES = Path(__file__).parent / "fixtures" / "logs"

DESKTOP = 13
VIEW = "scan"
HW = (64, 64)
#: The sheet has 42 steps; the first 14 carry every transition the tests need
#: (cover, four captive screws, SSD, cage, fan connector, the cooler itself)
#: and keep a full-scene recompile to a fraction of a second.
LAST_STEP = 14
COOLER = "cpu_cooler.fan.01"
SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in (1, 2, 3, 4))
CHASSIS = "chassis"
FAN_CONNECTOR = "connector.fan.01"
ANNOTATOR = "tester"
#: The sheet's last step: "Motherboard" -- the board is lifted out of the case.
BOARD_STEP = 42
BOARD = "motherboard.01"
#: What rides out inside it (``taxonomy.yaml``'s ``host_class``): D13 has four
#: RAM clips and one socket lever, and no step ever operates on them again.
BOARD_MOUNTED = (*(f"ram_latch.{i:02d}" for i in (1, 2, 3, 4)), "cpu_socket_lever.01")


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def rect(x0: int, y0: int, x1: int, y1: int, hw: tuple[int, int] = HW) -> np.ndarray:
    """A filled ``[x0, x1) x [y0, y1)`` rectangle as a bool mask."""
    mask = np.zeros(hw, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def cell(index: int, hw: tuple[int, int] = HW, grid: int = 8) -> np.ndarray:
    """A small rectangle in cell ``index`` of a ``grid`` x ``grid`` layout.

    Distinct, non-overlapping shapes keep every instance visible, so a seeded
    scene produces no ``empty_visible`` noise.
    """
    side = hw[0] // grid
    col, row = index % grid, (index // grid) % grid
    x0, y0 = col * side + 1, row * side + 1
    return rect(x0, y0, x0 + side - 2, y0 + side - 2, hw)


# --------------------------------------------------------------------------- #
# the database
# --------------------------------------------------------------------------- #
def _write_frames(db: Db, cache_dir: Path, steps, hw: tuple[int, int], missing=()) -> None:
    """One synthetic image per step, plus its ``frame`` row."""
    for step in steps:
        key = FrameKey(DESKTOP, step, VIEW)
        path = cache_path(str(cache_dir), key, VIEW_EXT[VIEW])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = np.full((*hw, 3), 20 + step, dtype=np.uint8)
        img[8:24, 8:24] = 200
        cv2.imwrite(path, img)
        aux = {"cache_path": path, "hw": [hw[0], hw[1]]}
        flags = {"missing": True} if step in missing else None
        db.upsert_frame(key, path, aux, None, flags)


def seed_db(db: Db, tax, cache_dir: Path, last_step: int = LAST_STEP,
            missing=(), hw: tuple[int, int] = HW) -> None:
    """Import the D13 sheet, truncate it, and write the frames."""
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{DESKTOP}.csv")
    imported = import_log(DESKTOP, rows, meta, tax)
    db.replace_steps(
        DESKTOP,
        [s for s in imported.steps if s.step <= last_step],
        [a for a in imported.actions if a.step <= last_step],
    )
    for key, rec in imported.instances.items():
        if key in SCREWS:  # what the S1 review records: captive, rides out with the cooler
            rec.parent = COOLER
            rec.attached = True
            rec.fastens = COOLER
        if key in BOARD_MOUNTED:  # the taxonomy's host_class, as S0 fills it in
            rec.parent = BOARD
            rec.attached = True
        db.upsert_instance(rec)
    _write_frames(db, cache_dir, range(1, last_step + 1), hw, missing=missing)


def make_session(tmp_path: Path, last_step: int = LAST_STEP, missing=(),
                 hw: tuple[int, int] = HW) -> AnnotationSession:
    """A session opened on the seeded D13 scanner view."""
    tax = load_taxonomy()
    db = Db(str(tmp_path / "tda.sqlite"))
    cache_dir = tmp_path / "cache"
    seed_db(db, tax, cache_dir, last_step=last_step, missing=missing, hw=hw)
    session = AnnotationSession(db, tax, TruthService(db, tax), str(cache_dir), ANNOTATOR)
    session.open(DESKTOP, VIEW)
    return session


# --------------------------------------------------------------------------- #
# seeding shapes
# --------------------------------------------------------------------------- #
def chassis_instances(session: AnnotationSession, step: int) -> list[str]:
    """Instances that need a chassis mask at ``step``, in a stable order."""
    db, tax = session.db, session.tax
    insts = instances_of(db, DESKTOP)
    state = state_of(db, tax, DESKTOP, step)
    return sorted(
        key
        for key, kind in needs_geom(insts, state, tax).items()
        if kind == "mask" and state[key].placement == "in_chassis"
    )


def seed_shapes(session: AnnotationSession, step: int, skip=(), grid: int = 8,
                anchor: int = LAST_STEP) -> list[str]:
    """Give every chassis instance of ``step`` its own rectangle, cheaply.

    Written straight to the database (one refresh at the end) rather than
    through ``commit_edit``, so a test that needs a fully drawn frame does not
    pay for one refresh sweep per instance.  The anchor is the last logical step
    of the scene, which selects the keyframe for every step where the instance
    is still in the chassis.  Returns the instances it drew, bottom-up.
    """
    db = session.db
    hw = (session.db.get_frame(FrameKey(DESKTOP, step, VIEW))["aux"]["hw"])
    hw = (int(hw[0]), int(hw[1]))
    keys = [k for k in chassis_instances(session, step) if k not in skip]
    order: list[tuple[str, str]] = []
    for index, key in enumerate(keys):
        db.add_keyframe(
            ShapeKeyframe(
                id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=1,
                anchor_step=anchor, placement="in_chassis", geom_type="mask",
                parts=[ShapePart("main", masks.encode_rle(cell(index, hw, grid)))],
            )
        )
        order.append((key, "main"))
    db.set_zorder(ZOrderRec(DESKTOP, VIEW, 1, order))
    session.refresh_all()
    return keys


def draw(session: AnnotationSession, instance: str, mask: np.ndarray, scope: str) -> dict:
    """``begin_edit`` + paint + ``commit_edit`` in one call."""
    session.begin_edit(instance)
    session.set_editing_mask(mask)
    return session.commit_edit(scope)
