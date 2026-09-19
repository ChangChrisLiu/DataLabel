"""Shared scene for the main-window tests (task 13b).

The database is the one :mod:`tests.test_session` builds -- the real D13 step
sheet truncated to :data:`LAST_STEP` logical steps -- plus a synthetic 64x64
frame per step in a temporary cache directory and an explicit pose segment, so
the window has something to open, zoom and store an ROI on.

Nothing here touches ``D:/DataSet``: ``paths.yaml`` is written into ``tmp_path``
and every derived location (``.cache`` for the settings INI, the log and the
crash sidecar; ``backups`` for the exit backup) hangs off it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import yaml

from tda.core import masks
from tda.core.cache import VIEW_EXT, cache_path
from tda.core.db import Db
from tda.core.logs import import_log, read_desktop_csv
from tda.core.model import FrameKey, ShapeKeyframe, ShapePart, ZOrderRec
from tda.core.states import needs_geom
from tda.core.taxonomy import load_taxonomy
from tda.core.truth import TruthService
from tda.core.truth_inputs import instances_of, state_of
from tda.models.sam_service import SamResult
from tda.ui.session import AnnotationSession

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "logs"

DESKTOP = 13
VIEW = "scan"
HW = (64, 64)
#: Enough steps to carry the cover, the four captive screws, the SSD, the cage,
#: the fan connector and the cooler itself -- the whole task-card vocabulary.
LAST_STEP = 14
#: Where the two-segment variant cuts; the window must reset the zoom here.
SEGMENT_CUT = 8
COOLER = "cpu_cooler.fan.01"
SCREWS = tuple(f"screw.cpu_cooler.{i:02d}" for i in (1, 2, 3, 4))


# --------------------------------------------------------------------------- #
# database
# --------------------------------------------------------------------------- #
def write_frames(db: Db, cache_dir: Path, steps, missing=()) -> None:
    """One synthetic 64x64 PNG per step, plus its ``frame`` row.

    Each step paints its bright square one cell further along, so consecutive
    frames really differ and the difference map has something to find.
    """
    for step in steps:
        key = FrameKey(DESKTOP, step, VIEW)
        if step in missing:
            # a step this view never photographed: the row exists, the file does not
            db.upsert_frame(key, "", {"hw": [HW[0], HW[1]]}, None, {"missing": True})
            continue
        path = cache_path(str(cache_dir), key, VIEW_EXT[VIEW])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        img = np.full((*HW, 3), 24, dtype=np.uint8)
        img[10:54, 10:54] = 90  # the "chassis" the ROI should find
        x = 14 + (step % 5) * 6
        img[20:32, x:x + 10] = 230
        cv2.imwrite(path, img)
        db.upsert_frame(key, path, {"cache_path": path, "hw": [HW[0], HW[1]]}, None)


def seed_pose_segments(db: Db, two: bool = False) -> None:
    """One pose segment over the whole scene, or two cut at :data:`SEGMENT_CUT`."""
    if two:
        db.set_pose_segment(DESKTOP, VIEW, 1, 1, SEGMENT_CUT - 1, SEGMENT_CUT - 1,
                            None, None)
        db.set_pose_segment(DESKTOP, VIEW, 2, SEGMENT_CUT, LAST_STEP, LAST_STEP,
                            None, None)
    else:
        db.set_pose_segment(DESKTOP, VIEW, 1, 1, LAST_STEP, LAST_STEP, None, None)


def seed_db(db: Db, tax, cache_dir: Path, last_step: int = LAST_STEP,
            missing=(), two_segments: bool = False) -> None:
    """Import the D13 sheet, truncate it, write the frames and the segments."""
    rows, meta = read_desktop_csv(FIXTURES / f"desktop_{DESKTOP}.csv")
    imported = import_log(DESKTOP, rows, meta, tax)
    db.upsert_desktop(DESKTOP, {"brand": meta.get("brand_model_raw") or "Dell"})
    db.replace_steps(
        DESKTOP,
        [s for s in imported.steps if s.step <= last_step],
        [a for a in imported.actions if a.step <= last_step],
    )
    for key, rec in imported.instances.items():
        if key in SCREWS:  # what the S1 review records: captive, rides out with it
            rec.parent = COOLER
            rec.attached = True
            rec.fastens = COOLER
        db.upsert_instance(rec)
    write_frames(db, cache_dir, range(1, last_step + 1), missing=missing)
    seed_pose_segments(db, two=two_segments)


def make_paths(tmp_path: Path) -> dict:
    """A ``paths.yaml`` dict whose every writable root sits inside ``tmp_path``."""
    return {
        "oak_root": str(tmp_path / "src" / "oak"),
        "scanner_root": str(tmp_path / "src" / "scan"),
        "rs_root": str(tmp_path / "src" / "rs"),
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(tmp_path / "annotations" / "tda.sqlite"),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
        "weights_dir": str(tmp_path / "weights"),
    }


def write_paths_yaml(tmp_path: Path) -> str:
    """Write :func:`make_paths` to disk and return the file name."""
    cfg = make_paths(tmp_path)
    out = tmp_path / "paths.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(out)


def make_db(tmp_path: Path, **kwargs) -> tuple[Db, dict, Any]:
    """``(db, paths, taxonomy)`` for a seeded scene under ``tmp_path``."""
    paths = make_paths(tmp_path)
    tax = load_taxonomy()
    Path(paths["db_path"]).parent.mkdir(parents=True, exist_ok=True)
    db = Db(paths["db_path"])
    seed_db(db, tax, Path(paths["cache_dir"]), **kwargs)
    return db, paths, tax


def make_session(tmp_path: Path, **kwargs) -> AnnotationSession:
    """A session opened on the seeded D13 scanner view."""
    db, paths, tax = make_db(tmp_path, **kwargs)
    session = AnnotationSession(db, tax, TruthService(db, tax),
                                paths["cache_dir"], "tester")
    session.open(DESKTOP, VIEW)
    return session


# --------------------------------------------------------------------------- #
# shapes
# --------------------------------------------------------------------------- #
def cell(index: int) -> np.ndarray:
    """A small rectangle in cell ``index`` of an 8x8 grid over the frame."""
    col, row = index % 8, (index // 8) % 8
    mask = np.zeros(HW, dtype=bool)
    mask[row * 8 + 1:row * 8 + 7, col * 8 + 1:col * 8 + 7] = True
    return mask


def freeze_mask(db: Db, view: str = VIEW, steps=(1, 2), instance: str = "chassis") -> None:
    """Confirm one *masked* instance per frame, so an export has something to write.

    :func:`seed_db` traces nothing, so a COCO export of the bare scene contains
    zero annotations -- which is exactly the state the exports now refuse. A
    mask and not a box: ``export_coco`` drops box-only rows unless asked for
    them. The stored ``input_hash`` is a placeholder, so a refreshing reader
    will re-check these frames and may raise a conflict; a test that only wants
    a non-empty export passes ``--allow-conflicts``.
    """
    mask = np.zeros(HW, dtype=bool)
    mask[HW[0] // 8:HW[0] // 2, HW[1] // 8:HW[1] // 2] = True
    rle = masks.encode_rle(mask)
    for step in steps:
        db.put_compiled(FrameKey(DESKTOP, step, view), instance, rle, 0.0, "visible",
                        "in_chassis", "verified", "seeded", verified_by="tester")


def chassis_instances(session: AnnotationSession, step: int) -> list[str]:
    """Instances that need a chassis mask at ``step``, in a stable order."""
    insts = instances_of(session.db, DESKTOP)
    state = state_of(session.db, session.tax, DESKTOP, step)
    return sorted(
        key for key, kind in needs_geom(insts, state, session.tax).items()
        if kind == "mask" and state[key].placement == "in_chassis"
    )


def seed_shapes(session: AnnotationSession, step: int, skip=(), seg: int = 1) -> None:
    """Give every chassis instance of ``step`` its own rectangle, cheaply."""
    order: list[tuple[str, str]] = []
    for index, key in enumerate(k for k in chassis_instances(session, step)
                                if k not in skip):
        session.db.add_keyframe(ShapeKeyframe(
            id=None, instance=key, desktop=DESKTOP, view=VIEW, pose_segment=seg,
            anchor_step=LAST_STEP, placement="in_chassis", geom_type="mask",
            parts=[ShapePart("main", masks.encode_rle(cell(index)))],
        ))
        order.append((key, "main"))
    session.db.set_zorder(ZOrderRec(DESKTOP, VIEW, seg, order))
    session.refresh_all()


# --------------------------------------------------------------------------- #
# a SAM queue that answers on demand
# --------------------------------------------------------------------------- #
def close_window(win) -> None:
    """Tear a window down completely: shut it, hide it, delete it, drain Qt.

    ``shutdown()`` alone leaves the object alive, and Qt keeps the application
    focus on a widget inside it -- which switches the keyboard off in whatever
    window a later test opens.  Every fixture goes through here.
    """
    from PySide6.QtWidgets import QApplication

    win.shutdown()
    win.hide()
    win.setParent(None)
    win.deleteLater()
    QApplication.processEvents()


class StubSamQueue:
    """``submit``/``stop`` compatible stand-in that never starts a thread.

    The callback is held until :meth:`flush`, which is what lets a test deliver
    a result *after* the frame has changed and check that it is dropped.
    """

    def __init__(self, mask_factory: Optional[Callable[[Any], np.ndarray]] = None) -> None:
        self.requests: list[Any] = []
        self._pending: list[tuple[Any, Callable]] = []
        self.mask_factory = mask_factory
        self.stopped = False

    def submit(self, req, cb, on_error=None) -> None:
        self.requests.append(req)
        self._pending.append((req, cb))
        self.on_error = on_error

    def stop(self, timeout: float = 30.0) -> None:
        self.stopped = True
        self._pending.clear()

    @property
    def running(self) -> bool:
        return not self.stopped

    def pending(self) -> int:
        return len(self._pending)

    def flush(self, multimask: Optional[bool] = None) -> int:
        """Answer every queued request; returns how many were answered."""
        pending, self._pending = self._pending, []
        for req, cb in pending:
            cb(self.result_for(req, multimask))
        return len(pending)

    def result_for(self, req, multimask: Optional[bool] = None) -> SamResult:
        """Three nested rectangles when ``multimask``, otherwise one."""
        h, w = req.image_crop.shape[:2]
        if self.mask_factory is not None:
            best = self.mask_factory(req)
            return SamResult(mask=best, score=0.9, ms=1.0)
        many = req.multimask if multimask is None else multimask
        candidates = []
        for i, inset in enumerate((4, 2, 1) if many else (4,)):
            mask = np.zeros((h, w), dtype=bool)
            mask[inset:max(inset + 1, h - inset), inset:max(inset + 1, w - inset)] = True
            candidates.append(mask)
        return SamResult(mask=candidates[0], score=0.9, ms=1.0,
                         candidates=candidates,
                         scores=[0.9 - 0.1 * i for i in range(len(candidates))])
