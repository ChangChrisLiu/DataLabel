"""Cutting a pose segment from the window: ``Ctrl+Shift+B``, and the audit's bar.

Spec 2.5 (v1.5) lets a view have breaks of its own: the camera was knocked, or
the chassis slid, in front of *this* lens.  Two things reach the annotator here:

* **the action.**  ``Ctrl+Shift+B`` proposes a new segment starting at the frame
  on screen.  It shows the two frames it is deciding between -- this one and the
  one before it -- says how many shapes the boundary cuts through, and asks
  whether to carry them across.  Nothing is written until OK.
* **the bar.**  When the camera audit has *proposed* a break at this frame, a
  non-modal strip says so and offers Accept / Reject.  ``Tab`` still flashes the
  neighbour, which is how the annotator decides.

Both go through the window's one ``leave_frame()`` gate.  A re-cut re-keys the
segment an uncommitted layer is being drawn in, so it is a move like any other;
there is no second gate here and no dialog of its own asking about unsaved work
-- the gate's answer is the answer.

A re-cut is **not** on the undo stack: it is a structural edit, like applying
the step table.  Its undo is to reject the break, which merges the two segments
back (:meth:`tda.core.db.Db.apply_recut`).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from tda.core.pose_breaks import ACCEPTED, KIND_MANUAL, PROPOSED, REJECTED
from tda.ui import app_compat as compat
from tda.ui import app_support as S
from tda.ui.app_widgets import Bar

__all__ = ["CARRY_BELOW_PX", "PoseMixin", "PoseSplitDialog", "proposal_text"]

#: Below this much measured movement the shapes are still roughly in place, so
#: carrying them across is the dialog's default (the same number
#: :mod:`tda.cli_pose` uses, so the two cannot drift).
CARRY_BELOW_PX = 25.0
#: Long edge of the two preview images in the dialog.
PREVIEW_PX = 360
#: Said when the frame on screen cannot start a new segment.
NO_SPLIT_HERE = ("第一帧之前没有可断开的位置 / step {step} is the first step of this "
                 "view: a segment already starts here")
NO_SEGMENTS = "这个视图还没有位姿段 / this view has no pose segments yet"


def proposal_text(row: dict) -> str:
    """The one line the proposal bar shows for a ``proposed`` break."""
    size = "" if row.get("magnitude_px") is None else f"（≈{row['magnitude_px']:.0f} px）"
    what = {"camera": "相机被碰 / camera move", "chassis": "机箱被移动 / chassis move"}.get(
        row.get("kind") or "", "位姿变化 / pose change")
    return (f"第 {row['step']} 步之前可能有{what}{size} — Tab 对比 / "
            f"possible move before step {row['step']}{size} — Tab to compare")


def _pixmap(rgb: Optional[np.ndarray]) -> QPixmap:
    """A scaled preview of one frame, or an empty pixmap when there is none."""
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return QPixmap()
    array = np.ascontiguousarray(rgb[:, :, :3])
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy()).scaled(
        PREVIEW_PX, PREVIEW_PX, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation)


class PoseSplitDialog(QDialog):
    """The two frames a boundary sits between, and the one question about it."""

    def __init__(self, parent: Optional[QWidget], step: int, before: Optional[np.ndarray],
                 after: Optional[np.ndarray], straddles: int, carry: bool,
                 note: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Split pose segment / 拆分位姿段")
        self.setModal(True)
        grid = QGridLayout()
        grid.addWidget(QLabel(f"step {step - 1}"), 0, 0)
        grid.addWidget(QLabel(f"step {step}  ← new segment starts here"), 0, 1)
        for column, rgb in ((0, before), (1, after)):
            label = QLabel()
            pixmap = _pixmap(rgb)
            if pixmap.isNull():
                label.setText("no image / 无图像")
            else:
                label.setPixmap(pixmap)
            grid.addWidget(label, 1, column)

        self.carry_box = QCheckBox(
            f"把跨越断点的 {straddles} 个形状复制到前一段 / carry {straddles} straddling "
            f"shapes into the earlier segment")
        self.carry_box.setChecked(bool(carry))
        self.carry_box.setToolTip(
            "Off: the frames before the boundary become missing_shape, which is the "
            "cue to redraw them in the new pose."
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        if note:
            hint = QLabel(note)
            hint.setWordWrap(True)
            layout.addWidget(hint)
        layout.addWidget(QLabel(
            "位姿段按视角独立；接受后该视角从这一步开始新段。/ Segments are per view: "
            "after OK this view starts a new segment at this step."))
        layout.addWidget(self.carry_box)
        layout.addWidget(buttons)

    def carry(self) -> bool:
        return bool(self.carry_box.isChecked())


class PoseMixin:
    """``Ctrl+Shift+B``, the audit's proposal bar, and the timeline's marks."""

    def _init_pose(self) -> None:
        """The non-modal bar the audit's proposals appear in."""
        self.pose_bar = Bar(self)
        self.pose_bar.add_button("接受 Accept", self.accept_pose_proposal)
        self.pose_bar.add_button("拒绝 Reject", self.reject_pose_proposal)
        self._central_layout.addWidget(self.pose_bar)

    # ------------------------------------------------------------ frame hook
    def on_frame_changed_pose(self, key) -> None:
        """Offer the audit's proposal for this frame, and mark the accepted ones."""
        row = self.pose_proposal(key)
        if row is None:
            self.pose_bar.hide()
        else:
            self.pose_bar.show_text(proposal_text(row))
        self.timeline.set_break_steps(self.accepted_break_steps())

    def pose_proposal(self, key=None) -> Optional[dict]:
        """The ``proposed`` break of the frame on screen, or ``None``."""
        if not compat.is_open(self.session):
            return None
        key = key if key is not None else self.session.current()
        row = self.db.pose_break(int(key.desktop), str(key.view), int(key.step))
        return row if row is not None and row["status"] == PROPOSED else None

    def accepted_break_steps(self) -> list[int]:
        """Steps this view's timeline draws a boundary mark at."""
        if not compat.is_open(self.session):
            return []
        return [int(r["step"]) for r in self.db.pose_breaks(
            int(self.session.desktop), str(self.session.view), status=ACCEPTED)]

    # --------------------------------------------------------------- actions
    @S.guard
    def act_split_pose(self) -> None:
        """``Ctrl+Shift+B``: start a new pose segment at the frame on screen.

        The whole operation sits inside ``leave_frame``: the gate is the one
        thing that decides whether an uncommitted layer may be left behind, and
        a re-cut both re-keys the segment that layer belongs to and re-opens the
        session on this frame afterwards.
        """
        self.leave_frame(lambda: self._split_pose_here(carry=None))

    @S.guard
    def accept_pose_proposal(self) -> None:
        """Accept the audit's proposal for this frame: the same dialog, pre-filled."""
        row = self.pose_proposal()
        if row is None:
            self.report("这一帧没有待确认的位姿断点 / no proposed break on this frame")
            return
        magnitude = row["magnitude_px"]
        carry = magnitude is not None and float(magnitude) < CARRY_BELOW_PX
        self.leave_frame(lambda: self._split_pose_here(carry=carry, proposal=row))

    @S.guard
    def reject_pose_proposal(self) -> None:
        """Reject the audit's proposal: the row stays, so it is not proposed again."""
        row = self.pose_proposal()
        if row is None:
            return
        self.leave_frame(lambda: self._reject(row))

    def _reject(self, row: dict) -> None:
        key = self.session.current()
        with self.db.transaction():
            self.db.set_pose_break_status(int(key.desktop), str(key.view),
                                          int(row["step"]), REJECTED)
            # rejecting one that was never accepted cuts nothing; rejecting one
            # that was merges the two segments back
            self.db.recut_view(int(key.desktop), str(key.view), annotator=self.annotator)
        self.pose_bar.hide()
        self._reopen_here(f"step {row['step']}: 位姿断点已拒绝 / pose break rejected")

    # ------------------------------------------------------------ the re-cut
    def _split_pose_here(self, carry: Optional[bool], proposal: Optional[dict] = None
                         ) -> None:
        """Ask, then cut, then re-open on the same frame."""
        if not compat.is_open(self.session):
            return
        key = self.session.current()
        steps = self.session.steps()
        if not self.db.pose_segments(int(key.desktop), str(key.view)):
            self.report(NO_SEGMENTS)
            return
        if not steps or int(key.step) <= min(steps):
            self.report(NO_SPLIT_HERE.format(step=key.step))
            return
        straddles = self.db.straddling_keyframes(int(key.desktop), str(key.view),
                                                 int(key.step))
        wanted = self.ask_pose_split(
            key, len(straddles),
            carry if carry is not None else len(straddles) == 0,
            note=(proposal or {}).get("note") or "")
        if wanted is None:
            return
        with self.db.transaction():
            self.db.add_pose_break(
                int(key.desktop), str(key.view), int(key.step), status=ACCEPTED,
                kind=(proposal or {}).get("kind") or KIND_MANUAL,
                magnitude_px=(proposal or {}).get("magnitude_px"),
                source=(proposal or {}).get("source") or f"manual:{self.annotator}",
                note=(proposal or {}).get("note") or "")
            self.db.set_pose_break_status(int(key.desktop), str(key.view),
                                          int(key.step), ACCEPTED)
            out = self.db.recut_view(int(key.desktop), str(key.view),
                                     carry_at=[int(key.step)] if wanted else (),
                                     annotator=self.annotator)
        self.pose_bar.hide()
        self._reopen_here(self._recut_message(key.step, out))

    def ask_pose_split(self, key, straddles: int, carry: bool, note: str = ""
                       ) -> Optional[bool]:
        """Show the split dialog; ``None`` when the annotator cancelled.

        A method rather than an inline dialog so a test can answer it without a
        modal event loop.
        """
        dialog = PoseSplitDialog(self, int(key.step), self.session.image_at(key.step - 1),
                                 self.session.image(), straddles, carry, note)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.carry()

    def _recut_message(self, step: int, out: dict) -> str:
        parts = [f"step {step}: 新位姿段 / new pose segment "
                 f"({len(out['ranges'])} segments)"]
        if out["carried"]:
            parts.append(f"{len(out['carried'])} shapes carried")
        if out["rechecked"]:
            parts.append(f"{len(out['rechecked'])} verified frames queued for re-check")
        return "; ".join(parts)

    def _reopen_here(self, message: str) -> None:
        """Re-open the session on the same frame: the segment it is in has moved."""
        key = self.session.current()
        self.session.open(int(key.desktop), str(key.view), force=True)
        if int(key.step) in self.session.steps():
            self.session.goto(int(key.step), force=True)
        self._segment = None          # the zoom belongs to the segment that has gone
        self.render_frame()
        self.review.refresh()
        self.timeline.refresh_statuses()
        self.report(message)
