"""The status bar: what the window says about itself, and how it says it.

Mixed into :class:`tda.ui.app.MainWindow`.  Four permanent labels -- zoom,
frame, tool, SAM -- plus one transient line, and the rules that keep the line
useful: it elides rather than widening the window, an important hint can hold
its place for a moment against a background job's answer, and every error also
goes to the log and to :meth:`last_error_message` so a later hint cannot erase
the only record of it.

The machine chooser's ``[done/total]`` lives here too, because it is the same
kind of thing: a number on screen that has to follow the work rather than the
launch.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy

from tda.ui import app_compat as compat

__all__ = ["KNOWN_FAILURES", "StatusMixin", "explain_exception"]

#: Everything else keeps the exception's own text, which is what a bug report
#: needs; these three are the ones that are *not* a bug.
KNOWN_FAILURES: tuple[tuple[str, str], ...] = (
    ("database is locked",
     "数据库正忙，稍后重试（另一个 tda 进程在写？）/ the database is busy"),
    ("disk full",
     "磁盘已满：先腾出空间再继续 / the disk is full"),
    ("out of memory",
     "显存/内存不足：关掉别的程序，或缩小视野再试 / out of memory"),
)


def explain_exception(exc: BaseException) -> str:
    """One sentence for the status bar, human where we know how to be."""
    text = str(exc)
    low = text.lower()
    for needle, sentence in KNOWN_FAILURES:
        if needle in low:
            return f"{sentence}（{type(exc).__name__}: {text}）"
    return f"{type(exc).__name__}: {text}"




class StatusMixin:
    """The four permanent labels, the transient line and the counters."""

    def _build_status_bar(self) -> None:
        self.zoom_label = QLabel("100%")
        self.frame_label = QLabel("")
        self.tool_label = QLabel("")
        self.sam_label = QLabel("")
        self.hint_label = QLabel("")
        # The Chinese hints start with a full-width glyph, which Qt draws hard
        # against the window edge without this.
        self.hint_label.setContentsMargins(8, 0, 4, 0)
        # A status line is one line: it elides, it never asks for width, and
        # the whole text is in the tooltip.  A refused confirmation on a start
        # frame put 2,550 characters here and the label's minimum width became
        # 30,612 px, which is the window asking to be six screens wide.
        self.hint_label.setSizePolicy(QSizePolicy.Policy.Ignored,
                                      QSizePolicy.Policy.Preferred)
        self.hint_label.setMinimumWidth(1)
        self.hint_label.setTextFormat(Qt.TextFormat.PlainText)
        bar = self.statusBar()
        for label in (self.zoom_label, self.frame_label, self.tool_label,
                      self.sam_label):
            bar.addPermanentWidget(label)
        bar.addWidget(self.hint_label, 1)


    # ----------------------------------------------------------- status bar
    def update_status(self) -> None:
        """Rewrite the four permanent status labels from the current state."""
        self.zoom_label.setText(f"{self.canvas.zoom_factor() * 100:.0f}%")
        self.frame_label.setText(self._frame_text())
        radius = getattr(self.active_tool, "radius", None)
        suffix = "" if radius is None else f" r{radius}"
        self.tool_label.setText(f"{self._tool_name}{suffix}")
        self.sam_label.setText(self.sam_status_text())

    def _frame_text(self) -> str:
        if not compat.is_open(self.session):
            return "no frame"
        if self.is_flashing():
            # What is on the canvas is not what the rest of the window is about,
            # and the annotator has to be able to see that at a glance.
            which = "另一帧" if self._flashing < 0 else f"step {self._flashing}"
            return f"正在对照 {which} / comparing — 松开 Tab 返回"
        key = self.session.current()
        steps = self.session.steps()
        total = max(steps) if steps else key.step
        status = self.session.frame_status(key.step)
        return f"D{key.desktop} · {key.view} · step {key.step}/{total} · {status}"

    def _view_counts(self) -> tuple[dict, dict]:
        """``(verified, frames)`` per ``(desktop, view)``; two whole-table scans.

        Read once and passed around: at 66 desktops, asking per row turned the
        combo box into 132 aggregate queries and 0.7 s of the start-up.
        """
        return self.db.count_per_view("verified"), self.db.count_per_view("frames")

    def refresh_desktop_counts(self) -> None:
        """Re-read ``[done/total]`` for the machine that is open.

        It was computed once at launch and never again, so the number the guide
        tells the annotator to watch stood still all day -- and it is per view,
        so switching view made it wrong rather than stale.  Only the current
        row is rewritten: the other 65 cannot have changed, and the two
        aggregate queries are the whole cost.
        """
        index = self.desktop_combo.currentIndex()
        if index < 0 or self.session.desktop is None:
            return
        text = self._desktop_text(int(self.session.desktop))
        if text != self.desktop_combo.itemText(index):
            blocked = self.desktop_combo.blockSignals(True)
            self.desktop_combo.setItemText(index, text)
            self.desktop_combo.blockSignals(blocked)

    def _desktop_text(self, desktop: int, counts: Optional[tuple] = None) -> str:
        """``D13 Dell OptiPlex 7020 [12/38]`` -- brand, model and this view's count.

        The model matters on a bench with four Dells on it: the brand alone does
        not tell two machines apart.
        """
        meta = self.db.get_desktop(desktop) or {}
        verified, frames = counts if counts is not None else self._view_counts()
        done = verified.get((desktop, self.session.view), 0)
        total = frames.get((desktop, self.session.view), 0)
        name = " ".join(str(meta.get(k) or "").strip()
                        for k in ("brand", "model_family") if meta.get(k))
        return " ".join(f"D{desktop} {name} [{done}/{total}]".split())

    def status_message(self) -> str:
        """The last transient line shown in the status bar."""
        return self._message

    def report(self, text: str, hold_ms: int = 0) -> None:
        """Show a transient line (a hint, a count, a refusal), elided to fit.

        ``hold_ms`` keeps it there against the *next* ordinary line: the flash
        hint was written the moment the annotator clicked and wiped a few
        milliseconds later by a difference-map result that had been running
        since the frame change, so the one message they needed was the one they
        never saw.  Clearing (an empty text) always wins -- that is a frame
        change, which every hint is about.
        """
        import time as _time

        now = _time.monotonic()
        if text and hold_ms <= 0 and now < getattr(self, "_hint_until", 0.0):
            return
        self._hint_until = (now + hold_ms / 1000.0) if text and hold_ms > 0 else 0.0
        self._message = str(text)
        self.hint_label.setToolTip(self._message)
        self._paint_hint()

    def _paint_hint(self) -> None:
        """Put as much of the message on screen as the bar is wide."""
        metrics = self.hint_label.fontMetrics()
        room = max(40, self.hint_label.width() - 12)
        self.hint_label.setText(
            metrics.elidedText(self._message, Qt.TextElideMode.ElideRight, room)
        )

    def last_error_message(self) -> str:
        """The last error shown, which a later hint does not erase."""
        return self._last_error

    def report_error(self, text: str) -> None:
        """Show an error line and log it.

        Never held back: :meth:`report`'s hold keeps an important *hint* from
        being wiped by a background job's answer, and an error is not the kind
        of line that should wait behind one.
        """
        self.logger.error("%s", text)
        self._last_error = str(text)
        self._hint_until = 0.0
        self.report(str(text))

    def report_exception(self, exc: BaseException, where: str = "") -> None:
        """What :func:`tda.ui.app_support.guard` calls; never raises itself.

        The rollback matters as much as the message: a slot that failed halfway
        through a write would otherwise leave the connection inside a
        transaction, and every later write would join it.
        """
        self.logger.exception("exception in %s: %s", where, exc)
        try:
            if self.db.conn.in_transaction:
                self.db.conn.rollback()
        except Exception:  # pragma: no cover - a rollback failure is terminal
            self.logger.exception("rollback after %s failed", where)
        # Through report_error, so that last_error_message() records it: a
        # failure that only reached the hint line was erased by the next hint
        # and there was no way to find out what had happened.
        self.report_error(explain_exception(exc))

