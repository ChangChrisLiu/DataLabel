"""Offscreen tests for the four dock panels (timeline, task card, instances, review).

The panels hold no business logic: every user gesture must end in a call on the
session.  So the tests drive a :class:`StubSession` -- an in-memory object that
satisfies :class:`tda.ui.session_api.SessionLike` and records every call -- and
assert on what the panels display and on which session methods they call with
which arguments.

Everything runs on the ``offscreen`` Qt platform plugin (see ``conftest.py``);
the env var is also set at import time because the ``QApplication`` is created
by a session fixture that may be built before the autouse fixture runs.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Optional

import pytest
from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from tda.core.compiler import CompiledFrame
from tda.core.model import FrameKey, Visibility
from tda.ui import session_api as api
from tda.ui.canvas.overlay import palette_color
from tda.ui.panels.instances import InstanceListPanel
from tda.ui.panels.review import STEP_ROLE, ReviewPanel
from tda.ui.panels.taskcard import KIND_ICONS, TaskCardPanel
from tda.ui.panels.timeline import TimelinePanel, status_brush


# --------------------------------------------------------------------------- #
# stub session
# --------------------------------------------------------------------------- #
class StubSession(QObject):
    """In-memory stand-in for ``AnnotationSession`` that records its calls."""

    sigFrameChanged = Signal(object)
    sigDirty = Signal(bool)
    sigProblems = Signal(list)
    sigEditingChanged = Signal(object)
    sigClosed = Signal()
    sigSweepProgress = Signal(int, int, int)
    sigSweepError = Signal(int, str)
    sigQueuesChanged = Signal()

    def __init__(self, thumb_dir: Optional[Path] = None) -> None:
        super().__init__()
        self.desktop = 13
        self.view = "scan"
        self._steps = [10, 11, 12, 13]
        self._status = {
            10: api.STATUS_UNLABELED,
            11: api.STATUS_AUTO,
            12: api.STATUS_VERIFIED,
            13: api.STATUS_CONFLICT,
        }
        self._step = 13
        self.calls: list[tuple] = []
        self.thumb_calls: list[int] = []
        self.confirm_result = True
        self.problems = ["missing_shape:screw.cpu_cooler.03"]
        # Keyed like the panel's cache: the same step number is a different
        # image in another view.  Step 11 deliberately has none.
        self._thumbs: dict[tuple[str, int], str] = {}
        if thumb_dir is not None:
            for view_index, view in enumerate(("scan", "oak1")):
                for step in (10, 12, 13):
                    path = thumb_dir / f"thumb_{view}_{step}.png"
                    pm = QPixmap(8, 8)
                    pm.fill(QColor(step, 40 + 60 * view_index, 40))
                    pm.save(str(path), "PNG")
                    self._thumbs[(view, step)] = str(path)
        self._rows = [
            {
                "key": "cpu_cooler.01",
                "cls": "cooler",
                "state": "installed",
                "placement": "in_chassis",
                "visibility": Visibility.VISIBLE.value,
                "z": 2,
                "hidden": False,
            },
            {
                "key": "screw.cpu_cooler.03",
                "cls": "screw",
                "state": "fastened",
                "placement": "in_chassis",
                "visibility": Visibility.OCCLUDED_PARTIAL.value,
                "z": 1,
                "hidden": True,
            },
            {
                "key": "chassis.01",
                "cls": "chassis",
                "state": "present",
                "placement": "in_chassis",
                "visibility": Visibility.VISIBLE.value,
                "z": 0,
                "hidden": False,
            },
        ]
        self._card = [
            {
                "instance": "cpu_cooler.01",
                "kind": api.KIND_ADD_SHAPE,
                "text": "Draw the cooler back inside the chassis",
                "done": True,
            },
            {
                "instance": "screw.cpu_cooler.03",
                "kind": api.KIND_ADD_SHAPE,
                "text": "Draw the attached screw",
                "done": False,
            },
            {
                "instance": "side_panel.01",
                "kind": api.KIND_STATE_ONLY,
                "text": "loosened -> fastened",
                "done": False,
            },
        ]
        self._queues = {
            api.QUEUE_CONFLICTS: [
                {"id": 7, "step": 12, "instance": "cpu_cooler.01", "sym_diff_px": 1234},
                {"id": 8, "step": 11, "instance": "chassis.01", "sym_diff_px": 20},
            ],
            api.QUEUE_NEEDS_REVIEW: [{"step": 11}, {"step": 10}],
            api.QUEUE_MISSING_SHAPE: [{"step": 13, "instance": "screw.cpu_cooler.03"}],
            api.QUEUE_UNEXPLAINED: [{"step": 12}],
        }

    # -- frames -------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return True

    def steps(self) -> list[int]:
        return list(self._steps)

    def frame_status(self, step: int) -> str:
        return self._status[step]

    def thumb_path(self, step: int) -> Optional[str]:
        self.thumb_calls.append(step)
        return self._thumbs.get((self.view, step))

    def switch_view(self, view: str, steps: Optional[list[int]] = None) -> None:
        """Open another view of the same machine, as the main window would."""
        self.view = view
        if steps is not None:
            self._steps = list(steps)
            self._status = {s: api.STATUS_AUTO for s in self._steps}
        self._step = self._steps[-1]
        self.sigFrameChanged.emit(self.current())

    def current(self) -> FrameKey:
        return FrameKey(self.desktop, self._step, self.view)

    def goto(self, step: int) -> None:
        self.calls.append(("goto", step))
        self._step = step
        self.sigFrameChanged.emit(self.current())

    def prev(self) -> None:
        self.calls.append(("prev",))

    def next(self) -> None:
        self.calls.append(("next",))

    def compiled(self) -> CompiledFrame:
        return CompiledFrame(key=self.current(), instances={}, problems=[], input_hash="")

    # -- content ------------------------------------------------------------
    def instance_rows(self) -> list[dict]:
        return [dict(r) for r in self._rows]

    def task_card(self) -> list[dict]:
        return [dict(r) for r in self._card]

    def task_neighbour(self) -> Optional[int]:
        later = [s for s in self._steps if s > self._step]
        return later[0] if later else None

    def overlay_layers(self):
        return {}, []

    # -- edits --------------------------------------------------------------
    def set_editing_mask(self, mask) -> None:
        self.calls.append(("set_editing_mask",))

    def push_stroke(self, before, after) -> None:
        self.calls.append(("push_stroke",))

    def preview(self, scope: str) -> dict:
        self.calls.append(("preview", scope))
        return {"steps": [self._step], "verified_steps": []}

    def set_unexplained(self, step: int, boxes) -> None:
        self.calls.append(("set_unexplained", step, list(boxes)))

    def begin_edit(self, instance: str) -> None:
        self.calls.append(("begin_edit", instance))

    def commit_edit(self, scope: str) -> None:
        self.calls.append(("commit_edit", scope))

    def set_visibility(self, instance: str, vis: str) -> None:
        self.calls.append(("set_visibility", instance, vis))
        self._write(instance, "visibility", vis)

    def set_hidden(self, instance: str, hidden: bool) -> None:
        self.calls.append(("set_hidden", instance, hidden))
        self._write(instance, "hidden", hidden)

    def _write(self, instance: str, field: str, value) -> None:
        """A real session persists these, so the stub must too."""
        for row in self._rows:
            if row["key"] == instance:
                row[field] = value

    def set_zorder_move(self, instance: str, above_of: str) -> None:
        self.calls.append(("set_zorder_move", instance, above_of))

    def confirm_frame(self) -> bool:
        self.calls.append(("confirm_frame",))
        if not self.confirm_result:
            self.sigProblems.emit(list(self.problems))
        return self.confirm_result

    # -- review -------------------------------------------------------------
    def queues(self) -> dict[str, list[dict]]:
        return {k: [dict(e) for e in v] for k, v in self._queues.items()}

    def retry_rechecks(self) -> int:
        self.calls.append(("retry_rechecks",))
        return 0

    def resolve_conflict(self, cid: int, resolution: str) -> None:
        self.calls.append(("resolve_conflict", cid, resolution))
        self._queues[api.QUEUE_CONFLICTS] = [
            e for e in self._queues[api.QUEUE_CONFLICTS] if e["id"] != cid
        ]


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def session(qapp, tmp_path: Path) -> StubSession:
    return StubSession(tmp_path)


def send_key(widget, key, modifiers=Qt.KeyboardModifier.NoModifier, text: str = "") -> bool:
    """Deliver one key press to ``widget`` exactly as Qt would."""
    event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers, text)
    return QApplication.sendEvent(widget, event)


def texts(list_widget) -> list[str]:
    return [list_widget.item(i).text() for i in range(list_widget.count())]


def show(panel, width: int = 260, height: int = 520):
    """Show a panel offscreen so that item rectangles are laid out."""
    panel.resize(width, height)
    panel.show()
    QApplication.processEvents()
    return panel


def click_item(view, item, double: bool = False) -> None:
    """Click an item through the real event path, at its centre in the viewport.

    A double click is sent as a click followed by the double-click sequence,
    which is both what Qt sees from a real mouse and what makes it reliable
    here: the very first synthetic press on a freshly shown window is
    swallowed, and ``mouseDClick`` needs a non-default delay or its two
    presses share a timestamp and stop being a double click.
    """
    view.scrollToItem(item)
    QApplication.processEvents()
    center = view.visualItemRect(item).center()
    target = view.viewport()
    QTest.mouseClick(
        target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center, 10
    )
    if double:
        QTest.mouseDClick(
            target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
            center, 10,
        )
    QApplication.processEvents()


# --------------------------------------------------------------------------- #
# session_api
# --------------------------------------------------------------------------- #
def test_stub_satisfies_the_protocol(session: StubSession) -> None:
    assert isinstance(session, api.SessionLike)


def test_constants_cover_the_spec_domains() -> None:
    assert api.VISIBILITY_VALUES == tuple(v.value for v in Visibility)
    assert len(api.VISIBILITY_VALUES) == 7
    assert api.FRAME_STATUSES == (
        "unlabeled",
        "auto",
        "verified",
        "needs_review",
        "conflict",
        "recheck",
        "missing",
    )
    assert api.TASK_KINDS == (
        "add_shape",
        "split_keyframe",
        "state_only",
        "add_bench_box",
        "remove_bench_box",
        "confirm",
    )
    assert api.COMMIT_SCOPES == ("keyframe", "frame_override", "split")
    assert api.QUEUE_NAMES == (
        "conflicts",
        "needs_review",
        "missing_shape",
        "unexplained",
    )
    assert api.RESOLUTIONS == ("keep_old", "accept_new")


# --------------------------------------------------------------------------- #
# TimelinePanel
# --------------------------------------------------------------------------- #
def test_timeline_lists_steps_descending_by_default(session: StubSession) -> None:
    panel = TimelinePanel(session)
    assert panel.is_descending() is True
    assert panel.item_steps() == [13, 12, 11, 10]
    assert "13" in texts(panel.list_widget())[0]


def test_timeline_order_toggle_keeps_the_current_step_selected(session: StubSession) -> None:
    panel = TimelinePanel(session)
    assert panel.current_step() == 13
    panel.order_toggle.setChecked(True)  # "Oldest first"
    assert panel.is_descending() is False
    assert panel.item_steps() == [10, 11, 12, 13]
    assert panel.current_step() == 13
    assert panel.list_widget().currentRow() == 3


def test_timeline_colours_follow_the_status(session: StubSession) -> None:
    panel = TimelinePanel(session)
    assert panel.step_brush(10).color() == status_brush(api.STATUS_UNLABELED).color()
    grey = status_brush(api.STATUS_UNLABELED).color()
    assert grey.red() == grey.green() == grey.blue()
    auto = status_brush(api.STATUS_AUTO).color()
    assert auto.red() > 180 and auto.green() > 150 and auto.blue() < 110
    verified = status_brush(api.STATUS_VERIFIED).color()
    assert verified.green() > verified.red() and verified.green() > verified.blue()
    for bad in (api.STATUS_CONFLICT, api.STATUS_NEEDS_REVIEW):
        red = status_brush(bad).color()
        assert red.red() > red.green() and red.red() > red.blue()
    assert status_brush(api.STATUS_CONFLICT).color() == status_brush(
        api.STATUS_NEEDS_REVIEW
    ).color()
    assert status_brush(api.STATUS_MISSING).style() == Qt.BrushStyle.BDiagPattern


def test_timeline_click_asks_for_that_step(session: StubSession) -> None:
    """It called an un-forced ``session.goto``, which raises on a dirty layer.

    Unreachable while the window re-wires ``itemClicked``, but a panel that can
    raise ``SessionRefusal`` out of a Qt slot the moment somebody forgets the
    re-wire is not a panel anybody should have to remember.
    """
    panel = show(TimelinePanel(session))
    seen: list[int] = []
    panel.sigOpenStep.connect(seen.append)
    lw = panel.list_widget()
    click_item(lw, lw.item(panel.item_steps().index(11)))
    assert seen == [11]
    assert session.calls == []
    assert panel.current_step() == 11


def test_timeline_follows_frame_changes(session: StubSession) -> None:
    panel = TimelinePanel(session)
    session._status[13] = api.STATUS_VERIFIED
    session._step = 11
    session.sigFrameChanged.emit(session.current())
    assert panel.current_step() == 11
    assert panel.list_widget().currentRow() == panel.item_steps().index(11)
    assert panel.step_brush(13).color() == status_brush(api.STATUS_VERIFIED).color()


def test_timeline_loads_thumbnails_lazily_and_caches_them(session: StubSession) -> None:
    panel = TimelinePanel(session)
    assert session.thumb_calls == []  # nothing loaded while the panel is hidden
    pm = panel.thumbnail(12)
    assert isinstance(pm, QPixmap) and not pm.isNull()
    assert pm.height() <= TimelinePanel.THUMB_SIZE
    assert session.thumb_calls == [12]
    assert panel.thumbnail(12) is pm  # cached: no second read
    assert session.thumb_calls == [12]
    # a step without a thumbnail gets a cached placeholder, not a repeated miss
    assert panel.thumbnail(11) is not None
    assert panel.thumbnail(11) is not None
    assert session.thumb_calls == [12, 11]


def test_timeline_loads_visible_thumbnails_when_shown(session: StubSession) -> None:
    panel = show(TimelinePanel(session), 200, 400)
    assert session.thumb_calls  # the visible rows were filled in


def test_timeline_thumbnail_cache_is_scoped_to_the_view(session: StubSession) -> None:
    panel = TimelinePanel(session)
    scan_12 = panel.thumbnail(12)
    assert session.thumb_calls == [12]

    session.switch_view("oak1", [11, 12, 13])
    assert panel.item_steps() == [13, 12, 11]  # the step list was rebuilt
    oak_12 = panel.thumbnail(12)
    assert oak_12 is not scan_12  # not the other view's image
    assert session.thumb_calls == [12, 12]

    session.switch_view("scan", [10, 11, 12, 13])
    assert panel.item_steps() == [13, 12, 11, 10]
    assert panel.thumbnail(12) is scan_12  # still cached under its own view
    assert session.thumb_calls == [12, 12]


# --------------------------------------------------------------------------- #
# TaskCardPanel
# --------------------------------------------------------------------------- #
def test_taskcard_lists_items_with_icons_and_highlights_the_first_open_one(
    session: StubSession,
) -> None:
    panel = TaskCardPanel(session)
    rows = texts(panel.list_widget())
    assert len(rows) == 3
    assert KIND_ICONS[api.KIND_ADD_SHAPE] in rows[0]
    assert "cpu_cooler.01" in rows[0] and "Draw the cooler" in rows[0]
    assert KIND_ICONS[api.KIND_STATE_ONLY] in rows[2]
    assert panel.current_index() == 1  # first not-done item
    assert panel.list_widget().currentRow() == 1
    assert panel.list_widget().item(0).font().strikeOut() is True
    assert panel.list_widget().item(1).font().bold() is True


def test_taskcard_buttons_only_report(session: StubSession) -> None:
    """The buttons emit; the window acts.  Calling the session from here let
    "Confirm" step the frame back over an uncommitted layer."""
    from tda.ui.panels.taskcard import SUGGESTED

    panel = TaskCardPanel(session)
    # Short captions keep the dock narrow; the full sentence is the tooltip.
    for button, key in ((panel.commit_button, "Enter"),
                        (panel.override_button, "Alt+Enter"),
                        (panel.split_button, "Ctrl+K"),
                        (panel.confirm_button, "Space")):
        assert key in button.toolTip()
        assert len(button.text()) <= 20

    scopes: list[str] = []
    confirms: list[int] = []
    panel.sigCommit.connect(scopes.append)
    panel.sigConfirm.connect(lambda: confirms.append(1))
    panel.commit_button.click()
    panel.override_button.click()
    panel.split_button.click()
    panel.confirm_button.click()
    assert scopes == [SUGGESTED, api.SCOPE_FRAME_OVERRIDE, api.SCOPE_SPLIT]
    assert confirms == [1]
    assert session.calls == []


def test_taskcard_methods_are_what_the_keys_call(session: StubSession) -> None:
    """The panel has no key table of its own; the window's ACTIONS calls these.

    ``commit(scope)`` is gone with it: it wrote straight to the session, so the
    window never cleared the layer, never dropped the crash sidecar and never
    offered the scope bar.
    """
    panel = TaskCardPanel(session)
    assert not hasattr(panel, "commit")
    panel.confirm()
    assert session.calls == [("confirm_frame",)]
    assert not hasattr(panel, "handle_key")


def test_taskcard_shows_problems_when_confirm_fails(session: StubSession) -> None:
    panel = TaskCardPanel(session)
    assert panel.problems_visible() is False
    session.confirm_result = False
    panel.confirm()          # what the window's act_confirm calls
    assert panel.problems_visible() is True
    assert panel.problems() == ["missing_shape:screw.cpu_cooler.03"]
    session.confirm_result = True
    panel.confirm()
    assert panel.problems_visible() is False


def test_taskcard_activation_only_reports_the_request(session: StubSession) -> None:
    """The window owns ``begin_edit``: it is the one that can refuse it."""
    panel = show(TaskCardPanel(session), 320, 320)
    seen: list[str] = []
    panel.sigRequestEdit.connect(seen.append)
    lw = panel.list_widget()
    click_item(lw, lw.item(1), double=True)
    assert session.calls == []
    assert seen == ["screw.cpu_cooler.03"]


def test_taskcard_problems_do_not_survive_a_frame_change(session: StubSession) -> None:
    panel = TaskCardPanel(session)
    session.confirm_result = False
    panel.confirm()
    assert panel.problems_visible() is True

    session.sigFrameChanged.emit(session.current())  # next frame
    assert panel.problems_visible() is False
    assert panel.problems() == []

    # a refusal that sends no problems must not resurrect the old list
    session.problems = []
    panel.confirm_button.click()
    assert panel.problems() == []


def test_taskcard_refreshes_on_frame_change(session: StubSession) -> None:
    panel = TaskCardPanel(session)
    session._card = [
        {"instance": "psu.01", "kind": api.KIND_CONFIRM, "text": "Duplicate frame", "done": False}
    ]
    session.sigFrameChanged.emit(session.current())
    rows = texts(panel.list_widget())
    assert len(rows) == 1 and "psu.01" in rows[0]
    assert panel.current_index() == 0


# --------------------------------------------------------------------------- #
# InstanceListPanel
# --------------------------------------------------------------------------- #
def test_instances_table_shows_its_four_columns_and_tells_the_rest(
        session: StubSession) -> None:
    """Class and placement moved into the tooltip so the dock can be narrow."""
    panel = InstanceListPanel(session)
    table = panel.table()
    assert table.rowCount() == 3
    assert table.columnCount() == len(InstanceListPanel.COLUMNS)
    swatch = table.item(0, 0)
    assert swatch.background().color() == QColor(*palette_color("cpu_cooler.01"))
    assert table.item(0, 1).text() == "cpu_cooler.01"
    assert table.item(0, 2).text() == "installed"
    tooltip = table.item(0, 1).toolTip()
    assert "cooler" in tooltip and "in_chassis" in tooltip
    assert Visibility.VISIBLE.value in tooltip
    hidden_col = InstanceListPanel.COLUMNS.index("Hidden")
    assert table.item(0, hidden_col).checkState() == Qt.CheckState.Unchecked
    assert table.item(1, hidden_col).checkState() == Qt.CheckState.Checked


def test_instances_checkbox_reports_hidden(session: StubSession) -> None:
    """The window applies it, so that the canvas overlay is repainted with it."""
    panel = InstanceListPanel(session)
    seen: list[tuple] = []
    panel.sigHiddenToggled.connect(lambda key, hidden: seen.append((key, hidden)))
    table = panel.table()
    hidden_col = InstanceListPanel.COLUMNS.index("Hidden")
    table.item(0, hidden_col).setCheckState(Qt.CheckState.Checked)
    assert seen == [("cpu_cooler.01", True)]
    assert session.calls == []


def test_instances_h_sends_the_value_the_session_holds(session: StubSession) -> None:
    """``H`` reads the session's row, not the checkbox somebody just clicked."""
    panel = InstanceListPanel(session)
    panel.select_instance("cpu_cooler.01")
    hidden_col = InstanceListPanel.COLUMNS.index("Hidden")
    panel.table().item(0, hidden_col).setCheckState(Qt.CheckState.Checked)
    session._rows[0]["hidden"] = True     # what the window's slot would produce
    panel.refresh()
    panel.toggle_hidden()
    assert session.calls == [("set_hidden", "cpu_cooler.01", False)]
    # and the table is re-read from the session afterwards, never patched locally
    assert panel.rows()[0]["hidden"] == session.instance_rows()[0]["hidden"]


def test_instances_toggle_hidden_flips_the_selected_row(session: StubSession) -> None:
    panel = InstanceListPanel(session)
    panel.select_instance("screw.cpu_cooler.03")
    panel.toggle_hidden()
    assert session.calls == [("set_hidden", "screw.cpu_cooler.03", False)]


def test_instances_v_cycles_visibility(session: StubSession) -> None:
    panel = InstanceListPanel(session)
    panel.select_instance("cpu_cooler.01")  # currently "visible"
    panel.cycle_visibility()
    assert session.calls == [
        ("set_visibility", "cpu_cooler.01", Visibility.OCCLUDED_PARTIAL.value)
    ]
    panel.cycle_visibility()  # again: it must advance
    assert session.calls == [
        ("set_visibility", "cpu_cooler.01", Visibility.OCCLUDED_PARTIAL.value),
        ("set_visibility", "cpu_cooler.01", Visibility.OCCLUDED_FULL.value),
    ]
    session.calls.clear()
    panel.select_instance("screw.cpu_cooler.03")  # "occluded_partial"
    panel.cycle_visibility()
    assert session.calls == [
        ("set_visibility", "screw.cpu_cooler.03", Visibility.OCCLUDED_FULL.value)
    ]


def test_instances_forget_a_selection_that_left_the_frame(session: StubSession) -> None:
    """Removed parts drop out of the table; H / V / 1-7 must not chase them."""
    panel = InstanceListPanel(session)
    panel.select_instance("screw.cpu_cooler.03")
    assert panel.selected_instance() == "screw.cpu_cooler.03"
    session._rows = [r for r in session._rows if r["key"] != "screw.cpu_cooler.03"]
    panel.refresh()
    assert panel.selected_instance() is None
    session.calls.clear()
    panel.toggle_hidden()
    panel.cycle_visibility()
    panel.set_visibility(api.VISIBILITY_VALUES[0])
    assert session.calls == []   # nothing acted on the key that is gone


def test_instances_set_visibility_covers_the_seven_values(session: StubSession) -> None:
    """``1``-``7`` are bound by the window's table; the panel takes a value."""
    panel = InstanceListPanel(session)
    panel.select_instance("cpu_cooler.01")
    for value in api.VISIBILITY_VALUES:
        panel.set_visibility(value)
    assert session.calls == [
        ("set_visibility", "cpu_cooler.01", v) for v in api.VISIBILITY_VALUES
    ]
    assert not hasattr(panel, "handle_key")


def test_instances_up_and_down_move_the_z_order(session: StubSession) -> None:
    panel = InstanceListPanel(session)
    seen: list[int] = []
    panel.sigReorder.connect(seen.append)
    panel.up_button.click()
    panel.down_button.click()
    assert seen == [-1, +1]          # the buttons report; the window acts
    assert session.calls == []

    panel.select_instance("screw.cpu_cooler.03")  # middle row
    panel.move_up()
    assert session.calls == [("set_zorder_move", "screw.cpu_cooler.03", "cpu_cooler.01")]
    session.calls.clear()
    panel.move_down()
    # moving a row down = the row below it goes above the selected one
    assert session.calls == [("set_zorder_move", "chassis.01", "screw.cpu_cooler.03")]
    session.calls.clear()
    panel.select_instance("cpu_cooler.01")  # top row: up does nothing
    panel.move_up()
    assert session.calls == []


def test_instances_ctrl_arrows_reorder(session: StubSession) -> None:
    panel = InstanceListPanel(session)
    panel.select_instance("screw.cpu_cooler.03")  # middle row
    panel.move_up()
    assert session.calls == [("set_zorder_move", "screw.cpu_cooler.03", "cpu_cooler.01")]
    session.calls.clear()
    panel.move_down()
    assert session.calls == [("set_zorder_move", "chassis.01", "screw.cpu_cooler.03")]
    session.calls.clear()
    # the bare arrows stay with the table's own row navigation
    send_key(panel.table(), Qt.Key.Key_Up)
    assert session.calls == []


def test_instances_double_click_only_reports_the_request(session: StubSession) -> None:
    """The window owns ``begin_edit``: it is the one that can refuse it."""
    panel = show(InstanceListPanel(session), 520, 320)
    seen: list[str] = []
    panel.sigRequestEdit.connect(seen.append)
    table = panel.table()
    click_item(table, table.item(2, 1), double=True)
    assert session.calls == []
    assert seen == ["chassis.01"]


def test_instances_refresh_on_frame_change(session: StubSession) -> None:
    panel = InstanceListPanel(session)
    session._rows = [dict(session._rows[0], key="psu.01", cls="psu")]
    session.sigFrameChanged.emit(session.current())
    table = panel.table()
    assert table.rowCount() == 1
    assert table.item(0, 1).text() == "psu.01"


# --------------------------------------------------------------------------- #
# ReviewPanel
# --------------------------------------------------------------------------- #
def test_review_has_one_tab_per_queue(session: StubSession) -> None:
    panel = ReviewPanel(session)
    tabs = panel.tabs()
    assert tabs.count() == 4
    assert [panel.queue_at(i) for i in range(4)] == list(api.QUEUE_NAMES)
    assert "2" in tabs.tabText(0)  # conflict count in the tab label
    conflicts = texts(panel.list_for(api.QUEUE_CONFLICTS))
    assert "12" in conflicts[0] and "cpu_cooler.01" in conflicts[0] and "1234" in conflicts[0]
    assert texts(panel.list_for(api.QUEUE_NEEDS_REVIEW)) == ["Step 11", "Step 10"]
    assert "screw.cpu_cooler.03" in texts(panel.list_for(api.QUEUE_MISSING_SHAPE))[0]
    assert texts(panel.list_for(api.QUEUE_UNEXPLAINED)) == ["Step 12"]


@pytest.mark.parametrize(
    "queue,row,step",
    [
        (api.QUEUE_CONFLICTS, 0, 12),
        (api.QUEUE_NEEDS_REVIEW, 1, 10),
        (api.QUEUE_MISSING_SHAPE, 0, 13),
        (api.QUEUE_UNEXPLAINED, 0, 12),
    ],
)
def test_review_activation_asks_for_the_frame(
    session: StubSession, queue: str, row: int, step: int
) -> None:
    """It called an un-forced ``session.goto``, which an uncommitted layer raised on."""
    panel = ReviewPanel(session)
    seen: list[int] = []
    panel.sigOpenStep.connect(seen.append)
    lw = panel.list_for(queue)
    lw.itemActivated.emit(lw.item(row))
    assert seen == [step]
    assert session.calls == []


def test_review_resolution_buttons_report_the_verdict(session: StubSession) -> None:
    """A resolution has three outcomes and only the window can tell which."""
    panel = ReviewPanel(session)
    seen: list[str] = []
    panel.sigResolve.connect(seen.append)
    lw = panel.list_for(api.QUEUE_CONFLICTS)
    lw.setCurrentRow(0)
    panel.keep_old_button.click()
    assert seen == [api.RESOLVE_KEEP_OLD]
    assert panel.selected_conflict() == 7
    panel.accept_new_button.click()
    assert seen == [api.RESOLVE_KEEP_OLD, api.RESOLVE_ACCEPT_NEW]
    assert session.calls == []


def test_review_has_no_confirm_of_its_own(session: StubSession) -> None:
    """``Enter`` is the window's binding and it goes through the task card."""
    panel = ReviewPanel(session)
    assert not hasattr(panel, "confirm")
    assert not hasattr(panel, "open_selected")
    assert not hasattr(panel, "_goto")


def test_review_rework_reports_the_selected_step(session: StubSession) -> None:
    panel = ReviewPanel(session)
    seen: list[int] = []
    panel.sigRework.connect(seen.append)
    lw = panel.list_for(api.QUEUE_CONFLICTS)
    lw.setCurrentRow(1)
    panel.rework()               # the window binds R to this
    assert seen == [11]
    lw.setCurrentRow(-1)
    panel.rework()
    assert seen == [11, session.current().step]  # falls back to the current frame
    assert not hasattr(panel, "handle_key")


def test_review_refreshes_on_frame_change(session: StubSession) -> None:
    panel = ReviewPanel(session)
    session._queues[api.QUEUE_UNEXPLAINED] = [{"step": 10}, {"step": 11}]
    session.sigFrameChanged.emit(session.current())
    assert texts(panel.list_for(api.QUEUE_UNEXPLAINED)) == ["Step 10", "Step 11"]


def test_review_snap_back_only_touches_the_list_that_was_clicked(session) -> None:
    """A refused click in one tab cleared a conflict selected in another.

    The panel restored the open frame in all four queues, so the annotator lost
    the conflict they had picked -- the `Keep old` / `Take new` keys then had
    nothing to act on.
    """
    panel = ReviewPanel(session)
    conflicts = panel.list_for(api.QUEUE_CONFLICTS)
    conflicts.setCurrentRow(1)
    unexplained = panel.list_for(api.QUEUE_UNEXPLAINED)
    unexplained.setCurrentRow(0)
    unexplained.itemActivated.emit(unexplained.item(0))     # the click that is refused

    panel.select_current_step()

    assert conflicts.currentRow() == 1, "another tab's selection was cleared"
    assert unexplained.currentRow() != 0 or int(
        unexplained.item(0).data(STEP_ROLE)) == session.current().step
