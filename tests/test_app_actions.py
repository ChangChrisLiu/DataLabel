"""The one shortcut table and everything generated from it (task 13b, spec 4.6).

:data:`tda.ui.app_actions.ACTIONS` is the single source for the key bindings,
the in-app cheat sheet and the ``快捷键`` section of ``docs/annotation_guide.md``.
The table-driven test below presses every declared key on a real main window and
checks that the slot it names is the one that runs -- a binding that exists only
in the documentation, or a slot the window does not have, fails here.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from app_scene import close_window, StubSamQueue, make_paths, make_session
from tda.ui import app_actions as A
from tda.ui.app import MainWindow

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "annotation_guide.md"
#: A floor, not the exact count: the table may grow, but it may not shrink to
#: the point where the brief's map is no longer covered.
ACTION_COUNT_MIN = 40


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    session = make_session(tmp_path)
    win = MainWindow(session, make_paths(tmp_path), "tester",
                     sam_queue=StubSamQueue())
    yield win
    close_window(win)


def key_event(spec: str, press: bool = True) -> QKeyEvent:
    """A ``QKeyEvent`` for a portable key string such as ``"Ctrl+Z"``."""
    combination = QKeySequence.fromString(spec)[0]
    kind = QKeyEvent.Type.KeyPress if press else QKeyEvent.Type.KeyRelease
    return QKeyEvent(kind, int(combination.key()), combination.keyboardModifiers())


# --------------------------------------------------------------------------- #
# the table itself
# --------------------------------------------------------------------------- #
def test_every_action_has_a_name_keys_and_chinese_label():
    assert len(A.ACTIONS) >= ACTION_COUNT_MIN
    groups = dict(A.GROUPS)
    for action in A.ACTIONS:
        assert action.name and action.keys and action.slot
        assert action.label and action.label_zh
        assert action.group in groups


def test_action_names_and_bindings_are_unique():
    names = [a.name for a in A.ACTIONS]
    assert len(names) == len(set(names))
    seen: dict[tuple[int, str], str] = {}
    for action in A.ACTIONS:
        for combo in A.combos_of(action):
            for mode in action.modes:
                assert (combo, mode) not in seen, (
                    f"{action.name} collides with {seen.get((combo, mode))}"
                )
                seen[(combo, mode)] = action.name


def test_views_are_f1_to_f4_and_digits_set_visibility():
    """The ruling of the brief: ``F1..F4`` are views, ``1..7`` visibility."""
    views = {a.args[0]: a.keys[0] for a in A.ACTIONS if a.slot == "act_set_view"}
    assert views == {"scan": "F1", "oak1": "F2", "oak2": "F3", "rs": "F4"}
    digits = sorted(a.keys[0] for a in A.ACTIONS if a.slot == "act_set_visibility")
    assert digits == [str(i) for i in range(1, 8)]


def test_lookup_matches_a_declared_key():
    action = A.action_for(Qt.Key.Key_B, Qt.KeyboardModifier.NoModifier,
                          A.MODE_ANNOTATE)
    assert action is not None and action.slot == "act_tool"
    assert A.action_for(Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier,
                        A.MODE_ANNOTATE) is None


def test_visibility_keys_are_annotate_only():
    assert A.action_for(Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier,
                        A.MODE_ANNOTATE) is not None
    assert A.action_for(Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier,
                        A.MODE_STEPS) is None


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("action", A.ACTIONS, ids=lambda a: a.name)
def test_every_shortcut_dispatches_to_its_slot(window, action):
    calls: list[tuple] = []
    assert hasattr(window, action.slot), f"{action.slot} is missing"
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(action.modes[0])
    for spec in action.keys:
        calls.clear()
        assert window.handle_key(key_event(spec)) is True, spec
        assert calls == [tuple(action.args)], f"{spec} did not reach {action.slot}"


def test_shortcuts_do_not_fire_while_a_line_edit_has_focus(window):
    calls: list[tuple] = []
    window.act_tool = lambda *args: calls.append(args)
    window.set_mode(A.MODE_ANNOTATE)
    editor = QLineEdit(window)
    editor.show()
    editor.setFocus()
    QApplication.processEvents()
    assert A.blocks_shortcuts(editor) is True
    assert window.handle_key(key_event("B")) is False
    assert calls == []


def test_shortcuts_do_not_fire_while_a_combo_has_focus(window):
    combo = QComboBox(window)
    combo.addItems(["a", "b"])
    combo.show()
    combo.setFocus()
    QApplication.processEvents()
    assert A.blocks_shortcuts(combo) is True
    assert window.handle_key(key_event("B")) is False


def test_tab_and_shift_tab_are_the_hold_actions(window):
    held = [a for a in A.ACTIONS if a.hold]
    assert [a.keys[0] for a in held] == ["Tab", "Shift+Tab"]
    seen: list[bool] = []
    window.act_flash_compare = lambda on: seen.append(on)
    window.set_mode(A.MODE_ANNOTATE)
    window.handle_key(key_event("Tab"))
    window.handle_key(key_event("Tab", press=False))
    assert seen == [True, False]

    other: list[bool] = []
    window.act_flash_other = lambda on: other.append(on)
    window.handle_key(key_event("Shift+Tab"))
    window.handle_key(key_event("Shift+Tab", press=False))
    assert other == [True, False]


# --------------------------------------------------------------------------- #
# generated documentation
# --------------------------------------------------------------------------- #
def test_cheat_sheet_names_every_action():
    html = A.cheat_sheet_html()
    for action in A.ACTIONS:
        assert action.label_zh in html
        assert action.keys[0] in html


def test_shortcut_markdown_is_a_table_of_every_action():
    text = A.shortcut_markdown()
    assert text.splitlines()[0].startswith("| 快捷键 |")
    for action in A.ACTIONS:
        assert action.label_zh in text


def test_annotation_guide_embeds_the_generated_table():
    """The guide's 快捷键 section is the table, not a hand-written copy."""
    body = GUIDE.read_text(encoding="utf-8")
    assert A.shortcut_markdown() in body
    prose = [l for l in body.splitlines() if not l.startswith("| `")]
    # A budget, not a fact: the table may grow freely, the prose may not drift.
    # Raised from 200 by the two FAQ entries about parts that leave inside
    # something else (board-mounted latches) and covers logged as `open`.
    assert len(prose) <= 212, "the prose is the part that has to stay readable"


# --------------------------------------------------------------------------- #
# holding a key (final review, item 3)
# --------------------------------------------------------------------------- #
def repeat_event(spec: str) -> QKeyEvent:
    """The auto-repeat Qt sends while a key is held down."""
    combination = QKeySequence.fromString(spec)[0]
    return QKeyEvent(QKeyEvent.Type.KeyPress, int(combination.key()),
                     combination.keyboardModifiers(), 0, 0, 0, "", True)


REPEATABLE = {"radius_down": "[", "radius_up": "]", "opacity_down": ",",
              "opacity_up": ".", "step_back": "PgDown", "step_forward": "PgUp",
              "zorder_up": "Ctrl+Up", "zorder_down": "Ctrl+Down"}


def test_the_repeatable_actions_are_marked_as_such():
    marked = {a.name for a in A.ACTIONS if a.repeat}
    assert marked == set(REPEATABLE), f"marked: {sorted(marked)}"
    assert not any(a.repeat and a.hold for a in A.ACTIONS)


@pytest.mark.parametrize("name", sorted(REPEATABLE), ids=sorted(REPEATABLE))
def test_holding_a_repeatable_key_keeps_firing(window, name):
    """``if event.isAutoRepeat(): return True`` killed every held key.

    Holding ``]`` to grow the brush, ``.`` to fade the overlay or ``PgDn`` to
    walk back through the machine did nothing at all after the first press.
    """
    action = next(a for a in A.ACTIONS if a.name == name)
    calls: list[tuple] = []
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(action.modes[0])

    assert window.handle_key(key_event(REPEATABLE[name])) is True
    for _ in range(3):
        assert window.handle_key(repeat_event(REPEATABLE[name])) is True

    assert calls == [tuple(action.args)] * 4


@pytest.mark.parametrize("spec,name", [("Return", "commit"), ("Space", "confirm"),
                                       ("B", "tool_brush"), ("A", "toggle_overlays")])
def test_holding_a_one_shot_key_fires_once(window, spec, name):
    """A held Enter must not commit forty times."""
    action = next(a for a in A.ACTIONS if a.name == name)
    calls: list[tuple] = []
    setattr(window, action.slot, lambda *args: calls.append(args))
    window.set_mode(A.MODE_ANNOTATE)

    assert window.handle_key(key_event(spec)) is True
    for _ in range(3):
        assert window.handle_key(repeat_event(spec)) is True

    assert calls == [tuple(action.args)]


def test_holding_tab_does_not_walk_the_focus_chain(window):
    """The repeat is consumed but not fired: Qt would move the focus instead."""
    seen: list[bool] = []
    window.act_flash_compare = lambda on: seen.append(on)
    window.set_mode(A.MODE_ANNOTATE)
    assert window.handle_key(key_event("Tab")) is True
    for _ in range(3):
        assert window.handle_key(repeat_event("Tab")) is True
    assert seen == [True]


def test_the_cheat_sheet_marks_the_repeatable_keys():
    html = A.cheat_sheet_html()
    text = A.shortcut_markdown()
    for name in REPEATABLE:
        action = next(a for a in A.ACTIONS if a.name == name)
        assert action.label_zh in html and action.label_zh in text
    assert "可长按" in html and "可长按" in text


# ---------------------------------------------------------------------------
# the session's newer refusals reach the annotator, not a traceback (F3 round 4)
# ---------------------------------------------------------------------------
def test_a_bench_box_on_a_view_with_no_staging_area_is_explained(window):
    """``commit_box`` refuses on a view with no staging-area ROI (spec 4.2 item
    1), and the scanner view of the fixture is one.  The window arms the tool
    from the task card, so the refusal has to come back as a sentence."""
    window.begin_bench_box("chassis")

    window.on_bench_box((1.0, 1.0, 9.0, 9.0))

    said = window.status_message()
    assert "堆放区" in said and "scan" in said, said
    assert "Traceback" not in said


def test_editing_a_label_studio_draft_is_refused_in_words(window):
    """``ls:`` keys are the team's old tracings; a commit would rewrite one in
    place and the draft nobody adopted would become somebody's annotation.
    The window asks for the edit the same way the task card does."""
    window.task_card.sigRequestEdit.emit("ls:Motherboard#1")

    said = window.status_message()
    assert "Label Studio" in said and "草稿" in said, said
    assert getattr(window.session, "editing_instance", None) is None


def test_an_ignore_step_is_shown_neutral_and_stepped_over(qapp, tmp_path):
    """A calibration shot is no moment of the teardown: it is not compiled,
    not confirmable and not exported, so ``PgDn`` must not stop on it -- but
    the timeline still shows the frame, in the colour of a step nobody has
    annotated (spec 4.2, 缺帧处理)."""
    import dataclasses

    from app_scene import DESKTOP, VIEW, make_db
    from tda.core.model import StepType
    from tda.core.truth import TruthService
    from tda.ui.session import AnnotationSession
    from tda.ui import session_api as api
    from tda.ui.panels.timeline import status_brush

    db, paths, tax = make_db(tmp_path)
    ignored = 13
    db.replace_steps(
        DESKTOP,
        [dataclasses.replace(rec, step_type=StepType.IGNORE.value)
         if rec.step == ignored else rec for rec in db.steps(DESKTOP)],
        db.actions(DESKTOP),
    )
    session = AnnotationSession(db, tax, TruthService(db, tax),
                                paths["cache_dir"], "tester")
    session.open(DESKTOP, VIEW)
    win = MainWindow(session, paths, "tester", sam_queue=StubSamQueue())
    try:
        assert ignored in win.timeline.item_steps(), "the frame is not on the timeline"
        assert session.frame_status(ignored) == api.STATUS_UNLABELED
        assert (win.timeline.step_brush(ignored).color()
                == status_brush(api.STATUS_UNLABELED).color())

        assert session.current().step == 14
        win.act_step(-1)          # PgDn: one step towards the start

        assert session.current().step == 12, "PgDn stopped on the ignore step"
    finally:
        close_window(win)
