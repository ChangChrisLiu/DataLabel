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
    assert len(prose) <= 200, "the prose is the part that has to stay readable"
