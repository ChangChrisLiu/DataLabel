"""The "＋ 添加零件 / Add parts" dialog of stage S1 (task U5b).

A part the picture shows but the log never names -- the clips of D13's two
empty RAM slots -- has no instance, so the detector learns it as background and
no state question can be asked about it. This dialog stages such parts on the
:class:`~tda.ui.steps_model.StepTableData` the panel edits; ``Apply`` writes
them and ``Revert`` drops them, like every other S1 edit.

It asks for five things, each pre-filled from the taxonomy so the usual answer
is one click:

* the class, by its Chinese name and key (:mod:`tda.ui.class_names`), plus the
  class's ``role`` / ``kind`` when it has one -- that is part of the key;
* how many (1-16);
* the part it rides on: the unique instance of the class's ``host_class`` when
  there is exactly one (a ``ram_latch`` -> ``motherboard.01``), else a pick;
* the state the picture shows it in (default: the class default);
* an optional note.

Below them, the keys it will create -- ``ram_latch.05 … ram_latch.08`` -- so
nothing about the numbering is a surprise. The dialog only *reads* the view
model; :meth:`values` is what the panel hands to
:meth:`~tda.ui.steps_model.StepTableData.add_extras`.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tda.core import extra as X
from tda.core.logs import CHASSIS_KEY
from tda.ui.class_names import class_zh, state_zh
from tda.ui.steps_model import StepTableData

__all__ = ["AddPartsDialog", "BUTTON", "HINT", "NO_PARENT", "TIP", "TITLE",
           "class_choices", "preview_text"]

TITLE = "＋ 添加零件 / Add parts"
#: The button above the Instances table that opens the dialog.
BUTTON = TITLE
#: The line next to it.
HINT = "画面里有、日志里没有的零件（例如空内存槽的卡扣）在这里加 / parts the log never names"
#: The button's tooltip.
TIP = ("新建日志里没有的实例：选类别、数量、装在哪个零件上、画面里的状态。"
       "按 Apply 写入，Revert 丢弃；没画过形状的可以在表里右键删除。")
#: The parent combo's first entry: the part rides on nothing.
NO_PARENT = "（无 / none）"
#: What the dialog is for, in the words of the annotator's question.
INTRO = ("画面里有、日志里没有的零件（例如空内存槽上的卡扣）：每一个都要标，"
         "不然检测器会把它当背景。它没有动作，一直是下面选的状态，"
         "直到它所在的零件被拆走。")
#: How many keys the preview spells out before it abbreviates.
PREVIEW_SHOWN = 6


def class_choices(data: StepTableData) -> list[tuple[str, str]]:
    """``(label, class)`` for every class an extra may have, taxonomy order.

    The chassis is unique per desktop and is left out.
    """
    return [(f"{class_zh(cls)}  {cls}", cls) for cls in data.tax.classes
            if cls != CHASSIS_KEY]


def preview_text(keys: list[str], parent: Optional[str]) -> str:
    """The line under the form: the keys the batch will create, and its host."""
    if not keys:
        return "（这个类别不能添加 / this class cannot be added）"
    shown = keys if len(keys) <= PREVIEW_SHOWN else [*keys[:PREVIEW_SHOWN - 1], "…", keys[-1]]
    text = f"将新建 / will create: {', '.join(shown)}"
    if parent:
        text += f"\n跟 {parent} 一起离开机箱 / leaves the chassis with {parent}"
    else:
        text += "\n不挂在任何零件上：一直留在机箱里 / rides on nothing: stays in the chassis"
    return text


class AddPartsDialog(QDialog):
    """Pick a class, a count, a parent, a starting state and a note."""

    def __init__(self, data: StepTableData, parent: Optional[QWidget] = None,
                 cls: Optional[str] = None) -> None:
        super().__init__(parent)
        self.data = data
        self.setWindowTitle(TITLE)
        self.class_combo = QComboBox(self)
        for label, name in class_choices(data):
            self.class_combo.addItem(label, name)
        self.disc_label = QLabel("", self)
        self.disc_combo = QComboBox(self)
        self.count_spin = QSpinBox(self)
        self.count_spin.setRange(X.MIN_COUNT, X.MAX_COUNT)
        self.count_spin.setValue(1)
        self.parent_combo = QComboBox(self)
        self.state_combo = QComboBox(self)
        self.note_edit = QLineEdit(self)
        self.note_edit.setPlaceholderText("可选 / optional，例如：空槽的卡扣")
        self.preview = QLabel("", self)
        self.preview.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("添加 / Add")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._build()
        if cls:
            index = self.class_combo.findData(cls)
            if index >= 0:
                self.class_combo.setCurrentIndex(index)
        self.class_combo.currentIndexChanged.connect(self._on_class)
        self.disc_combo.currentIndexChanged.connect(self._refresh_preview)
        self.count_spin.valueChanged.connect(self._refresh_preview)
        self.parent_combo.currentIndexChanged.connect(self._refresh_preview)
        self._on_class()

    def _build(self) -> None:
        intro = QLabel(INTRO, self)
        intro.setWordWrap(True)
        form = QFormLayout()
        form.addRow("类别 / class", self.class_combo)
        form.addRow(self.disc_label, self.disc_combo)
        form.addRow("数量 / count", self.count_spin)
        form.addRow("装在 / rides on", self.parent_combo)
        form.addRow("画面里的状态 / state", self.state_combo)
        form.addRow("备注 / note", self.note_edit)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.preview)
        layout.addWidget(self.buttons)

    # -- reading ------------------------------------------------------------ #
    @property
    def cls(self) -> str:
        return str(self.class_combo.currentData() or "")

    def _attrs(self) -> dict:
        name, vocabulary = self.data.discriminator_of(self.cls)
        value = self.disc_combo.currentText() if vocabulary else ""
        return {name: value} if name and value else {}

    def _parent(self) -> Optional[str]:
        value = self.parent_combo.currentData()
        return str(value) if value else None

    def keys(self) -> list[str]:
        """The keys the dialog would create as it stands."""
        return self.data.preview_extras(self.cls, self.count_spin.value(), self._attrs())

    def values(self) -> dict:
        """The keyword arguments of :meth:`StepTableData.add_extras`."""
        return {
            "cls": self.cls,
            "count": int(self.count_spin.value()),
            "parent": self._parent(),
            "state": str(self.state_combo.currentData() or "") or None,
            "note": self.note_edit.text().strip(),
            "attrs": self._attrs(),
        }

    # -- keeping the form consistent --------------------------------------- #
    def _on_class(self, *_args) -> None:
        cls = self.cls
        name, vocabulary = self.data.discriminator_of(cls)
        self.disc_combo.blockSignals(True)
        self.disc_combo.clear()
        self.disc_combo.addItems(vocabulary)
        self.disc_combo.blockSignals(False)
        self.disc_label.setText(f"{name} / 细分" if vocabulary else "")
        self.disc_combo.setVisible(bool(vocabulary))
        self.disc_label.setVisible(bool(vocabulary))
        defaults = self.data.extra_defaults(cls) if cls else {
            "parent": None, "parents": [], "state": "", "states": []}
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem(NO_PARENT, "")
        for key in defaults["parents"]:
            rec = self.data.instances.get(key)
            label = f"{class_zh(rec.cls if rec else None, rec.attrs if rec else None, key)}  {key}"
            self.parent_combo.addItem(label, key)
        index = self.parent_combo.findData(defaults["parent"] or "")
        self.parent_combo.setCurrentIndex(max(index, 0))
        self.parent_combo.blockSignals(False)
        self.state_combo.clear()
        for state in defaults["states"]:
            self.state_combo.addItem(f"{state_zh(state)}  {state}", state)
        index = self.state_combo.findData(defaults["state"])
        self.state_combo.setCurrentIndex(max(index, 0))
        self._refresh_preview()

    def _refresh_preview(self, *_args) -> None:
        keys = self.keys()
        self.preview.setText(preview_text(keys, self._parent()))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(keys))
