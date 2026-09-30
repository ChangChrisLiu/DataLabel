"""A stand-in for the ``rfdetr`` package: no network, no weights, no GPU (task U4).

:func:`install` puts a module named ``rfdetr`` into ``sys.modules`` that does
to the process what rfdetr 1.10.1 was measured to do (``_u4`` probe, SAM
loaded first), at the moment it does it:

* on ``from rfdetr import RFDETR`` (the import): ``torch``'s float32 matmul
  precision to "high" (TF32), ``PIL.Image.MAX_IMAGE_PIXELS`` to 2e9,
  ``numpy.complex_`` added, sklearn's ``KMP_DUPLICATE_LIB_OK`` /
  ``KMP_INIT_AT_FORK`` set, and warning filters from code that runs *as*
  rfdetr, scipy, urllib3 and requests -- which is how
  :func:`tda.models.detector.keep_process_state` tells them from everyone
  else's;
* on the first ``predict``: supervision's filter.

The model answers every tile through ``answer(rgb) -> [(xyxy, conf,
class_id), ...]``; the default finds the bright square the synthetic scene
paints (pixels >= 200) as a screw at 0.9, and adds two boxes that must never
come out: a screw at 0.09, under the merge floor, and one of the head's extra
slot (``class_id == len(class_names)``) at 0.95.
"""
from __future__ import annotations

import sys
import types
from typing import Callable, Optional, Sequence

import numpy as np

L3_CLASSES = ("screw", "connector", "ram_latch", "psu_latch", "cpu_socket_lever",
              "drive_latch", "ram_module", "cpu")
PHANTOM = (2.0, 2.0, 8.0, 8.0)          # tile pixels, conf 0.09
EXTRA_SLOT = (30.0, 30.0, 40.0, 40.0)   # tile pixels, the (num_classes + 1)th logit


def _code_as(name: str, source: str) -> types.ModuleType:
    """A module whose functions run with ``__name__ == name`` (for the filter watch)."""
    module = types.ModuleType(name)
    exec(compile(source, f"<fake {name}>", "exec"), module.__dict__)
    return module


_IMPORT_TIME = '''
import os, warnings
import numpy, PIL.Image, torch


def patch():
    torch.set_float32_matmul_precision("high")          # rfdetr/detr.py
    PIL.Image.MAX_IMAGE_PIXELS = 2_000_000_000           # rfdetr/datasets/o365.py
    setattr(numpy, "complex_", numpy.complex128)         # rfdetr/__init__
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")  # sklearn/__init__
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")     # sklearn, joblib
    warnings.filterwarnings("ignore", message="u4 fake rfdetr")
'''

_OTHERS = {
    "scipy.sparse": 'import warnings\ndef run():\n'
                    '    warnings.filterwarnings("ignore", message="u4 fake matrix subclass")\n',
    "urllib3": 'import warnings\ndef run():\n'
               '    warnings.simplefilter("always", ResourceWarning, append=True)\n',
    "requests": 'import warnings\ndef run():\n'
                '    warnings.simplefilter("default", ImportWarning, append=True)\n',
    "supervision.utils.internal": 'import warnings\ndef run():\n'
                                  '    warnings.simplefilter("always", UnicodeWarning)\n',
}


class Detections:
    """What ``supervision.Detections`` offers the backend: xyxy, confidence, class_id."""

    def __init__(self, rows: Sequence) -> None:
        self.xyxy = np.asarray([r[0] for r in rows], np.float32).reshape(-1, 4)
        self.confidence = np.asarray([r[1] for r in rows], np.float32)
        self.class_id = np.asarray([r[2] for r in rows], int)

    def __len__(self) -> int:
        return len(self.xyxy)


def bright_square(rgb: np.ndarray, names: int = len(L3_CLASSES)) -> list:
    """The scene's moving square as a screw, plus the two boxes that must not survive."""
    rows = [(PHANTOM, 0.09, 0), (EXTRA_SLOT, 0.95, names)]
    ys, xs = np.nonzero(np.asarray(rgb)[..., 0] >= 200)
    if xs.size:
        rows.insert(0, ((float(xs.min()), float(ys.min()), float(xs.max()) + 1.0,
                         float(ys.max()) + 1.0), 0.9, 0))
    return rows


class FakeModel:
    def __init__(self, fake: types.ModuleType, path: str, device: str) -> None:
        self._fake = fake
        self.path, self.device = path, device
        self.class_names = list(fake.CLASS_NAMES)
        self.model = types.SimpleNamespace(resolution=fake.RESOLUTION)
        self.dtype = None

    def inference(self, compile: bool = True, batch_size: int = 1, dtype=None, **_kw) -> None:
        self.dtype = dtype
        self._fake.log.append(("inference", compile, str(dtype)))

    def predict(self, images, threshold: float = 0.5, include_source_image: bool = True,
                **_kw):
        fake = self._fake
        if not fake.predicted:
            fake.predicted = True
            fake.OTHERS["supervision.utils.internal"].run()   # "import supervision"
        fake.log.append(("predict", len(images), float(threshold), include_source_image))
        return [Detections(fake.ANSWER(np.asarray(img))) for img in images]


def install(monkeypatch=None, *, answer: Optional[Callable] = None,
            class_names: Sequence[str] = L3_CLASSES, resolution: int = 640,
            fail_import: bool = False, fail_load: Optional[Exception] = None,
            after_import: Sequence[Callable] = (),
            during: Sequence[Callable] = ()) -> types.ModuleType:
    """Put the fake ``rfdetr`` in ``sys.modules`` (through ``monkeypatch`` when given).

    ``after_import`` are called right after the import's side effects (inside
    the import's snapshot), ``during`` inside ``from_checkpoint`` (inside the
    model's) -- where SAM's loader thread really does set things of its own.
    """
    import importlib.machinery

    fake = types.ModuleType("rfdetr")
    # found by importlib.util.find_spec, as the installed package is
    fake.__spec__ = importlib.machinery.ModuleSpec("rfdetr", None)
    fake.CLASS_NAMES = tuple(class_names)
    fake.RESOLUTION = int(resolution)
    fake.ANSWER = answer or (lambda rgb: bright_square(rgb, len(fake.CLASS_NAMES)))
    fake.OTHERS = {name: _code_as(name, src) for name, src in _OTHERS.items()}
    fake.log = []
    fake.imported = False
    fake.predicted = False
    body = _code_as("rfdetr", _IMPORT_TIME)

    class RFDETR:
        @classmethod
        def from_checkpoint(cls, path, **kwargs):
            fake.log.append(("from_checkpoint", str(path), dict(kwargs)))
            for other in during:
                other()
            if fail_load is not None:
                raise fail_load
            return FakeModel(fake, str(path), str(kwargs.get("device")))

    def __getattr__(name):
        if name != "RFDETR":
            raise AttributeError(name)
        if not fake.imported:          # the import's side effects, once
            fake.imported = True
            body.patch()
            for other in ("scipy.sparse", "urllib3", "requests"):
                fake.OTHERS[other].run()
            for hook in after_import:
                hook()
        if fail_import:
            raise ImportError("u4: this rfdetr does not import")
        return RFDETR

    fake.__getattr__ = __getattr__
    if monkeypatch is not None:
        monkeypatch.setitem(sys.modules, "rfdetr", fake)
    else:
        sys.modules["rfdetr"] = fake
    return fake
