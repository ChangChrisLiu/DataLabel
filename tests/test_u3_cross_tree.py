"""U3: on frames that ask for no screw, this tree arms exactly what main 95d3386 did.

``git archive`` puts main's ``tda/``, its configs and its test scene in a
temporary directory; ``tests/u3_payload_driver.py`` walks the same synthetic
scene in both trees -- here with a stub detector that finds three screws on
**every** frame -- and writes, per frame, everything that makes up the prompt:
the box on the canvas and in both SAM tools, the chip, the cross, the rank,
the status line, the ``Shift+C`` alternates and the SAM request a click sends.

On every frame whose card asks for no screw the two are compared field by
field and must be equal.  On the frame that does ask for screws they must
**differ** -- the proof that the stub was live and it is the rule, not an idle
detector, that kept it off the others.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MAIN = "95d3386"
DRIVER = REPO / "tests" / "u3_payload_driver.py"


def _main_tree(dest: Path) -> Path:
    try:
        blob = subprocess.run(
            ["git", "-C", str(REPO), "archive", "--format=tar", MAIN, "tda", "configs",
             "tests/app_scene.py", "tests/fixtures"],
            check=True, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"main {MAIN} cannot be extracted here: {exc}")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest)
    shutil.copy2(DRIVER, dest / "tests" / DRIVER.name)
    return dest


def _run(tree: Path, out: Path, work: Path, *extra: str) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["TMP"] = env["TEMP"] = str(work.parent)
    done = subprocess.run([sys.executable, str(tree / "tests" / DRIVER.name), str(out),
                           str(work), *extra], cwd=str(tree), env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_frames_without_a_screw_arm_byte_for_byte_what_main_did(tmp_path):
    main_tree = _main_tree(tmp_path / "main_src")
    before = _run(main_tree, tmp_path / "main.json", tmp_path / "w_main")
    after = _run(REPO, tmp_path / "u3.json", tmp_path / "w_u3", "--detector")
    assert sorted(before) == sorted(after)
    plain = [s for s in before if not after[s]["open_screw_rows"]]
    screwed = [s for s in before if after[s]["open_screw_rows"]]
    assert len(plain) >= 10 and screwed, (plain, screwed)
    for step in plain:
        assert after[step] == before[step], f"frame {step} differs from main"
    # the detector was live: where the card asks for screws, rank 1 is its box
    armed = [s for s in screwed if after[s]["held"]["window"] is not None]
    assert armed, "no frame with an open screw row armed a box"
    for step in armed:
        assert after[step]["chip"] == "SAM 提示框（检测器：螺丝）"
        assert after[step] != before[step]
