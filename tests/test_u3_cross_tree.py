"""U3: on frames that ask for no screw of their own, this tree arms exactly what main 95d3386 did.

``git archive`` puts main's ``tda/``, its configs and its test scene in a
temporary directory; ``tests/u3_payload_driver.py`` walks the same synthetic
scene in both trees -- here with a stub detector that finds three screws on
**every** frame -- and writes, per frame, everything that makes up the prompt:
the box on the canvas and in both SAM tools, the chip, the cross, the rank,
the status line, the ``Shift+C`` alternates and the SAM request a click sends.

In this scene no screw comes back on its own: frame 12's four are captive,
back in with the fan (round 2), and frame 14 has no neighbour.  So **every**
frame must equal main field by field -- frame 12 included, its box the
difference map's over the fan.  The driver's ``live_check`` then takes the
screws' ``parent`` off frame 12's card and the detector's screw must arm: the
proof that the stub was live and it is the rule, not an idle detector, that
kept it off.
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
def test_frames_without_a_screw_of_their_own_arm_byte_for_byte_what_main_did(tmp_path):
    main_tree = _main_tree(tmp_path / "main_src")
    before = _run(main_tree, tmp_path / "main.json", tmp_path / "w_main")
    after = _run(REPO, tmp_path / "u3.json", tmp_path / "w_u3", "--detector")
    live = after.pop("live_check")
    assert sorted(before) == sorted(after) and len(before) == 14
    for step in before:
        assert after[step] == before[step], f"frame {step} differs from main"
    # frame 12 asks for four screws, all captive -- and still armed main's box
    assert len(after["12"]["open_screw_rows"]) == 4
    assert after["12"]["held"]["window"] is not None
    assert after["12"]["chip"] == "SAM 提示框（程序猜的位置）"
    # the stub was live: the same frame with screws of their own arms its screw
    assert live["chip"] == "SAM 提示框（检测器：螺丝）"
    assert live["held"]["window"] == [26.0, 20.0, 36.0, 32.0]
    assert len(set(json.dumps(v) for v in live["held"].values())) == 1
