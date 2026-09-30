"""U3/U4: on frames that ask for no screw of their own, this tree arms exactly what main did.

``git archive`` puts main's ``tda/``, its configs and its test scene in a
temporary directory; ``tests/u3_payload_driver.py`` walks the same synthetic
scene in both trees, once per view -- scan, oak1 and oak2 (U4) -- and writes,
per frame, everything that makes up the prompt: the box on the canvas and in
both SAM tools, the chip, the cross, the rank, the status line, the
``Shift+C`` alternates and the SAM request a click sends.  Main runs without
a detector; this tree runs the real RF-DETR backend over a fake ``rfdetr``
that finds the scene's bright square as a screw on **every** frame (plus a
0.09 box and one of the head's extra slot, which must never come out).

In this scene no screw comes back on its own: frame 12's four are captive,
back in with the fan (U3 round 2), and frame 14 has no neighbour.  So
**every** frame of every view must equal main field by field -- frame 12
included, its box the difference map's over the fan.  The driver's
``live_check`` then takes the screws' ``parent`` off frame 12's card and the
detector's screw must arm: the proof that the backend was live and it is the
rule, not an idle detector, that kept it off.

Main is 72363fc (U4's base), which on these frames arms what 95d3386 -- the
tree before any detector -- did (U3's version of this test).
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
MAIN = "72363fc"
DRIVER = REPO / "tests" / "u3_payload_driver.py"
VIEWS = ("scan", "oak1", "oak2")
#: Where the square is on frame 12, native pixels (x 14 + (12 % 5) * 6).
SQUARE_12 = [26.0, 20.0, 36.0, 32.0]


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
                           str(work), "--views", ",".join(VIEWS), *extra], cwd=str(tree),
                          env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_frames_without_a_screw_of_their_own_arm_byte_for_byte_what_main_did(tmp_path):
    main_tree = _main_tree(tmp_path / "main_src")
    before = _run(main_tree, tmp_path / "main.json", tmp_path / "w_main")
    after = _run(REPO, tmp_path / "u4.json", tmp_path / "w_u4", "--detector")
    assert sorted(before) == sorted(after) == sorted(VIEWS)
    for view in VIEWS:
        live = after[view].pop("live_check")
        assert sorted(before[view]) == sorted(after[view]) and len(before[view]) == 14, view
        for step in before[view]:
            assert after[view][step] == before[view][step], f"{view} frame {step} differs"
        # frame 12 asks for four screws, all captive -- and still armed main's box
        assert len(after[view]["12"]["open_screw_rows"]) == 4
        assert after[view]["12"]["held"]["window"] is not None
        assert after[view]["12"]["chip"] == "SAM 提示框（程序猜的位置）"
        # the backend was live: the same frame with screws of their own arms the square
        assert live["det_state"] == "on", (view, live)
        assert live["chip"] == "SAM 提示框（检测器：螺丝）", (view, live)
        assert len(set(json.dumps(v) for v in live["held"].values())) == 1
        box = live["held"]["window"]
        # scan is PNG (exact); oak1/oak2 are JPEG, whose square may blur a pixel
        tolerance = 0.0 if view == "scan" else 2.0
        assert max(abs(a - b) for a, b in zip(box, SQUARE_12)) <= tolerance, (view, box)
        # what came out of the merge: the square alone -- not the 0.09 box, not
        # the head's extra slot (either would be cached: the store floor is 0.01)
        assert len(live["dets"]) == 1 and live["dets"][0][4:] == ["screw", 0.9], live["dets"]
