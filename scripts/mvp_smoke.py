"""Drive the annotator on real data and report what it costs.

::

    D:\\Anaconda\\envs\\tda\\python.exe scripts/mvp_smoke.py            # numbers + images
    ... scripts/mvp_smoke.py --frames 5 --no-sam --no-shots

What it does: copy the real database to ``<cache>/../.cache/tmp``, open the
window on one desktop/view, accept the proposed ROI, then for each of a few
frames run the loop an annotator runs -- activate the task card's item, prompt
SAM with a point inside the difference map's box, commit, confirm, step back --
timing every part of it.  Afterwards it changes frame 40 times to see whether
anything leaks, saves three screenshots and deletes the copy.

**It never opens the live database for writing.**  The copy is taken through
SQLite's backup API on a read-only connection, every path it writes to is under
``.cache/tmp``, and the copy is removed in a ``finally``.

The screenshots are rendered with the real Qt platform plugin plus
``WA_DontShowOnScreen``, because the offscreen plugin ships no fonts and turns
every label -- Chinese ones above all -- into a row of boxes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULTS = {"desktop": 13, "view": "scan", "frames": 3, "leak_steps": 40}
#: The part that proves the task card and the frame agree: on D13 the cooler is
#: installed at step 12, so it must be drawable and committable there.
DEFAULT_DRAW = ("12:cpu_cooler.fan.01",)
SHOT_LIMIT_MB = 1.5
#: A 1920x1200 grab of a 12 MP frame is ~2.5 MB of PNG; the documentation only
#: needs to show the layout, so the images are scaled before they are saved.
SHOT_WIDTH = 1280


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=str(REPO / "configs" / "paths.yaml"))
    ap.add_argument("--desktop", type=int, default=DEFAULTS["desktop"])
    ap.add_argument("--view", default=DEFAULTS["view"])
    ap.add_argument("--frames", type=int, default=DEFAULTS["frames"])
    ap.add_argument("--leak-steps", type=int, default=DEFAULTS["leak_steps"])
    ap.add_argument("--annotator", default="smoke")
    ap.add_argument("--out", default=None, help="where the JSON report goes")
    ap.add_argument("--img-dir", default=str(REPO / "docs" / "img"))
    ap.add_argument("--no-sam", dest="sam", action="store_false", default=True)
    ap.add_argument("--no-shots", dest="shots", action="store_false", default=True)
    ap.add_argument("--shot-width", type=int, default=SHOT_WIDTH,
                    help="screenshots are scaled to this width to stay under 1.5 MB")
    ap.add_argument("--draw", action="append", default=None, metavar="STEP:INSTANCE",
                    help="draw one named instance on one step "
                         f"(default for D13: {DEFAULT_DRAW[0]})")
    ap.add_argument("--settings", default=None,
                    help="INI file to use (default: a temporary one, so the run "
                         "never touches the annotator's own window state)")
    ap.add_argument("--keep-db", action="store_true", help="leave the copy behind")
    args = ap.parse_args(argv)
    if args.draw is None:
        args.draw = list(DEFAULT_DRAW) if int(args.desktop) == 13 else []
    return args


def copy_database(src: str, dst: Path) -> float:
    """SQLite backup of a **read-only** connection; returns the size in MB."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    target = sqlite3.connect(str(dst))
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    return dst.stat().st_size / 1e6


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:  # pragma: no cover - psutil is not required
        return float("nan")


class Smoke:
    """One run; every measurement lands in :attr:`report`."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.report: dict[str, Any] = {"frames": [], "screenshots": []}
        self.peak = rss_mb()

    def note_peak(self) -> None:
        self.peak = max(self.peak, rss_mb())

    # -- the app ------------------------------------------------------------
    def run(self) -> dict:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        from tda import pipeline as P
        from tda.core.taxonomy import load_taxonomy
        from tda.core.truth import TruthService
        from tda.ui.app import MainWindow
        from tda.ui.session import AnnotationSession

        config = P.load_paths(self.args.paths)
        tmp = Path(P.require(config, "cache_dir")).parent / ".cache" / "tmp"
        copy = tmp / "tda_smoke.sqlite"
        self.report["db_mb"] = copy_database(P.require(config, "db_path"), copy)
        # The frames still come from the real cache, but the INI, the log and
        # the sidecars go to a directory of their own: a smoke run must not
        # move the annotator's last frame or their window layout.
        state = Path(self.args.settings) if self.args.settings else tmp / "state"
        config = dict(config, db_path=str(copy), backup_dir=str(tmp / "backups"),
                      app_dir=str(state))
        self.report["app_state_dir"] = str(state)

        app = QApplication.instance() or QApplication(sys.argv[:1])
        started = time.perf_counter()
        db = P.open_db(config, str(copy))
        tax = load_taxonomy()
        session = AnnotationSession(db, tax, TruthService(db, tax),
                                    config["cache_dir"], self.args.annotator)
        session.open(int(self.args.desktop), str(self.args.view))
        window = MainWindow(session, config, self.args.annotator)
        if self.args.shots:
            # Real fonts, no window on screen: the offscreen plugin has no font
            # database and renders every label -- Chinese first -- as boxes.
            window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.resize(1920, 1200)
        window.show()
        QApplication.processEvents()
        self.report["time_to_first_frame_ms"] = (time.perf_counter() - started) * 1000
        self.report["platform"] = app.platformName()
        self.note_peak()

        try:
            self.drive(window, app)
        finally:
            closed = time.perf_counter()
            window.close()
            self.report["close_ms"] = (time.perf_counter() - closed) * 1000
            db.close()
            self.note_peak()
            self.report["peak_rss_mb"] = self.peak
            if not self.args.keep_db:
                copy.unlink(missing_ok=True)
                shutil.rmtree(tmp / "backups", ignore_errors=True)
        return self.report

    # -- the loop -----------------------------------------------------------
    def drive(self, window, app) -> None:
        from PySide6.QtWidgets import QApplication

        image = window.canvas.image_rgb()
        self.report["image_hw"] = None if image is None else list(image.shape[:2])
        self.report["start_step"] = window.session.current().step
        self.report["steps"] = len(window.session.steps())
        self.report["roi_proposed"] = bool(window.roi_editing)
        started = time.perf_counter()
        if window.roi_editing:
            window.act_commit()
        self.report["roi_accept_ms"] = (time.perf_counter() - started) * 1000
        self.report["roi"] = list(window.roi() or [])

        if self.args.sam:
            started = time.perf_counter()
            from tda.models.sam_service import SamQueue, SamService

            window.set_sam_queue(SamQueue(SamService()), owns=True)
            self.report["sam_load_ms"] = (time.perf_counter() - started) * 1000
        self.note_peak()

        for _ in range(int(self.args.frames)):
            self.report["frames"].append(self.one_frame(window, app))
            self.note_peak()

        self.report["drawn"] = [self.draw_named(window, spec)
                                for spec in self.args.draw]
        if self.args.shots:
            self.report["cjk_shot"] = self.shoot_blocked_hint(window)
        self.note_peak()

        latencies = []
        for index in range(int(self.args.leak_steps)):
            started = time.perf_counter()
            window.act_step(-1 if index % 2 == 0 else +1)
            QApplication.processEvents()
            latencies.append((time.perf_counter() - started) * 1000)
        self.report["frame_change_ms"] = latencies
        self.report["rss_after_leak_check_mb"] = rss_mb()
        self.report["live_threads"] = _thread_names()
        self.note_peak()

    def one_frame(self, window, app) -> dict:
        from PySide6.QtWidgets import QApplication

        frame: dict[str, Any] = {"step": window.session.current().step}
        card = [row for row in window.session.task_card() if row.get("instance")]
        frame["task_items"] = len(card)
        frame["kinds"] = sorted({str(row.get("kind", "")) for row in card})
        if not card:
            window.act_step(-1)
            return frame

        instance = str(card[0]["instance"])
        frame["instance"] = instance
        started = time.perf_counter()
        window.on_request_edit(instance)
        frame["begin_edit_ms"] = (time.perf_counter() - started) * 1000

        window.act_fit_roi()
        if self.args.sam and window.sam_available:
            frame.update(self.prompt_sam(window))
        if self.args.shots and len(self.report["screenshots"]) < 3:
            self.shoot(window, len(self.report["screenshots"]) + 1)

        started = time.perf_counter()
        window.act_commit()
        if window.scope_bar.isVisible():
            window.act_commit()
        frame["commit_ms"] = (time.perf_counter() - started) * 1000
        frame["committed"] = window.session.editing_instance is None
        frame["refusal"] = window.last_error_message() if not frame["committed"] else ""

        started = time.perf_counter()
        frame["confirmed"] = bool(window.act_confirm())
        frame["confirm_ms"] = (time.perf_counter() - started) * 1000
        frame["problems"] = window.task_card.problems()[:3]
        if not frame["confirmed"]:
            window.act_clear_edit()
            window.act_step(-1)
        QApplication.processEvents()
        return frame

    def draw_named(self, window, spec: str) -> dict:
        """``STEP:INSTANCE`` -- draw one named part where it really is installed.

        The task card's own item is whatever the session offers; this is the
        explicit version, used to check that a part that *is* in the chassis on
        the frame being looked at can be drawn and committed there.
        """
        from PySide6.QtWidgets import QApplication

        step_text, _, instance = str(spec).partition(":")
        out: dict[str, Any] = {"step": int(step_text), "instance": instance}
        window.act_clear_edit()
        window.session.goto(int(step_text))
        QApplication.processEvents()
        window.on_request_edit(instance)
        out["editing"] = window.session.editing_instance == instance
        if not out["editing"]:
            out["error"] = window.status_message()
            return out
        window.act_fit_roi()
        if self.args.sam and window.sam_available:
            out.update(self.prompt_sam(window))
        started = time.perf_counter()
        window.act_commit()
        if window.scope_bar.isVisible():
            out["scope_bar"] = window.scope_bar_text()
            window.act_commit()
        out["commit_ms"] = (time.perf_counter() - started) * 1000
        out["committed"] = window.session.editing_instance is None
        out["keyframes"] = len(window.session.db.keyframes(
            int(self.args.desktop), str(self.args.view), instance))
        if not out["committed"]:
            out["error"] = window.last_error_message()
            window.act_clear_edit()
        return out

    def prompt_sam(self, window) -> dict:
        from PySide6.QtWidgets import QApplication

        out: dict[str, Any] = {}
        roi = window.roi() or (0, 0, *reversed(window.canvas.image_hw()))
        blobs = (window.assist_result or {}).get("unexplained") or []
        if blobs:
            box = blobs[0].box
            window.begin_add_shape(blobs[0])
        else:
            box = roi
        out["prompt_box"] = [int(v) for v in box]
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        window.act_tool("sam_point")
        before = int(window.overlay.editing.sum())
        started = time.perf_counter()
        window.sam_point.on_press(cx, cy, None)
        deadline = started + 60.0
        while time.perf_counter() < deadline:
            QApplication.processEvents()
            if int(window.overlay.editing.sum()) != before:
                break
            time.sleep(0.002)
        out["sam_round_trip_ms"] = (time.perf_counter() - started) * 1000
        out["sam_mask_px"] = int(window.overlay.editing.sum())
        out["sam_candidates"] = window.sam_point.candidate_count
        return out

    def shoot_blocked_hint(self, window) -> dict:
        """One screenshot with Chinese on screen, so CJK rendering is evidenced.

        The blocked-navigation hint is the natural candidate: it is the message
        the annotator sees most often, it is bilingual, and producing it needs
        nothing but an uncommitted stroke and a ``PgDn``.
        """
        from PySide6.QtWidgets import QApplication

        card = [row for row in window.session.task_card() if row.get("instance")]
        if not card:
            return {"skipped": "no task item to edit"}
        window.on_request_edit(str(card[0]["instance"]))
        mask = window.session.editing_mask()
        if mask is None:
            return {"skipped": "no editing layer"}
        painted = mask.copy()
        painted[:] = False
        x0, y0, x1, y1 = window.roi() or (0, 0, mask.shape[1], mask.shape[0])
        painted[(y0 + y1) // 2:(y0 + y1) // 2 + 40, (x0 + x1) // 2:(x0 + x1) // 2 + 40] = True
        window.set_editing_mask(painted, undoable=True)
        QApplication.processEvents()             # let the diff map settle first
        window.act_step(-1)                      # refused: the hint appears
        text = window.status_message()           # read it before anything else
        out = Path(self.args.img_dir) / f"mvp_d{self.args.desktop}_hint.png"
        self.shoot_to(window, out)
        window.act_clear_edit()
        return {"path": str(out), "status": text,
                "has_cjk": any("一" <= ch <= "鿿" for ch in text)}

    def shoot(self, window, index: int) -> None:
        from PySide6.QtCore import Qt

        out = Path(self.args.img_dir) / f"mvp_d{self.args.desktop}_{index}.png"
        size = self.shoot_to(window, out)
        self.report["screenshots"].append({"path": str(out), "mb": round(size, 3),
                                           "over_limit": size > SHOT_LIMIT_MB})

    def shoot_to(self, window, out: Path) -> float:
        """Grab the window, scale it down and save it; returns the size in MB."""
        from PySide6.QtCore import Qt

        out.parent.mkdir(parents=True, exist_ok=True)
        shot = window.grab()
        if shot.width() > int(self.args.shot_width):
            shot = shot.scaledToWidth(int(self.args.shot_width),
                                      Qt.TransformationMode.SmoothTransformation)
        shot.save(str(out), "PNG")
        return out.stat().st_size / 1e6


def _thread_names() -> list[str]:
    import threading

    return sorted(t.name for t in threading.enumerate() if t.is_alive())


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.shots:
        os.environ.pop("QT_QPA_PLATFORM", None)   # real fonts for the screenshots
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    report = Smoke(args).run()
    text = json.dumps(report, indent=1, ensure_ascii=False, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    # A Windows console is cp1252 by default and the report quotes the Chinese
    # status line: print it in the console's encoding rather than exit 1 on it.
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(encoding, "replace").decode(encoding) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
