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
import cProfile
import io
import json
import os
import pstats
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
#: How many lines of each ``--profile`` table go into the report.
PROFILE_ROWS = 15


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
    ap.add_argument("--start-step", type=int, default=None, metavar="K",
                    help="stand on this logical step instead of the last one; "
                         "annotation runs backwards, so an early teardown step "
                         "is where most of the machine is still drawn")
    ap.add_argument("--instances", type=int, default=0, metavar="N",
                    help="before timing anything, give N instances of the start "
                         "frame a synthetic rectangle inside the ROI -- what a "
                         "half-annotated late step costs (0: draw nothing)")
    ap.add_argument("--profile", action="store_true",
                    help="cProfile one commit, one Space and one frame change; "
                         f"the top {PROFILE_ROWS} cumulative rows go into the report")
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


def note(what: str) -> None:
    """Say where the run has got to, on stderr.

    A 12 MP run with SAM loaded takes minutes and used to print nothing until
    the end, so "still working" and "wedged on a modal dialog nobody can click"
    looked exactly alike. The report itself still goes to stdout untouched.
    """
    sys.stderr.write(f"[smoke {time.strftime('%H:%M:%S')}] {what}\n")
    sys.stderr.flush()


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
        note("copying the database")
        self.report["db_mb"] = copy_database(P.require(config, "db_path"), copy)
        # The frames still come from the real cache, but the INI, the log and
        # the sidecars go to a directory of their own: a smoke run must not
        # move the annotator's last frame or their window layout.
        state = Path(self.args.settings) if self.args.settings else tmp / "state"
        config = dict(config, db_path=str(copy), backup_dir=str(tmp / "backups"),
                      app_dir=str(state))
        self.report["app_state_dir"] = str(state)

        note("opening the session")
        app = QApplication.instance() or QApplication(sys.argv[:1])
        started = time.perf_counter()
        db = P.open_db(config, str(copy))
        tax = load_taxonomy()
        session = AnnotationSession(db, tax, TruthService(db, tax),
                                    config["cache_dir"], self.args.annotator)
        session.open(int(self.args.desktop), str(self.args.view))
        if self.args.start_step is not None:
            session.goto(int(self.args.start_step), force=True)
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
            note("closing the window")
            # A run must never wedge on the "Uncommitted edit" question: it is
            # a modal dialog, the platform plugin is offscreen and there is
            # nobody to answer it, so the process simply stops for ever. Any
            # layer still open here belongs to a measurement, not to an
            # annotator, and is dropped through the session's own API so no
            # window guard can refuse it.
            try:
                session.clear_edit()
            except Exception as exc:  # noqa: BLE001 - closing must not raise
                self.report["clear_edit_failed"] = f"{type(exc).__name__}: {exc}"
            QApplication.processEvents()
            closed = time.perf_counter()
            window.close()
            self.report["close_ms"] = (time.perf_counter() - closed) * 1000
            # Every worker the window owns has to be joined by close(): the diff
            # map, the truth sweeper, the SAM queue and the ROI proposer all
            # hold something the window is about to drop.
            self.report["threads_after_close"] = _thread_names()
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
            # The rectangle is measured on three frames of the pose segment, on
            # a worker, so the window is usable at once and the box arrives a
            # moment later.  An annotator drags or waits; a smoke run has to
            # wait, or it would press Enter on an empty draft and report that
            # the detector found nothing.
            note("waiting for the chassis measurement")
            self.report["roi_measured"] = bool(window.wait_for_roi_proposal())
            self.report["roi_measure_ms"] = (time.perf_counter() - started) * 1000
            window.act_commit()
        self.report["roi_accept_ms"] = (time.perf_counter() - started) * 1000
        self.report["roi"] = list(window.roi() or [])

        note(f"ROI {self.report['roi']}")
        if int(self.args.instances) > 0:
            note(f"seeding {self.args.instances} instances")
            self.report["seeded"] = self.seed_instances(window, int(self.args.instances))

        if self.args.sam:
            note("loading SAM")
            started = time.perf_counter()
            from tda.models.sam_service import SamQueue, SamService

            window.set_sam_queue(SamQueue(SamService()), owns=True)
            self.report["sam_load_ms"] = (time.perf_counter() - started) * 1000
        self.note_peak()

        for index in range(int(self.args.frames)):
            note(f"frame {index + 1}/{self.args.frames} at step {window.session.current().step}")
            self.report["frames"].append(self.one_frame(window, app))
            self.note_peak()

        note("drawing the named instances")
        self.report["drawn"] = [self.draw_named(window, spec)
                                for spec in self.args.draw]
        note("checking the prompt points")
        self.report["prompt_points"] = self.check_prompt_points(window)
        note("timeline jumps")
        window.act_clear_edit()
        window.session.clear_edit()
        self.report["timeline_jump_ms"] = self.time_timeline_jumps(window)
        if self.args.profile:
            note("profiling one commit, one Space and one frame change")
            self.report["profiles"] = self.profile_ops(window)
        if self.args.shots:
            self.report["cjk_shot"] = self.shoot_blocked_hint(window)
            self.report["refused_space_shot"] = self.shoot_refused_space(window)
        self.note_peak()

        note(f"leak check: {self.args.leak_steps} frame changes")
        # A step is refused while an editing layer is open -- including one a
        # SAM result repopulated after the measurement that asked for it -- and
        # a refusal is very fast, so without this the report would have called
        # "nothing happened" a 2 ms frame change.
        window.act_clear_edit()
        window.session.clear_edit()
        QApplication.processEvents()
        latencies = []
        moved = 0
        for index in range(int(self.args.leak_steps)):
            before = window.session.current().step
            started = time.perf_counter()
            window.act_step(-1 if index % 2 == 0 else +1)
            QApplication.processEvents()
            latencies.append((time.perf_counter() - started) * 1000)
            moved += int(window.session.current().step != before)
        self.report["frame_change_ms"] = latencies
        self.report["frame_changes_that_moved"] = moved
        self.report["rss_after_leak_check_mb"] = rss_mb()
        self.report["live_threads"] = _thread_names()
        self.note_peak()

    # -- the half-annotated frame -------------------------------------------
    def seed_instances(self, window, count: int) -> dict:
        """Give ``count`` instances of the open frame a rectangle inside the ROI.

        The measured budgets are "a commit on a **late** step", and a late step
        is one where most of the machine is already drawn: the truth table then
        holds one row per instance and every one of them is re-derived and
        re-encoded whenever the frame's inputs move.  A smoke run on a desktop
        nobody has annotated measures the 1-4-shape case instead, which is the
        cheap one.

        The shapes are written straight into the copy -- one ``add_keyframe``
        per instance and one ``set_zorder``, the way ``tests/session_scene.py``
        seeds a scene -- rather than through ``commit_edit``, because the point
        is to *arrive* at a drawn frame, not to time forty commits.  They are
        laid out as a grid of non-overlapping rectangles inside the ROI, so
        nothing is hidden and the frame compiles without ``empty_visible``.
        """
        import numpy as np

        from tda.core import masks
        from tda.core.model import ShapeKeyframe, ShapePart, ZOrderRec
        from tda.core.states import needs_geom
        from tda.core.truth_inputs import (
            frame_hw,
            instances_of,
            pose_segment_of,
            state_of,
        )

        session = window.session
        key = session.current()
        db, tax = session.db, session.tax
        state = state_of(db, tax, key.desktop, key.step)
        wanted = [
            inst
            for inst, kind in sorted(needs_geom(instances_of(db, key.desktop),
                                                state, tax).items())
            if kind == "mask" and state[inst].placement == "in_chassis"
        ][:int(count)]
        out: dict[str, Any] = {"asked": int(count), "available": len(wanted)}
        if not wanted:
            return out

        hw = frame_hw(db, key, session.truth.cache_dir)
        seg = pose_segment_of(db, key)
        row = db.pose_segment_for(key) or {}
        anchor = int(row.get("end_step") or max(session.steps()))
        x0, y0, x1, y1 = window.roi() or (0, 0, hw[1], hw[0])
        grid = 1
        while grid * grid < len(wanted):
            grid += 1
        cell_w, cell_h = (x1 - x0) // grid, (y1 - y0) // grid
        order: list[tuple[str, str]] = []
        started = time.perf_counter()
        for index, instance in enumerate(wanted):
            col, row_i = index % grid, index // grid
            mask = np.zeros(hw, dtype=bool)
            mx0, my0 = x0 + col * cell_w + 1, y0 + row_i * cell_h + 1
            mask[my0:my0 + cell_h - 2, mx0:mx0 + cell_w - 2] = True
            db.add_keyframe(ShapeKeyframe(
                id=None, instance=instance, desktop=key.desktop, view=key.view,
                pose_segment=seg, anchor_step=anchor, placement="in_chassis",
                geom_type="mask",
                parts=[ShapePart("main", masks.encode_rle(mask))],
            ))
            order.append((instance, "main"))
        db.set_zorder(ZOrderRec(key.desktop, key.view, seg, order))
        session._invalidate()
        out["drawn"] = len(order)
        out["anchor_step"] = anchor
        out["write_ms"] = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        session.compiled()               # the frame the timings start from
        out["first_compile_ms"] = (time.perf_counter() - started) * 1000
        out["rows"] = len(session.compiled().instances)
        return out

    # -- the profile ---------------------------------------------------------
    def profile_ops(self, window) -> dict:
        """cProfile one commit, one Space and one frame change on this frame.

        One run each, after the timings above: a profile is read for *where*
        the time goes, and the wall clocks next to it are the ones measured
        without the profiler's overhead.
        """
        from PySide6.QtWidgets import QApplication

        out: dict[str, Any] = {}
        window.act_clear_edit()
        card = [r for r in window.session.task_card() if r.get("instance")]
        instance = str(card[0]["instance"]) if card else None
        if instance is not None:
            window.on_request_edit(instance)
            mask = window.session.editing_mask()
            if mask is not None:
                painted = mask.copy()
                x0, y0, x1, y1 = window.roi() or (0, 0, mask.shape[1], mask.shape[0])
                painted[(y0 + y1) // 2:(y0 + y1) // 2 + 120,
                        (x0 + x1) // 2:(x0 + x1) // 2 + 120] ^= True
                window.set_editing_mask(painted, undoable=True)
                out["commit"] = self._profiled(window.act_commit)
                window.act_clear_edit()
        out["confirm"] = self._profiled(window.act_confirm)
        window.act_clear_edit()
        window.session.clear_edit()
        QApplication.processEvents()
        out["step_before_change"] = window.session.current().step
        out["frame_change"] = self._profiled(lambda: window.act_step(-1))
        QApplication.processEvents()
        out["step_after_change"] = window.session.current().step
        return out

    @staticmethod
    def _profiled(call) -> dict:
        """``{"ms", "rows"}`` -- one call's wall clock and its profile table."""
        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.enable()
        try:
            call()
        finally:
            profiler.disable()
        elapsed = (time.perf_counter() - started) * 1000
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(
            PROFILE_ROWS
        )
        lines = [line.rstrip() for line in buffer.getvalue().splitlines() if line.strip()]
        return {"ms": elapsed, "rows": lines}

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
        warned = False
        window.act_commit()
        # The two non-modal bars an annotator answers with a second Enter: the
        # scope suggestion and the area warning.  Sampled over the *whole*
        # sequence -- the warning follows an accepted scope, so looking only
        # after the first Enter missed it -- because "how often does the size
        # warning fire on real masks?" is what a smoke run should answer.
        for _ in range(2):
            warned = warned or bool(window.warn_bar.isVisible())
            if not (window.scope_bar.isVisible() or window.warn_bar.isVisible()):
                break
            window.act_commit()
        frame["area_warned"] = warned
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
        out["area_warned"] = bool(window.warn_bar.isVisible())
        if window.warn_bar.isVisible():
            out["area_warning"] = window.warn_bar_text()
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

    def check_prompt_points(self, window) -> dict:
        """The second instance's first prompt must carry exactly one point.

        Final review item 1: the points used to accumulate for the whole
        session, so every mask after the first commit was a union over
        everything that had been clicked.
        """
        from PySide6.QtWidgets import QApplication

        if not (self.args.sam and window.sam_available):
            return {"skipped": "SAM not loaded"}
        card = [row for row in window.session.task_card() if row.get("instance")]
        if len(card) < 2:
            return {"skipped": "the frame has fewer than two card items"}
        out: dict[str, Any] = {}
        for index, row in enumerate(card[:2]):
            window.on_request_edit(str(row["instance"]))
            window.act_tool("sam_point")
            x0, y0, x1, y1 = window.roi() or (0, 0, 100, 100)
            window.sam_point.on_press((x0 + x1) / 2.0, (y0 + y1) / 2.0, None)
            QApplication.processEvents()
            out[f"instance_{index + 1}_points"] = len(window.sam_point.points)
            out[f"instance_{index + 1}_candidates"] = window.sam_point.candidate_count
            window.act_clear_edit()
        out["ok"] = out.get("instance_2_points") == 1
        return out

    def time_timeline_jumps(self, window, jumps: int = 6) -> list:
        """Clicking a row in the timeline, the way an annotator reaches a frame."""
        from PySide6.QtWidgets import QApplication

        steps = sorted(window.session.steps())
        if len(steps) < 4:
            return []
        picks = [steps[i * len(steps) // (jumps + 1)] for i in range(1, jumps + 1)]
        out = []
        for step in picks:
            started = time.perf_counter()
            window.timeline_goto(int(step))
            QApplication.processEvents()
            if window.session.current().step != int(step):
                continue          # refused: that is not a jump time
            out.append((time.perf_counter() - started) * 1000)
        return out

    def shoot_refused_space(self, window) -> dict:
        """The start frame with 60+ missing shapes, refused: is it readable?

        Final review item 11: this is the frame whose refusal put 2,550
        characters into the status bar and listed every part twice.
        """
        from PySide6.QtWidgets import QApplication

        window.session.goto(max(window.session.steps()), force=True)
        QApplication.processEvents()
        window.act_confirm()
        QApplication.processEvents()
        text = window.status_message()
        out = Path(self.args.img_dir) / f"mvp_d{self.args.desktop}_refused.png"
        self.shoot_to(window, out)
        return {"path": str(out), "status_len": len(text),
                "problems_listed": len(window.task_card.problem_rows()),
                "hint_min_width_px": window.hint_label.minimumSizeHint().width()}

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
