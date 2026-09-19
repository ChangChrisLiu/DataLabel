"""The command line as the annotator actually runs it: a real child process.

Every other CLI test calls :func:`tda.cli.main` in-process, where ``tda.cli`` is
an ordinary import.  ``python -m tda.cli`` is not that: the module runs as
``__main__``, so anything that imported ``tda.cli`` a second time got a *second*
copy of its classes -- and an ``except Locked`` in one copy does not catch the
``Locked`` of the other.  That turned "somebody else holds the lock" into a
21-line traceback for half the subcommands, which no in-process test could see.

So these tests spawn ``sys.executable`` and read what comes back: the exit code,
and the fact that a refusal is one line without the word ``Traceback``.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Every subcommand that takes the single-user lock, with the arguments it needs
#: to get as far as taking it.
LOCKING_COMMANDS = [
    ("load-index",),
    ("import-logs",),
    ("import-ls", "--export", "nowhere.json"),
    ("infer-relations",),
    ("constraints",),
    ("check", "--desktop", "1"),
    ("export-coco", "--desktops", "1"),
    ("export-vlm", "--desktops", "1"),
]

#: The two spellings of the entry point that must both work.
ENTRY_POINTS = [("-m", "tda.cli"), ("-m", "tda")]


def run_cli(workspace: Path, entry: tuple[str, ...], *argv: str) -> subprocess.CompletedProcess:
    """Run one subcommand in a child process, with stdout and stderr captured."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, *entry, "--paths", str(workspace / "paths.yaml"), *argv],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180,
    )


def output_lines(proc: subprocess.CompletedProcess) -> list[str]:
    """Every non-empty line the child printed, on either stream."""
    return [line for line in (proc.stdout + proc.stderr).splitlines() if line.strip()]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A paths.yaml, an empty database, and a *fresh* lock held by somebody else."""
    db_path = tmp_path / "annotations" / "tda.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "db_path": str(db_path),
        "backup_dir": str(tmp_path / "backups"),
        "raw_logs_dir": str(tmp_path / "raw_logs"),
    }
    (tmp_path / "paths.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

    from tda.core.db import Db

    Db(str(db_path)).close()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (tmp_path / "annotations" / "tda.sqlite.lock").write_text(
        json.dumps({"annotator": "chang", "ts": stamp, "pid": 1}), encoding="utf-8"
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# the lock
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", ENTRY_POINTS, ids=lambda e: e[-1])
@pytest.mark.parametrize("command", LOCKING_COMMANDS, ids=lambda c: c[0])
def test_a_held_lock_is_one_line_and_exit_three(workspace, entry, command):
    proc = run_cli(workspace, entry, *command)
    lines = output_lines(proc)
    assert "Traceback" not in proc.stdout + proc.stderr
    assert lines == [line for line in lines if "chang" in line], lines
    assert len(lines) == 1, lines
    assert proc.returncode == 3, (proc.returncode, lines)


# --------------------------------------------------------------------------- #
# an unusable paths.yaml
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("command", [
    ("build-index",),
    ("load-index",),
    ("build-cache", "--desktops", "1"),
    ("constraints",),
    ("check", "--desktop", "1"),
    ("export-coco", "--desktops", "1"),
])
@pytest.mark.parametrize("broken", ["missing", "unreadable", "invalid"])
def test_an_unusable_paths_yaml_is_one_line_and_exit_one(tmp_path, command, broken):
    if broken == "missing":
        pass  # nothing is written at all
    elif broken == "unreadable":
        (tmp_path / "paths.yaml").mkdir()  # a directory is not a file
    else:
        (tmp_path / "paths.yaml").write_text("db_path: [unclosed\n", encoding="utf-8")

    proc = run_cli(tmp_path, ("-m", "tda.cli"), *command)
    lines = output_lines(proc)
    assert "Traceback" not in proc.stdout + proc.stderr
    assert len(lines) == 1, lines
    assert lines[0].startswith(f"[{command[0]}] cannot read "), lines
    assert proc.returncode == 1, (proc.returncode, lines)
