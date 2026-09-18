"""Thumbnails the manifest no longer knows about, and folders nobody wanted.

``build_thumbs`` writes one JPEG per step of a desktop's ``manifest.json``. It
never removed one, so re-running the cache after a step was dropped (a burst
re-decided away, a desktop re-indexed shorter) left a thumbnail the timeline can
still open: the picture of a step that no longer exists. They are deleted now
and counted as ``removed``.

The other half is the sidecar: a desktop whose every thumbnail failed created
``<cache>/thumbs/<view>/D<nn>/`` for a plan describing nothing, which then reads
as a built desktop on the next run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_cache_thumbs import _frame, _imwrite, _make_cache  # noqa: F401

from tda.core.cache_thumbs import build_thumbs, thumb_path
from tda.core.model import FrameKey

VIEW = "scan"
DESKTOP = 7


def _thumb_dir(cache: str) -> Path:
    return Path(cache) / "thumbs" / VIEW / f"D{DESKTOP:02d}"


def _built(cache: str) -> set[str]:
    folder = _thumb_dir(cache)
    return {p.name for p in folder.glob("s*.jpg")} if folder.exists() else set()


# --------------------------------------------------------------------------- #
# orphans
# --------------------------------------------------------------------------- #
def test_a_thumbnail_whose_step_left_the_manifest_is_deleted(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2, 3))
    build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert _built(cache) == {"s001.jpg", "s002.jpg", "s003.jpg"}

    # step 2 is dropped from the cache and its manifest
    ddir = Path(cache) / VIEW / f"D{DESKTOP:02d}"
    manifest = json.loads((ddir / "manifest.json").read_text(encoding="utf-8"))
    del manifest["2"]
    (ddir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    os.remove(ddir / "s002.png")

    stats = build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert stats["removed"] == 1
    assert _built(cache) == {"s001.jpg", "s003.jpg"}


def test_nothing_is_removed_when_the_manifest_is_unchanged(tmp_path):
    cache = _make_cache(tmp_path, steps=(1, 2))
    build_thumbs(cache, views=(VIEW,), auto_roi=False)
    stats = build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert stats["removed"] == 0
    assert _built(cache) == {"s001.jpg", "s002.jpg"}


def test_the_sidecar_and_foreign_files_are_left_alone(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,))
    build_thumbs(cache, views=(VIEW,), auto_roi=False)
    keep = _thumb_dir(cache) / "notes.txt"
    keep.write_text("mine", encoding="utf-8")

    stats = build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert stats["removed"] == 0
    assert keep.exists()
    assert (_thumb_dir(cache) / "thumbs.json").exists()


def test_a_desktop_with_no_thumbnails_yet_removes_nothing(tmp_path):
    cache = _make_cache(tmp_path, steps=(1,))
    stats = build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert stats["removed"] == 0


# --------------------------------------------------------------------------- #
# a folder for nothing
# --------------------------------------------------------------------------- #
def test_a_desktop_whose_every_thumbnail_failed_creates_no_folder(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1, 2))
    import tda.core.cache_thumbs as ct

    def boom(img, dest, quality):
        raise OSError("no room on the cache volume")

    monkeypatch.setattr(ct, "_write_jpeg", boom)
    stats = build_thumbs(cache, views=(VIEW,), auto_roi=False)
    assert stats["failed"] == 2 and stats["written"] == 0
    assert not _thumb_dir(cache).exists()


def test_a_partially_built_desktop_still_records_its_plan(tmp_path, monkeypatch):
    cache = _make_cache(tmp_path, steps=(1, 2))
    build_thumbs(cache, views=(VIEW,), auto_roi=False)  # both land
    import tda.core.cache_thumbs as ct

    monkeypatch.setattr(ct, "_write_jpeg", lambda *a: (_ for _ in ()).throw(OSError("x")))
    build_thumbs(cache, views=(VIEW,), auto_roi=False, force=True)
    assert (_thumb_dir(cache) / "thumbs.json").exists()
