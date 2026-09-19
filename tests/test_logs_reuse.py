"""The narrow physical rule: "displace X" then "remove X" is one part, not two.

Spec 3.2 says every row that names a target is a new instance unless the strict
reuse test allows otherwise, and it says so because an earlier, looser rule
merged 419 actions onto the wrong instance.  But a sheet that swings the PSU out
of the way at step 10 (``Open Power Module``, a ``displace``) and lifts it out at
step 20 (``Power module``, a ``remove``) is talking about one physical power
supply, and drafting ``psu.01`` and ``psu.02`` asks the annotator to draw the
same part twice on every frame in between.

So one extra door, deliberately narrow: an **unnumbered** successful ``remove``
may reuse an earlier instance only when that instance came from an unnumbered
row of the same class *and* discriminator, is not removed yet, was last
successfully given a non-terminal verb, and is the only such candidate on the
desktop.  Every reuse is reported, so nothing merges silently.

The three real shapes from the Drive sheets are here as fixtures-in-test: D22's
PSU (merges), D36's two numbered optical drives and D48's two drive cages (both
stay two parts).  ``test_logs.py`` keeps the regression tests of the strict rule.
"""
from __future__ import annotations

import pytest
from test_logs import synth  # noqa: F401  (the shared row builder)

from tda.core.taxonomy import load_taxonomy


@pytest.fixture(scope="module")
def tax():
    return load_taxonomy()


def targets(li) -> list[str]:
    return [a.target for a in li.actions]


def reuse_issues(li) -> list[str]:
    return [text for text in li.issues if " reuses " in text]


# --------------------------------------------------------------------------- #
# the real shapes
# --------------------------------------------------------------------------- #
def test_d22_psu_displace_then_remove_is_one_instance(tax):
    """D22 step 12 ``Open Power Module``, step 20 ``Power module``."""
    li = synth(["Open Power Module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.01"]
    assert [a.verb for a in li.actions] == ["displace", "remove"]
    assert sorted(k for k in li.instances if k.startswith("psu")) == ["psu.01"]
    assert li.instances["psu.01"].raw_names == ["Open Power Module", "Power module"]


def test_the_reuse_is_reported_for_stage_s1(tax):
    li = synth(["Open Power Module", "Power module"], tax=tax)
    lines = reuse_issues(li)
    assert len(lines) == 1
    assert "step 3" in lines[0]  # row 1 is the implicit "Initial Conditions"
    assert "reuses psu.01" in lines[0]
    assert "displace" in lines[0] and "remove" in lines[0]


def test_d36_two_numbered_optical_drives_stay_two_parts(tax):
    """``Optical drive 1`` / ``Optical drive 2``: numbered, so the rule stands off."""
    li = synth(["Optical drive 1", "Optical drive 2"], tax=tax)
    assert targets(li) == ["optical_drive.01", "optical_drive.02"]
    assert reuse_issues(li) == []


def test_d48_two_drive_cages_stay_two_parts(tax):
    """``HDD cover`` (of=hdd, remove) and ``Open optical drive`` (of=optical)."""
    li = synth(["HDD cover", "Open optical drive"], tax=tax)
    assert targets(li) == ["drive_cage.01", "drive_cage.02"]
    assert li.instances["drive_cage.01"].attrs["of"] == "hdd"
    assert li.instances["drive_cage.02"].attrs["of"] == "optical"
    assert reuse_issues(li) == []


# --------------------------------------------------------------------------- #
# every condition of the rule, one at a time
# --------------------------------------------------------------------------- #
def test_a_second_remove_starts_a_new_instance(tax):
    """D42 style: two plain ``Optical drive`` removes really are two drives."""
    li = synth(["Optical drive", "Optical drive"], tax=tax)
    assert targets(li) == ["optical_drive.01", "optical_drive.02"]
    assert reuse_issues(li) == []


def test_two_displaced_candidates_are_left_to_the_human(tax):
    """Ambiguity is not resolved by picking one: both stay, nothing is reported."""
    li = synth(["Open Power Module", "Open Power Module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02", "psu.03"]
    assert reuse_issues(li) == []


def test_a_failed_remove_row_does_not_reuse(tax):
    """An attempt that did not happen must not absorb the part it failed on."""
    li = synth(["Open Power Module", "Try to remove power module (failed)"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02"]
    assert reuse_issues(li) == []


def test_an_instance_with_no_successful_verb_is_not_a_candidate(tax):
    """D45 style: a failed attempt leaves nothing to continue from."""
    li = synth(["Try to remove power module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02"]
    assert reuse_issues(li) == []


def test_a_different_discriminator_is_a_different_part(tax):
    """A displaced heatsink is never the fan that is removed next."""
    li = synth(["Heatsink", "CPU fan"], tax=tax)
    keys = [k for k in li.instances if k.startswith("cpu_cooler.")]
    assert len(keys) == 2
    assert reuse_issues(li) == []


def test_a_numbered_remove_keeps_the_strict_rule(tax):
    """A numbered row is judged by the strict test alone, which needs a numbered
    candidate: the unnumbered cover it follows is not one."""
    li = synth(["Open RAM cover", "Remove RAM cover 1"], tax=tax)
    assert targets(li) == ["cover.01", "cover.02"]
    assert reuse_issues(li) == []


def test_a_numbered_first_row_is_not_an_unnumbered_candidate(tax):
    """The candidate itself must have come from an unnumbered row."""
    li = synth(["Open Power Module 1", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02"]
    assert reuse_issues(li) == []


def test_the_rule_does_not_touch_repeated_unnumbered_connectors(tax):
    """D61 style: seven ``disconnect`` rows are still seven connectors."""
    li = synth(["Case - motherboard connector"] * 7, tax=tax)
    assert len(set(targets(li))) == 7
    assert reuse_issues(li) == []


def test_a_reused_instance_receives_no_verb_twice(tax):
    """The one-operation-per-instance invariant must stay quiet after a reuse."""
    li = synth(["Open Power Module", "Power module"], tax=tax)
    assert not any("INVARIANT" in text for text in li.issues)
