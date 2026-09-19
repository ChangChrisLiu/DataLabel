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


def test_d45_a_failed_attempt_then_the_real_removal_is_one_instance(tax):
    """A failed attempt changes no state, so the part is still there to remove."""
    li = synth(["Try to remove power module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.01"]
    assert sorted(k for k in li.instances if k.startswith("psu")) == ["psu.01"]
    assert any("reuses psu.01" in text and "failed remove" in text
               for text in reuse_issues(li))


def test_d36_the_bracketed_failed_marker_does_not_split_the_identity(tax):
    """``(failed)`` is how the action went, not which part it was about."""
    li = synth(["Try to remove the power module (failed)", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.01"]
    assert "qualifier" not in li.instances["psu.01"].attrs


def test_a_real_qualifier_still_splits_the_identity(tax):
    """Only the result marker goes; a qualifier that names the part stays."""
    li = synth(["Open RAM cover (left)", "Remove RAM cover"], tax=tax)
    assert targets(li) == ["cover.01", "cover.02"]
    assert li.instances["cover.01"].attrs["qualifier"] == "left"
    assert reuse_issues(li) == []


def test_a_failed_attempt_on_something_already_removed_is_not_a_candidate(tax):
    """"Still present" is a condition: a second candidate makes it ambiguous."""
    li = synth(["Try to remove power module", "Try to remove power module",
                "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02", "psu.03"]
    assert reuse_issues(li) == []


def test_the_step_type_of_a_failed_row_is_unchanged(tax):
    """Dropping the qualifier must not stop the row counting as an attempt."""
    li = synth(["Try to remove the power module (failed)"], tax=tax)
    assert [s.step_type for s in li.steps][-1] == "failed"
    assert [a.result for a in li.actions] == ["failed"]


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


def test_a_failed_attempt_does_not_trip_the_one_verb_invariant(tax):
    """D35/D36/D45: `failed remove` then `remove` is one part acted on once.

    The invariant guards against two different parts being merged onto one key,
    and an attempt that did not happen is not an operation -- so counting it
    made the check cry wolf at exactly the three merges ruling A asked for.
    """
    li = synth(["Try to remove power module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.01"]
    assert not any("INVARIANT" in text for text in li.issues)


def _checked(tax, ops) -> list[str]:
    """Run the one-verb-per-instance guard over a hand-made history."""
    from tda.core.logs import _Importer
    from tda.core.model import InstanceRec

    importer = _Importer(99, {}, tax)
    importer.out.instances["psu.01"] = InstanceRec(key="psu.01", desktop=99, cls="psu")
    for step, verb, result in ops:
        importer.ledger.record("psu.01", step, verb, result)
    importer._check_operations()
    return [text for text in importer.out.issues if "INVARIANT" in text]


def test_the_invariant_still_fires_for_two_successful_removes(tax):
    """A real double-remove on one key can only mean two parts were merged."""
    found = _checked(tax, [(3, "remove", "success"), (9, "remove", "success")])
    assert len(found) == 1
    assert "psu.01" in found[0] and "remove" in found[0]


def test_the_invariant_ignores_an_attempt_that_did_not_happen(tax):
    assert _checked(tax, [(3, "remove", "failed"), (9, "remove", "success")]) == []
    assert _checked(tax, [(3, "remove", "failed"), (9, "remove", "failed")]) == []


# --------------------------------------------------------------------------- #
# ambiguity is reported, not silently declined
# --------------------------------------------------------------------------- #
def test_several_candidates_raise_an_issue_instead_of_going_quiet(tax):
    """Declining is right; declining in silence is how a part grows unnoticed."""
    li = synth(["Open Power Module", "Open Power Module", "Power module"], tax=tax)
    assert targets(li) == ["psu.01", "psu.02", "psu.03"]
    assert reuse_issues(li) == []
    ambiguous = [t for t in li.issues if "candidates for an unnumbered remove" in t]
    assert len(ambiguous) == 1
    assert "psu.01" in ambiguous[0] and "psu.02" in ambiguous[0]
    assert "left unmerged" in ambiguous[0]


def test_no_candidates_says_nothing(tax):
    li = synth(["Power module"], tax=tax)
    assert not any("candidates for an unnumbered remove" in t for t in li.issues)
