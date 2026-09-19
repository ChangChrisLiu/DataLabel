from tda.core.model import FrameKey, Similarity, Visibility, Placement, StepType, VIEWS


def test_frame_key_ordering_and_hash():
    a = FrameKey(13, 2, "scan")
    b = FrameKey(13, 3, "scan")
    assert a < b
    assert len({a, b, FrameKey(13, 2, "scan")}) == 2


def test_similarity_identity():
    assert Similarity().is_identity()
    assert not Similarity(tx=0.5).is_identity()


def test_enums_roundtrip():
    assert Visibility("too_small") is Visibility.TOO_SMALL
    assert Placement("on_bench").value == "on_bench"
    assert StepType("dupli") is StepType.DUPLI
    assert VIEWS == ("scan", "oak1", "oak2", "rs")


def test_step_type_semantics_are_one_table():
    """What a step type means is decided here and read everywhere else."""
    from tda.core import model as M

    assert M.SKIP_STEP_TYPES == frozenset({StepType.IGNORE.value})
    assert M.NO_CHANGE_STEP_TYPES == frozenset(
        {StepType.IGNORE.value, StepType.INITIAL.value, StepType.DUPLI.value}
    )
    assert M.step_is_annotatable(StepType.NORMAL.value)
    assert not M.step_is_annotatable(StepType.IGNORE.value)
    # a step the table has no row for is an ordinary one
    assert M.step_is_annotatable(None) and M.step_is_annotatable("")


def test_both_exports_read_the_step_table_from_the_model():
    from tda.core import model as M
    from tda.core.export import coco

    assert coco.SKIP_STEP_TYPES is M.SKIP_STEP_TYPES
    assert coco.NO_CHANGE_STEP_TYPES is M.NO_CHANGE_STEP_TYPES
