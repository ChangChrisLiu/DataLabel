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
