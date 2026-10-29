from src.metrics import hit_at_k, reciprocal_rank


def test_hit_at_k():
    assert hit_at_k({"a"}, ["x", "a", "y"], 2) == 1.0
    assert hit_at_k({"a"}, ["x", "y", "a"], 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank({"a"}, ["a", "b"]) == 1.0
    assert reciprocal_rank({"a"}, ["b", "a"]) == 0.5
    assert reciprocal_rank({"a"}, ["b", "c"]) == 0.0
