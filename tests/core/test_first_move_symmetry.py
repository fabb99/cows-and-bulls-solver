from collections import Counter

from src.core.feedback import calculate_feedback
from src.core.generator import generate_all_permutations


def test_first_move_buckets_are_isomorphic():
    perms = generate_all_permutations()
    sizes = lambda g: Counter(calculate_feedback(g, s) for s in perms)
    assert sizes("0123") == sizes("8901") == sizes("4567")