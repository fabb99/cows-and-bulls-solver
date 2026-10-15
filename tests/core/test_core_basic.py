from src.core.feedback import calculate_feedback
from src.core.generator import generate_all_feedbacks, generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.core.engine import calculate_buckets, filter_solutions
from src.core.state import state_id


def test_feedback_examples():
    assert calculate_feedback("0123", "0123") == (4, 0)
    assert calculate_feedback("0123", "3210") == (0, 4)
    assert calculate_feedback("0123", "0132") == (2, 2)
    assert calculate_feedback("0123", "4567") == (0, 0)


def test_universe_sizes():
    assert len(generate_all_permutations()) == 5040
    assert len(generate_all_feedbacks()) == 14


def test_lookup_symmetry_and_buckets():
    domain = generate_all_permutations()[:60]
    lookup = FeedbackLookup(domain)
    for a in domain[:10]:
        for b in domain[:10]:
            assert lookup.get_feedback(a, b) == lookup.get_feedback(b, a)
    buckets = calculate_buckets(domain[0], domain, lookup)
    assert sum(len(v) for v in buckets.values()) == len(domain)
    fb = lookup.get_feedback(domain[0], domain[5])
    assert filter_solutions(domain[0], fb, domain, lookup) == buckets[fb]


def test_state_id_order_independent():
    assert state_id(["0123", "4567"]) == state_id(["4567", "0123"])