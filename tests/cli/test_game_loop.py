import pytest

from src.cli.game_loop import fb_tuple_to_string, parse_feedback, play_autoplay
from src.core.generator import generate_all_feedbacks, generate_all_permutations
from src.core.lookup_table import FeedbackLookup
from src.strategies.entropy import EntropyStrategy


@pytest.mark.parametrize(
    "text, expected",
    [("++--", (2, 2)), ("-+", (1, 1)), ("0", (0, 0)), ("++++", (4, 0)), (" +- ", (1, 1))],
)
def test_parse_valid(text, expected):
    assert parse_feedback(text) == expected


@pytest.mark.parametrize("text", ["", "+++-", "+-x", "+++++", "1", "++--+"])
def test_parse_invalid(text):
    assert parse_feedback(text) is None


def test_string_roundtrip_for_all_feedbacks():
    for fb in generate_all_feedbacks():
        text = fb_tuple_to_string(fb) or "0"
        assert parse_feedback(text) == fb


def test_autoplay_returns_turns_and_checks_secret():
    domain = generate_all_permutations()[:40]
    strategy = EntropyStrategy(domain, FeedbackLookup(domain))
    turns = play_autoplay(strategy, domain[7], domain, verbose=False)
    assert 1 <= turns <= 6
    with pytest.raises(ValueError):
        play_autoplay(strategy, "9876", domain, verbose=False)