import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askpatients.bayesian.entropy import (
    shannon_entropy, expected_entropy_for_question, select_min_entropy_question,
)


def test_entropy_of_uniform_distribution_is_log_n():
    dist = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    h = shannon_entropy(dist)
    assert math.isclose(h, math.log(4), rel_tol=1e-6)


def test_entropy_of_degenerate_distribution_is_near_zero():
    dist = {"A": 1.0 - 3e-9, "B": 1e-9, "C": 1e-9, "D": 1e-9}
    h = shannon_entropy(dist)
    assert h < 1e-4


def test_entropy_peaked_lower_than_uniform():
    peaked = {"A": 0.9, "B": 0.05, "C": 0.05}
    uniform = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    assert shannon_entropy(peaked) < shannon_entropy(uniform)


def test_entropy_empty_distribution_is_zero():
    assert shannon_entropy({}) == 0.0


def test_expected_entropy_for_question_returns_posterior_and_entropy():
    joints = [{"A": 0.4, "B": 0.1}, {"A": 0.3, "B": 0.2}]
    h, posterior = expected_entropy_for_question(joints)
    assert math.isclose(sum(posterior.values()), 1.0, rel_tol=1e-6)
    assert math.isclose(h, shannon_entropy(posterior), rel_tol=1e-9)


def test_select_min_entropy_question_picks_lowest():
    entropies = {"q1": 1.2, "q2": 0.3, "q3": 0.9}
    assert select_min_entropy_question(entropies) == "q2"


def test_select_min_entropy_question_ties_break_by_first_seen():
    entropies = {"q1": 0.5, "q2": 0.5, "q3": 0.9}
    assert select_min_entropy_question(entropies) == "q1"


def test_select_min_entropy_question_empty_raises():
    try:
        select_min_entropy_question({})
        assert False, "expected ValueError"
    except ValueError:
        pass
