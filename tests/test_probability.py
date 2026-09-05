import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askpatients.bayesian.probability import (
    normalize_distribution, joint_probability, posterior_given_question,
    bayesian_update_with_observed_response,
)


def test_normalize_distribution_sums_to_one():
    d = normalize_distribution({"A": 2.0, "B": 1.0, "C": 1.0})
    assert math.isclose(sum(d.values()), 1.0, rel_tol=1e-9)
    assert d["A"] > d["B"] == d["C"]


def test_normalize_distribution_handles_zeros_with_floor():
    d = normalize_distribution({"A": 0.0, "B": 0.0}, floor=1e-6)
    assert math.isclose(sum(d.values()), 1.0, rel_tol=1e-6)
    assert d["A"] == d["B"]


def test_normalize_distribution_empty():
    assert normalize_distribution({}) == {}


def test_joint_probability_eq2():
    prior = {"A": 0.6, "B": 0.4}
    likelihood = {"A": 0.8, "B": 0.1}
    joint = joint_probability(likelihood, prior)
    assert math.isclose(joint["A"], 0.6 * 0.8)
    assert math.isclose(joint["B"], 0.4 * 0.1)


def test_joint_probability_missing_disease_is_zero():
    prior = {"A": 0.5, "B": 0.5}
    likelihood = {"A": 0.9}  # B omitted
    joint = joint_probability(likelihood, prior)
    assert joint["B"] == 0.0


def test_posterior_given_question_eq3_normalizes():
    # Two hypothetical responses, each with per-disease joint probabilities.
    joints = [
        {"A": 0.4, "B": 0.1},
        {"A": 0.3, "B": 0.2},
    ]
    posterior = posterior_given_question(joints)
    assert math.isclose(sum(posterior.values()), 1.0, rel_tol=1e-9)
    # A's numerator (0.7) > B's numerator (0.3) -> A should dominate.
    assert posterior["A"] > posterior["B"]
    assert math.isclose(posterior["A"], 0.7 / 1.0)
    assert math.isclose(posterior["B"], 0.3 / 1.0)


def test_posterior_given_question_all_zero_falls_back_to_uniform():
    joints = [{"A": 0.0, "B": 0.0}]
    posterior = posterior_given_question(joints)
    assert math.isclose(posterior["A"], posterior["B"])
    assert math.isclose(sum(posterior.values()), 1.0)


def test_posterior_given_question_empty_list():
    assert posterior_given_question([]) == {}


def test_bayesian_update_shifts_belief_toward_higher_likelihood_disease():
    prior = {"A": 0.5, "B": 0.5}
    # Observed response strongly favors A over B.
    likelihood = {"A": 0.9, "B": 0.1}
    posterior = bayesian_update_with_observed_response(prior, likelihood)
    assert posterior["A"] > posterior["B"]
    assert math.isclose(sum(posterior.values()), 1.0, rel_tol=1e-6)


def test_bayesian_update_is_order_independent_of_prior_scale():
    # Renormalizing an unnormalized prior first shouldn't change the result.
    prior_a = normalize_distribution({"A": 2.0, "B": 2.0})
    prior_b = normalize_distribution({"A": 5.0, "B": 5.0})
    likelihood = {"A": 0.7, "B": 0.2}
    post_a = bayesian_update_with_observed_response(prior_a, likelihood)
    post_b = bayesian_update_with_observed_response(prior_b, likelihood)
    assert math.isclose(post_a["A"], post_b["A"], rel_tol=1e-6)
