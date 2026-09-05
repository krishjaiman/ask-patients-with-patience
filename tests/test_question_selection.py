"""
NOTE on test design: Equation (3) of the paper, as written, marginalizes
over hypothesized responses by an *unweighted* sum:
    P(d_i | q_k) = sum_l P(r_k^l, d_i) / sum_j sum_l P(r_k^l, d_j)
It does not multiply by an explicit prior probability of each response
occurring. One consequence (a property of the paper's formula itself,
not a bug in this reproduction) is that a *perfectly symmetric* set of
hypothesized responses -- e.g. one response that strongly favors disease
A and another, equally-weighted, response that strongly favors disease B
by the same margin -- averages back out to the prior, since the two
hypothesized scenarios cancel. A genuinely informative question, in
practice, skews the *aggregate* mass toward one outcome (its plausible
responses don't cancel symmetrically); the tests below use that more
realistic, asymmetric shape to exercise the "more informative -> lower
expected entropy -> selected" behavior end-to-end.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from askpatients.bayesian.entropy import expected_entropy_for_question, select_min_entropy_question
from askpatients.bayesian.probability import joint_probability


def _entropy_for_candidate(prior, per_response_likelihoods):
    """Helper: given a prior and a list of {disease: likelihood} dicts (one
    per hypothetical response), compute the candidate question's H_{q_k}."""
    joints = [joint_probability(lik, prior) for lik in per_response_likelihoods]
    h, _ = expected_entropy_for_question(joints)
    return h


def test_uninformative_question_leaves_entropy_at_uniform_prior():
    prior = {"Disease A": 0.5, "Disease B": 0.5}
    q_responses = [
        {"Disease A": 0.5, "Disease B": 0.5},
        {"Disease A": 0.5, "Disease B": 0.5},
    ]
    h = _entropy_for_candidate(prior, q_responses)
    assert math.isclose(h, math.log(2), rel_tol=1e-6)


def test_consistently_skewed_question_lowers_entropy_below_uniform():
    prior = {"Disease A": 0.5, "Disease B": 0.5}
    uniform_h = math.log(2)
    skewed_responses = [
        {"Disease A": 0.9, "Disease B": 0.1},
        {"Disease A": 0.8, "Disease B": 0.2},
        {"Disease A": 0.7, "Disease B": 0.15},
    ]
    h = _entropy_for_candidate(prior, skewed_responses)
    assert h < uniform_h


def test_selection_picks_the_more_informative_question():
    prior = {"Disease A": 0.5, "Disease B": 0.5}
    candidates = {
        "vague question": [
            {"Disease A": 0.5, "Disease B": 0.5},
            {"Disease A": 0.5, "Disease B": 0.5},
        ],
        "informative question": [
            {"Disease A": 0.9, "Disease B": 0.1},
            {"Disease A": 0.8, "Disease B": 0.2},
        ],
    }
    entropies = {q: _entropy_for_candidate(prior, resp) for q, resp in candidates.items()}
    selected = select_min_entropy_question(entropies)
    assert selected == "informative question"


def test_three_disease_case_selects_the_more_informative_of_two_questions():
    prior = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    candidates = {
        "q_noisy": [
            {"A": 0.33, "B": 0.33, "C": 0.34},
            {"A": 0.34, "B": 0.33, "C": 0.33},
        ],
        "q_focused": [
            {"A": 0.8, "B": 0.1, "C": 0.1},
            {"A": 0.7, "B": 0.15, "C": 0.15},
            {"A": 0.75, "B": 0.1, "C": 0.1},
        ],
    }
    entropies = {q: _entropy_for_candidate(prior, resp) for q, resp in candidates.items()}
    assert select_min_entropy_question(entropies) == "q_focused"
