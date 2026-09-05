"""
Bayesian probability computations implementing Equations (1)-(3) of the
paper. This module contains NO LLM calls -- it is pure, deterministic
Python/numpy so the probabilistic model itself can be independently
unit-tested and audited. LLM calls only ever *supply inputs* to these
functions (a prior distribution, per-disease response likelihoods);
every actual probability computation happens here.

WHY Bayesian updating: the paper's core claim is that Dr.APP's
diagnostic reasoning should be transparent and grounded rather than a
black-box LLM guess. Representing the diagnosis as an explicit
probability distribution P_t(D) over a candidate disease set, and
updating it via Bayes' rule as new patient responses arrive, is what
makes the "confidence" of a diagnosis at any point in the dialogue a
well-defined, inspectable number (see Section 2.3, Eq. 1; Section 2.6,
Eq. 2-3) instead of an opaque LLM claim.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

Distribution = Dict[str, float]


def normalize_distribution(probs: Mapping[str, float], floor: float = 1e-6) -> Distribution:
    """
    Renormalize an arbitrary non-negative weighting over diseases into a
    valid probability distribution summing to 1 (Eq. 1's constraint
    sum_i P_t(d_i | ...) = 1). A small `floor` is applied to every entry
    before renormalizing so that no disease is ever assigned exactly zero
    probability (which would make it permanently unrecoverable under
    Bayesian updating, and would break log(.) in the entropy computation).
    """
    if not probs:
        return {}
    floored = {k: max(float(v), floor) for k, v in probs.items()}
    total = sum(floored.values())
    if total <= 0:
        # Degenerate input (e.g. all zeros): fall back to a uniform prior.
        n = len(floored)
        return {k: 1.0 / n for k in floored}
    return {k: v / total for k, v in floored.items()}


def joint_probability(response_likelihood: Mapping[str, float],
                       prior: Mapping[str, float]) -> Distribution:
    """
    Equation (2):
        P(r_k^l, d_i) = P(r_k^l | Gamma(d_i)) * P_t(d_i)

    `response_likelihood` maps disease -> P(r_k^l | Gamma(d_i)), the
    LLM-estimated conditional probability of observing a *specific*
    candidate response r_k^l given each disease's grounded clinical
    profile Gamma(d_i). `prior` is the current disease distribution
    P_t(D). Diseases missing from `response_likelihood` are treated as
    having zero response likelihood for this particular response (i.e.
    they contribute nothing to this response's joint probability, but
    are NOT dropped from the overall computation -- see
    `posterior_given_question`, which sums over ALL diseases in the
    denominator).
    """
    joint: Distribution = {}
    for disease, p_prior in prior.items():
        p_resp_given_disease = float(response_likelihood.get(disease, 0.0))
        joint[disease] = p_resp_given_disease * float(p_prior)
    return joint


def posterior_given_question(per_response_joints: List[Distribution]) -> Distribution:
    """
    Equation (3):
        P(d_i | q_k) = [ sum_l P(r_k^l, d_i) ] / [ sum_j sum_l P(r_k^l, d_j) ]

    `per_response_joints` is the list (length L, one per simulated
    plausible response r_k^1..r_k^L) of joint-probability dicts produced
    by `joint_probability`. This marginalizes out which specific
    response was hypothesized, giving the "virtual next-step disease
    probability distribution" the paper uses to score a *candidate
    question* (rather than a specific hypothetical answer) before it is
    actually asked.
    """
    if not per_response_joints:
        return {}
    diseases = set()
    for joint in per_response_joints:
        diseases.update(joint.keys())

    numerators: Dict[str, float] = {d: 0.0 for d in diseases}
    for joint in per_response_joints:
        for d in diseases:
            numerators[d] += joint.get(d, 0.0)

    denominator = sum(numerators.values())
    if denominator <= 0:
        # No signal at all from any hypothesized response for any disease:
        # fall back to a uniform posterior rather than dividing by zero.
        n = len(diseases)
        return {d: 1.0 / n for d in diseases} if n else {}

    return {d: v / denominator for d, v in numerators.items()}


def bayesian_update_with_observed_response(
    prior: Mapping[str, float],
    response_likelihood: Mapping[str, float],
    floor: float = 1e-6,
) -> Distribution:
    """
    Applies the *actual* (not hypothetical) patient response after
    q*_{t+1} has been asked, updating P_t(D) -> P_{t+1}(D). This is the
    same Bayes-rule mechanics as Eq. 2-3 but collapsed to a single
    observed response instead of marginalizing over L hypothetical ones:

        P_{t+1}(d_i) propto P(r_observed | Gamma(d_i)) * P_t(d_i)

    normalized over all diseases. This is what actually advances the
    dialogue's belief state each turn (Section 2.6, final paragraph:
    "the user's response r_{t+1} is incorporated into the dialogue ...
    updating P_{t+1}(D)").
    """
    joint = joint_probability(response_likelihood, prior)
    return normalize_distribution(joint, floor=floor)
