"""Scoring according to the paper's printed Equations 3-5.

This is entropy of an aggregated virtual distribution, NOT standard expected
posterior entropy. If response likelihoods form a complete normalized partition,
aggregation recovers the prior and cannot distinguish questions. Keep this as the
paper-formula baseline; evaluate standard expected information gain separately.
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Tuple

from .probability import Distribution


def shannon_entropy(distribution: Mapping[str, float], floor: float = 1e-12) -> float:
    """
    Equation (4): H = - sum_i P(d_i) * log P(d_i)

    Uses natural log (paper does not specify base; base is a matter of
    units/scale only and does not change the argmin in Eq. 5 -- see
    README "Implementation decisions"). A numerical floor prevents
    log(0) for any disease the posterior assigned exactly zero mass.
    """
    if not distribution:
        return 0.0
    total = sum(distribution.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for p in distribution.values():
        p_norm = max(p / total, floor)
        h -= p_norm * math.log(p_norm)
    return h


def expected_entropy_for_question(per_response_joints: List[Distribution]) -> Tuple[float, Distribution]:
    """
    Computes H_{q_k} for one candidate question: first marginalizes the
    per-hypothesized-response joint distributions into the "virtual
    next-step" posterior P(d_i | q_k) (Eq. 3, via
    bayesian.probability.posterior_given_question), then takes its
    entropy (Eq. 4). Returns (entropy, posterior) so callers can inspect
    the resulting distribution as well as its scalar uncertainty.
    """
    from .probability import posterior_given_question

    posterior = posterior_given_question(per_response_joints)
    return shannon_entropy(posterior), posterior


def select_min_entropy_question(question_entropies: Mapping[str, float]) -> str:
    """
    Equation (5): q*_{t+1} = argmin_{q_k in Q_{t+1}} H_{q_k}

    Ties are broken by insertion order (first-seen candidate wins),
    which keeps selection deterministic given deterministic candidate
    generation -- important for reproducibility of experiments.
    """
    if not question_entropies:
        raise ValueError("No candidate questions to select from.")
    best_q = None
    best_h = math.inf
    for q, h in question_entropies.items():
        if h < best_h:
            best_h = h
            best_q = q
    assert best_q is not None
    return best_q
