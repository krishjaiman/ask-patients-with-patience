"""
Shannon-entropy-based Bayesian active question selection: Equations (4)-(5).

WHY entropy: entropy of a probability distribution quantifies how
*uncertain* it is -- a peaked distribution (one disease dominant) has
low entropy, a flat/uniform distribution has high entropy. The paper
uses this as a proxy for diagnostic confidence (Section 3.4.2).

WHY minimum expected entropy determines the question: for each
candidate follow-up question q_k, we don't know in advance what the
patient will answer, but we can *simulate* plausible answers (Section
2.5) and compute what the resulting disease distribution P(d_i | q_k)
would look like on average (Eq. 3). The question whose *expected*
resulting distribution is most peaked (lowest entropy, Eq. 4) is the
one expected to be most informative -- i.e. to most reduce diagnostic
uncertainty. Choosing q*_{t+1} = argmin_k H_{q_k} (Eq. 5) is a concrete,
inspectable instantiation of Bayesian active learning / maximum
expected information gain, rather than an LLM "vibe check" of which
question seems best.
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
