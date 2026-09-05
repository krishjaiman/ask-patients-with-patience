"""
AskPatients: a research reproduction of

  Zhu, Pan, Liu, Liu, Wu. "Ask Patients with Patience: Enabling LLMs for
  Human-Centric Medical Dialogue with Grounded Reasoning" (EMNLP 2025).

This package is a faithful, from-scratch reimplementation of the paper's
Dr.APP pipeline: symptom mapping -> disease probability prediction ->
question generation -> one-more-step response simulation -> Bayesian
conditional-probability computation -> expected-entropy question
selection -> human-centric question rewriting -> patient interaction.

See README.md at the repository root for the full paper-to-code mapping
and a list of implementation decisions taken where the paper under-
specifies details.

RESEARCH / EDUCATIONAL USE ONLY. Nothing in this repository is a medical
device or a substitute for professional medical advice.
"""

__version__ = "0.1.0"
