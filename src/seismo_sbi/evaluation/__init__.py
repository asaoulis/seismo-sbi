"""Evaluation harness: pipeline build, held-out validation and posterior metrics.

``inference`` builds the evaluation pipeline and posterior and loads a real observation;
``validation`` runs the held-out validation and TARP coverage; ``posterior_metrics`` scores
posteriors against the truth.
"""
