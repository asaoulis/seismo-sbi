"""Evaluation harness: pipeline build, held-out validation and tensor comparison.

``inference`` builds the evaluation pipeline and posterior and loads a real observation;
``validation`` runs the held-out validation and TARP coverage; ``moment_tensor`` holds the
tensor-comparison primitives. Imports are lazy inside functions, so importing this package
pulls in neither torch nor the pipeline classes.
"""
