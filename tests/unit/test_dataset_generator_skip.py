"""Unit tests for DatasetGenerator skip-on-failure + systemic-failure guard.

Dependency-free: the wrapped "simulator" is a plain callable, no Instaseis/CPS.

A small fraction of sampled sources can fall outside a forward model's valid
domain (e.g. instaseis 'Element not found'); the generator must SKIP those after
retries (rather than abort the whole run) yet still ABORT if the failure fraction
is high (a systemic problem such as a wrong DB path).
"""

import pytest

from seismo_sbi.instaseis_simulator.dataset_generator import ParallelSimulationRunner


class _Runner(ParallelSimulationRunner):
    """Concrete ParallelSimulationRunner (the abstract method is unused here)."""

    def run_and_save_simulations(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _always_fail(*args, **kwargs):
    raise ValueError("Element not found")


class TestErrorWrapper:

    def test_success_returns_true(self):
        seen = []
        runner = _Runner(lambda *a, **k: seen.append(a), num_parallel_jobs=1)
        assert runner.simulator("job") is True
        assert seen == [("job",)]

    def test_skip_after_retries_returns_false_not_raises(self):
        runner = _Runner(_always_fail, num_parallel_jobs=1)
        # Must NOT raise — a doomed sample is skipped, not fatal.
        assert runner.simulator("job") is False

    def test_retries_then_succeeds_returns_true(self):
        state = {"n": 0}

        def fail_twice(*a, **k):
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("transient")

        runner = _Runner(fail_twice, num_parallel_jobs=1)
        assert runner.simulator("job") is True
        assert state["n"] == 3


class TestSkipGuard:

    def test_no_skips_is_noop(self):
        _Runner._guard_against_excessive_skips([True, True, True])

    def test_few_skips_allowed(self):
        # 1/10 = 10% < 20% threshold
        _Runner._guard_against_excessive_skips([True] * 9 + [False])

    def test_excessive_skips_raise(self):
        with pytest.raises(RuntimeError, match="systemic"):
            _Runner._guard_against_excessive_skips([True] * 5 + [False] * 5)  # 50%


class TestRunParallelSequential:

    def test_sequential_skips_rare_failures(self):
        """num_parallel_jobs=1 -> sequential path: rare failures skipped, no crash."""
        def sim(i):
            if i == 3:           # 1 of 10 fails
                raise ValueError("Element not found")
        runner = _Runner(sim, num_parallel_jobs=1)
        runner.run_parallel_simulations([(i,) for i in range(10)])  # must not raise

    def test_sequential_aborts_on_systemic_failure(self):
        runner = _Runner(_always_fail, num_parallel_jobs=1)
        with pytest.raises(RuntimeError, match="systemic"):
            runner.run_parallel_simulations([(i,) for i in range(10)])  # 100% fail
