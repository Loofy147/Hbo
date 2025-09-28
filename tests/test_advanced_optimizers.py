# tests/test_advanced_optimizers.py
"""
Integration tests for Advanced Optimizers using the Optuna adapter.

Notes:
- These are lightweight smoke/integration tests (fast) intended to run in CI.
- They create uniquely-named Optuna studies to avoid name collisions.
- Requires `optuna` installed (add to dev-requirements.txt).
"""

import time
from typing import Any, Dict, Tuple

import pytest

from hpo.optimizers.advanced_optimizers import MultiObjectiveOptimizer, AdaptiveBudgetOptimizer
from hpo.adapters.optuna_adapter import OptunaHPOSystem


class TestSearchSpace:
    """
    Minimal SearchSpace builder used for tests.
    The real project will have its own SearchSpace; this test only needs the
    methods used by MultiObjectiveOptimizer when build_search_space=True.
    """

    def __init__(self):
        self.space = {}

    def add_int(self, name: str, low: int, high: int):
        self.space[name] = ("int", low, high)

    def add_categorical(self, name: str, choices):
        self.space[name] = ("cat", list(choices))

    def add_uniform(self, name: str, low: float, high: float, log: bool = False):
        self.space[name] = ("uniform", low, high, log)


def unique_study_name(prefix: str) -> str:
    """Return a unique study name for Optuna to avoid clashes."""
    return f"{prefix}_{int(time.time() * 1000)}"


def test_optuna_integration_multiobjective_smoke():
    """
    Smoke test: run MultiObjectiveOptimizer.optimize_model_tradeoffs using OptunaHPOSystem.
    We expect a BestTrial-like object returned (not None) and a float value.
    """
    ss_builder = TestSearchSpace  # pass builder class so optimizer will build canonical dims
    opt = MultiObjectiveOptimizer(rng_seed=42)

    best, hpo_inst = opt.optimize_model_tradeoffs(
        HPOSystem=OptunaHPOSystem,
        SearchSpace=ss_builder,
        n_trials=5,
        sampler="TPE",
        study_name=unique_study_name("test_multiobj"),
        build_search_space=True,
        evaluator=None,  # use internal weighted simulation (fast)
    )

    assert best is not None, "Expected best trial, got None"
    assert isinstance(best.value, float), "Best trial value should be float"
    # user_attrs may be empty depending on HPO internals, but if present should include accuracy or speed keys
    assert hasattr(best, "params")
    assert isinstance(best.params, dict)


def test_adaptive_budget_with_optuna_smoke():
    """
    Smoke test: run AdaptiveBudgetOptimizer using Optuna adapter.
    The test uses a simple objective that mirrors the internal simulator of MultiObjectiveOptimizer.
    The objective uses trial.suggest_* to be fully compatible with Optuna.
    """

    # Reuse the scoring logic from MultiObjectiveOptimizer by instantiating it.
    simulator = MultiObjectiveOptimizer(rng_seed=7)

    def objective(trial: Any) -> float:
        # Use Optuna trial.suggest_* API to get params
        mc = int(trial.suggest_int("model_complexity", 1, 10))
        bs = int(trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]))
        lr = float(trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True))
        rg = float(trial.suggest_float("regularization", 0.0, 0.1))
        acc, sp, mem, et, em = simulator._simulate_scores(mc, bs, lr, rg)
        score = simulator._weighted_score(acc, sp, mem)
        # Optionally set user attrs if trial supports it (Optuna trial does support setting user attrs)
        try:
            trial.set_user_attr("accuracy", float(acc))
            trial.set_user_attr("estimated_time", float(et))
        except Exception:
            pass
        return float(score)

    # Prepare search space instance (not used by Optuna adapter but kept for API compatibility)
    ss_instance = TestSearchSpace()
    ss_instance.add_int("model_complexity", 1, 10)
    ss_instance.add_categorical("batch_size", [16, 32, 64, 128, 256])
    ss_instance.add_uniform("learning_rate", 1e-5, 1e-1, log=True)
    ss_instance.add_uniform("regularization", 0.0, 0.1)

    adapt = AdaptiveBudgetOptimizer(initial_budget=3, budget_increment=2, performance_threshold=1e-6, no_improve_limit=2)

    best_overall, all_trials = adapt.optimize_with_adaptive_budget(
        HPOSystem=OptunaHPOSystem,
        SearchSpace=TestSearchSpace,
        objective_function=objective,
        search_space_instance=ss_instance,
        max_total_budget=8,
        sampler="TPE",
        study_name_prefix=unique_study_name("test_adapt"),
    )

    assert best_overall is not None, "Adaptive optimizer should return a best overall trial"
    assert isinstance(best_overall.value, float)
    # all_trials should be a list (possibly empty if the adapter doesn't expose trials)
    assert isinstance(all_trials, list)