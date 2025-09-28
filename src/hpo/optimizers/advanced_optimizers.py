# src/hpo/optimizers/advanced_optimizers.py
"""
Advanced Optimizers (production-ready interface)

This module contains:
- MultiObjectiveOptimizer: combine multiple objectives (accuracy, speed, memory)
  into a single weighted objective and run it through a provided HPOSystem.
- AdaptiveBudgetOptimizer: run HPO in rounds and adapt the budget based on
  observed improvement.

Design goals:
- No mocks/stand-ins: caller must provide a real HPOSystem class and a SearchSpace builder.
- Minimal assumptions about external HPO API; supports Optuna-like trial objects,
  but will work with any HPOSystem that follows the documented constructor/optimize contract.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


@dataclass
class BestTrial:
    """Normalized representation of a best trial returned by the HPO system."""

    value: float
    params: Dict[str, Any]
    user_attrs: Dict[str, Any]
    trial_id: Optional[int] = None


class MultiObjectiveOptimizer:
    """
    Multi-objective optimizer wrapper (production usage).

    Usage:
      - Provide your HPOSystem class and a SearchSpace builder (project-specific).
      - The HPOSystem must accept at least the kwargs:
            search_space, objective_function, direction, n_trials, sampler, study_name, verbose
        and expose an `optimize()` method that returns a best-trial-like object
        (with .value, .params or .values, and optional .user_attrs).
      - SearchSpace must be a builder object (or the caller can build it externally
        and pass a prepared instance via the SearchSpace parameter).

    This class only provides the objective function wrapper and utilities to
    normalize the best trial returned by the HPOSystem.
    """

    def __init__(self, objectives_weights: Optional[Dict[str, float]] = None, rng_seed: Optional[int] = None):
        self.objectives_weights = objectives_weights or {"accuracy": 0.6, "speed": 0.3, "memory": 0.1}
        self.rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random
        logger.info("MultiObjectiveOptimizer initialized with weights=%s", self.objectives_weights)

    # ---- Replace or override this simulation with a real evaluator in production ----
    def _simulate_scores(self, model_complexity: int, batch_size: int, learning_rate: float, regularization: float) -> Tuple[float, float, float, float, float]:
        """Simulation helper (kept minimal). Replace by real evaluation if available."""
        base_accuracy = 0.7 + (model_complexity / 10.0) * 0.2
        if 0.001 <= learning_rate <= 0.01:
            base_accuracy += 0.08
        if 0.01 <= regularization <= 0.05:
            base_accuracy += 0.05
        accuracy = float(min(0.99, base_accuracy + self.rng.normal(0, 0.02)))

        base_time = 10.0 + model_complexity * 2.0 + (batch_size / 32.0) * 3.0
        speed_score = float(max(1e-6, 1.0 / base_time))

        memory_usage = model_complexity * 100 + batch_size * 10
        memory_score = float(max(1e-6, 1.0 / (memory_usage / 1000.0)))

        return accuracy, speed_score, memory_score, base_time, memory_usage

    def _weighted_score(self, accuracy: float, speed_score: float, memory_score: float) -> float:
        w = self.objectives_weights
        return float(accuracy * w.get("accuracy", 0.6) + speed_score * w.get("speed", 0.3) + memory_score * w.get("memory", 0.1))

    def optimize_model_tradeoffs(
        self,
        HPOSystem: Any,
        SearchSpace: Any,
        n_trials: int = 30,
        sampler: str = "TPE",
        study_name: str = "multi_objective",
        build_search_space: bool = True,
        evaluator: Optional[Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]]] = None,
    ) -> Tuple[Optional[BestTrial], Any]:
        """
        Run HPO to explore tradeoffs.

        Parameters
        ----------
        HPOSystem: class or callable
            Instantiable with kwargs:
              search_space, objective_function, direction, n_trials, sampler, study_name, verbose
            and provides optimize() -> best_trial
        SearchSpace: class or pre-built instance
            If build_search_space is True, SearchSpace is expected to be a builder class (callable)
            that we can instantiate and add standard dimensions to. Otherwise, pass an already-built instance.
        n_trials: int
        sampler: str
        study_name: str
        build_search_space: bool
            Whether to construct the SearchSpace here (True) or assume caller passed a prepared instance (False).
        evaluator: Optional callable(params: Dict[str,Any]) -> (score: float, user_attrs: Dict)
            If provided, this evaluator will be used instead of the builtin `_simulate_scores` to compute the
            objectives. It should return a scalar score (higher is better) and a dict of user attributes
            (e.g. accuracy, time, memory).
        """

        def objective_function(trial: Any) -> float:
            # Robust param extraction (support Optuna-like and param-dict trial shims)
            try:
                model_complexity = int(trial.suggest_int("model_complexity", 1, 10))
                batch_size = int(trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]))
                learning_rate = float(trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True))
                regularization = float(trial.suggest_float("regularization", 0.0, 0.1))
                params = {"model_complexity": model_complexity, "batch_size": batch_size, "learning_rate": learning_rate, "regularization": regularization}
            except Exception:
                params = getattr(trial, "params", {}) or getattr(trial, "suggested_params", {}) or {}

            # If user provided evaluator, use it (recommended for real HPO)
            if evaluator is not None:
                score, user_attrs = evaluator(params)
            else:
                mc = int(params.get("model_complexity", 5))
                bs = int(params.get("batch_size", 64))
                lr = float(params.get("learning_rate", 1e-3))
                rg = float(params.get("regularization", 0.01))
                acc, sp, mem, est_time, est_mem = self._simulate_scores(mc, bs, lr, rg)
                score = self._weighted_score(acc, sp, mem)
                user_attrs = {"accuracy": acc, "speed_score": sp, "memory_score": mem, "estimated_time": est_time, "estimated_memory": est_mem}

            # record user attrs if trial supports it
            try:
                for k, v in (user_attrs or {}).items():
                    trial.set_user_attr(k, v)
            except Exception:
                # best-effort: attach attrs to the trial object
                try:
                    setattr(trial, "user_attrs", user_attrs)
                except Exception:
                    pass

            return float(score)

        # Build or accept search space
        if build_search_space:
            search_space = SearchSpace()
            # Try to add canonical dims; if SearchSpace interface differs user can pass prebuilt instance
            try:
                search_space.add_int("model_complexity", 1, 10)
                search_space.add_categorical("batch_size", [16, 32, 64, 128, 256])
                search_space.add_uniform("learning_rate", 1e-5, 1e-1, log=True)
                search_space.add_uniform("regularization", 0.0, 0.1)
            except Exception:
                logger.debug("SearchSpace builder didn't support standard methods; assume caller provided a prepared instance.")
        else:
            search_space = SearchSpace  # type: ignore

        hpo = HPOSystem(
            search_space=search_space,
            objective_function=objective_function,
            direction="maximize",
            n_trials=n_trials,
            sampler=sampler,
            study_name=study_name,
            verbose=False,
        )

        best = hpo.optimize()

        if best is None:
            return None, hpo

        # Normalize to BestTrial (best may be Optuna trial or custom object)
        try:
            user_attrs = getattr(best, "user_attrs", {}) or getattr(best, "attrs", {}) or {}
            params = getattr(best, "params", {}) or getattr(best, "values", {}) or {}
            best_trial = BestTrial(value=float(best.value), params=dict(params), user_attrs=dict(user_attrs), trial_id=getattr(best, "trial_id", None))
        except Exception:
            best_trial = BestTrial(value=float(getattr(best, "value", 0.0)), params=getattr(best, "params", {}) or {}, user_attrs=getattr(best, "user_attrs", {}) or {}, trial_id=None)

        return best_trial, hpo


class AdaptiveBudgetOptimizer:
    """
    Adaptive budget optimizer suitable for production HPO loops.

    Call `optimize_with_adaptive_budget` with your HPOSystem class and an objective function
    (for example, the same evaluator used above).
    """

    def __init__(self, initial_budget: int = 100, budget_increment: int = 50, performance_threshold: float = 0.01, no_improve_limit: int = 2):
        self.initial_budget = int(initial_budget)
        self.budget_increment = int(budget_increment)
        self.performance_threshold = float(performance_threshold)
        self.no_improve_limit = int(no_improve_limit)
        self.current_budget = int(initial_budget)
        self.best_performance_history: List[Dict[str, Any]] = []

    def optimize_with_adaptive_budget(
        self,
        HPOSystem: Any,
        SearchSpace: Any,
        objective_function: Callable[[Any], float],
        search_space_instance: Any,
        max_total_budget: int = 500,
        sampler: str = "TPE",
        study_name_prefix: str = "adaptive_round",
    ) -> Tuple[Optional[BestTrial], List[Any]]:
        """
        Parameters:
        - HPOSystem: your real HPOSystem class
        - SearchSpace: builder class or whatever your HPOSystem expects as search_space param
        - objective_function: callable accepting a trial-like object or params and returning float score
        - search_space_instance: the prepared/constructed search space instance to pass to each HPOSystem
        - max_total_budget: total trials budget across rounds
        """

        all_trials: List[Any] = []
        best_overall: Optional[BestTrial] = None
        rounds = 0
        total_trials_used = 0
        no_improve_rounds = 0

        while total_trials_used < max_total_budget:
            rounds += 1
            remaining_budget = min(self.current_budget, max_total_budget - total_trials_used)
            if remaining_budget <= 0:
                break

            logger.info("Starting adaptive round %d with budget=%d", rounds, remaining_budget)

            hpo = HPOSystem(
                search_space=search_space_instance,
                objective_function=objective_function,
                direction="maximize",
                n_trials=remaining_budget,
                sampler=sampler,
                study_name=f"{study_name_prefix}_{rounds}",
                verbose=False,
            )

            round_best = hpo.optimize()

            # attempt to collect trials from HPOSystem (best-effort)
            if hasattr(hpo, "trials") and getattr(hpo, "trials") is not None:
                try:
                    all_trials.extend(hpo.trials)
                except Exception:
                    pass
            else:
                try:
                    all_trials.extend(getattr(hpo, "study").trials)
                except Exception:
                    pass

            total_trials_used += remaining_budget

            if round_best is None:
                logger.warning("Round %d returned no best trial; stopping adaptive loop", rounds)
                break

            # normalize round_best
            try:
                rb_val = float(round_best.value)
                rb_params = getattr(round_best, "params", {}) or {}
                rb_attrs = getattr(round_best, "user_attrs", {}) or {}
                rb = BestTrial(value=rb_val, params=dict(rb_params), user_attrs=dict(rb_attrs), trial_id=getattr(round_best, "trial_id", None))
            except Exception:
                rb = BestTrial(value=float(getattr(round_best, "value", 0.0)), params=getattr(round_best, "params", {}) or {}, user_attrs=getattr(round_best, "user_attrs", {}) or {}, trial_id=None)

            improvement = 0.0 if best_overall is None else (rb.value - best_overall.value)

            if best_overall is None or rb.value > best_overall.value:
                logger.info("New overall best at round %d: %.6f (prev=%.6f)", rounds, rb.value, best_overall.value if best_overall else float("-inf"))
                best_overall = rb

            self.best_performance_history.append(
                {"round": rounds, "round_best": rb.value, "best_overall": best_overall.value if best_overall else None, "improvement": improvement, "total_trials_used": total_trials_used, "timestamp": time.time()}
            )

            if improvement >= self.performance_threshold:
                logger.info("Improvement %.6f >= threshold %.6f -> increase budget by %d", improvement, self.performance_threshold, self.budget_increment)
                self.current_budget += self.budget_increment
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                logger.info("No sufficient improvement in round %d (%.6f). No_improve_rounds=%d", rounds, improvement, no_improve_rounds)
                if no_improve_rounds >= self.no_improve_limit:
                    logger.info("Stopping adaptive optimization after %d rounds without improvement", no_improve_rounds)
                    break

        return best_overall, all_trials