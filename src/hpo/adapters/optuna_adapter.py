# src/hpo/adapters/optuna_adapter.py
"""
Optuna adapter for the project's HPOSystem interface.

This adapter wraps Optuna's API and exposes a minimal HPOSystem-like class
compatible with the MultiObjectiveOptimizer and AdaptiveBudgetOptimizer above.

Usage:
    from hpo.adapters.optuna_adapter import OptunaHPOSystem
    optuna_hpo = OptunaHPOSystem(search_space=search_space_instance, objective_function=objective_fn, n_trials=50)
    best = optuna_hpo.optimize()
"""

from __future__ import annotations

from typing import Any, Callable, Optional
import optuna


class OptunaHPOSystem:
    """
    Minimal adapter around Optuna to provide an `optimize()`-returning-best-trial API.

    Parameters expected to be similar to MultiObjectiveOptimizer usage:
      - search_space: (ignored here) we expect the objective to create suggestions via trial.suggest_*
        OR the caller will craft an objective that uses the provided search_space instance.
      - objective_function: callable(trial) -> float
      - direction: 'maximize' / 'minimize'
      - n_trials: number of trials
      - sampler: string (optional, e.g. 'TPE' will select optuna.samplers.TPESampler)
    """

    def __init__(self, search_space: Any, objective_function: Callable[[Any], float], direction: str = "maximize", n_trials: int = 50, sampler: Optional[str] = None, study_name: str = "optuna_hpo", verbose: bool = False):
        self.search_space = search_space
        self.objective_function = objective_function
        self.direction = direction
        self.n_trials = int(n_trials)
        self.sampler = sampler
        self.study_name = study_name
        self.verbose = verbose

        if direction == "maximize":
            self.study = optuna.create_study(direction="maximize", study_name=study_name)
        else:
            self.study = optuna.create_study(direction="minimize", study_name=study_name)

        # choose sampler if requested
        if sampler is not None:
            if sampler.upper() == "TPE":
                self.study.sampler = optuna.samplers.TPESampler()
            elif sampler.upper() == "CMAES":
                self.study.sampler = optuna.samplers.CmaEsSampler()
            # else use default

    def optimize(self) -> Optional[Any]:
        """
        Runs the optuna study optimization loop and returns the best trial object.
        The returned object will be an optuna.trial.FrozenTrial (Optuna's best_trial)
        which is compatible with the normalizing logic in advanced_optimizers.
        """
        self.study.optimize(self.objective_function, n_trials=self.n_trials)
        try:
            return self.study.best_trial
        except Exception:
            return None