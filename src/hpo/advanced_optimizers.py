"""
Advanced Optimizers Module
--------------------------

This module provides high-level, reusable optimization strategies built on top
of the core hpo.Study class. It contains implementations for finding a Pareto
front in multi-objective problems and for using successive halving for
efficient pruning.
"""

import logging
from typing import Any, Callable
import optuna

# --- Logger setup ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class ParetoFrontOptimizer:
    """
    A high-level optimizer to find the Pareto front for multi-objective problems.
    This class encapsulates the logic for setting up and running a multi-objective study.
    """
    def find_pareto_front(self, hpo_system: Any, search_space: Any,
                          objective_function: Callable,
                          directions: list, n_trials: int = 70):
        """
        Runs a multi-objective HPO loop to find the Pareto front.
        """
        hpo = hpo_system(
            search_space=search_space,
            objective_function=objective_function,
            directions=directions,
            n_trials=n_trials,
            sampler=optuna.samplers.TPESampler(),
            study_name='pareto_front_optimization'
        )

        pareto_front_trials = hpo.optimize()
        return pareto_front_trials, hpo


class SuccessiveHalvingOptimizer:
    """
    An optimizer that uses the Successive Halving algorithm via a pruner.
    """
    def __init__(self, n_steps: int = 20):
        self.n_steps = n_steps
        logger.info(f'Initialized SuccessiveHalvingOptimizer with n_steps={self.n_steps}.')

    def optimize_with_pruning(self, hpo_system: Any, search_space: Any,
                               objective_function: Callable,
                               n_trials: int = 50):
        """
        Runs an HPO loop with a pruner-compatible objective function.
        """
        def objective_with_pruning(trial):
            last_value = 0.0
            for step in range(1, self.n_steps + 1):
                value = objective_function(trial, step)
                trial.report(value, step)
                last_value = value

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return last_value

        hpo = hpo_system(
            search_space=search_space,
            objective_function=objective_with_pruning,
            direction='maximize',
            n_trials=n_trials,
            pruner=optuna.pruners.SuccessiveHalvingPruner(),
            study_name='successive_halving_optimization'
        )

        best_trial = hpo.optimize()
        return best_trial, hpo