"""
A Tree-structured Parzen Estimator (TPE) sampler.
"""
from __future__ import annotations
from typing import Dict, Any, List
from types import SimpleNamespace

import numpy as np
from scipy.stats import gaussian_kde

from .base import BaseOptimizer
from ..study import SearchSpace

class TPESampler(BaseOptimizer):
    """
    A Tree-structured Parzen Estimator (TPE) sampler.

    TPE is a sequential model-based optimization (SMBO) algorithm that models
    the probability of observing good and bad results and proposes new candidates
    based on the ratio of these probabilities (Expected Improvement).
    """
    def __init__(self, search_space: SearchSpace, n_startup_trials: int = 10,
                 n_ei_candidates: int = 24, gamma: float = 0.25, direction: str = 'maximize'):
        self.search_space = search_space
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.direction = direction
        self.trials: List[SimpleNamespace] = []

    def tell(self, config: Dict[str, Any], result: float) -> None:
        """
        Stores the result of a completed trial.
        """
        trial_result = SimpleNamespace(params=config, value=result, state='COMPLETE')
        self.trials.append(trial_result)

    def ask(self) -> Dict[str, Any]:
        """
        Suggests a new set of hyperparameters based on the TPE algorithm.

        If the number of completed trials is less than `n_startup_trials`, it
        falls back to random sampling.
        """
        if len(self.trials) < self.n_startup_trials:
            return self.search_space.sample()

        # Sort trials by performance
        is_maximize = self.direction == 'maximize'
        sorted_trials = sorted(self.trials, key=lambda t: t.value, reverse=is_maximize)

        # Split into good and bad trials
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]

        best_params = None
        best_ei = -np.inf

        # Generate and evaluate candidates
        for _ in range(self.n_ei_candidates):
            candidate = self.search_space.sample()

            # Calculate probability densities for good and bad groups
            good_density = self._compute_density(candidate, good_trials)
            bad_density = self._compute_density(candidate, bad_trials)

            if bad_density > 0:
                ei = good_density / bad_density
                if ei > best_ei:
                    best_ei = ei
                    best_params = candidate

        return best_params if best_params is not None else self.search_space.sample()

    def _compute_density(self, candidate: Dict[str, Any], trials: List[SimpleNamespace]) -> float:
        """
        Computes the probability density for a given candidate based on a set of trials.
        """
        if not trials:
            return 1.0

        log_density = 0.0
        for name, value in candidate.items():
            param_config = self.search_space.params[name]
            param_values = [t.params[name] for t in trials if name in t.params]

            if not param_values:
                continue

            if param_config['type'] in ['uniform', 'int', 'float']:
                if len(param_values) < 2:
                    log_density += 0.0
                else:
                    try:
                        kde = gaussian_kde(param_values)
                        density = kde.pdf([value])[0]
                        log_density += np.log(max(density, 1e-10))
                    except (np.linalg.LinAlgError, ValueError):
                        log_density += 0.0

            elif param_config['type'] == 'categorical':
                count = param_values.count(value)
                num_choices = len(param_config.get('choices', []))
                prob = (count + 1) / (len(param_values) + num_choices)
                log_density += np.log(prob)

        return np.exp(log_density)