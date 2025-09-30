from __future__ import annotations
from typing import Dict, Any, Callable, List, Optional

import numpy as np

from unified_quantum_hpo.kaehler_optimizer import KaehlerHPOOptimizer
from hpo.study import TrialObject


class QuantumHPOSampler:
    """
    A sampler that wraps the KaehlerHPOOptimizer to make it compatible with the HPO engine.
    """
    def __init__(self, n_startup_trials: int = 10, direction: str = 'minimize'):
        self.n_startup_trials = n_startup_trials
        self.direction = direction
        self.optimizer: Optional[KaehlerHPOOptimizer] = None
        self.objective_fn: Optional[Callable[[Dict[str, float]], float]] = None

    def set_objective(self, objective_fn: Callable[[Dict[str, float]], float], search_space):
        """
        Sets the objective function and initializes the KaehlerHPOOptimizer.
        This method must be called by the Study before optimization begins.
        """
        self.objective_fn = objective_fn

        def objective_adapter(config: Dict[str, float]) -> float:
            """Adapts the dict-based call from Kaehler to the TrialObject-based user objective."""
            trial_obj = TrialObject(search_space, config)
            return self.objective_fn(trial_obj)

        # The Kaehler optimizer expects a function that returns a loss (lower is better).
        # If we are maximizing, we need to negate the output of the user's objective.
        if self.direction == 'maximize':
            wrapped_objective = lambda config: -objective_adapter(config)
        else:
            wrapped_objective = objective_adapter

        self.optimizer = KaehlerHPOOptimizer(
            config_space=search_space,
            objective_fn=wrapped_objective
        )

    def suggest(self, trials: List[Any], search_space) -> Dict[str, float]:
        """
        Suggests the next hyperparameters.

        For the first `n_startup_trials`, it samples randomly. After that, it uses
        the KaehlerHPOOptimizer to suggest the next point based on the best trial so far.
        """
        if len(trials) < self.n_startup_trials:
            return search_space.sample()

        if not self.optimizer:
            raise RuntimeError(
                "The QuantumHPOSampler's objective function has not been set. "
                "The Study object is responsible for calling `set_objective` on the sampler."
            )

        # Find the best trial so far to use as the starting point for the gradient flow.
        if self.direction == 'maximize':
            best_trial = max(trials, key=lambda t: t.value)
        else:
            best_trial = min(trials, key=lambda t: t.value)

        # Get the current best configuration and map it to the Kähler manifold.
        z_current = self.optimizer.complexify_config(best_trial.params)

        # Use the circular echo integrator to find the next point.
        z_next = self.optimizer.integrate_with_circular_echo(
            z_current,
            echo_strength=0.01,
            neumann_steps=3,
            metric_method='diag'  # Use 'diag' for better performance
        )

        # Convert the new point back to a configuration dictionary.
        next_config = self.optimizer._z_to_config(z_next)

        # It's possible for the optimizer to suggest a point outside the defined bounds.
        # We should clip the values to be safe.
        for name, param in search_space.params.items():
            if param['type'] in ['uniform', 'int']:
                low, high = param['bounds']
                if name in next_config:
                    next_config[name] = np.clip(next_config[name], low, high)
                    if param['type'] == 'int':
                        next_config[name] = int(round(next_config[name]))

        return next_config