from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional, Union

import numpy as np

from .optimizers.tpe import RandomSampler

# A simple shim for SearchSpace to make the file self-contained for now.
class SearchSpace:
    """A minimal SearchSpace implementation for testing purposes."""
    def __init__(self):
        self.params: Dict[str, Any] = {}

    def add_uniform(self, name: str, low: float, high: float):
        self.params[name] = {'type': 'uniform', 'bounds': (low, high)}

    def add_int(self, name: str, low: int, high: int):
        self.params[name] = {'type': 'int', 'bounds': (low, high)}

    def sample(self) -> Dict[str, float]:
        """Samples a configuration from the defined space."""
        config = {}
        for name, p_info in self.params.items():
            if p_info['type'] in ['uniform', 'int']:
                low, high = p_info['bounds']
                if p_info['type'] == 'uniform':
                    config[name] = np.random.uniform(low, high)
                else: # int
                    config[name] = np.random.randint(low, high + 1)
        return config

    @property
    def parameters(self) -> List[str]:
        return list(self.params.keys())

    def to_array(self, configs: List[Dict[str, float]]) -> np.ndarray:
        """Converts a list of configs to a numpy array."""
        arr = []
        for c in configs:
            v = [float(c.get(p, 0.0)) for p in self.parameters]
            arr.append(v)
        return np.array(arr, dtype=float)


@dataclass
class Trial:
    """Represents the result of a single trial."""
    params: Dict[str, Any]
    value: float
    state: str = "COMPLETE"

class TrialObject:
    """An object passed to the objective function to suggest parameters."""
    def __init__(self, search_space: SearchSpace, params: Dict[str, Any]):
        self._search_space = search_space
        self.params = params

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return self.params[name]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self.params[name]


class Study:
    """A minimal Study class to orchestrate an HPO process."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective_function: Callable,
        direction: str = 'minimize',
        n_trials: int = 100,
        sampler: Optional[Any] = None,
        study_name: str = "hpo_study"
    ):
        self.search_space = search_space
        self.objective = objective_function
        self.direction = direction
        self.n_trials = n_trials
        self.sampler = sampler or RandomSampler()
        self.study_name = study_name
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

    def optimize(self) -> Optional[Trial]:
        """Runs the optimization loop."""
        # Special handling for samplers that need the objective function.
        if hasattr(self.sampler, 'set_objective') and callable(self.sampler.set_objective):
            self.sampler.set_objective(self.objective, self.search_space)

        for i in range(self.n_trials):
            # Get new parameters from the sampler.
            params = self.sampler.suggest(self.trials, self.search_space)

            # Create a TrialObject to pass to the user's objective function.
            trial_obj = TrialObject(self.search_space, params)
            value = self.objective(trial_obj)

            # Store the result.
            trial_result = Trial(params=params, value=value)
            self.trials.append(trial_result)

            # Update the best trial.
            if self.best_trial is None:
                self.best_trial = trial_result
            else:
                if self.direction == 'maximize' and trial_result.value > self.best_trial.value:
                    self.best_trial = trial_result
                elif self.direction == 'minimize' and trial_result.value < self.best_trial.value:
                    self.best_trial = trial_result

            print(f"Trial {i+1}/{self.n_trials} finished. Value: {value:.6f}. Best value: {self.best_trial.value:.6f}")

        return self.best_trial