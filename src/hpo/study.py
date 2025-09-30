"""
The central user-facing API for running HPO studies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional, Union

import numpy as np

from .optimizers.base import BaseOptimizer
from .optimizers.random import RandomSampler

# --- Data Structures ---

@dataclass
class Trial:
    """Represents a single evaluation of a hyperparameter configuration."""
    trial_id: int
    params: Dict[str, Any]
    value: Optional[float] = None
    state: str = "PENDING"  # PENDING, RUNNING, COMPLETE, FAILED

class SearchSpace:
    """
    Defines the search space for an HPO study.
    Provides methods for adding parameters and sampling configurations.
    """
    def __init__(self):
        self.params: Dict[str, Dict[str, Any]] = {}

    def add_float(self, name: str, low: float, high: float, log: bool = False):
        self.params[name] = {'type': 'float', 'bounds': (low, high), 'log': log}

    def add_int(self, name: str, low: int, high: int):
        self.params[name] = {'type': 'int', 'bounds': (low, high)}

    def add_categorical(self, name: str, choices: List[Any]):
        self.params[name] = {'type': 'categorical', 'choices': choices}

    def sample(self) -> Dict[str, Any]:
        """Samples a random configuration from the defined space."""
        config = {}
        for name, p_info in self.params.items():
            if p_info['type'] == 'float':
                if p_info['log']:
                    config[name] = np.exp(np.random.uniform(np.log(p_info['bounds'][0]), np.log(p_info['bounds'][1])))
                else:
                    config[name] = np.random.uniform(p_info['bounds'][0], p_info['bounds'][1])
            elif p_info['type'] == 'int':
                config[name] = np.random.randint(p_info['bounds'][0], p_info['bounds'][1] + 1)
            elif p_info['type'] == 'categorical':
                config[name] = np.random.choice(p_info['choices'])
        return config

# --- Main Study Class ---

class Study:
    """
    The main class for orchestrating an HPO study.

    This class manages the optimization loop, interacts with the provided optimizer,
    and stores the results of all trials.
    """
    def __init__(
        self,
        search_space: SearchSpace,
        optimizer: Optional[BaseOptimizer] = None,
        direction: str = 'maximize'
    ):
        self.search_space = search_space
        self.optimizer = optimizer or RandomSampler(self.search_space)
        self.direction = direction
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

    def optimize(self, objective: Callable[[Dict[str, Any]], float], n_trials: int):
        """
        Runs the optimization loop for a given number of trials.

        Args:
            objective: A function that takes a dictionary of hyperparameters
                       and returns a score to be optimized.
            n_trials: The number of trials to run.
        """
        print(f"🚀 Starting optimization study with {n_trials} trials.")
        print(f"   - Optimizer: {self.optimizer.__class__.__name__}")
        print(f"   - Direction: {self.direction}")

        for i in range(n_trials):
            # 1. Ask the optimizer for a new configuration
            params = self.optimizer.ask()

            trial = Trial(trial_id=i, params=params, state="RUNNING")
            self.trials.append(trial)

            try:
                # 2. Evaluate the objective function
                value = objective(params)
                trial.value = value
                trial.state = "COMPLETE"

                # 3. Tell the optimizer the result
                self.optimizer.tell(params, value)

                # 4. Update the best trial
                if self.best_trial is None:
                    self.best_trial = trial
                else:
                    is_maximize = self.direction == 'maximize'
                    if is_maximize and trial.value > self.best_trial.value:
                        self.best_trial = trial
                    elif not is_maximize and trial.value < self.best_trial.value:
                        self.best_trial = trial

                print(f"Trial {i+1}/{n_trials} finished. Value: {value:.6f}. Best value: {self.best_trial.value:.6f}")

            except Exception as e:
                trial.state = "FAILED"
                print(f"Trial {i+1}/{n_trials} failed: {e}")

        print("\n🎉 Optimization finished!")
        if self.best_trial:
            print(f"Best trial #{self.best_trial.trial_id}:")
            print(f"  - Value: {self.best_trial.value:.6f}")
            print("  - Params:")
            for k, v in self.best_trial.params.items():
                print(f"    - {k}: {v}")

        return self.best_trial