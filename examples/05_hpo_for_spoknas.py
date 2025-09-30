"""
Example 5: Hyperparameter Optimization for SpokNAS
----------------------------------------------------

This example demonstrates how to use the HPO engine to optimize the
hyperparameters of the SpokNAS neural architecture search process itself.
"""
import sys
import os
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Optional

# Adjust the path to import from the root 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from spoknas.optimizer import SpokNASOptimizer
from spoknas.fitness import evaluate_multifidelity
from spoknas.controller import ExperimentController
from sklearn.ensemble import RandomForestRegressor

# --- Minimal HPO Framework (similar to the previous task) ---

class SearchSpace:
    """A minimal SearchSpace for defining HPO search ranges."""
    def __init__(self):
        self.params: Dict[str, Any] = {}

    def add_float(self, name: str, low: float, high: float):
        self.params[name] = {'type': 'float', 'bounds': (low, high)}

    def add_int(self, name: str, low: int, high: int):
        self.params[name] = {'type': 'int', 'bounds': (low, high)}

    def sample(self) -> Dict[str, float]:
        config = {}
        for name, p_info in self.params.items():
            low, high = p_info['bounds']
            if p_info['type'] == 'float':
                config[name] = np.random.uniform(low, high)
            else: # int
                config[name] = np.random.randint(low, high + 1)
        return config

@dataclass
class Trial:
    params: Dict[str, Any]
    value: float

class TrialObject:
    """An object passed to the objective function to suggest parameters."""
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def suggest_float(self, name: str, low: float, high: float) -> float:
        return self.params[name]

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return self.params[name]

class Study:
    """A minimal Study class to orchestrate the meta-optimization."""
    def __init__(self, search_space: SearchSpace, objective_function: Callable, n_trials: int):
        self.search_space = search_space
        self.objective = objective_function
        self.n_trials = n_trials
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None

    def optimize(self):
        print("🚀 Starting HPO for SpokNAS...")
        for i in range(self.n_trials):
            print(f"\n--- HPO Trial {i+1}/{self.n_trials} ---")
            params = self.search_space.sample()
            trial_obj = TrialObject(params)

            # Run the objective function (which runs SpokNAS)
            value = self.objective(trial_obj)

            trial_result = Trial(params=params, value=value)
            self.trials.append(trial_result)

            if self.best_trial is None or trial_result.value > self.best_trial.value:
                self.best_trial = trial_result

            print(f"HPO Trial {i+1} finished. Fitness: {value:.4f}. Best fitness so far: {self.best_trial.value:.4f}")

        print("\n🎉 HPO for SpokNAS finished!")
        print(f"Best fitness found: {self.best_trial.value:.4f}")
        print("Best hyperparameters:")
        for k, v in self.best_trial.params.items():
            print(f"  - {k}: {v:.4f}")

# --- Objective Function for HPO ---

def spoknas_objective(trial: TrialObject) -> float:
    """
    This function wraps the SpokNAS process and serves as the objective
    for the outer HPO loop.
    """
    # 1. Get hyperparameters for SpokNAS from the HPO trial
    mutation_rate = trial.suggest_float('mutation_rate', 0.01, 0.2)
    crossover_rate = trial.suggest_float('crossover_rate', 0.4, 0.9)
    elitism = trial.suggest_int('elitism', 1, 4)

    print(f"Running SpokNAS with: mutation={mutation_rate:.3f}, crossover={crossover_rate:.3f}, elitism={elitism}")

    # 2. Set up SpokNAS components (similar to trainer_main.py)
    cfg = yaml.safe_load(open('config.yaml'))
    profile = cfg['profiles']['cpu_small'] # Use a small profile for quick HPO trials
    layer_lib = cfg['optimizer']['layer_library']

    surrogate = RandomForestRegressor(n_estimators=cfg['surrogate']['n_estimators'])
    optimizer = SpokNASOptimizer(
        layer_lib,
        fitness_fn=evaluate_multifidelity,
        population_size=profile.get('pop_size', 8),
        elitism=elitism,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        surrogate_enabled=False, # Disable surrogate for this example to focus on core GA params
        surrogate_model=surrogate
    )
    controller = ExperimentController()

    class DataManager:
        def __init__(self, num_samples=100, num_classes=5, img_size=16, num_channels=3):
            self.x_full = np.random.randn(num_samples, num_channels, img_size, img_size).astype(np.float32)
            self.y_full = np.random.randint(0, num_classes, (num_samples,)).astype(int)

    data_manager = DataManager()

    # 3. Run the SpokNAS optimization
    best_genome, history = optimizer.run_with_controller(
        generations=profile.get('generations', 5), # Fewer generations for a quick HPO trial
        num_islands=profile.get('num_islands', 2),
        migrate_every=profile.get('migrate_every', 5),
        migration_k=profile.get('migration_k', 1),
        controller=controller,
        data_manager=data_manager,
        train_idx=list(range(80)),
        val_idx=list(range(80, 100)),
        fitness_cfg=cfg['fitness']
    )

    # 4. Return the best fitness found by this SpokNAS run
    best_fitness = best_genome.get('fitness', 0.0)
    return best_fitness

def main():
    """
    Main function to set up and run the HPO study on SpokNAS.
    """
    # 1. Define the search space for SpokNAS hyperparameters
    search_space = SearchSpace()
    search_space.add_float('mutation_rate', 0.01, 0.2)
    search_space.add_float('crossover_rate', 0.4, 0.9)
    search_space.add_int('elitism', 1, 4)

    # 2. Create and run the HPO study
    study = Study(
        search_space=search_space,
        objective_function=spoknas_objective,
        n_trials=5 # Run 5 HPO trials to find the best SpokNAS config
    )
    study.optimize()

if __name__ == "__main__":
    main()