"""
Example 4: Unified Quantum HPO
----------------------------------

This example demonstrates how to use the new Unified Quantum HPO system,
which leverages Kähler geometry and quantum-inspired techniques for optimization.
"""

import sys
import os

# Adjust the path to import from the root 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hpo.study import Study, SearchSpace
from hpo.optimizers.quantum_hpo_sampler import QuantumHPOSampler

def objective(trial):
    """
    A simple objective function to be minimized.
    The true minimum is at (x=2, y=-1).
    """
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return (x - 2)**2 + (y + 1)**2

def main():
    """
    Run the optimization study using the QuantumHPOSampler.
    """
    print("\nRunning Example: Unified Quantum HPO")
    print("Goal: Minimize a simple quadratic function using the KaehlerHPOOptimizer.")
    print("-------------------------------------------------------------------------")

    # 1. Define the search space
    search_space = SearchSpace()
    search_space.add_uniform("x", -5, 5)
    search_space.add_uniform("y", -5, 5)

    # 2. Create a study with the QuantumHPOSampler
    study = Study(
        search_space=search_space,
        objective_function=objective,
        direction='minimize',
        n_trials=30,
        sampler=QuantumHPOSampler(n_startup_trials=10, direction='minimize'),
        study_name='quantum_hpo_example'
    )

    best_trial = study.optimize()

    # 3. Print the results
    if best_trial:
        print("\n----- Analysis -----")
        print(f"Optimal value found: {best_trial.value:.6f}")
        print(f"Optimal params: x={best_trial.params['x']:.4f}, y={best_trial.params['y']:.4f}")

if __name__ == "__main__":
    main()