"""
Example 3: Integrating with Advanced Third-Party Samplers
---------------------------------------------------------

This example shows how to extend the HPO engine by using a more
sophisticated sampler from a third-party library like Optuna.
This demonstrates the flexibility of the `Study` class.
"""

from hpo import Study, SearchSpace
from hpo.visualization import plot_optimization_history

try:
    from optuna.samplers import CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

def objective(trial):
    """
    A simple quadratic objective function to minimize.
    """
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -5, 5)
    return (x - 2)**2 + (y + 1)**2

def main():
    """
    Run an optimization study using Optuna's CMA-ES sampler.
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna is not installed. Please run 'pip install optuna' to run this example.")
        return

    print("\nRunning Example: Advanced Sampler Integration (Optuna's CMA-ES)")
    print("Goal: Minimize a simple quadratic function using a powerful external sampler.")
    print("-------------------------------------------------------------------------")

    # 1. Define the search space
    search_space = SearchSpace()
    search_space.add_uniform("x", -10, 10)
    search_space.add_uniform("y", -5, 5)

    # 2. Create a study with an external sampler
    # The HPO engine is designed to be flexible. You can pass any sampler
    # object that has a `suggest` method matching the expected interface.
    # Here, we use the Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
    # sampler from Optuna, a powerful algorithm for numerical optimization.
    study = Study(
        search_space=search_space,
        objective_function=objective,
        direction='minimize',
        n_trials=100,
        sampler=CmaEsSampler(),  # Pass the advanced sampler instance here
        study_name='advanced_sampler_example'
    )

    study.optimize()

    # 3. Visualize the results
    plot_optimization_history(study, save_path='advanced_optimization.png')
    print("\n✅ Visualization saved to 'advanced_optimization.png'")
    print("Notice how the search points cluster around the optimal solution (x=2, y=-1).")


if __name__ == "__main__":
    main()