"""
Example 1: Basic Mathematical Optimization
-------------------------------------------

This example demonstrates how to use the HPO engine to find the maximum
value of a simple 2D mathematical function.

The objective is to maximize: f(x, y) = -(x - 3)^2 - (y + 2)^2 + 10
The known optimal solution is at (x=3, y=-2), with a value of 10.
"""

import time
from hpo import Study, SearchSpace
from hpo.visualization import plot_optimization_history

def objective(trial):
    """
    The objective function to be maximized.

    Args:
        trial (TrialObject): The trial object provided by the Study.

    Returns:
        float: The value of the function for the given hyperparameters.
    """
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)

    # Simulate some computational work
    time.sleep(0.01)

    # The function to maximize
    value = -(x - 3)**2 - (y + 2)**2 + 10
    return value

def main():
    """
    Run the mathematical optimization study.
    """
    print("Running Example: Basic Mathematical Optimization")
    print("Goal: Maximize f(x, y) = -(x - 3)^2 - (y + 2)^2 + 10")
    print("--------------------------------------------------")

    # 1. Define the search space
    search_space = SearchSpace()
    search_space.add_uniform('x', -10, 10)
    search_space.add_uniform('y', -10, 10)

    # 2. Create and run the study
    study = Study(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=50,
        study_name='mathematical_example'
    )

    best_trial = study.optimize()

    # 3. Print and analyze the results
    if best_trial:
        print("\n----- Analysis -----")
        print(f"Optimal value found: {best_trial.value:.6f} (Expected: 10.0)")
        print(f"Optimal params: x={best_trial.params['x']:.4f}, y={best_trial.params['y']:.4f} (Expected: x=3, y=-2)")

        error_x = abs(best_trial.params['x'] - 3.0)
        error_y = abs(best_trial.params['y'] + 2.0)

        if error_x < 0.1 and error_y < 0.1:
            print("✅ Solution found with high accuracy!")
        else:
            print("⚠️ Solution is an approximation. More trials might improve accuracy.")

    # 4. Visualize the optimization process
    plot_optimization_history(study, save_path='mathematical_optimization.png')


if __name__ == "__main__":
    main()