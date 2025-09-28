"""
HPO Engine: Advanced Strategies Showcase
------------------------------------------

This script demonstrates how to use the high-level optimizer classes from
the `advanced_optimizers` module with the core `hpo.Study` system.

This example covers:
1.  **Successive Halving:** Efficiently finding the best model for a single
    objective by pruning unpromising trials early.
2.  **Pareto Front Optimization:** Finding the optimal trade-off between
    competing objectives (e.g., accuracy vs. latency).
"""

import logging
import time
import sys
import os
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

# --- Setup Project Path ---
# This allows the script to be run from the root directory
sys.path.insert(0, os.path.abspath('./src'))

# --- Core HPO Library Imports ---
from hpo import Study, SearchSpace
from hpo.advanced_optimizers import ParetoFrontOptimizer, SuccessiveHalvingOptimizer

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# === Part 1: Successive Halving for Efficient Single-Objective Tuning ===

def objective_for_pruning(trial, step):
    """
    This is the core objective function passed to the SuccessiveHalvingOptimizer.
    It simulates training a model for a single "step" (e.g., an epoch).
    """
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
    }

    model = RandomForestClassifier(random_state=42, **params, n_jobs=-1)

    # Simulate training for one "step" by adjusting n_estimators
    model.n_estimators = (step + 1) * 10
    model.fit(X_train, y_train)
    return accuracy_score(y_test, model.predict(X_test))

def run_successive_halving_example():
    """
    Demonstrates using the SuccessiveHalvingOptimizer to find the best
    RandomForestClassifier, pruning unpromising trials early.
    """
    logging.info("\n" + "="*70)
    logging.info("    🚀 Part 1: Successive Halving for Efficient Single-Objective Tuning")
    logging.info("="*70)

    # Define the search space using our simple builder API
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 50, 500)
    search_space.add_int('max_depth', 5, 50)

    # Instantiate and run the high-level optimizer
    optimizer = SuccessiveHalvingOptimizer(n_steps=15)
    best_trial, _ = optimizer.optimize_with_pruning(
        hpo_system=Study,
        search_space=search_space,
        objective_function=objective_for_pruning,
        n_trials=30
    )

    logging.info("🏆 Successive Halving Complete!")
    if best_trial:
        logging.info(f"   - Best Trial Value: {best_trial.value:.4f}")
        logging.info(f"   - Best Trial Params: {best_trial.params}")


# === Part 2: Pareto Front Optimization for Accuracy vs. Latency ===

def multi_objective_fn(trial: optuna.Trial):
    """
    A multi-objective function that balances model accuracy and inference latency.
    """
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
    }

    model = RandomForestClassifier(random_state=42, **params, n_jobs=-1)

    start_time = time.time()
    model.fit(X_train, y_train)
    latency = (time.time() - start_time) * 1000
    accuracy = accuracy_score(y_test, model.predict(X_test))

    trial.set_user_attr("latency_ms", latency)
    return accuracy, latency

def run_pareto_front_example():
    """
    Demonstrates finding the Pareto front for a multi-objective problem.
    """
    logging.info("\n" + "="*70)
    logging.info("    🚀 Part 2: Pareto Front Optimization (Accuracy vs. Latency)")
    logging.info("="*70)

    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 500)
    search_space.add_int('max_depth', 3, 30)

    optimizer = ParetoFrontOptimizer()
    pareto_trials, _ = optimizer.find_pareto_front(
        hpo_system=Study,
        search_space=search_space,
        objective_function=multi_objective_fn,
        directions=['maximize', 'minimize'],
        n_trials=40
    )

    logging.info(f"🏆 Pareto Front Optimization Complete! Found {len(pareto_trials)} optimal solutions.")
    for i, trial in enumerate(pareto_trials):
        logging.info(f"  - Solution {i+1}: Accuracy={trial.values[0]:.4f}, Latency={trial.values[1]:.2f}ms")


# === Main Execution Block ===

def main():
    """
    Main function to run all advanced examples.
    """
    run_successive_halving_example()
    run_pareto_front_example()
    logging.info("\n✅ All examples completed successfully.")


if __name__ == "__main__":
    main()