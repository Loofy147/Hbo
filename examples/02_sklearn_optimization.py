"""
Example 2: Scikit-learn Model Optimization with Pruning
-------------------------------------------------------

This example demonstrates how to use the HPO engine to optimize the
hyperparameters of a Scikit-learn RandomForestClassifier, including how
to report intermediate values for early stopping (pruning).
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

from hpo import Study, SearchSpace
from hpo.visualization import plot_optimization_history

# Load a dataset
try:
    X, y = load_breast_cancer(return_X_y=True)
    print("Loaded Breast Cancer dataset.")
except ImportError:
    print("Scikit-learn is not installed. Please install it to run this example.")
    exit()

def objective(trial):
    """
    An objective function that trains a RandomForestClassifier and reports
    intermediate scores to enable pruning.
    """
    # Define hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # Use a single train/test split to make the pruning demonstration faster
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        warm_start=True,  # Important for iterative training
        random_state=42,
        n_jobs=-1
    )

    # Simulate iterative training by increasing n_estimators and reporting scores
    n_steps = 10
    final_accuracy = 0
    # Create a schedule of estimator counts to train on
    estimator_schedule = np.linspace(10, n_estimators, n_steps, dtype=int)

    for i, n_est in enumerate(estimator_schedule):
        model.n_estimators = n_est
        model.fit(X_train, y_train)
        intermediate_accuracy = model.score(X_test, y_test)
        final_accuracy = intermediate_accuracy

        # Report the intermediate score. The study's pruner will use this
        # to decide whether to stop the trial early.
        trial.report(intermediate_accuracy, step=i + 1)

    return final_accuracy

def main():
    """
    Run the Scikit-learn optimization study with a pruner enabled.
    """
    print("\nRunning Example: Scikit-learn Optimization with Pruning")
    print("Goal: Maximize accuracy, with early stopping of unpromising trials.")
    print("--------------------------------------------------------------------")

    # 1. Define the search space
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 200)
    search_space.add_int('max_depth', 3, 25)
    search_space.add_int('min_samples_split', 2, 20)
    search_space.add_categorical('max_features', ['sqrt', 'log2'])

    # 2. Create and run the study with a pruner
    print("\nNote: A 'MedianPruner' is enabled. Poorly performing trials may be stopped early.")
    study = Study(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=50,  # Increased trials to better see pruning effect
        pruner='Median',  # Enable the Median Pruner
        study_name='sklearn_rf_pruning_example'
    )

    study.optimize()

    # 3. Visualize the results
    plot_optimization_history(study, save_path='sklearn_optimization_pruning.png')
    print("\n✅ Visualization saved to 'sklearn_optimization_pruning.png'")

if __name__ == "__main__":
    main()