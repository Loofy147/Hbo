"""
Example 2: Scikit-learn Model Optimization
-------------------------------------------

This example demonstrates how to use the HPO engine to optimize the
hyperparameters of a Scikit-learn RandomForestClassifier on a real-world
dataset (Breast Cancer).
"""

from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    The objective function to optimize a RandomForestClassifier.

    Args:
        trial (TrialObject): The trial object provided by the Study.

    Returns:
        float: The mean cross-validated accuracy score.
    """
    # Define hyperparameters to be suggested by the trial
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 3, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])

    # Create the Scikit-learn model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate the model using cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return scores.mean()

def main():
    """
    Run the Scikit-learn optimization study.
    """
    print("\nRunning Example: Scikit-learn RandomForest Optimization")
    print("Goal: Maximize the cross-validated accuracy of a RandomForestClassifier.")
    print("--------------------------------------------------------------------")

    # 1. Define the search space
    search_space = SearchSpace()
    search_space.add_int('n_estimators', 10, 200)
    search_space.add_int('max_depth', 3, 25)
    search_space.add_int('min_samples_split', 2, 20)
    search_space.add_categorical('max_features', ['sqrt', 'log2'])

    # 2. Create and run the study
    study = Study(
        search_space=search_space,
        objective_function=objective,
        direction='maximize',
        n_trials=25,  # A smaller number of trials for a quick example
        study_name='sklearn_rf_example'
    )

    study.optimize()

    # 3. Visualize the results
    plot_optimization_history(study, save_path='sklearn_optimization.png')

if __name__ == "__main__":
    main()