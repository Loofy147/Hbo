import pytest
import numpy as np
from hpo import Study, SearchSpace

# A simple objective function for testing
def objective_function(trial):
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_int('y', 1, 5)
    return -(x - 2)**2 - (y - 3)**2

@pytest.fixture
def search_space():
    """A fixture for a simple search space."""
    space = SearchSpace()
    space.add_uniform('x', -10, 10)
    space.add_int('y', 1, 5)
    return space

def test_study_creation(search_space):
    """
    Tests the successful creation of a Study object.
    """
    study = Study(
        search_space=search_space,
        objective_function=objective_function,
        n_trials=10,
        study_name="test_study"
    )
    assert study.n_trials == 10
    assert study.study_name == "test_study"
    assert study.direction == 'maximize'
    assert len(study.trials) == 0

def test_study_optimization_run(search_space):
    """
    Tests a full optimization run to ensure it completes without errors.
    """
    study = Study(
        search_space=search_space,
        objective_function=objective_function,
        n_trials=5, # Keep it short for testing
        verbose=False # Disable printing during tests
    )

    best_trial = study.optimize()

    assert len(study.trials) == 5
    assert study.stats['n_complete'] == 5
    assert best_trial is not None
    assert 'x' in best_trial.params
    assert 'y' in best_trial.params

def test_study_direction_maximize(search_space):
    """
    Tests that the study correctly finds the maximum value.
    """
    np.random.seed(42)
    study = Study(
        search_space=search_space,
        objective_function=objective_function,
        direction='maximize',
        n_trials=30,
        verbose=False
    )

    best_trial = study.optimize()

    # The true maximum is 0 at x=2, y=3.
    # The best value found should be close to 0 (i.e., negative and small).
    assert -1.0 < best_trial.value <= 0.0
    # Check if the best params are reasonably close to the optimum
    assert abs(best_trial.params['x'] - 2.0) < 5.0 # Loose bound due to randomness
    assert best_trial.params['y'] == 3

def test_study_direction_minimize(search_space):
    """
    Tests that the study correctly finds the minimum value when direction is 'minimize'.
    """
    np.random.seed(42)
    # For minimization, we want to find the point furthest from the maximum.
    def opposite_objective(trial):
        return -objective_function(trial) # Return positive values

    study = Study(
        search_space=search_space,
        objective_function=opposite_objective,
        direction='minimize',
        n_trials=30,
        verbose=False
    )

    best_trial = study.optimize()

    # The true minimum of the inverted function is 0.
    assert 0.0 <= best_trial.value < 1.0
    assert abs(best_trial.params['x'] - 2.0) < 5.0
    assert best_trial.params['y'] == 3

def test_study_dataframe_conversion(search_space):
    """
    Tests the conversion of trial results to a pandas DataFrame.
    """
    pytest.importorskip("pandas") # Skip test if pandas is not installed

    study = Study(
        search_space=search_space,
        objective_function=objective_function,
        n_trials=3,
        verbose=False
    )
    study.optimize()

    df = study.get_trials_dataframe()

    assert len(df) == 3
    assert 'value' in df.columns
    assert 'x' in df.columns
    assert 'y' in df.columns
    assert 'state' in df.columns
    assert df['state'].iloc[0] == 'COMPLETE'