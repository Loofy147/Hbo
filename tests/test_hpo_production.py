import pytest
import numpy as np
import pandas as pd
import os
import json
from hpo import (
    Parameter,
    ParameterType,
    OptimizationConfig,
    BayesianOptimizer,
    TPEOptimizer,
    HyperbandOptimizer,
    Trial,
)

# A simple objective function for testing: f(x, y) = (x-3)^2 + (y-5)^2
# Minimum is at x=3, y=5
def objective_function(params):
    x = params.get('x', 0)
    y = params.get('y', 0)
    return {'loss': (x - 3)**2 + (y - 5)**2}

# Another objective for multi-fidelity: f(x, epochs) = (x-3)^2 + 1/epochs
def hyperband_objective_function(params):
    x = params.get('x', 0)
    epochs = params.get('epochs', 1)
    return {'loss': (x - 3)**2 + 1.0 / epochs}


@pytest.fixture
def base_config():
    """Fixture for a basic optimization configuration."""
    return OptimizationConfig(
        parameters=[
            Parameter(name='x', param_type=ParameterType.FLOAT, low=0.0, high=5.0),
            Parameter(name='y', param_type=ParameterType.INTEGER, low=0, high=10),
        ],
        objective_name='loss',
        direction='minimize',
        n_trials=15,
        random_seed=42,
    )

@pytest.fixture
def hyperband_config():
    """Fixture for a Hyperband configuration."""
    return OptimizationConfig(
        parameters=[
            Parameter(name='x', param_type=ParameterType.FLOAT, low=0.0, high=5.0),
        ],
        objective_name='loss',
        direction='minimize',
        n_trials=20, # n_trials is not directly used by hyperband's logic, but for base class
        random_seed=42,
        multi_fidelity_enabled=True,
        resource_attr='epochs',
        max_resource=27,
        min_resource=1,
        reduction_factor=3,
    )

def test_bayesian_optimizer(base_config):
    """Test the BayesianOptimizer."""
    pytest.importorskip("skopt")
    optimizer = BayesianOptimizer(config=base_config, study_name="test_bayesian")
    best_trial = optimizer.optimize(objective_function)

    assert best_trial is not None
    assert 'x' in best_trial.parameters
    assert 'y' in best_trial.parameters
    # After 15 trials, it should be reasonably close to the minimum (3, 5)
    assert abs(best_trial.parameters['x'] - 3.0) < 1.0
    assert abs(best_trial.parameters['y'] - 5.0) <= 2 # Integer param

def test_tpe_optimizer(base_config):
    """Test the TPEOptimizer."""
    pytest.importorskip("optuna")
    optimizer = TPEOptimizer(config=base_config, study_name="test_tpe")
    best_trial = optimizer.optimize(objective_function)

    assert best_trial is not None
    assert 'x' in best_trial.parameters
    assert 'y' in best_trial.parameters
    # After 15 trials, it should be reasonably close to the minimum (3, 5)
    assert abs(best_trial.parameters['x'] - 3.0) < 1.0
    assert abs(best_trial.parameters['y'] - 5.0) <= 2

def test_hyperband_optimizer(hyperband_config):
    """Test the HyperbandOptimizer."""
    optimizer = HyperbandOptimizer(config=hyperband_config, study_name="test_hyperband")
    best_trial = optimizer.optimize(hyperband_objective_function)

    assert best_trial is not None
    assert 'x' in best_trial.parameters
    # The best trial should have been evaluated at the highest resource level
    assert best_trial.metadata['resource'] == hyperband_config.max_resource
    # It should be close to the true minimum for x=3
    assert abs(best_trial.parameters['x'] - 3.0) < 1.0

def test_save_and_load_study(base_config, tmp_path):
    """Test saving and loading a study."""
    optimizer = TPEOptimizer(config=base_config, study_name="test_save_load")
    optimizer.optimize(objective_function)

    filepath = os.path.join(tmp_path, "study.json")
    optimizer.save_study(filepath)

    assert os.path.exists(filepath)

    with open(filepath, 'r') as f:
        data = json.load(f)

    assert data['study_name'] == "test_save_load"
    assert len(data['trials']) == base_config.n_trials
    assert 'parameters' in data['best_trial']
    assert 'metrics' in data['best_trial']
    assert data['config']['objective_name'] == 'loss'

def test_optimization_history(base_config):
    """Test getting optimization history as a DataFrame."""
    pytest.importorskip("pandas")
    optimizer = TPEOptimizer(config=base_config, study_name="test_history")
    optimizer.optimize(objective_function)

    history_df = optimizer.get_optimization_history()
    assert isinstance(history_df, pd.DataFrame)
    assert len(history_df) == base_config.n_trials
    assert 'trial_id' in history_df.columns
    assert 'x' in history_df.columns
    assert 'y' in history_df.columns
    assert 'loss' in history_df.columns
    assert 'status' in history_df.columns