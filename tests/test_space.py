import pytest
import numpy as np
from hpo.space import SearchSpace, TrialResult

def test_trial_result_creation():
    """
    Tests the basic creation of a TrialResult object.
    """
    params = {'x': 1, 'y': 2.0}
    result = TrialResult(trial_id=1, params=params, value=100.0, duration=0.5)
    assert result.trial_id == 1
    assert result.params == params
    assert result.value == 100.0
    assert result.state == 'COMPLETE'
    assert result.duration == 0.5

def test_search_space_add_uniform():
    """
    Tests adding a uniform parameter to the search space.
    """
    space = SearchSpace()
    space.add_uniform('lr', 0.001, 0.1, log=True)
    assert 'lr' in space.params
    assert space.params['lr']['type'] == 'uniform'
    assert space.params['lr']['log'] is True

def test_search_space_add_int():
    """
    Tests adding an integer parameter to the search space.
    """
    space = SearchSpace()
    space.add_int('n_layers', 1, 5)
    assert 'n_layers' in space.params
    assert space.params['n_layers']['type'] == 'int'
    assert space.params['n_layers']['high'] == 5

def test_search_space_add_categorical():
    """
    Tests adding a categorical parameter to the search space.
    """
    space = SearchSpace()
    choices = ['adam', 'sgd']
    space.add_categorical('optimizer', choices)
    assert 'optimizer' in space.params
    assert space.params['optimizer']['type'] == 'categorical'
    assert space.params['optimizer']['choices'] == choices

def test_search_space_sampling():
    """
    Tests the sampling from a complete search space.
    """
    space = SearchSpace()
    space.add_uniform('x', -1.0, 1.0)
    space.add_int('y', 1, 10)
    space.add_categorical('z', ['a', 'b', 'c'])

    sample = space.sample()

    assert isinstance(sample, dict)
    assert -1.0 <= sample['x'] <= 1.0
    assert 1 <= sample['y'] <= 10
    assert sample['z'] in ['a', 'b', 'c']
    assert isinstance(sample['y'], int)

def test_log_uniform_sampling():
    """
    Tests that log-uniform sampling produces values in the correct range.
    """
    space = SearchSpace()
    space.add_uniform('log_param', 0.001, 1.0, log=True)

    samples = [space.sample()['log_param'] for _ in range(100)]

    for s in samples:
        assert 0.001 <= s <= 1.0

    # Check that the distribution is roughly log-uniform
    # (more values in the lower end of the range)
    low_end_count = sum(1 for s in samples if s < 0.1)
    high_end_count = sum(1 for s in samples if s > 0.9)
    assert low_end_count > high_end_count

def test_multiple_samples():
    """
    Tests generating multiple samples at once.
    """
    space = SearchSpace()
    space.add_int('a', 0, 100)

    samples = space.sample(n_samples=5)

    assert isinstance(samples, list)
    assert len(samples) == 5
    assert isinstance(samples[0], dict)
    assert 'a' in samples[0]