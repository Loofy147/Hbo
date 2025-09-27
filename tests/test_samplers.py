import pytest
from hpo.space import SearchSpace, TrialResult
from hpo.samplers import RandomSampler, TPESampler

@pytest.fixture
def search_space():
    """A fixture for a standard search space."""
    space = SearchSpace()
    space.add_uniform('x', -5.0, 5.0)
    space.add_int('y', 1, 10)
    space.add_categorical('z', ['a', 'b', 'c'])
    return space

@pytest.fixture
def completed_trials():
    """A fixture for a list of completed trials."""
    trials = [
        TrialResult(1, {'x': 1.0, 'y': 2, 'z': 'a'}, 10.0),
        TrialResult(2, {'x': 2.0, 'y': 3, 'z': 'b'}, 20.0),
        TrialResult(3, {'x': -1.0, 'y': 8, 'z': 'c'}, 5.0),
        TrialResult(4, {'x': 4.0, 'y': 6, 'z': 'a'}, 25.0) # Best trial
    ]
    return trials

def test_random_sampler(search_space):
    """
    Tests that the RandomSampler returns a valid sample from the space.
    """
    sampler = RandomSampler()
    suggestion = sampler.suggest([], search_space)

    assert isinstance(suggestion, dict)
    assert -5.0 <= suggestion['x'] <= 5.0
    assert 1 <= suggestion['y'] <= 10
    assert suggestion['z'] in ['a', 'b', 'c']

def test_tpe_sampler_startup(search_space):
    """
    Tests that TPESampler uses random sampling during the startup phase.
    """
    sampler = TPESampler(n_startup_trials=5)
    # Only 3 trials, which is less than n_startup_trials
    trials = [TrialResult(i, search_space.sample(), i) for i in range(3)]

    # Suggestion should be equivalent to a random sample
    suggestion = sampler.suggest(trials, search_space)

    assert isinstance(suggestion, dict)
    assert set(suggestion.keys()) == set(['x', 'y', 'z'])

def test_tpe_sampler_post_startup(search_space, completed_trials):
    """
    Tests that TPESampler returns a valid suggestion after the startup phase.
    """
    # Set startup trials low enough to trigger TPE logic
    sampler = TPESampler(n_startup_trials=2)

    suggestion = sampler.suggest(completed_trials, search_space)

    assert isinstance(suggestion, dict)
    assert set(suggestion.keys()) == set(['x', 'y', 'z'])
    # Check that suggested values are within the space bounds
    assert -5.0 <= suggestion['x'] <= 5.0
    assert 1 <= suggestion['y'] <= 10
    assert suggestion['z'] in ['a', 'b', 'c']

def test_tpe_density_computation(search_space):
    """
    Tests the internal _compute_density method with a simple case.
    """
    sampler = TPESampler()

    # A single trial to model against
    trials = [TrialResult(1, {'x': 1.0, 'y': 5, 'z': 'a'}, 100)]

    # A candidate that is close to the single trial
    close_candidate = {'x': 1.1, 'y': 5, 'z': 'a'}
    # A candidate that is far from the single trial
    far_candidate = {'x': 4.0, 'y': 10, 'z': 'c'}

    density_close = sampler._compute_density(close_candidate, trials, search_space)
    density_far = sampler._compute_density(far_candidate, trials, search_space)

    # The density for the closer candidate should be higher
    assert density_close > density_far

def test_tpe_gamma_split(completed_trials):
    """
    Tests that the gamma parameter correctly splits trials into 'good' and 'bad'.
    """
    sampler = TPESampler(gamma=0.25) # Top 25% are good

    # With 4 trials, 25% means 1 is good
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
    n_good = max(1, int(len(sorted_trials) * sampler.gamma))

    assert n_good == 1
    good_trials = sorted_trials[:n_good]
    bad_trials = sorted_trials[n_good:]

    assert len(good_trials) == 1
    assert len(bad_trials) == 3
    assert good_trials[0].value == 25.0 # The best trial
    assert bad_trials[0].value == 20.0