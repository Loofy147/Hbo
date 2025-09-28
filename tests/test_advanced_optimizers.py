import pytest
from hpo.study import Study
from hpo.space import SearchSpace

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="optuna is not installed")
def test_optuna_sampler_integration():
    """
    Tests that the study can be run with a sampler from the Optuna library.
    """
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return x**2

    search_space = SearchSpace()
    search_space.add_uniform("x", -10, 10)

    # The test is primarily to ensure this doesn't crash.
    # We are not checking the performance of the sampler itself.
    study = Study(
        search_space=search_space,
        objective_function=objective,
        n_trials=10,
        sampler=optuna.samplers.RandomSampler() # Using an optuna sampler
    )
    study.optimize()
    assert len(study.trials) == 10
    assert study.best_trial is not None