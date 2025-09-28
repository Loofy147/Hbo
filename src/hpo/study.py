"""
Study Module
------------

This module provides the main user-facing class, `Study`, which serves as a
high-level wrapper around the powerful Optuna optimization framework.
"""

import optuna
from .space import SearchSpace

class Study:
    """
    Manages the hyperparameter optimization process.

    This class is a high-level wrapper around an `optuna.Study` object, providing
    a simplified interface for defining search spaces and running optimization, while
    exposing the full power of the Optuna backend to the user's objective function.
    """
    def __init__(self, search_space, objective_function, direction='maximize', directions=None,
                 n_trials=100, sampler=None, pruner=None,
                 study_name=None, verbose=True):

        if not isinstance(search_space, SearchSpace):
            raise TypeError("search_space must be an instance of hpo.SearchSpace")

        self.search_space = search_space
        self.user_objective = objective_function
        self.n_trials = n_trials
        self.verbose = verbose

        if directions is None:
            self.directions = [direction]
        else:
            self.directions = directions

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        else:
            optuna.logging.set_verbosity(optuna.logging.INFO)

        self.optuna_study = optuna.create_study(
            directions=self.directions,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner
        )

    def _objective_wrapper(self, trial: optuna.Trial):
        """
        A wrapper that translates our simple SearchSpace definitions into the
        correct calls on the underlying Optuna trial object.
        """
        original_suggest_float = trial.suggest_float
        original_suggest_int = trial.suggest_int
        original_suggest_categorical = trial.suggest_categorical

        # The new methods must accept the user-facing arguments (low, high, etc.)
        # even if they are unused, to match the calling signature.
        def new_suggest_float(name, low, high, *, log=False):
            if name not in self.search_space.params:
                raise ValueError(f"Hyperparameter '{name}' is not defined in the SearchSpace.")
            config = self.search_space.params[name]
            return original_suggest_float(name, low=config['low'], high=config['high'], log=config.get('log', False))

        def new_suggest_int(name, low, high, **kwargs):
            if name not in self.search_space.params:
                raise ValueError(f"Hyperparameter '{name}' is not defined in the SearchSpace.")
            config = self.search_space.params[name]
            return original_suggest_int(name, low=config['low'], high=config['high'])

        def new_suggest_categorical(name, choices):
            if name not in self.search_space.params:
                raise ValueError(f"Hyperparameter '{name}' is not defined in the SearchSpace.")
            config = self.search_space.params[name]
            return original_suggest_categorical(name, choices=config['choices'])

        trial.suggest_float = new_suggest_float
        trial.suggest_int = new_suggest_int
        trial.suggest_categorical = new_suggest_categorical

        return self.user_objective(trial)

    def optimize(self):
        """
        Starts the optimization process.
        """
        self.optuna_study.optimize(
            self._objective_wrapper,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose
        )

        if self.is_multi_objective:
            return self.get_pareto_front()
        else:
            return self.best_trial

    @property
    def best_trial(self):
        return self.optuna_study.best_trial

    @property
    def best_params(self):
        return self.optuna_study.best_params

    @property
    def best_value(self):
        return self.optuna_study.best_value

    @property
    def is_multi_objective(self):
        return len(self.optuna_study.directions) > 1

    def get_pareto_front(self):
        if not self.is_multi_objective:
            raise TypeError("Pareto front is only available for multi-objective studies.")
        return self.optuna_study.best_trials

    @property
    def trials(self):
        return self.optuna_study.trials

    def get_trials_dataframe(self):
        return self.optuna_study.trials_dataframe()