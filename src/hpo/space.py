import time
import numpy as np

class TrialResult:
    """
    Represents the result of a single trial execution.

    Attributes:
        trial_id (int): A unique identifier for the trial.
        params (dict): A dictionary of hyperparameters used in the trial.
        value (float): The objective value returned by the trial.
        state (str): The final state of the trial (e.g., 'COMPLETE', 'FAILED').
        duration (float): The execution time of the trial in seconds.
        timestamp (float): The completion time of the trial.
    """
    def __init__(self, trial_id, params, value, state='COMPLETE', duration=0.0):
        self.trial_id = trial_id
        self.params = params.copy()
        self.value = float(value)
        self.state = state
        self.duration = duration
        self.timestamp = time.time()

class SearchSpace:
    """
    Defines the hyperparameter search space for an optimization study.

    This class provides methods to add different types of hyperparameters,
    such as uniform, integer, and categorical, and to sample from the defined space.
    """
    def __init__(self):
        self.params = {}
        self.param_types = {}

    def add_uniform(self, name, low, high, log=False):
        """
        Adds a continuous hyperparameter with a uniform distribution.

        Args:
            name (str): The name of the hyperparameter.
            low (float): The lower bound of the distribution.
            high (float): The upper bound of the distribution.
            log (bool): If True, sample from a log-uniform distribution.
        """
        self.params[name] = {'type': 'uniform', 'low': low, 'high': high, 'log': log}
        self.param_types[name] = 'uniform'
        return self

    def add_int(self, name, low, high):
        """
        Adds a discrete hyperparameter that takes integer values.

        Args:
            name (str): The name of the hyperparameter.
            low (int): The lower bound of the range (inclusive).
            high (int): The upper bound of the range (inclusive).
        """
        self.params[name] = {'type': 'int', 'low': low, 'high': high}
        self.param_types[name] = 'int'
        return self

    def add_categorical(self, name, choices):
        """
        Adds a categorical hyperparameter from a list of choices.

        Args:
            name (str): The name of the hyperparameter.
            choices (list): A list of possible values for the hyperparameter.
        """
        self.params[name] = {'type': 'categorical', 'choices': choices}
        self.param_types[name] = 'categorical'
        return self

    def sample(self, n_samples=1):
        """
        Generates random samples from the defined search space.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            A single dictionary if n_samples is 1, otherwise a list of dictionaries.
        """
        samples = []
        for _ in range(n_samples):
            sample = {}
            for name, config in self.params.items():
                if config['type'] == 'uniform':
                    if config.get('log', False):
                        sample[name] = np.exp(np.random.uniform(
                            np.log(config['low']), np.log(config['high'])))
                    else:
                        sample[name] = np.random.uniform(config['low'], config['high'])
                elif config['type'] == 'int':
                    sample[name] = np.random.randint(config['low'], config['high'] + 1)
                elif config['type'] == 'categorical':
                    sample[name] = np.random.choice(config['choices'])
            samples.append(sample)
        return samples[0] if n_samples == 1 else samples