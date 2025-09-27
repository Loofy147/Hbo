import numpy as np

class RandomSampler:
    """
    A simple sampler that suggests hyperparameters completely at random.

    This sampler is useful for establishing a baseline or for the initial
    startup phase of an optimization process.
    """
    def suggest(self, trials, search_space):
        """
        Suggests a new set of hyperparameters by sampling randomly from the search space.

        Args:
            trials (list): A list of completed trials (not used by this sampler).
            search_space (SearchSpace): The search space to sample from.

        Returns:
            dict: A dictionary of suggested hyperparameter values.
        """
        return search_space.sample()

class TPESampler:
    """
    A Tree-structured Parzen Estimator (TPE) sampler.

    TPE is a sequential model-based optimization (SMBO) algorithm that models
    the probability of observing good and bad results and proposes new candidates
    based on the ratio of these probabilities (Expected Improvement).

    Attributes:
        n_startup_trials (int): The number of random trials to run before using TPE.
        n_ei_candidates (int): The number of candidates to sample for Expected Improvement.
        gamma (float): The fraction of top-performing trials to use as the 'good' set.
    """
    def __init__(self, n_startup_trials=10, n_ei_candidates=24, gamma=0.25):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma

    def suggest(self, trials, search_space):
        """
        Suggests a new set of hyperparameters based on the TPE algorithm.

        If the number of completed trials is less than `n_startup_trials`, it
        falls back to random sampling.

        Args:
            trials (list): A list of completed `TrialResult` objects.
            search_space (SearchSpace): The search space definition.

        Returns:
            dict: A dictionary of suggested hyperparameter values.
        """
        complete_trials = [t for t in trials if t.state == 'COMPLETE']

        if len(complete_trials) < self.n_startup_trials:
            return search_space.sample()

        # Sort trials by performance (assuming maximization)
        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)

        # Split into good and bad trials
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        good_trials = sorted_trials[:n_good]
        bad_trials = sorted_trials[n_good:]

        best_params = None
        best_ei = -np.inf

        # Generate and evaluate candidates
        for _ in range(self.n_ei_candidates):
            candidate = search_space.sample()

            # Calculate probability densities for good and bad groups
            good_density = self._compute_density(candidate, good_trials, search_space)
            bad_density = self._compute_density(candidate, bad_trials, search_space)

            if bad_density > 0:
                ei = good_density / bad_density
                if ei > best_ei:
                    best_ei = ei
                    best_params = candidate

        return best_params if best_params is not None else search_space.sample()

    def _compute_density(self, candidate, trials, search_space):
        """
        Computes the probability density for a given candidate based on a set of trials.

        This is a simplified implementation of Kernel Density Estimation (KDE).
        """
        if not trials:
            return 1.0

        log_density = 0.0

        for name, value in candidate.items():
            param_config = search_space.params[name]
            param_values = [t.params[name] for t in trials if name in t.params]

            if not param_values:
                continue

            if param_config['type'] in ['uniform', 'int']:
                # Simplified KDE: use inverse of minimum distance as a proxy for density
                if len(param_values) > 1:
                    distances = [abs(value - pv) for pv in param_values]
                    # Add a small epsilon to avoid division by zero
                    min_distance = min(distances) + 1e-10
                    density = 1.0 / min_distance
                    log_density += np.log(max(density, 1e-10))
                else:
                    # Not enough data to estimate density, neutral contribution
                    log_density += 0.0

            elif param_config['type'] == 'categorical':
                # Calculate relative frequency with Laplace smoothing
                count = param_values.count(value)
                num_choices = len(param_config['choices'])
                prob = (count + 1) / (len(param_values) + num_choices)
                log_density += np.log(prob)

        return np.exp(log_density)