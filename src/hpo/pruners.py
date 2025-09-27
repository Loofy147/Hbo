import numpy as np

class MedianPruner:
    """
    A pruner that stops trials performing worse than the median of previous trials.

    This pruner compares the intermediate values of a trial against the median
    of all previously reported values at the same step and prunes if the current
    trial's performance is below the median.

    Attributes:
        n_startup_trials (int): Number of trials to complete before pruning is active.
        n_warmup_steps (int): Minimum number of steps before a trial can be pruned.
    """
    def __init__(self, n_startup_trials=5, n_warmup_steps=10):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.step_values = {}  # Stores {step: [values]}

    def should_prune(self, trial_id, step, value, completed_trials):
        """
        Determines whether a trial should be pruned at a given step.

        Args:
            trial_id (int): The ID of the current trial.
            step (int): The current step (e.g., epoch or batch number).
            value (float): The objective value at the current step.
            completed_trials (list): A list of all completed trials.

        Returns:
            bool: True if the trial should be pruned, False otherwise.
        """
        # Do not prune during startup or warmup phases
        if len(completed_trials) < self.n_startup_trials or step < self.n_warmup_steps:
            return False

        # Record the value for the current step
        if step not in self.step_values:
            self.step_values[step] = []
        self.step_values[step].append(value)

        # Check if there are enough values to compute a median
        step_history = self.step_values[step]
        if len(step_history) < 2:
            return False

        # Prune if the current value is worse than the median
        median_value = np.median(step_history)
        return value < median_value

class ASHAPruner:
    """
    Asynchronous Successive Halving Algorithm (ASHA) pruner.

    ASHA is an aggressive early-stopping algorithm that promotes promising trials
    to higher "rungs" of evaluation (e.g., more training epochs) while stopping
    underperforming ones.

    Attributes:
        min_resource (int): The minimum resource (e.g., epochs) allocated to a trial.
        max_resource (int): The maximum resource a trial can be allocated.
        reduction_factor (int): The factor by which the number of trials is reduced at each rung.
    """
    def __init__(self, min_resource=1, max_resource=100, reduction_factor=3):
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.reduction_factor = reduction_factor

        # Calculate the rungs (evaluation levels)
        self.rungs = []
        resource = max_resource
        while resource >= min_resource:
            self.rungs.append(resource)
            resource //= reduction_factor
        self.rungs = sorted(self.rungs)

        # Store results for each rung
        self.rung_results = {rung: [] for rung in self.rungs}

    def should_prune(self, trial_id, step, value):
        """
        Determines whether a trial should be pruned at a given step (resource level).

        Args:
            trial_id (int): The ID of the current trial.
            step (int): The current resource level (e.g., number of epochs).
            value (float): The objective value at the current step.

        Returns:
            bool: True if the trial should be pruned, False otherwise.
        """
        # Find the appropriate rung for the current step
        current_rung = None
        for rung in self.rungs:
            if step >= rung:
                current_rung = rung
                break

        if current_rung is None:
            return False

        # Add the current trial's performance to the rung
        self.rung_results[current_rung].append(value)

        # Check if the rung is full enough to make a pruning decision
        rung_values = self.rung_results[current_rung]
        if len(rung_values) >= self.reduction_factor:
            # Determine the performance threshold for this rung
            threshold_index = len(rung_values) // self.reduction_factor
            # Sort descending to find the cut-off point
            sorted_values = sorted(rung_values, reverse=True)
            threshold = sorted_values[threshold_index - 1] if threshold_index > 0 else sorted_values[0]

            # Prune if the current trial's value is below the threshold
            return value < threshold

        return False