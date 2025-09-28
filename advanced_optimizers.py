import time
import logging
from typing import Any, Callable, List, Tuple
import numpy as np
import optuna

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


class ParetoFrontOptimizer:
    """
    Pareto Front Optimizer for True Multi-Objective Optimization.

    This optimizer finds the Pareto front, a set of non-dominated solutions,
    for multiple objectives like accuracy, speed, and memory. This is more advanced
    than combining them into a single weighted score.
    """

    def __init__(self):
        logger.info('Initialized ParetoFrontOptimizer.')

    def find_pareto_front(self, hpo_system: Any, search_space_generator: Any, n_trials: int = 50):
        """
        Runs a multi-objective HPO loop to find the Pareto front.

        HPOSystem and SearchSpace are passed as arguments for better testability.
        """

        def multi_objective_function(trial) -> Tuple[float, float, float]:
            # Suggest hyperparameters
            model_complexity = trial.suggest_int('model_complexity', 1, 10)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
            regularization = trial.suggest_float('regularization', 0.0, 0.1)

            # --- Simulate Metrics ---
            # 1. Accuracy (higher is better)
            base_accuracy = 0.7 + (model_complexity / 10.0) * 0.2
            if 0.001 <= learning_rate <= 0.01:
                base_accuracy += 0.08
            if 0.01 <= regularization <= 0.05:
                base_accuracy += 0.05
            accuracy = float(min(0.98, base_accuracy + np.random.normal(0, 0.03)))

            # 2. Speed Score (higher is better)
            base_time = 10.0 + model_complexity * 2.0 + (batch_size / 32.0) * 3.0
            speed_score = float(max(0.1, 1.0 / base_time))

            # 3. Memory Score (higher is better, derived from lower usage)
            memory_usage = model_complexity * 100 + batch_size * 10
            memory_score = float(max(0.1, 1.0 / (memory_usage / 1000.0)))

            # Store individual metrics for analysis, although they are also the return values
            trial.set_user_attr('accuracy', accuracy)
            trial.set_user_attr('speed_score', speed_score)
            trial.set_user_attr('memory_score', memory_score)

            # For true multi-objective, return a tuple of the objectives
            return accuracy, speed_score, memory_score

        # Build search space
        search_space = search_space_generator()
        search_space.add_int('model_complexity', 1, 10)
        search_space.add_categorical('batch_size', [16, 32, 64, 128, 256])
        search_space.add_uniform('learning_rate', 1e-5, 1e-1, log=True)
        search_space.add_uniform('regularization', 0.0, 0.1)

        # Run HPO for multi-objective optimization
        hpo = hpo_system(
            search_space=search_space,
            objective_function=multi_objective_function,
            # Provide a direction for each objective
            directions=['maximize', 'maximize', 'maximize'],
            n_trials=n_trials,
            sampler='TPE', # TPE supports multi-objective
            study_name='pareto_front_optimization'
        )

        # The optimization process is the same, but the result interpretation differs
        hpo.optimize()

        # The result is the set of best trials (the Pareto front)
        pareto_front_trials = hpo.get_pareto_front()
        return pareto_front_trials, hpo


class SuccessiveHalvingOptimizer:
    """
    An optimizer that uses the Successive Halving algorithm via a Pruner.

    This approach is highly efficient for budget allocation. It works by:
    1.  Allocating a small budget to many hyperparameter configurations.
    2.  Periodically reporting intermediate results (e.g., accuracy after each epoch).
    3.  "Pruning" (stopping) the worst-performing configurations early.
    4.  Continuing with only the promising configurations, allocating them more budget.

    This implementation wraps a user-defined objective function to simulate
    this behavior over a fixed number of "steps" (e.g., epochs).
    """

    def __init__(self, n_steps: int = 20):
        """
        Args:
            n_steps (int): The total number of steps (e.g., epochs) a full trial should run.
        """
        self.n_steps = n_steps
        logger.info(f'Initialized SuccessiveHalvingOptimizer with n_steps={n_steps}.')

    def optimize_with_pruning(self, hpo_system: Any, search_space_generator: Any,
                               create_objective_fn: Callable[[Any], Callable[[int], float]],
                               n_trials: int = 50):
        """
        Runs an HPO loop with a pruner-compatible objective function.

        Args:
            hpo_system: The HPO system to use (e.g., a wrapper around Optuna).
            search_space_generator: A function that returns a new SearchSpace object.
            create_objective_fn: A function that takes a trial and returns an objective
                                 function. This objective function should accept a step
                                 number and return the performance at that step.
            n_trials: The total number of trials to run.
        """

        def objective_with_pruning(trial):
            # Create a model-specific objective function for this trial
            # This function simulates performance improving over steps
            objective_fn = create_objective_fn(trial)

            last_value = 0.0
            for step in range(1, self.n_steps + 1):
                # Get the performance at the current step
                value = objective_fn(step)

                # Report the intermediate value to the pruner
                trial.report(value, step)
                last_value = value

                # Check if the trial should be pruned
                if trial.should_prune():
                    # Optuna will raise a Pruned trial exception
                    raise optuna.exceptions.TrialPruned()

            return last_value

        # Build search space
        search_space = search_space_generator()

        # Instantiate the HPO system, ensuring a pruner is active
        hpo = hpo_system(
            search_space=search_space,
            objective_function=objective_with_pruning,
            direction='maximize',
            n_trials=n_trials,
            sampler='TPE',
            pruner='ASHA',  # Using a pruner is essential for this optimizer
            study_name='successive_halving_optimization'
        )

        best_trial = hpo.optimize()
        return best_trial, hpo