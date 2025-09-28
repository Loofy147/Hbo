import unittest
import sys
import os
import optuna

# Add project root to path to allow direct import of modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from advanced_optimizers import ParetoFrontOptimizer, SuccessiveHalvingOptimizer

# --- Mock Objects for Testing ---

class MockTrial:
    """A mock of Optuna's Trial object supporting pruning."""
    def __init__(self, params=None, values=None):
        self.params = params or {}
        self.user_attrs = {}
        self._suggestions = {
            'model_complexity': 5,
            'batch_size': 32,
            'learning_rate': 0.01,
            'regularization': 0.05,
            'pruning_propensity': 0.75 # Default to a "good" trial
        }
        self.state = 'RUNNING'
        self.intermediate_values = {}

        if values is not None:
            self.values = values if isinstance(values, (list, tuple)) else [values]
            self.value = self.values[0]
        else:
            self.value = None
            self.values = None

    def suggest_int(self, name, low, high):
        if name in self.params: return self.params[name]
        return self._suggestions.get(name, low)

    def suggest_categorical(self, name, choices):
        if name in self.params: return self.params[name]
        return self._suggestions.get(name, choices[0])

    def suggest_float(self, name, low, high, log=False):
        if name in self.params: return self.params[name]
        return self._suggestions.get(name, low)

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value

    def report(self, value, step):
        self.intermediate_values[step] = value

    def should_prune(self):
        # A simple logic: "bad" trials (low propensity) are pruned after the first step.
        is_bad_trial = self.suggest_float('pruning_propensity', 0, 1) < 0.5
        is_past_first_step = len(self.intermediate_values) > 1
        if is_bad_trial and is_past_first_step:
            self.state = 'PRUNED'
            return True
        return False

class MockHPOSystem:
    """A mock of the HPOSystem supporting pruning."""
    def __init__(self, search_space, objective_function, n_trials=10, **kwargs):
        self.search_space = search_space
        self.objective_function = objective_function
        self.n_trials = n_trials
        self.directions = kwargs.get('directions', ['maximize'])
        self.pruner = kwargs.get('pruner')
        self.trials = []

    def optimize(self):
        for i in range(self.n_trials):
            # To test pruning, create a mix of "good" and "bad" trials.
            propensity = 0.25 if i % 2 == 0 else 0.75  # Alternate bad/good
            trial = MockTrial(params={'pruning_propensity': propensity})

            try:
                final_value = self.objective_function(trial)
                trial.value = final_value
                trial.values = [final_value] if not isinstance(final_value, (list, tuple)) else final_value
                trial.state = 'COMPLETE'
            except optuna.exceptions.TrialPruned:
                trial.state = 'PRUNED'

            self.trials.append(trial)

        completed_trials = [t for t in self.trials if t.state == 'COMPLETE']
        if not completed_trials: return None

        return max(completed_trials, key=lambda t: t.value)

    def get_pareto_front(self):
        completed = [t for t in self.trials if t.state == 'COMPLETE']
        if not completed: return []
        # Simplified mock logic
        return completed[:2]

class MockSearchSpace:
    """A mock of the SearchSpace class."""
    def __init__(self): self.params = {}
    def add_int(self, name, low, high): return self
    def add_categorical(self, name, choices): return self
    def add_uniform(self, name, low, high, log=False): return self
    def add_float(self, name, low, high, log=False):
        self.params[name] = ('float', low, high, log)
        return self

# --- Unit Tests ---

class TestParetoFrontOptimizer(unittest.TestCase):
    """Tests for the ParetoFrontOptimizer."""
    def test_optimizer_finds_pareto_front(self):
        optimizer = ParetoFrontOptimizer()
        pareto_front, _ = optimizer.find_pareto_front(
            hpo_system=MockHPOSystem,
            search_space_generator=MockSearchSpace,
            n_trials=10
        )
        self.assertIsInstance(pareto_front, list)

class TestSuccessiveHalvingOptimizer(unittest.TestCase):
    """Tests for the SuccessiveHalvingOptimizer."""

    def test_optimizer_with_pruning(self):
        """Verify that the optimizer runs and some trials are pruned."""
        optimizer = SuccessiveHalvingOptimizer(n_steps=10)

        def create_objective_for_trial(trial):
            pruning_propensity = trial.suggest_float('pruning_propensity', 0, 1)
            def objective(step):
                return 0.1 if pruning_propensity < 0.5 else 0.5 + (step / 20.0)
            return objective

        best_trial, hpo_instance = optimizer.optimize_with_pruning(
            hpo_system=MockHPOSystem,
            search_space_generator=MockSearchSpace,
            create_objective_fn=create_objective_for_trial,
            n_trials=20
        )

        self.assertIsNotNone(best_trial, "Should return a best trial")
        self.assertEqual(best_trial.state, 'COMPLETE')

        pruned_trials = [t for t in hpo_instance.trials if t.state == 'PRUNED']
        completed_trials = [t for t in hpo_instance.trials if t.state == 'COMPLETE']

        self.assertGreater(len(pruned_trials), 0, "Some trials should have been pruned")
        self.assertGreater(len(completed_trials), 0, "Some trials should have completed")
        self.assertEqual(len(pruned_trials) + len(completed_trials), 20)

        a_pruned_trial = pruned_trials[0]
        self.assertGreater(len(a_pruned_trial.intermediate_values), 0)
        self.assertLess(len(a_pruned_trial.intermediate_values), optimizer.n_steps)

if __name__ == '__main__':
    unittest.main()