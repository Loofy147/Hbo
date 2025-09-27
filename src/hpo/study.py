import time
import pandas as pd
from .samplers import RandomSampler, TPESampler
from .pruners import MedianPruner, ASHAPruner
from .space import TrialResult

class TrialObject:
    """
    An object passed to the objective function, providing an interface for the trial.

    This object allows the objective function to suggest hyperparameter values dynamically
    and access information about the trial.

    Attributes:
        trial_id (int): The unique identifier for the trial.
        params (dict): A dictionary of hyperparameters for the current trial.
    """
    def __init__(self, trial_id, params):
        self.trial_id = trial_id
        self.params = params

    def suggest_float(self, name, low, high, log=False):
        """Suggests a float value for a hyperparameter."""
        if name in self.params:
            return self.params[name]
        # This is a fallback, real suggestion should happen in the Study
        return np.random.uniform(low, high)

    def suggest_int(self, name, low, high):
        """Suggests an integer value for a hyperparameter."""
        if name in self.params:
            return self.params[name]
        return np.random.randint(low, high + 1)

    def suggest_categorical(self, name, choices):
        """Suggests a categorical value for a hyperparameter."""
        if name in self.params:
            return self.params[name]
        return np.random.choice(choices)

class Study:
    """
    Manages the hyperparameter optimization process.

    A study orchestrates the optimization, running multiple trials to find the
    best hyperparameters for a given objective function.

    Attributes:
        study_name (str): The name of the study.
        direction (str): The optimization direction ('maximize' or 'minimize').
        search_space (SearchSpace): The hyperparameter search space.
        objective_function (callable): The function to optimize.
        n_trials (int): The total number of trials to run.
        sampler (object): The sampler algorithm to use.
        pruner (object): The pruner algorithm to use.
        trials (list): A list of all `TrialResult` objects.
        best_trial (TrialResult): The best trial found so far.
    """
    def __init__(self, search_space, objective_function, direction='maximize',
                 n_trials=100, sampler='TPE', pruner=None,
                 study_name='hpo-study', verbose=True):

        self.search_space = search_space
        self.objective_function = objective_function
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name
        self.verbose = verbose

        # Initialize components
        if sampler == 'TPE':
            self.sampler = TPESampler()
        else:
            self.sampler = RandomSampler()

        if pruner == 'ASHA':
            self.pruner = ASHAPruner()
        elif pruner == 'Median':
            self.pruner = MedianPruner()
        else:
            self.pruner = None

        self.trials = []
        self.best_trial = None
        self.start_time = None

        self.stats = {
            'n_complete': 0,
            'n_pruned': 0,
            'n_failed': 0,
            'total_time': 0
        }

        if self.verbose:
            print(f"✅ HPO study created: {study_name}")
            print(f"📊 Search Space: {len(search_space.params)} parameters")
            print(f"🎯 Direction: {direction}")
            print(f"🔬 Number of Trials: {n_trials}")

    def optimize(self):
        """
        Starts the optimization process.
        """
        self.start_time = time.time()
        if self.verbose:
            print(f"\n🚀 Starting optimization...")

        for trial_num in range(self.n_trials):
            try:
                # Suggest hyperparameters
                params = self.sampler.suggest(self.trials, self.search_space)
                method = 'TPE' if isinstance(self.sampler, TPESampler) and trial_num >= self.sampler.n_startup_trials else 'Random'

                trial_id = f"trial_{trial_num:03d}"
                trial_obj = TrialObject(trial_id, params)

                start_time = time.time()
                try:
                    value = self.objective_function(trial_obj)
                    state = 'COMPLETE'
                except Exception as e:
                    if self.verbose:
                        print(f"❌ Trial {trial_num} failed: {e}")
                    value = float('-inf') if self.direction == 'maximize' else float('inf')
                    state = 'FAILED'

                duration = time.time() - start_time

                trial_result = TrialResult(trial_id, params, value, state, duration)
                self.trials.append(trial_result)

                self._update_best_trial(trial_result)
                self._update_stats()

                if self.verbose:
                    self._print_trial_result(trial_result, trial_num, method)

            except KeyboardInterrupt:
                if self.verbose:
                    print("\n⏹️ Optimization stopped by user.")
                break
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error in trial {trial_num}: {e}")
                continue

        self.stats['total_time'] = time.time() - self.start_time
        if self.verbose:
            self.print_summary()

        return self.best_trial

    def _update_best_trial(self, trial):
        if trial.state != 'COMPLETE':
            return

        is_better = False
        if self.best_trial is None:
            is_better = True
        elif self.direction == 'maximize':
            is_better = trial.value > self.best_trial.value
        else: # minimize
            is_better = trial.value < self.best_trial.value

        if is_better:
            self.best_trial = trial

    def _update_stats(self):
        self.stats['n_complete'] = sum(1 for t in self.trials if t.state == 'COMPLETE')
        self.stats['n_pruned'] = sum(1 for t in self.trials if t.state == 'PRUNED')
        self.stats['n_failed'] = sum(1 for t in self.trials if t.state == 'FAILED')

    def _print_trial_result(self, trial, trial_num, method):
        if trial.state == 'COMPLETE':
            is_best = trial == self.best_trial
            icon = "🎯" if is_best else "  "

            params_str = []
            for name, value in list(trial.params.items())[:3]:
                if isinstance(value, float):
                    params_str.append(f"{name}={value:.4f}")
                else:
                    params_str.append(f"{name}={value}")
            params_display = " | ".join(params_str)
            if len(trial.params) > 3:
                params_display += "..."

            print(f"{icon} #{trial_num:3d} ({method:>6s}) | "
                  f"Value: {trial.value:.4f} | {params_display} | "
                  f"Time: {trial.duration:.1f}s")

    def print_summary(self):
        print("\n" + "="*70)
        print("🏆 Optimization Summary")
        print("="*70)

        if self.best_trial:
            print(f"🎯 Best Value: {self.best_trial.value:.6f}")
            print(f"🏅 Best Trial: {self.best_trial.trial_id}")
            print(f"\n⚙️ Best Parameters:")
            for name, value in self.best_trial.params.items():
                if isinstance(value, float):
                    print(f"   {name}: {value:.6f}")
                else:
                    print(f"   {name}: {value}")
        else:
            print("❌ No successful trials found.")

        print(f"\n📊 Statistics:")
        print(f"   Total Trials: {len(self.trials)}")
        print(f"   Completed: {self.stats['n_complete']}")
        print(f"   Failed: {self.stats['n_failed']}")
        print(f"   Total Time: {self.stats['total_time']:.1f}s")

    def get_trials_dataframe(self):
        """Returns the trial results as a pandas DataFrame."""
        if not self.trials:
            return pd.DataFrame()

        data = []
        for trial in self.trials:
            row = {
                'trial_id': trial.trial_id,
                'value': trial.value,
                'state': trial.state,
                'duration': trial.duration,
                **trial.params
            }
            data.append(row)

        return pd.DataFrame(data)