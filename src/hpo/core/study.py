import time
import pandas as pd
from typing import Callable, Optional, Union

# HPO library components
from ..storage.base import BaseStorage
from ..storage.sqlite import SQLiteStorage
from ..optimizers.tpe import TPESampler, RandomSampler # Will be refactored to a base class later
from ..pruners.median import MedianPruner # Will be refactored to a base class later
from .trial import Trial, TrialState, TrialPruned
from .parameter import SearchSpace

class TrialObject:
    """
    An object passed to the objective function, providing an interface for the trial.

    This object allows the objective function to suggest hyperparameter values, report
    intermediate results for pruning, and access information about the trial.
    """
    def __init__(self, study: 'Study', trial: Trial):
        self._study = study
        self._trial = trial

    @property
    def params(self):
        """Returns the dictionary of hyperparameters for the current trial."""
        return self._trial.params

    def report(self, value: float, step: int):
        """
        Reports an intermediate objective value for the trial.

        This allows the pruner to evaluate the trial's performance at intermediate
        steps and decide whether to stop it early.
        """
        # Pruning logic will be enhanced later.
        # For now, this is a placeholder for the new architecture.
        # if self._study.pruner and self._study.pruner.should_prune(...):
        #     raise TrialPruned(...)
        pass

    def should_prune(self) -> bool:
        """Asks the study's pruner if the trial should be pruned."""
        return False

class Study:
    """
    Manages the hyperparameter optimization process using a persistent backend.

    A study orchestrates the optimization, running multiple trials to find the
    best hyperparameters for a given objective function. It loads and saves
    its state, allowing for interruption and resumption.

    Args:
        study_name: The name of the study.
        storage: A storage backend instance or a URL to a database file.
        direction: The optimization direction ('maximize' or 'minimize').
        sampler: The hyperparameter sampling algorithm to use.
        pruner: The trial pruning algorithm to use.
    """
    def __init__(self,
                 study_name: str,
                 storage: Union[str, BaseStorage],
                 direction: str = 'maximize',
                 sampler=None,
                 pruner=None,
                 verbose: bool = True):

        self.study_name = study_name
        self.verbose = verbose

        if isinstance(storage, str):
            self.storage = SQLiteStorage(storage)
        else:
            self.storage = storage

        # Sampler and Pruner setup (to be improved with factories/base classes)
        self.sampler = sampler if sampler is not None else TPESampler()
        self.pruner = pruner

        # Load or create the study in the backend
        study_id = self.storage.get_study_id(study_name)
        if study_id is None:
            self.study_id = self.storage.create_study(study_name, direction)
            if self.verbose:
                print(f"✅ Created new study '{study_name}' with ID {self.study_id}.")
        else:
            self.study_id = study_id
            if self.verbose:
                print(f"✅ Loaded existing study '{study_name}' with ID {self.study_id}.")

        self.direction = self.storage.get_study_direction(self.study_id)

    def optimize(self, objective: Callable[[TrialObject], float], n_trials: int, search_space: SearchSpace):
        """
        Starts or resumes the optimization process.
        """
        if self.verbose:
            print(f"\n🚀 Starting optimization for {n_trials} trials...")

        for trial_num in range(n_trials):
            # 1. Get past trials from storage to inform the sampler
            past_trials = self.storage.get_all_trials(self.study_id)

            # 2. Suggest new parameters
            params = self.sampler.suggest(past_trials, search_space)

            # 3. Create a new trial in the storage backend
            trial_id = self.storage.create_trial(self.study_id)
            self.storage.set_trial_params(trial_id, params)

            # Reconstruct the Trial object for the objective function
            current_trial = Trial(
                trial_id=trial_id,
                study_id=self.study_id,
                state=TrialState.RUNNING,
                params=params
            )
            trial_obj = TrialObject(self, current_trial)

            try:
                value = objective(trial_obj)
                state = TrialState.COMPLETE
                self.storage.update_trial(trial_id, state=state.value, value=value)
                if self.verbose:
                    print(f"✅ Trial #{trial_id} complete. Value: {value:.4f}")

            except TrialPruned:
                state = TrialState.PRUNED
                self.storage.update_trial(trial_id, state=state.value)
                if self.verbose:
                    print(f" pruned Trial #{trial_id} pruned.")
            except Exception as e:
                state = TrialState.FAILED
                self.storage.update_trial(trial_id, state=state.value)
                if self.verbose:
                    print(f"❌ Trial #{trial_id} failed: {e}")

    @property
    def best_trial(self) -> Optional[Trial]:
        """
        Retrieves the best trial from the study so far.
        """
        all_trials = self.storage.get_all_trials(self.study_id)
        completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]

        if not completed_trials:
            return None

        if self.direction == 'maximize':
            return max(completed_trials, key=lambda t: t.value)
        else:
            return min(completed_trials, key=lambda t: t.value)

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Returns the trial results as a pandas DataFrame."""
        all_trials = self.storage.get_all_trials(self.study_id)
        if not all_trials:
            return pd.DataFrame()

        data = []
        for trial in all_trials:
            row = {
                'trial_id': trial.trial_id,
                'value': trial.value,
                'state': trial.state.value,
                **trial.params
            }
            data.append(row)

        return pd.DataFrame(data)

    def print_summary(self):
        """Prints a summary of the optimization results."""
        df = self.get_trials_dataframe()
        best = self.best_trial

        print("\n" + "="*70)
        print("🏆 Optimization Summary")
        print("="*70)

        if best:
            print(f"🎯 Best Value: {best.value:.6f}")
            print(f"🏅 Best Trial: #{best.trial_id}")
            print(f"\n⚙️ Best Parameters:")
            for name, value in best.params.items():
                print(f"   {name}: {value}")
        else:
            print("❌ No successful trials found.")

        if not df.empty:
            print(f"\n📊 Statistics:")
            print(f"   Total Trials: {len(df)}")
            for state in TrialState:
                count = (df['state'] == state.value).sum()
                if count > 0:
                    print(f"   {state.name}: {count}")