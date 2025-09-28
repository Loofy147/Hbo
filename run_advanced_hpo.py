import sys
import os
import logging
import optuna
import numpy as np

# --- Setup Paths and Logging ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'hpo_production_system')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Import HPO System and Advanced Optimizers ---
try:
    from hpo_production_system import HPOSystem
    from hpo.space import SearchSpace
    from advanced_optimizers import ParetoFrontOptimizer, SuccessiveHalvingOptimizer
    logging.info("Successfully imported all required modules.")
except ImportError as e:
    logging.error(f"Failed to import modules. Please check paths and dependencies: {e}")
    sys.exit(1)

# --- Adapter Class to Bridge New Optimizers with Original HPOSystem ---

class HPOSystemAdapter(HPOSystem):
    """
    An adapter to make the original HPOSystem compatible with modern Optuna versions.
    It intercepts calls, translates arguments, and bypasses broken legacy methods.
    """
    def __init__(self, search_space, objective_function, n_trials, directions=None, **kwargs):

        # 1. Handle direction vs directions for multi-objective
        if directions:
            kwargs['direction'] = directions[0]
            self.study_directions = directions
        else:
            self.study_directions = [kwargs.get('direction', 'maximize')]

        # 2. Handle Pruner Instantiation - The Definitive Fix
        # The legacy _get_pruner method is broken. We must bypass it entirely.
        pruner_instance = None
        pruner_arg = kwargs.pop('pruner', None)  # Remove pruner from kwargs
        if isinstance(pruner_arg, str) and pruner_arg.lower() == 'asha':
            # The legacy system requests 'ASHA', but modern Optuna uses this name.
            pruner_instance = optuna.pruners.SuccessiveHalvingPruner()
            logging.info("Adapter is creating and injecting the correct Pruner instance.")

        # 3. Convert SearchSpace object to the dict format the parent expects
        if isinstance(search_space, SearchSpace):
            hpo_system_space = {name: (config['type'], config['low'], config['high'])
                                for name, config in search_space.params.items() if config['type'] in ['int', 'float']}
            hpo_system_space.update({name: ('categorical', config['choices'])
                                     for name, config in search_space.params.items() if config['type'] == 'categorical'})
        else:
            hpo_system_space = search_space

        # Initialize the parent, but explicitly pass `pruner=None` to avoid its broken logic.
        super().__init__(search_space=hpo_system_space,
                         objective_function=objective_function,
                         n_trials=n_trials,
                         pruner=None, # This is the key to bypassing the broken method
                         **kwargs)

        # Now, after the parent is initialized, we manually inject the correct pruner.
        if pruner_instance:
            self.pruner = pruner_instance

    def _objective_wrapper(self, trial):
        return self.objective_function(trial)

    def optimize(self):
        print(f"🚀 Starting Study '{self.study_name}' with {self.n_trials} trials.")
        print(f"   - Directions: {self.study_directions}")
        if self.pruner:
            print(f"   - Pruner: {self.pruner.__class__.__name__}")

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            directions=self.study_directions,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True
        )
        self.study.optimize(self.objective_function, n_trials=self.n_trials, n_jobs=self.n_jobs, show_progress_bar=True)

        print("\n🎉 Optimization complete!")
        return self.study.best_trial if len(self.study_directions) == 1 else self.study.best_trials

    def get_pareto_front(self):
        return self.study.best_trials

# --- Examples ---

def run_pareto_front_optimization():
    logging.info("\n" + "="*60 + "\n🚀 Starting Example 1: Pareto Front Optimization\n" + "="*60)
    optimizer = ParetoFrontOptimizer()
    try:
        pareto_front, _ = optimizer.find_pareto_front(
            hpo_system=HPOSystemAdapter,
            search_space_generator=SearchSpace,
            n_trials=30
        )
        logging.info(f"🏆 Found {len(pareto_front)} non-dominated solutions:")
        for i, trial in enumerate(pareto_front):
            logging.info(f"  - Sol {i+1}: Vals={trial.values}, Params={trial.params}")
    except Exception as e:
        logging.error(f"An error occurred during Pareto Front optimization: {e}", exc_info=True)

def run_successive_halving_optimization():
    logging.info("\n" + "="*60 + "\n🚀 Starting Example 2: Successive Halving with Pruning\n" + "="*60)
    optimizer = SuccessiveHalvingOptimizer(n_steps=10)

    def create_objective_for_trial(trial):
        complexity = trial.suggest_int('model_complexity', 1, 10)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        def objective(step):
            base = 1.0 / (1 + complexity * 0.3)
            progress = np.log1p(step * lr * 10)
            return base * progress + np.random.normal(0, 0.02)
        return objective

    def search_space_gen():
        ss = SearchSpace()
        ss.add_int('model_complexity', 1, 10)
        ss.add_uniform('learning_rate', 1e-4, 1e-1, log=True)
        return ss

    try:
        best_trial, hpo = optimizer.optimize_with_pruning(
            hpo_system=HPOSystemAdapter,
            search_space_generator=search_space_gen,
            create_objective_fn=create_objective_for_trial,
            n_trials=50
        )

        pruned = [t for t in hpo.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        completed = [t for t in hpo.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logging.info("🏆 Successive Halving Complete!")
        logging.info(f"   - Completed: {len(completed)}, Pruned: {len(pruned)}")
        if best_trial:
            logging.info(f"   - Best Trial: Val={best_trial.value:.4f}, Params={best_trial.params}")

    except Exception as e:
        logging.error(f"An error occurred during Successive Halving optimization: {e}", exc_info=True)

if __name__ == "__main__":
    # Clean up previous runs for a fresh start
    if os.path.exists("hpo_studies.db"):
        os.remove("hpo_studies.db")
        logging.info("Removed previous hpo_studies.db for a clean run.")

    run_pareto_front_optimization()
    run_successive_halving_optimization()