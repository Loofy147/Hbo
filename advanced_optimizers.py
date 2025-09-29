import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import concurrent.futures
from pathlib import Path
import sqlite3
import hashlib
import time
from datetime import datetime, timedelta
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


# ==============================================================================
# 1. Enhanced Configuration System
# ==============================================================================

@dataclass
class HyperParameter:
    """Enhanced hyperparameter definition"""
    name: str
    param_type: str  # 'int', 'float', 'categorical', 'bool', 'log_float'
    bounds: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    conditional_on: Optional[Dict[str, Any]] = None  # Conditional parameters

    def validate_value(self, value: Any) -> bool:
        """Validate if value is within parameter constraints"""
        try:
            if self.param_type == 'int':
                return isinstance(value, (int, np.integer)) and self.bounds[0] <= value <= self.bounds[1]
            elif self.param_type == 'float' or self.param_type == 'log_float':
                return isinstance(value, (int, float, np.number)) and self.bounds[0] <= value <= self.bounds[1]
            elif self.param_type == 'categorical':
                return value in self.choices
            elif self.param_type == 'bool':
                return isinstance(value, bool)
            return False
        except:
            return False

    def sample_value(self, rng: np.random.RandomState = None) -> Any:
        """Sample a random value from parameter space"""
        if rng is None:
            rng = np.random.RandomState()

        if self.param_type == 'int':
            return rng.randint(self.bounds[0], self.bounds[1] + 1)
        elif self.param_type == 'float':
            return rng.uniform(self.bounds[0], self.bounds[1])
        elif self.param_type == 'log_float':
            log_low, log_high = np.log(self.bounds[0]), np.log(self.bounds[1])
            return np.exp(rng.uniform(log_low, log_high))
        elif self.param_type == 'categorical':
            return rng.choice(self.choices)
        elif self.param_type == 'bool':
            return rng.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


class ConfigurationSpace:
    """Enhanced configuration space with conditional parameters"""

    def __init__(self, parameters: List[HyperParameter]):
        self.parameters = {p.name: p for p in parameters}
        self.conditional_graph = self._build_conditional_graph()

    def _build_conditional_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for conditional parameters"""
        graph = {}
        for param_name, param in self.parameters.items():
            if param.conditional_on:
                for parent_name in param.conditional_on.keys():
                    if parent_name not in graph:
                        graph[parent_name] = []
                    graph[parent_name].append(param_name)
        return graph

    def sample_configuration(self, rng: np.random.RandomState = None) -> Dict[str, Any]:
        """Sample a valid configuration respecting conditional dependencies"""
        if rng is None:
            rng = np.random.RandomState()

        config = {}

        # Topological sort to handle dependencies
        visited = set()

        def sample_param(param_name: str):
            if param_name in visited:
                return
            visited.add(param_name)

            param = self.parameters[param_name]

            # Check if parameter should be included based on conditionals
            if param.conditional_on:
                should_include = True
                for parent_name, required_value in param.conditional_on.items():
                    if parent_name not in config or config[parent_name] != required_value:
                        should_include = False
                        break

                if not should_include:
                    return

            # Sample value
            config[param_name] = param.sample_value(rng)

            # Recursively sample dependent parameters
            if param_name in self.conditional_graph:
                for child_name in self.conditional_graph[param_name]:
                    sample_param(child_name)

        # Sample all root parameters
        for param_name in self.parameters:
            sample_param(param_name)

        return config

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration and return errors if invalid"""
        errors = []

        for param_name, param in self.parameters.items():
            if param_name in config:
                # Validate value
                if not param.validate_value(config[param_name]):
                    errors.append(f"Invalid value for {param_name}: {config[param_name]}")

                # Check conditionals
                if param.conditional_on:
                    for parent_name, required_value in param.conditional_on.items():
                        if parent_name not in config:
                            errors.append(f"Parameter {param_name} requires {parent_name} to be set")
                        elif config[parent_name] != required_value:
                            errors.append(f"Parameter {param_name} requires {parent_name}={required_value}")

        return len(errors) == 0, errors


# ==============================================================================
# 2. Persistent Storage and Experiment Tracking
# ==============================================================================

class ExperimentDatabase:
    """SQLite-based experiment tracking with full persistence"""

    def __init__(self, db_path: str = "hpo_experiments.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Studies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    study_name TEXT UNIQUE NOT NULL,
                    direction TEXT NOT NULL,
                    objective_name TEXT NOT NULL,
                    config_space TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'running',
                    metadata TEXT
                )
            """)

            # Trials table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    parameters TEXT NOT NULL,
                    metrics TEXT,
                    status TEXT DEFAULT 'running',
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    duration_seconds REAL,
                    error_message TEXT,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id),
                    UNIQUE(study_id, trial_number)
                )
            """)

            # Intermediate values table (for pruning)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trial_intermediate_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
                    UNIQUE(trial_id, step)
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_study_id ON trials(study_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_intermediate_trial_id ON trial_intermediate_values(trial_id)")

            conn.commit()

    def create_study(self, study_name: str, direction: str, objective_name: str,
                    config_space: ConfigurationSpace, metadata: Dict = None) -> str:
        """Create a new study"""
        study_id = f"study_{hashlib.md5(study_name.encode()).hexdigest()[:8]}_{int(time.time())}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO studies
                (study_id, study_name, direction, objective_name, config_space, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                study_id, study_name, direction, objective_name,
                json.dumps([p.__dict__ for p in config_space.parameters.values()]),
                json.dumps(metadata or {})
            ))

        return study_id

    def save_trial(self, study_id: str, trial_number: int, parameters: Dict[str, Any],
                  metrics: Dict[str, float] = None, status: str = "running",
                  error_message: str = None, duration: float = None) -> str:
        """Save trial to database"""
        trial_id = f"trial_{study_id}_{trial_number:06d}"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if status == "running":
                cursor.execute("""
                    INSERT INTO trials
                    (trial_id, study_id, trial_number, parameters, status)
                    VALUES (?, ?, ?, ?, ?)
                """, (trial_id, study_id, trial_number, json.dumps(parameters), status))
            else:
                cursor.execute("""
                    UPDATE trials
                    SET metrics=?, status=?, completed_at=CURRENT_TIMESTAMP,
                        duration_seconds=?, error_message=?
                    WHERE trial_id=?
                """, (json.dumps(metrics or {}), status, duration, error_message, trial_id))

        return trial_id

    def get_study_trials(self, study_id: str) -> pd.DataFrame:
        """Get all trials for a study"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT trial_id, trial_number, parameters, metrics, status,
                       started_at, completed_at, duration_seconds, error_message
                FROM trials
                WHERE study_id = ?
                ORDER BY trial_number
            """
            df = pd.read_sql_query(query, conn, params=(study_id,))

            # Parse JSON columns
            df['parameters'] = df['parameters'].apply(json.loads)
            df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else {})

            return df

    def save_intermediate_value(self, trial_id: str, step: int, value: float):
        """Save intermediate value for pruning"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trial_intermediate_values
                (trial_id, step, value) VALUES (?, ?, ?)
            """, (trial_id, step, value))


# ==============================================================================
# 3. Enhanced Bayesian Optimization with Multiple Acquisitions
# ==============================================================================

class EnhancedBayesianOptimizer:
    """Advanced Bayesian optimization with multiple acquisition functions"""

    def __init__(self, config_space: ConfigurationSpace, acquisition_function: str = "ei",
                 xi: float = 0.01, kappa: float = 2.576, n_initial_points: int = 10):
        self.config_space = config_space
        self.acquisition_function = acquisition_function
        self.xi = xi  # for EI
        self.kappa = kappa  # for UCB
        self.n_initial_points = n_initial_points
        self.param_order = sorted(self.config_space.parameters.keys())

        self.X_observed = []
        self.y_observed = []
        self.gp_model = None

        # Try to import scikit-optimize
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern

            kernel = Matern(length_scale=1.0, nu=2.5)
            self.gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            self.has_gp = True
        except ImportError:
            self.has_gp = False

    def suggest(self) -> Dict[str, Any]:
        """Suggest next configuration to evaluate"""
        if len(self.X_observed) < self.n_initial_points or not self.has_gp:
            return self.config_space.sample_configuration()

        # Fit GP model
        X_array = self._configs_to_array(self.X_observed)
        self.gp_model.fit(X_array, self.y_observed)

        # Optimize acquisition function
        best_config = None
        best_acq_value = float('-inf') if self.acquisition_function != 'pi' else 0

        # Multiple random starts for acquisition optimization
        for _ in range(100):
            candidate_config = self.config_space.sample_configuration()
            candidate_array = self._configs_to_array([candidate_config])[0]

            acq_value = self._calculate_acquisition(candidate_array)

            if (self.acquisition_function != 'pi' and acq_value > best_acq_value) or \
               (self.acquisition_function == 'pi' and acq_value < best_acq_value):
                best_acq_value = acq_value
                best_config = candidate_config

        return best_config or self.config_space.sample_configuration()

    def tell(self, config: Dict[str, Any], objective_value: float):
        """Update model with new observation"""
        self.X_observed.append(config)
        self.y_observed.append(objective_value)

    def _configs_to_array(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        """Convert configurations to numpy array"""
        arrays = []

        for config in configs:
            vector = []
            for param_name in self.param_order:
                param = self.config_space.parameters[param_name]
                if param_name in config:
                    value = config[param_name]

                    if param.param_type in ['int', 'float', 'log_float']:
                        if param.log_scale or param.param_type == 'log_float':
                            vector.append(np.log(value))
                        else:
                            vector.append(value)
                    elif param.param_type == 'categorical':
                        # One-hot encoding for categorical
                        for choice in param.choices:
                            vector.append(1.0 if choice == value else 0.0)
                    elif param.param_type == 'bool':
                        vector.append(1.0 if value else 0.0)
                else:
                    # Handle missing parameters (conditional)
                    if param.param_type in ['int', 'float', 'log_float']:
                        vector.append(0.0)
                    elif param.param_type == 'categorical':
                        vector.extend([0.0] * len(param.choices))
                    elif param.param_type == 'bool':
                        vector.append(0.0)

            arrays.append(vector)

        return np.array(arrays)

    def _calculate_acquisition(self, x: np.ndarray) -> float:
        """Calculate acquisition function value"""
        if not self.has_gp or self.gp_model is None:
            return 0.0

        x = x.reshape(1, -1)
        mu, sigma = self.gp_model.predict(x, return_std=True)

        if self.acquisition_function == "ei":  # Expected Improvement
            if len(self.y_observed) == 0:
                return 0.0

            f_best = min(self.y_observed)  # Assuming minimization
            with np.errstate(divide='warn'):
                Z = (f_best - mu - self.xi) / sigma
                ei = (f_best - mu - self.xi) * self._normal_cdf(Z) + sigma * self._normal_pdf(Z)
                return ei[0] if sigma > 0 else 0.0

        elif self.acquisition_function == "ucb":  # Upper Confidence Bound
            return -(mu - self.kappa * sigma)[0]  # Negative for minimization

        elif self.acquisition_function == "pi":  # Probability of Improvement
            if len(self.y_observed) == 0:
                return 0.5

            f_best = min(self.y_observed)
            Z = (f_best - mu - self.xi) / sigma
            return self._normal_cdf(Z)[0]

        return 0.0

    def _normal_cdf(self, x):
        """Standard normal CDF"""
        return 0.5 * (1 + np.tanh(x / np.sqrt(2)))

    def _normal_pdf(self, x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


# ==============================================================================
# 4. Advanced Multi-Fidelity Optimization (BOHB Implementation)
# ==============================================================================

class BOHBOptimizer:
    """Bayesian Optimization and HyperBand (BOHB)"""

    def __init__(self, config_space: ConfigurationSpace, min_budget: float = 1,
                 max_budget: float = 27, eta: int = 3, num_samples: int = 64,
                 random_fraction: float = 1/3):
        self.config_space = config_space
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.num_samples = num_samples
        self.random_fraction = random_fraction

        # Initialize HyperBand parameters
        self.s_max = int(np.log(max_budget / min_budget) / np.log(eta)) if max_budget > min_budget else 0
        self.B = (self.s_max + 1) * max_budget

        # Bayesian optimization component
        self.kde_models = {}  # KDE models for each budget level
        self.observations = {}  # Observations for each budget level

        # Current state
        self.active_bracket = None
        self.bracket_counter = 0
        self.configs_to_run = []
        self.param_order = sorted(self.config_space.parameters.keys())

        try:
            from sklearn.neighbors import KernelDensity
            self.has_kde = True
        except ImportError:
            self.has_kde = False

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        """Suggest next configuration and budget"""
        if not self.configs_to_run:
            self._generate_bracket()

        if self.configs_to_run:
            config, budget = self.configs_to_run.pop(0)
            return config, budget

        # Fallback if generation fails
        return self.config_space.sample_configuration(), self.max_budget

    def tell(self, config: Dict[str, Any], budget: float, objective_value: float):
        """Update model with new observation"""
        if budget not in self.observations:
            self.observations[budget] = []

        self.observations[budget].append((config, objective_value))

        self._update_kde_models(budget)

    def _generate_bracket(self):
        """Generate configurations for the next rung or start a new bracket."""
        if self.active_bracket is None:
            # Start a new bracket
            s = self.s_max - (self.bracket_counter % (self.s_max + 1))
            self.bracket_counter += 1

            n = int(np.ceil((self.s_max + 1) * self.eta**s / (s + 1)))
            r = self.max_budget * self.eta**(-s)

            initial_configs = self._get_initial_configs(n)

            self.active_bracket = {'s': s, 'rung': 0, 'configs': initial_configs, 'budget_base': r}
        else:
            # Promote configs from the previous rung
            rung_configs = self.active_bracket['configs']
            prev_rung = self.active_bracket['rung']
            prev_budget = self.active_bracket['budget_base'] * (self.eta ** prev_rung)

            results = []
            for config in rung_configs:
                found_res = [obs[1] for obs in self.observations.get(prev_budget, []) if obs[0] == config]
                if found_res:
                    results.append((config, found_res[0]))
                else:
                    results.append((config, float('inf')))

            results.sort(key=lambda x: x[1])  # Assuming minimization

            num_to_promote = int(len(rung_configs) / self.eta)
            promoted_configs = [res[0] for res in results[:num_to_promote]]

            self.active_bracket['rung'] += 1
            self.active_bracket['configs'] = promoted_configs

        # Check if bracket is finished
        if self.active_bracket['rung'] > self.active_bracket['s'] or not self.active_bracket['configs']:
            self.active_bracket = None
            return self._generate_bracket()  # Recurse to start a new bracket

        # Generate jobs for the current rung
        rung = self.active_bracket['rung']
        budget = self.active_bracket['budget_base'] * (self.eta ** rung)
        configs = self.active_bracket['configs']

        self.configs_to_run = [(config, budget) for config in configs]

    def _get_initial_configs(self, n: int) -> List[Dict[str, Any]]:
        """Get initial configurations for a bracket using random sampling or KDE."""
        if not self.has_kde:
            return [self.config_space.sample_configuration() for _ in range(n)]

        n_random = int(n * self.random_fraction)
        configs = [self.config_space.sample_configuration() for _ in range(n_random)]

        n_kde = n - n_random
        if n_kde <= 0:
            return configs

        best_budget_with_model = None
        for budget in sorted(self.kde_models.keys(), reverse=True):
            best_budget_with_model = budget
            break

        if best_budget_with_model is None:
            configs.extend([self.config_space.sample_configuration() for _ in range(n_kde)])
            return configs

        good_kde, bad_kde = self.kde_models[best_budget_with_model]

        kde_configs = []
        samples_tried = 0
        max_tries = n_kde * 20

        while len(kde_configs) < n_kde and samples_tried < max_tries:
            samples_tried += 1
            sample_array = bad_kde.sample(1)[0]

            good_log_pdf = good_kde.score_samples(sample_array.reshape(1, -1))[0]
            bad_log_pdf = bad_kde.score_samples(sample_array.reshape(1, -1))[0]

            ratio = np.exp(good_log_pdf - bad_log_pdf)

            if np.random.rand() < ratio:
                config = self._array_to_config(sample_array)
                is_valid, _ = self.config_space.validate_configuration(config)
                if is_valid:
                    kde_configs.append(config)

        configs.extend(kde_configs)

        if len(configs) < n:
            configs.extend([self.config_space.sample_configuration() for _ in range(n - len(configs))])

        return configs

    def _update_kde_models(self, budget: float):
        """Update KDE models for a given budget."""
        if not self.has_kde or budget not in self.observations:
            return

        observations = self.observations[budget]

        if len(observations) < self.num_samples:
            return

        observations.sort(key=lambda x: x[1])

        split_index = max(1, int(len(observations) * 0.25))
        good_configs = [obs[0] for obs in observations[:split_index]]
        bad_configs = [obs[0] for obs in observations[split_index:]]

        good_array = self._configs_to_array(good_configs)
        bad_array = self._configs_to_array(bad_configs)

        if good_array.shape[0] == 0 or bad_array.shape[0] == 0:
            return

        from sklearn.neighbors import KernelDensity
        good_kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(good_array)
        bad_kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(bad_array)

        self.kde_models[budget] = (good_kde, bad_kde)

    def _configs_to_array(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        """Convert configurations to numpy array"""
        arrays = []
        for config in configs:
            vector = []
            for param_name in self.param_order:
                param = self.config_space.parameters[param_name]
                if param_name in config:
                    value = config[param_name]
                    if param.param_type in ['int', 'float', 'log_float']:
                        if param.log_scale or param.param_type == 'log_float':
                            vector.append(np.log(value))
                        else:
                            vector.append(value)
                    elif param.param_type == 'categorical':
                        # One-hot encoding for categorical
                        for choice in param.choices:
                            vector.append(1.0 if choice == value else 0.0)
                    elif param.param_type == 'bool':
                        vector.append(1.0 if value else 0.0)
                else:
                    # Handle missing parameters (conditional)
                    if param.param_type in ['int', 'float', 'log_float']:
                        vector.append(0.0)
                    elif param.param_type == 'categorical':
                        vector.extend([0.0] * len(param.choices))
                    elif param.param_type == 'bool':
                        vector.append(0.0)
            arrays.append(vector)
        return np.array(arrays)

    def _array_to_config(self, array: np.ndarray) -> Dict[str, Any]:
        """Convert numpy array back to configuration dictionary"""
        config = {}
        current_idx = 0
        for param_name in self.param_order:
            param = self.config_space.parameters[param_name]
            if param.param_type in ['int', 'float', 'log_float']:
                value = array[current_idx]
                if param.log_scale or param.param_type == 'log_float':
                    value = np.exp(value)
                if param.param_type == 'int':
                    value = int(round(value))
                if param.bounds:
                    value = np.clip(value, param.bounds[0], param.bounds[1])
                config[param_name] = value
                current_idx += 1
            elif param.param_type == 'categorical':
                one_hot_vector = array[current_idx : current_idx + len(param.choices)]
                choice_idx = np.argmax(one_hot_vector)
                config[param_name] = param.choices[choice_idx]
                current_idx += len(param.choices)
            elif param.param_type == 'bool':
                config[param_name] = True if array[current_idx] > 0.5 else False
                current_idx += 1
        return config


# ==============================================================================
# 5. Advanced Pruning and Early Stopping
# ==============================================================================

class MedianPruner:
    """Median-based pruning for multi-step optimization"""

    def __init__(self, n_startup_trials: int = 5, n_warmup_steps: int = 10,
                 interval_steps: int = 1):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.interval_steps = interval_steps
        self.trial_histories = {}

    def should_prune(self, trial_id: str, step: int, value: float) -> bool:
        """Decide if trial should be pruned"""
        # Store intermediate value
        if trial_id not in self.trial_histories:
            self.trial_histories[trial_id] = {}
        self.trial_histories[trial_id][step] = value

        # Don't prune during warmup
        if step < self.n_warmup_steps:
            return False

        # Don't prune if we don't have enough trials yet
        if len(self.trial_histories) < self.n_startup_trials:
            return False

        # Only prune at specified intervals
        if step % self.interval_steps != 0:
            return False

        # Collect values at this step from all trials
        step_values = []
        for trial_hist in self.trial_histories.values():
            if step in trial_hist:
                step_values.append(trial_hist[step])

        if len(step_values) < self.n_startup_trials:
            return False

        # Calculate median
        median_value = np.median(step_values)

        # Prune if current value is worse than median
        return value > median_value  # Assuming minimization


class PercentilePruner:
    """Percentile-based pruning"""

    def __init__(self, percentile: float = 50.0, n_startup_trials: int = 5,
                 n_warmup_steps: int = 10):
        self.percentile = percentile
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.trial_histories = {}

    def should_prune(self, trial_id: str, step: int, value: float) -> bool:
        """Decide if trial should be pruned"""
        if trial_id not in self.trial_histories:
            self.trial_histories[trial_id] = {}
        self.trial_histories[trial_id][step] = value

        if (step < self.n_warmup_steps or
            len(self.trial_histories) < self.n_startup_trials):
            return False

        # Collect values at this step
        step_values = []
        for trial_hist in self.trial_histories.values():
            if step in trial_hist:
                step_values.append(trial_hist[step])

        if len(step_values) < self.n_startup_trials:
            return False

        threshold = np.percentile(step_values, self.percentile)
        return value > threshold


# ==============================================================================
# 6. Integration Framework
# ==============================================================================

class HPOStudy:
    """Main interface for hyperparameter optimization studies"""

    def __init__(self, study_name: str, config_space: ConfigurationSpace,
                 objective_name: str = "loss", direction: str = "minimize",
                 storage_url: str = None, pruner=None):

        self.study_name = study_name
        self.config_space = config_space
        self.objective_name = objective_name
        self.direction = direction

        # Initialize storage
        if storage_url:
            self.storage = ExperimentDatabase(storage_url)
        else:
            self.storage = ExperimentDatabase()

        # Create study in database
        self.study_id = self.storage.create_study(
            study_name, direction, objective_name, config_space
        )

        # Initialize optimizer (default to Bayesian)
        self.optimizer = EnhancedBayesianOptimizer(config_space)
        self.pruner = pruner

        # Trial management
        self.trial_number = 0
        self.best_trial = None
        self.best_value = float('inf') if direction == 'minimize' else float('-inf')

        # Callbacks
        self.callbacks = []

    def optimize(self, objective_func: Callable, n_trials: int = 100,
                timeout: Optional[int] = None, n_jobs: int = 1,
                callbacks: List[Callable] = None) -> Dict[str, Any]:
        """Run optimization"""

        self.callbacks = callbacks or []
        start_time = time.time()

        if n_jobs == 1:
            # Sequential optimization
            for _ in range(n_trials):
                if timeout and (time.time() - start_time) > timeout:
                    break

                trial_result = self._run_single_trial(objective_func)

                # Execute callbacks
                for callback in self.callbacks:
                    callback(trial_result, self)

        else:
            # Parallel optimization
            self._run_parallel_optimization(objective_func, n_trials, n_jobs, timeout)

        return {
            'best_trial': self.best_trial,
            'best_value': self.best_value,
            'n_trials': self.trial_number,
            'study_id': self.study_id
        }

    def _run_single_trial(self, objective_func: Callable) -> Dict[str, Any]:
        """Run a single trial"""
        self.trial_number += 1

        # Get suggestion from optimizer
        if isinstance(self.optimizer, BOHBOptimizer):
            # Multi-fidelity optimizer interface
            config, budget = self.optimizer.suggest()
        else:
            # Standard optimizer interface
            config = self.optimizer.suggest()
            budget = None

        # Validate configuration
        is_valid, errors = self.config_space.validate_configuration(config)
        if not is_valid:
            return {
                'trial_number': self.trial_number,
                'config': config,
                'status': 'failed',
                'error': f"Invalid configuration: {errors}"
            }

        # Save trial start
        trial_id = self.storage.save_trial(
            self.study_id, self.trial_number, config, status="running"
        )

        try:
            start_time = time.time()

            # Create trial context for pruning
            trial_context = TrialContext(trial_id, self.pruner, self.storage)

            # Execute objective function
            if budget is not None:
                config['_budget'] = budget

            result = objective_func(config, trial_context)

            duration = time.time() - start_time

            # Process result
            if isinstance(result, dict):
                metrics = result
                objective_value = metrics.get(self.objective_name)
            else:
                objective_value = float(result)
                metrics = {self.objective_name: objective_value}

            # Update best trial
            is_better = (
                (self.direction == 'minimize' and objective_value < self.best_value) or
                (self.direction == 'maximize' and objective_value > self.best_value)
            )

            if is_better:
                self.best_value = objective_value
                self.best_trial = {
                    'trial_number': self.trial_number,
                    'config': config,
                    'metrics': metrics,
                    'value': objective_value
                }

            # Save completed trial
            self.storage.save_trial(
                self.study_id, self.trial_number, config,
                metrics=metrics, status="completed", duration=duration
            )

            # Update optimizer
            if hasattr(self.optimizer, 'tell'):
                if budget is not None:
                    self.optimizer.tell(config, budget, objective_value)
                else:
                    self.optimizer.tell(config, objective_value)

            return {
                'trial_number': self.trial_number,
                'config': config,
                'metrics': metrics,
                'value': objective_value,
                'duration': duration,
                'status': 'completed'
            }

        except PruneTrialException:
            # Trial was pruned
            duration = time.time() - start_time
            self.storage.save_trial(
                self.study_id, self.trial_number, config,
                status="pruned", duration=duration
            )

            return {
                'trial_number': self.trial_number,
                'config': config,
                'status': 'pruned',
                'duration': duration
            }

        except Exception as e:
            # Trial failed
            duration = time.time() - start_time
            error_msg = str(e)

            self.storage.save_trial(
                self.study_id, self.trial_number, config,
                status="failed", error_message=error_msg, duration=duration
            )

            return {
                'trial_number': self.trial_number,
                'config': config,
                'status': 'failed',
                'error': error_msg,
                'duration': duration
            }

    def _run_parallel_optimization(self, objective_func: Callable, n_trials: int,
                                 n_jobs: int, timeout: Optional[int]):
        """Run optimization in parallel"""
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []

            for _ in range(n_trials):
                if timeout and (time.time() - start_time) > timeout:
                    break

                future = executor.submit(self._run_single_trial, objective_func)
                futures.append(future)

            # Process completed trials
            for future in concurrent.futures.as_completed(futures):
                try:
                    trial_result = future.result()

                    # Execute callbacks
                    for callback in self.callbacks:
                        callback(trial_result, self)

                except Exception as e:
                    logging.error(f"Parallel trial failed: {e}")

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get trials as pandas DataFrame"""
        return self.storage.get_study_trials(self.study_id)

    def add_callback(self, callback: Callable):
        """Add callback function"""
        self.callbacks.append(callback)


class TrialContext:
    """Context object passed to objective function for pruning support"""

    def __init__(self, trial_id: str, pruner, storage: ExperimentDatabase):
        self.trial_id = trial_id
        self.pruner = pruner
        self.storage = storage

    def report(self, step: int, value: float):
        """Report intermediate value"""
        if self.storage:
            self.storage.save_intermediate_value(self.trial_id, step, value)

        if self.pruner and self.pruner.should_prune(self.trial_id, step, value):
            raise PruneTrialException(f"Trial {self.trial_id} pruned at step {step}")


class PruneTrialException(Exception):
    """Exception raised when trial should be pruned"""
    pass


# ==============================================================================
# 7. Usage Examples and Factory Functions
# ==============================================================================

def create_sklearn_study(model_class, X_train, y_train, X_val, y_val,
                        metric: str = "accuracy") -> HPOStudy:
    """Create HPO study for sklearn models"""

    # Define common parameter spaces for different models
    if "RandomForest" in str(model_class):
        parameters = [
            HyperParameter("n_estimators", "int", bounds=(10, 500)),
            HyperParameter("max_depth", "int", bounds=(3, 20)),
            HyperParameter("min_samples_split", "int", bounds=(2, 20)),
            HyperParameter("min_samples_leaf", "int", bounds=(1, 10)),
            HyperParameter("max_features", "categorical", choices=["auto", "sqrt", "log2"])
        ]
    elif "SVC" in str(model_class):
        parameters = [
            HyperParameter("C", "log_float", bounds=(0.001, 1000)),
            HyperParameter("gamma", "log_float", bounds=(1e-5, 1)),
            HyperParameter("kernel", "categorical", choices=["rbf", "poly", "sigmoid"])
        ]
    else:
        # Generic parameters
        parameters = [
            HyperParameter("param1", "float", bounds=(0.0, 1.0))
        ]

    config_space = ConfigurationSpace(parameters)

    def objective(config: Dict[str, Any], trial_context: TrialContext) -> float:
        """Objective function for sklearn model"""
        # Remove budget if present
        config_clean = {k: v for k, v in config.items() if not k.startswith('_')}

        # Train model
        model = model_class(**config_clean)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_val)

        if metric == "accuracy":
            from sklearn.metrics import accuracy_score
            score = accuracy_score(y_val, y_pred)
            return -score  # Negative for minimization
        elif metric == "mse":
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_val, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    study = HPOStudy(
        study_name=f"sklearn_{model_class.__name__}",
        config_space=config_space,
        objective_name="score",
        direction="minimize"
    )

    return study


def quick_optimize(objective_func: Callable, parameter_space: Dict[str, Dict],
                   n_trials: int = 50, direction: str = "minimize",
                   optimizer_type: str = "bayesian") -> Dict[str, Any]:
    """Quick optimization with minimal setup"""

    # Convert parameter space to ConfigurationSpace
    parameters = []
    for param_name, param_spec in parameter_space.items():
        param_type = param_spec.get('type', 'float')

        if param_type in ['int', 'float', 'log_float']:
            param = HyperParameter(
                name=param_name,
                param_type=param_type,
                bounds=(param_spec['low'], param_spec['high']),
                log_scale=param_spec.get('log_scale', False)
            )
        elif param_type in ['categorical', 'choice']:
            param = HyperParameter(
                name=param_name,
                param_type='categorical',
                choices=param_spec['choices']
            )
        elif param_type == 'bool':
            param = HyperParameter(
                name=param_name,
                param_type='bool'
            )
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

        parameters.append(param)

    config_space = ConfigurationSpace(parameters)

    # Create study
    study = HPOStudy(
        study_name="quick_optimize",
        config_space=config_space,
        direction=direction
    )

    # Set optimizer
    if optimizer_type == "bohb":
        study.optimizer = BOHBOptimizer(config_space)
    # else default Bayesian optimizer is already set

    # Define wrapper objective
    def wrapped_objective(config: Dict[str, Any], trial_context: TrialContext):
        return objective_func(config)

    # Run optimization
    result = study.optimize(wrapped_objective, n_trials=n_trials)

    return result


# ==============================================================================
# 8. Complete Working Example
# ==============================================================================