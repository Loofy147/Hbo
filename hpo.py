"""
Enhanced Hyperparameter Optimization System
Comprehensive improvements for modern HPO needs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import json
import pickle
import logging
import time
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from pathlib import Path
import hashlib
from enum import Enum

# Third-party imports with fallbacks
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Optuna not available. Some advanced features will be disabled.")

try:
    from skopt import Optimizer as SkoptOptimizer
    from skopt.space import Real, Integer, Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    warnings.warn("scikit-optimize not available. BayesianOptimizer will not be available.")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# =============================================================================
# Core Data Structures
# =============================================================================

class ParameterType(Enum):
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    LOG_UNIFORM = "log_uniform"

@dataclass
class Parameter:
    """Enhanced parameter definition with distribution support"""
    name: str
    param_type: ParameterType
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Optional[Any] = None
    prior: Optional[str] = None  # For Bayesian priors

    def __post_init__(self):
        if self.param_type in [ParameterType.INTEGER, ParameterType.FLOAT, ParameterType.LOG_UNIFORM]:
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter {self.name} requires low and high bounds")
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name} requires choices")

@dataclass
class Trial:
    """Trial result with enhanced metadata"""
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "COMPLETED"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error: Optional[str] = None

    @property
    def primary_metric(self) -> float:
        """Get the primary optimization metric"""
        if not self.metrics:
            return float('nan')
        if 'loss' in self.metrics:
            return self.metrics['loss']
        elif 'score' in self.metrics:
            return self.metrics['score']
        else:
            return list(self.metrics.values())[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to a serializable dictionary."""
        d = asdict(self)
        d['start_time'] = self.start_time.isoformat() if self.start_time else None
        d['end_time'] = self.end_time.isoformat() if self.end_time else None
        return d

@dataclass
class OptimizationConfig:
    """Comprehensive optimization configuration"""
    parameters: List[Parameter]
    objective_name: str = "loss"
    direction: str = "minimize"  # "minimize" or "maximize"
    n_trials: int = 100
    timeout: Optional[int] = None
    n_jobs: int = 1
    random_seed: Optional[int] = None

    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_rounds: int = 10
    early_stopping_threshold: float = 1e-4

    # Multi-fidelity
    multi_fidelity_enabled: bool = False
    resource_attr: str = "epochs"
    max_resource: int = 100
    min_resource: int = 1
    reduction_factor: int = 3

    # Advanced options
    pruning_enabled: bool = True
    warmup_steps: int = 5
    save_intermediate: bool = True

    # Integration settings
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization, handling enums."""
        data = asdict(self)
        for param in data['parameters']:
            if 'param_type' in param and isinstance(param['param_type'], Enum):
                param['param_type'] = param['param_type'].value
        return data
        for param in data['parameters']:
            if isinstance(param.get('param_type'), Enum):
                param['param_type'] = param['param_type'].value
        return data

# =============================================================================
# Base Optimizer Interface
# =============================================================================

class BaseOptimizer(ABC):
    """Abstract base class for all optimizers"""

    def __init__(self, config: OptimizationConfig, study_name: str = "hpo_study"):
        self.config = config
        self.study_name = study_name
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        self._setup_tracking()

    def _setup_tracking(self):
        """Setup experiment tracking"""
        if self.config.wandb_enabled and HAS_WANDB:
            wandb.init(
                project=self.config.wandb_project or "hpo-experiments",
                name=self.study_name,
                config=self.config.to_dict()
            )

    def _random_suggest(self) -> Dict[str, Any]:
        """Fallback random parameter suggestion"""
        params = {}
        for param in self.config.parameters:
            if param.param_type == ParameterType.FLOAT:
                if param.log_scale:
                    params[param.name] = np.exp(np.random.uniform(
                        np.log(param.low), np.log(param.high)))
                else:
                    params[param.name] = np.random.uniform(param.low, param.high)
            elif param.param_type == ParameterType.INTEGER:
                params[param.name] = np.random.randint(param.low, param.high + 1)
            elif param.param_type == ParameterType.CATEGORICAL:
                params[param.name] = np.random.choice(param.choices)
            elif param.param_type == ParameterType.LOG_UNIFORM:
                params[param.name] = np.exp(np.random.uniform(
                    np.log(param.low), np.log(param.high)))
        return params

    @abstractmethod
    def suggest_parameters(self, trial_id: str) -> Dict[str, Any]:
        """Suggest next set of parameters to try"""
        pass

    @abstractmethod
    def tell_result(self, trial: Trial):
        """Report trial result back to optimizer"""
        pass

    def optimize(self,
                 objective_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 callbacks: Optional[List[Callable]] = None) -> Trial:
        """Main optimization loop"""

        if self.config.n_jobs > 1:
            warnings.warn(
                f"{self.__class__.__name__} does not support parallel execution. "
                "Running sequentially. Override optimize() for parallel behavior."
            )

        callbacks = callbacks or []
        start_time = time.time()

        no_improvement_count = 0
        best_metric = float('inf') if self.config.direction == 'minimize' else float('-inf')

        self.logger.info(f"Starting optimization with {self.config.n_trials} trials")

        for trial_idx in range(self.config.n_trials):
            if self.config.timeout and (time.time() - start_time) > self.config.timeout:
                self.logger.info(f"Timeout reached after {trial_idx} trials")
                break

            trial_id = f"trial_{trial_idx:04d}_{int(time.time() * 1000) % 100000}"
            params = {}

            try:
                params = self.suggest_parameters(trial_id)

                trial_start_time = datetime.now()
                result = objective_func(params)
                trial_end_time = datetime.now()

                trial = Trial(
                    trial_id=trial_id,
                    parameters=params,
                    metrics=result if isinstance(result, dict) else {'loss': result},
                    start_time=trial_start_time,
                    end_time=trial_end_time,
                    duration=(trial_end_time - trial_start_time).total_seconds(),
                    status="COMPLETED"
                )

                self.trials.append(trial)
                self.tell_result(trial)

                current_metric = trial.primary_metric
                is_better = (
                    (self.config.direction == 'minimize' and current_metric < best_metric) or
                    (self.config.direction == 'maximize' and current_metric > best_metric)
                )

                if is_better:
                    best_metric = current_metric
                    self.best_trial = trial
                    no_improvement_count = 0
                    self.logger.info(f"New best trial: {trial_id} with metric {best_metric:.6f}")
                else:
                    no_improvement_count += 1

                self.logger.info(
                    f"Trial {trial_idx + 1}/{self.config.n_trials}: "
                    f"{self.config.objective_name}={current_metric:.6f}, "
                    f"Best={best_metric:.6f}"
                )

                if self.config.wandb_enabled and HAS_WANDB:
                    wandb.log({
                        "trial": trial_idx,
                        "current_metric": current_metric,
                        "best_metric": best_metric,
                        "duration": trial.duration,
                        **params,
                        **trial.metrics
                    })

                for callback in callbacks:
                    callback(trial, self)

                if (self.config.early_stopping_enabled and
                    no_improvement_count >= self.config.early_stopping_rounds):
                    self.logger.info(f"Early stopping triggered after {no_improvement_count} rounds without improvement")
                    break

            except Exception as e:
                self.logger.error(f"Trial {trial_id} failed: {str(e)}", exc_info=True)
                failed_trial = Trial(
                    trial_id=trial_id,
                    parameters=params,
                    metrics={},
                    status="FAILED",
                    error=str(e)
                )
                self.trials.append(failed_trial)
                self.tell_result(failed_trial)

        self.logger.info(f"Optimization completed. Best metric: {self.best_trial.primary_metric if self.best_trial else 'N/A'}")
        return self.best_trial

    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as a flattened DataFrame."""
        records = []
        for trial in self.trials:
            record = trial.to_dict()
            params = record.pop('parameters', {})
            metrics = record.pop('metrics', {})
            record.update(params)
            record.update(metrics)
            records.append(record)
        return pd.DataFrame(records)

    def save_study(self, filepath: str):
        """Save optimization study to a JSON file."""
        study_data = {
            'config': self.config.to_dict(),
            'study_name': self.study_name,
            'trials': [t.to_dict() for t in self.trials],
            'best_trial': self.best_trial.to_dict() if self.best_trial else None
        }

        with open(filepath, 'w') as f:
            json.dump(study_data, f, indent=2)

# =============================================================================
# Advanced Optimizers
# =============================================================================

class BayesianOptimizer(BaseOptimizer):
    """Bayesian Optimization using scikit-optimize."""

    def __init__(self, config: OptimizationConfig, study_name: str = "bayesian_study"):
        super().__init__(config, study_name)
        if not HAS_SKOPT:
            raise ImportError("scikit-optimize is required for BayesianOptimizer")

        self._setup_skopt_space()

        self.optimizer = SkoptOptimizer(
            dimensions=self.space,
            random_state=self.config.random_seed,
            n_initial_points=self.config.warmup_steps,
            acq_func="gp_hedge",  # Default robust acquisition function
        )
        self.param_names = [p.name for p in self.config.parameters]

    def _setup_skopt_space(self):
        """Setup scikit-optimize search space."""
        self.space = []
        for param in self.config.parameters:
            if param.param_type == ParameterType.FLOAT:
                self.space.append(Real(param.low, param.high, prior='log-uniform' if param.log_scale else 'uniform', name=param.name))
            elif param.param_type == ParameterType.INTEGER:
                self.space.append(Integer(param.low, param.high, name=param.name))
            elif param.param_type == ParameterType.CATEGORICAL:
                self.space.append(Categorical(param.choices, name=param.name))
            elif param.param_type == ParameterType.LOG_UNIFORM:
                self.space.append(Real(param.low, param.high, prior='log-uniform', name=param.name))

    def suggest_parameters(self, trial_id: str) -> Dict[str, Any]:
        """Suggest parameters using the internal skopt optimizer."""
        suggested_point = self.optimizer.ask()
        return dict(zip(self.param_names, suggested_point))

    def tell_result(self, trial: Trial):
        """Update the Bayesian model with the trial result."""
        if trial.status == "FAILED":
            # Tell optimizer with a high penalty value
            y_value = float('inf') if self.config.direction == 'minimize' else float('-inf')
        else:
            y_value = trial.primary_metric

        if self.config.direction == 'maximize':
            y_value = -y_value

        x_point = [trial.parameters[name] for name in self.param_names]
        self.optimizer.tell(x_point, y_value)


class TPEOptimizer(BaseOptimizer):
    """Tree-structured Parzen Estimator using Optuna."""

    def __init__(self, config: OptimizationConfig, study_name: str = "tpe_study"):
        super().__init__(config, study_name)
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for TPEOptimizer")

        direction = "minimize" if config.direction == "minimize" else "maximize"
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=config.random_seed)
        )
        self._trial_map: Dict[str, optuna.trial.Trial] = {}

    def suggest_parameters(self, trial_id: str) -> Dict[str, Any]:
        """Suggest parameters using TPE from Optuna."""
        optuna_trial = self.study.ask()
        self._trial_map[trial_id] = optuna_trial

        params = {}
        for param in self.config.parameters:
            if param.param_type == ParameterType.FLOAT:
                params[param.name] = optuna_trial.suggest_float(param.name, param.low, param.high, log=param.log_scale)
            elif param.param_type == ParameterType.INTEGER:
                params[param.name] = optuna_trial.suggest_int(param.name, param.low, param.high)
            elif param.param_type == ParameterType.CATEGORICAL:
                params[param.name] = optuna_trial.suggest_categorical(param.name, param.choices)
            elif param.param_type == ParameterType.LOG_UNIFORM:
                params[param.name] = optuna_trial.suggest_float(param.name, param.low, param.high, log=True)

        return params

    def tell_result(self, trial: Trial):
        """Report result to Optuna study."""
        optuna_trial = self._trial_map.pop(trial.trial_id, None)
        if optuna_trial is None:
            self.logger.warning(f"Optuna trial not found for trial_id: {trial.trial_id}")
            return

        if trial.status == "COMPLETED":
            self.study.tell(optuna_trial, trial.primary_metric, state=optuna.trial.TrialState.COMPLETE)
        else:
            self.study.tell(optuna_trial, None, state=optuna.trial.TrialState.FAIL)


class HyperbandOptimizer(BaseOptimizer):
    """Hyperband multi-fidelity optimization."""

    def __init__(self, config: OptimizationConfig, study_name: str = "hyperband_study"):
        super().__init__(config, study_name)

        if not config.multi_fidelity_enabled:
            raise ValueError("HyperbandOptimizer requires multi_fidelity_enabled=True in OptimizationConfig")

        self.max_resource = config.max_resource
        self.min_resource = config.min_resource
        self.reduction_factor = config.reduction_factor

        self.s_max = int(np.log(self.max_resource / self.min_resource) / np.log(self.reduction_factor))
        self.B = (self.s_max + 1) * self.max_resource

    def suggest_parameters(self, trial_id: str) -> Dict[str, Any]:
        """Suggests random parameters for a new configuration."""
        return self._random_suggest()

    def tell_result(self, trial: Trial):
        """Updates the best trial found so far."""
        if self.best_trial is None:
            self.best_trial = trial
            return

        if trial.status == "COMPLETED":
            is_better = (
                (self.config.direction == 'minimize' and trial.primary_metric < self.best_trial.primary_metric) or
                (self.config.direction == 'maximize' and trial.primary_metric > self.best_trial.primary_metric)
            )
            if is_better:
                self.best_trial = trial

    def optimize(self,
                 objective_func: Callable[[Dict[str, Any]], Dict[str, Any]],
                 callbacks: Optional[List[Callable]] = None) -> Trial:
        """Hyperband optimization algorithm."""
        callbacks = callbacks or []
        trial_counter = 0

        for s in range(self.s_max, -1, -1):
            n = int(np.ceil(self.B / self.max_resource * self.reduction_factor**s / (s + 1)))
            r = self.max_resource * self.reduction_factor**(-s)

            self.logger.info(f"Starting bracket s={s}: n={n}, r={r:.2f}")

            # Get initial configurations for this bracket
            configurations = [self.suggest_parameters(f"s{s}_t{i}") for i in range(n)]

            for i in range(s + 1):
                n_i = len(configurations)
                r_i = r * self.reduction_factor**i

                if n_i == 0:
                    break

                self.logger.info(f"  Rung i={i}: Evaluating {n_i} configs with resource r_i={r_i:.2f}")

                results = []
                for params in configurations:
                    trial_id = f"trial_{trial_counter:04d}"
                    trial_counter += 1

                    params_with_resource = params.copy()
                    params_with_resource[self.config.resource_attr] = int(r_i)

                    try:
                        trial_start_time = datetime.now()
                        metric_result = objective_func(params_with_resource)
                        trial_end_time = datetime.now()

                        trial = Trial(
                            trial_id=trial_id,
                            parameters=params, # Store original params without resource
                            metrics=metric_result,
                            metadata={'resource': r_i, 'bracket': s},
                            start_time=trial_start_time,
                            end_time=trial_end_time,
                            duration=(trial_end_time - trial_start_time).total_seconds(),
                            status="COMPLETED"
                        )
                        results.append(trial)

                    except Exception as e:
                        self.logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
                        trial = Trial(
                            trial_id=trial_id,
                            parameters=params,
                            metrics={},
                            status="FAILED",
                            error=str(e),
                            metadata={'resource': r_i, 'bracket': s},
                        )

                    self.trials.append(trial)
                    self.tell_result(trial)
                    for callback in callbacks:
                        callback(trial, self)

                # Promote the top configurations
                completed_trials = [t for t in results if t.status == "COMPLETED"]

                if not completed_trials:
                    self.logger.warning("  No trials completed successfully in this rung. Stopping bracket.")
                    break

                # Sort by metric
                reverse = self.config.direction == 'maximize'
                completed_trials.sort(key=lambda t: t.primary_metric, reverse=reverse)

                k = n_i // self.reduction_factor
                promoted_trials = completed_trials[:k]
                configurations = [t.parameters for t in promoted_trials]

                if not configurations:
                    self.logger.info("  No configurations promoted to the next rung.")
                    break

        self.logger.info(f"Hyperband optimization finished. Best metric: {self.best_trial.primary_metric if self.best_trial else 'N/A'}")
        return self.best_trial