"""
HPO core utilities: ConfigurationSpace, ExperimentDatabase, EnhancedBayesianOptimizer, BOHBOptimizer

This file is intended to be the core foundation. It provides:
- deterministic vectorization of configs for surrogates
- SQLite-backed experiment DB with safe pragmas
- GP-based Bayesian optimizer (EI/UCB/PI) with correct math
- BOHB scheduler skeleton with brackets, promotions, and task_id tracking
"""

from __future__ import annotations

import json
import math
import time
import uuid
import sqlite3
import threading
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from hpo.optimizers.bohb_kde import BOHB_KDE
from hpo.warmstart.warm_start_manager import WarmStartManager
from hpo.configuration import ConfigurationSpace

# Optional imports
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import norm as _scipy_norm
    def _norm_pdf(x): return _scipy_norm.pdf(x)
    def _norm_cdf(x): return _scipy_norm.cdf(x)
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    from math import erf
    def _norm_pdf(x): return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    def _norm_cdf(x): return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))

logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)




# -----------------------
# 3) Enhanced Bayesian optimizer (GP + EI/UCB/PI)
# -----------------------
class EnhancedBayesianOptimizer:
    """
    Gaussian Process surrogate with multiple acquisition functions.
    """

    def __init__(self, config_space: ConfigurationSpace, acquisition: str = "ei", xi: float = 0.01, kappa: float = 2.576, n_initial: int = 8, minimize: bool = True):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for EnhancedBayesianOptimizer")
        self.cs = config_space
        self.acquisition = acquisition.lower()
        self.xi = xi
        self.kappa = kappa
        self.n_initial = int(n_initial)
        self.minimize = bool(minimize)

        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=3)

    def tell(self, config: Dict[str, Any], value: float):
        vec = self.cs.to_array([config])[0]
        self.X.append(vec)
        self.y.append(float(value))

    def suggest(self, n_candidates: int = 200) -> Dict[str, Any]:
        if len(self.X) < self.n_initial:
            return self.cs.sample_configuration()
        X_arr = np.vstack(self.X)
        y_arr = np.array(self.y)
        self.gp.fit(X_arr, y_arr)

        candidates = [self.cs.sample_configuration() for _ in range(n_candidates)]
        Xc = self.cs.to_array(candidates)
        mu, sigma = self.gp.predict(Xc, return_std=True)
        sigma = np.maximum(sigma, 1e-12)

        if self.minimize:
            f_best = np.min(y_arr)
            if self.acquisition == "ei":
                gamma = (f_best - mu - self.xi) / sigma
                acq = (f_best - mu - self.xi) * _norm_cdf(gamma) + sigma * _norm_pdf(gamma)
            elif self.acquisition == "ucb":
                acq = -(mu - self.kappa * sigma)
            elif self.acquisition == "pi":
                gamma = (f_best - mu - self.xi) / sigma
                acq = _norm_cdf(gamma)
            else:
                raise ValueError("Unknown acquisition")
        else:
            f_best = np.max(y_arr)
            if self.acquisition == "ei":
                gamma = (mu - f_best - self.xi) / sigma
                acq = (mu - f_best - self.xi) * _norm_cdf(gamma) + sigma * _norm_pdf(gamma)
            elif self.acquisition == "ucb":
                acq = mu + self.kappa * sigma
            elif self.acquisition == "pi":
                gamma = (mu - f_best - self.xi) / sigma
                acq = _norm_cdf(gamma)
            else:
                raise ValueError("Unknown acquisition")

        best_idx = int(np.argmax(acq))
        return candidates[best_idx]


# -----------------------
# 4) BOHB (BO + HyperBand skeleton)
# -----------------------
class BOHBOptimizer:
    """
    BOHB-like scheduler that generates brackets (HyperBand) and performs successive halving.
    This is an orchestration skeleton: it samples configs from config_space, schedules them
    at budgets, and promotes top performers.
    """

    def __init__(self,
                 config_space: ConfigurationSpace,
                 min_budget: float = 1.0,
                 max_budget: float = 27.0,
                 eta: int = 3,
                 minimize: bool = True,
                 use_kde: bool = False,
                 warm_start_manager: Optional[WarmStartManager] = None,
                 current_meta_features: Optional[Dict[str, Any]] = None):
        self.cs = config_space
        self.min_budget = float(min_budget)
        self.max_budget = float(max_budget)
        self.eta = int(eta)
        self.minimize = bool(minimize)
        self.s_max = int(math.floor(math.log(self.max_budget / self.min_budget) / math.log(self.eta)))
        self.B = (self.s_max + 1) * self.max_budget

        self.brackets = {}
        self.task_queue: List[Tuple[str, Dict[str, Any], float]] = []
        self.task_map = {}
        self.results_lock = threading.Lock()
        self.task_counter = 0

        # Integration for KDE and Warm-Start
        self.kde_optimizer = None
        if use_kde:
            budgets = [self.max_budget * (self.eta ** -i) for i in range(self.s_max + 1)]
            self.kde_optimizer = BOHB_KDE(self.cs, budgets=budgets)

        self.warm_start_manager = warm_start_manager
        if self.warm_start_manager and current_meta_features:
            # For simplicity, we assume the first key is the one to use for finding similar studies
            meta_key = list(current_meta_features.keys())[0]
            self.warm_start_manager.find_and_train(current_meta_features, meta_key=meta_key)

    def _next_task_id(self) -> str:
        self.task_counter += 1
        return f"t_{self.task_counter}_{uuid.uuid4().hex[:6]}"

    def _generate_bracket(self, s: int) -> str:
        R = self.max_budget
        n = int(math.ceil(self.B / R * (self.eta**s) / (s + 1)))
        r = float(R * (self.eta**(-s)))
        bracket_id = f"br_{int(time.time()*1000)}_{s}_{uuid.uuid4().hex[:6]}"
        logger.info("Creating bracket %s: s=%d n=%d r=%.4f", bracket_id, s, n, r)

        # Set up rungs
        rungs = []
        cur_n, cur_r = n, r
        for i in range(s + 1):
            rungs.append({"level": i, "n": cur_n, "r": cur_r, "results": []})
            cur_n = int(math.floor(cur_n / self.eta))
            cur_r *= self.eta
        self.brackets[bracket_id] = {"s": s, "rungs": rungs, "completed": False}

        # --- Integrated Configuration Generation ---
        selected_configs = []
        if self.kde_optimizer:
            # Fit models before proposing
            self.kde_optimizer.fit_all()

            # Generate a pool of candidates
            n_candidates = max(n * 10, 100)
            candidates = [self.cs.sample_configuration() for _ in range(n_candidates)]

            # Warm-start ranking if available
            if self.warm_start_manager and self.warm_start_manager.meta_learner.trained:
                candidates = self.warm_start_manager.rank_candidates(candidates, {})

            # Propose from the pool using KDE
            selected_configs = self.kde_optimizer.propose(candidates, top_k=n)

        # Fallback to random sampling if KDE is not used or didn't return enough configs
        if len(selected_configs) < n:
            num_needed = n - len(selected_configs)
            random_configs = [self.cs.sample_configuration() for _ in range(num_needed)]
            selected_configs.extend(random_configs)

        # Enqueue the selected configurations
        for cfg in selected_configs:
            task_id = self._next_task_id()
            cfg_internal = dict(cfg)
            cfg_internal["__hbo_task_id"] = task_id
            self.task_queue.append((task_id, cfg_internal, r))
            self.task_map[task_id] = (bracket_id, 0, cfg_internal)

        return bracket_id

    def suggest(self) -> Tuple[Dict[str, Any], float]:
        if not self.task_queue:
            next_s = (len(self.brackets)) % (self.s_max + 1)
            self._generate_bracket(next_s)
        task_id, cfg, budget = self.task_queue.pop(0)
        return cfg, budget

    def tell(self, config: Dict[str, Any], budget: float, objective_value: float):
        task_id = config.get("__hbo_task_id")
        if not task_id:
            logger.warning("tell() called with config missing __hbo_task_id - ignoring")
            return
        with self.results_lock:
            mapping = self.task_map.get(task_id)
            if not mapping:
                logger.warning("Unknown task_id %s in tell()", task_id)
                return
            bracket_id, rung_level, cfg = mapping
            bracket = self.brackets.get(bracket_id)
            if not bracket:
                logger.warning("Unknown bracket %s", bracket_id)
                return
            rung = bracket["rungs"][rung_level]
            rung["results"].append({"task_id": task_id, "config": cfg, "budget": budget, "value": float(objective_value)})

            # Pass observation to KDE optimizer if it's being used
            if self.kde_optimizer:
                self.kde_optimizer.observe(config=cfg, budget=budget, value=float(objective_value))

            expected = rung["n"]
            if len(rung["results"]) >= expected:
                logger.info("Rung %d complete for bracket %s (n=%d) -> promotion", rung_level, bracket_id, expected)
                next_level = rung_level + 1
                if next_level < len(bracket["rungs"]):
                    sorted_results = sorted(rung["results"], key=lambda x: x["value"], reverse=not self.minimize)
                    promote_k = int(math.floor(len(sorted_results) / self.eta))
                    if promote_k > 0:
                        promoted = sorted_results[:promote_k]
                        next_r = bracket["rungs"][next_level]["r"]
                        for item in promoted:
                            new_task_id = self._next_task_id()
                            cfg_copy = dict(item["config"])
                            cfg_copy["__hbo_task_id"] = new_task_id
                            self.task_queue.append((new_task_id, cfg_copy, next_r))
                            self.task_map[new_task_id] = (bracket_id, next_level, cfg_copy)
                        logger.info("Promoted %d configs to rung %d at budget=%.4f", len(promoted), next_level, next_r)
                else:
                    bracket["completed"] = True
                    logger.info("Bracket %s completed", bracket_id)

    def num_pending(self) -> int:
        return len(self.task_queue)