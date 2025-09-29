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
# 1) HyperParameter & ConfigurationSpace
# -----------------------
@dataclass
class HyperParameter:
    name: str
    param_type: str  # 'int', 'float', 'log_float', 'categorical', 'bool'
    bounds: Optional[Tuple[Union[int, float], Union[int, float]]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    conditional_on: Optional[Dict[str, Any]] = None

    def validate_value(self, value: Any) -> bool:
        try:
            if self.param_type == "int":
                if not isinstance(value, (int, np.integer)): return False
                return self.bounds[0] <= value <= self.bounds[1]
            if self.param_type in ("float", "log_float"):
                if not isinstance(value, (int, float, np.number)): return False
                return self.bounds[0] <= float(value) <= self.bounds[1]
            if self.param_type == "categorical":
                return value in (self.choices or [])
            if self.param_type == "bool":
                return isinstance(value, bool)
            return False
        except Exception:
            return False

    def sample_value(self, rng: Optional[np.random.RandomState] = None) -> Any:
        if rng is None:
            rng = np.random.RandomState()
        if self.param_type == "int":
            low, high = int(self.bounds[0]), int(self.bounds[1])
            return int(rng.randint(low, high + 1))
        if self.param_type == "float":
            low, high = float(self.bounds[0]), float(self.bounds[1])
            return float(rng.uniform(low, high))
        if self.param_type == "log_float":
            low, high = float(self.bounds[0]), float(self.bounds[1])
            return float(np.exp(rng.uniform(np.log(low), np.log(high))))
        if self.param_type == "categorical":
            return rng.choice(self.choices)
        if self.param_type == "bool":
            return bool(rng.choice([False, True]))
        raise ValueError(f"Unknown parameter type: {self.param_type}")


class ConfigurationSpace:
    """
    Holds HyperParameter objects and provides:
    - sample_configuration(rng)
    - validate_configuration(cfg)
    - to_array(configs) -> np.ndarray for surrogate inputs
    """

    def __init__(self, parameters: List[HyperParameter]):
        # preserve insertion order
        self.parameters: Dict[str, HyperParameter] = {p.name: p for p in parameters}
        self.conditional_graph = self._build_conditional_graph()
        # derived info for vectorization
        self._vector_length = None
        self._compute_vector_length()

    def _build_conditional_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for name, param in self.parameters.items():
            if param.conditional_on:
                for parent in param.conditional_on.keys():
                    graph.setdefault(parent, []).append(name)
        return graph

    def sample_configuration(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        if rng is None:
            rng = np.random.RandomState()
        config: Dict[str, Any] = {}
        remaining = set(self.parameters.keys())
        progressed = True
        while remaining and progressed:
            progressed = False
            for name in list(remaining):
                p = self.parameters[name]
                if not p.conditional_on:
                    config[name] = p.sample_value(rng)
                    remaining.remove(name)
                    progressed = True
                else:
                    parents_satisfied = all(parent in config for parent in p.conditional_on.keys())
                    if parents_satisfied:
                        ok = True
                        for parent, required in (p.conditional_on or {}).items():
                            if config.get(parent) != required:
                                ok = False
                                break
                        if ok:
                            config[name] = p.sample_value(rng)
                        remaining.remove(name)
                        progressed = True
        return config

    def validate_configuration(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for name, param in self.parameters.items():
            if name in config:
                if not param.validate_value(config[name]):
                    errors.append(f"Invalid value for {name}: {config[name]}")
                if param.conditional_on:
                    for parent, required in param.conditional_on.items():
                        if parent not in config:
                            errors.append(f"{name} requires parent {parent} to be present")
                        elif config[parent] != required:
                            errors.append(f"{name} requires {parent} == {required}")
        return (len(errors) == 0, errors)

    def _compute_vector_length(self):
        length = 0
        for name, param in self.parameters.items():
            if param.param_type in ("int", "float", "log_float", "bool"):
                length += 1
            elif param.param_type == "categorical":
                length += len(param.choices or [])
        self._vector_length = length

    def to_array(self, configs: List[Dict[str, Any]]) -> np.ndarray:
        if self._vector_length is None:
            self._compute_vector_length()
        result = np.zeros((len(configs), self._vector_length), dtype=float)
        for i, cfg in enumerate(configs):
            idx = 0
            for name, param in self.parameters.items():
                if param.param_type in ("int", "float", "log_float", "bool"):
                    if name in cfg:
                        v = cfg[name]
                        if v is None:
                            result[i, idx] = 0.0
                        else:
                            if param.param_type == "bool":
                                result[i, idx] = 1.0 if bool(v) else 0.0
                            else:
                                val = float(v)
                                if param.log_scale or param.param_type == "log_float":
                                    val = max(val, 1e-12)
                                    result[i, idx] = math.log(val)
                                else:
                                    result[i, idx] = val
                    else:
                        result[i, idx] = 0.0
                    idx += 1
                elif param.param_type == "categorical":
                    choices = param.choices or []
                    one_hot = [0.0] * len(choices)
                    if name in cfg:
                        val = cfg[name]
                        for k, ch in enumerate(choices):
                            if ch == val:
                                one_hot[k] = 1.0
                                break
                    for v in one_hot:
                        result[i, idx] = v
                        idx += 1
                else:
                    # Unknown parameter types are not represented in the vector, so we do nothing.
                    # This aligns with the behavior of _compute_vector_length.
                    pass
        return result


# -----------------------
# 2) ExperimentDatabase (SQLite)
# -----------------------
class ExperimentDatabase:
    """
    Lightweight SQLite-backed experiment DB with simple concurrency settings.
    Stores studies and trials. Use one connection per operation.
    """

    def __init__(self, db_path: str = "hpo_experiments.db"):
        self.db_path = db_path
        self._init_lock = threading.Lock()
        self._is_memory = self.db_path == ":memory:"
        self._conn = None
        if self._is_memory:
            # For in-memory db, we need a single connection that persists.
            # Using check_same_thread=False to allow multi-threaded access,
            # which seems intended given the presence of threading.Lock.
            self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _get_conn(self):
        if self._is_memory:
            return self._conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self):
        with self._init_lock:
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS studies (
                        study_id TEXT PRIMARY KEY,
                        study_name TEXT UNIQUE,
                        direction TEXT,
                        objective_name TEXT,
                        config_space TEXT,
                        metadata TEXT,
                        created_at INTEGER,
                        completed_at INTEGER,
                        status TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trials (
                        trial_id TEXT PRIMARY KEY,
                        study_id TEXT,
                        trial_number INTEGER,
                        parameters TEXT,
                        metrics TEXT,
                        status TEXT,
                        started_at INTEGER,
                        completed_at INTEGER,
                        duration_seconds REAL,
                        error_message TEXT,
                        FOREIGN KEY(study_id) REFERENCES studies(study_id)
                    )
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_trials_study ON trials(study_id)")
                conn.commit()

    def create_study(self, study_name: str, direction: str, objective_name: str, config_space: ConfigurationSpace, metadata: Dict = None) -> str:
        meta_json = json.dumps(metadata or {})
        cfg_json = json.dumps([vars(p) for p in config_space.parameters.values()])
        study_id = f"study_{hashlib.md5((study_name+str(time.time())).encode()).hexdigest()[:10]}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO studies (study_id,study_name,direction,objective_name,config_space,metadata,created_at,status) VALUES (?,?,?,?,?,?,?,?)",
                        (study_id, study_name, direction, objective_name, cfg_json, meta_json, int(time.time()), "running"))
            conn.commit()
        return study_id

    def save_trial(self, study_id: str, trial_number: int, parameters: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None, status: str = "running", error_message: Optional[str] = None, duration_seconds: Optional[float] = None) -> str:
        trial_id = f"trial_{study_id}_{trial_number:06d}"
        with self._get_conn() as conn:
            cur = conn.cursor()
            if status == "running":
                cur.execute("INSERT OR REPLACE INTO trials (trial_id,study_id,trial_number,parameters,status,started_at) VALUES (?,?,?,?,?,?)",
                            (trial_id, study_id, trial_number, json.dumps(parameters), status, int(time.time())))
            else:
                cur.execute("UPDATE trials SET metrics=?, status=?, completed_at=?, duration_seconds=?, error_message=? WHERE trial_id=?",
                            (json.dumps(metrics or {}), status, int(time.time()), duration_seconds, error_message, trial_id))
            conn.commit()
        return trial_id

    def get_study_trials(self, study_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT trial_number, parameters, metrics, status, started_at, completed_at, duration_seconds FROM trials WHERE study_id=? ORDER BY trial_number", (study_id,))
            rows = cur.fetchall()
        out = []
        for r in rows:
            trial_number, params_j, metrics_j, status, started_at, completed_at, duration_seconds = r
            out.append({
                "trial_number": trial_number,
                "parameters": json.loads(params_j),
                "metrics": json.loads(metrics_j) if metrics_j else {},
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration_seconds
            })
        return out

    def get_top_configs(self, study_id: str, metric_name: str, top_k: int = 3, minimize: bool = True) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT parameters, metrics FROM trials WHERE study_id=? AND metrics IS NOT NULL", (study_id,))
            rows = cur.fetchall()
        entries = []
        for params_j, metrics_j in rows:
            m = json.loads(metrics_j)
            if metric_name in m:
                entries.append((json.loads(params_j), m[metric_name]))
        if not entries:
            return []
        entries.sort(key=lambda x: x[1], reverse=not minimize)
        return [e[0] for e in entries[:top_k]]

    def find_similar_studies(self, meta_key: str, meta_value: Any) -> List[str]:
        res = []
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT study_id, metadata FROM studies")
            rows = cur.fetchall()
        for study_id, meta_j in rows:
            try:
                md = json.loads(meta_j or "{}")
                if isinstance(md, dict) and md.get(meta_key) == meta_value:
                    res.append(study_id)
            except Exception:
                continue
        return res


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

    def __init__(self, config_space: ConfigurationSpace, min_budget: float = 1.0, max_budget: float = 27.0, eta: int = 3, minimize: bool = True):
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

    def _next_task_id(self) -> str:
        self.task_counter += 1
        return f"t_{self.task_counter}_{uuid.uuid4().hex[:6]}"

    def _generate_bracket(self, s: int) -> str:
        R = self.max_budget
        n = int(math.ceil(self.B / R * (self.eta ** s) / (s + 1)))
        r = float(R * (self.eta ** (-s)))
        bracket_id = f"br_{int(time.time()*1000)}_{s}_{uuid.uuid4().hex[:6]}"
        logger.info("Creating bracket %s: s=%d n=%d r=%.4f", bracket_id, s, n, r)
        rungs = []
        cur_n = n
        cur_r = r
        for i in range(s + 1):
            rungs.append({"level": i, "n": cur_n, "r": cur_r, "results": []})
            cur_n = int(math.floor(cur_n / self.eta))
            cur_r = cur_r * self.eta
        self.brackets[bracket_id] = {"s": s, "rungs": rungs, "completed": False}
        for i in range(n):
            cfg = self.cs.sample_configuration()
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