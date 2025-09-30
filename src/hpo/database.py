from __future__ import annotations

import json
import sqlite3
import threading
import hashlib
import time
from typing import Any, Dict, List, Optional

from hpo.configuration import ConfigurationSpace


class ExperimentDatabase:
    """
    Lightweight SQLite-backed experiment DB with simple concurrency settings.
    Stores studies and trials. Use one connection per operation.
    """

    def __init__(self, db_path: str = "hpo_experiments.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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