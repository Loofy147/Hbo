import sqlite3
import threading
import json
from typing import List, Dict, Any, Optional

from .base import BaseStorage
from ..core.trial import Trial, TrialState

class SQLiteStorage(BaseStorage):
    """
    A storage backend that uses SQLite for persistence.

    This class implements the BaseStorage interface to provide a durable, file-based
    storage solution for HPO studies.

    Args:
        database_url (str): The path to the SQLite database file.
    """
    def __init__(self, database_url: str):
        if database_url.startswith("sqlite:///"):
            self.database_url = database_url[len("sqlite:///"):]
        else:
            self.database_url = database_url
        self._local = threading.local()
        self._init_db()

    def _get_conn(self):
        """Establishes and returns a database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.database_url, timeout=10)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        conn = self._get_conn()
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_name TEXT NOT NULL UNIQUE,
                    direction TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    value REAL,
                    FOREIGN KEY (study_id) REFERENCES studies (study_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trial_params (
                    param_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id INTEGER NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value TEXT NOT NULL, -- Stored as JSON string
                    FOREIGN KEY (trial_id) REFERENCES trials (trial_id)
                )
            """)

    def create_study(self, study_name: str, direction: str) -> int:
        conn = self._get_conn()
        try:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO studies (study_name, direction) VALUES (?, ?)",
                    (study_name, direction)
                )
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Study name already exists, return its ID instead.
            return self.get_study_id(study_name)

    def get_study_id(self, study_name: str) -> Optional[int]:
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "SELECT study_id FROM studies WHERE study_name = ?",
                (study_name,)
            )
            row = cursor.fetchone()
            return row['study_id'] if row else None

    def get_study_direction(self, study_id: int) -> Optional[str]:
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "SELECT direction FROM studies WHERE study_id = ?",
                (study_id,)
            )
            row = cursor.fetchone()
            return row['direction'] if row else None

    def create_trial(self, study_id: int) -> int:
        conn = self._get_conn()
        with conn:
            cursor = conn.execute(
                "INSERT INTO trials (study_id, state) VALUES (?, ?)",
                (study_id, 'RUNNING') # Default state
            )
            return cursor.lastrowid

    def set_trial_params(self, trial_id: int, params: Dict[str, Any]) -> None:
        conn = self._get_conn()
        with conn:
            for name, value in params.items():
                conn.execute(
                    "INSERT INTO trial_params (trial_id, param_name, param_value) VALUES (?, ?, ?)",
                    (trial_id, name, json.dumps(value))
                )

    def update_trial(self, trial_id: int, state: str, value: Optional[float] = None) -> None:
        conn = self._get_conn()
        with conn:
            conn.execute(
                "UPDATE trials SET state = ?, value = ? WHERE trial_id = ?",
                (state, value, trial_id)
            )

    def get_all_trials(self, study_id: int) -> List[Trial]:
        conn = self._get_conn()
        with conn:
            trials_cursor = conn.execute(
                "SELECT trial_id, state, value FROM trials WHERE study_id = ?",
                (study_id,)
            )
            trials_rows = trials_cursor.fetchall()

            results = []
            for trial_row in trials_rows:
                trial_id = trial_row['trial_id']

                params_cursor = conn.execute(
                    "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
                    (trial_id,)
                )
                params = {row['param_name']: json.loads(row['param_value']) for row in params_cursor.fetchall()}

                trial = Trial(
                    trial_id=trial_id,
                    study_id=study_id,
                    state=TrialState(trial_row['state']),
                    value=trial_row['value'],
                    params=params
                )
                results.append(trial)

            return results