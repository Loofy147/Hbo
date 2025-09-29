import sqlite3
import json
import numpy as np
from typing import Dict, Any, List, Tuple
import hashlib
import time

class MetaDB:
    def __init__(self, path='hpo_meta.db'):
        self.path = path
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as conn:
            cur = conn.cursor()
            cur.execute('''CREATE TABLE IF NOT EXISTS studies (study_id TEXT PRIMARY KEY, name TEXT UNIQUE, meta JSON, created_at INTEGER)''')
            cur.execute('''CREATE TABLE IF NOT EXISTS best_configs (id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT, config JSON, metrics JSON, vector BLOB, created_at INTEGER)''')
            conn.commit()

    def insert_study(self, name: str, meta: Dict[str,Any]) -> str:
        study_id = 's_'+hashlib.md5((name+str(time.time())).encode()).hexdigest()[:8]
        with sqlite3.connect(self.path) as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO studies (study_id,name,meta,created_at) VALUES (?,?,?,?)', (study_id,name,json.dumps(meta),int(time.time())))
            conn.commit()
        return study_id

    def insert_best_config(self, study_id: str, config: Dict[str,Any], metrics: Dict[str,Any], meta_vector: List[float]):
        vec = np.array(meta_vector,dtype=np.float32).tobytes()
        with sqlite3.connect(self.path) as conn:
            cur = conn.cursor()
            cur.execute('INSERT INTO best_configs (study_id,config,metrics,vector,created_at) VALUES (?,?,?,?,?)', (study_id,json.dumps(config),json.dumps(metrics),vec,int(time.time())))
            conn.commit()

    def _read_vectors(self) -> List[Tuple[int,str,bytes]]:
        with sqlite3.connect(self.path) as conn:
            cur = conn.cursor()
            cur.execute('SELECT id,study_id,vector FROM best_configs')
            return cur.fetchall()

    def find_similar_studies(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str,float]]:
        q = np.array(query_vector,dtype=np.float32)
        rows = self._read_vectors()
        sims = []
        for id_, study_id, vec in rows:
            v = np.frombuffer(vec,dtype=np.float32)
            # cosine
            denom = (np.linalg.norm(q) * np.linalg.norm(v))
            score = float(np.dot(q, v) / denom) if denom>0 else 0.0
            sims.append((study_id, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def get_top_configs(self, study_id: str, k: int) -> List[Dict]:
        """
        Retrieves the top k configurations for a given study_id.
        NOTE: This is a placeholder implementation. A real implementation would
        need to define what "best" means (e.g., based on a specific metric).
        Here, we just retrieve the most recently added ones.
        """
        with sqlite3.connect(self.path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT config FROM best_configs WHERE study_id = ? ORDER BY id DESC LIMIT ?",
                (study_id, k)
            )
            rows = cur.fetchall()
            return [json.loads(row[0]) for row in rows]