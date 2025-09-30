import numpy as np
import joblib
from typing import List, Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class MetaLearner:
    """
    MetaLearner predicts expected score for (meta_features, config_vector).
    It can be used to rank candidate configs for warm-start.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=6)
        self.trained = False

    def fit(self, meta_vectors: np.ndarray, config_vectors: np.ndarray, scores: np.ndarray):
        X = np.hstack([meta_vectors, config_vectors])
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, scores)
        self.trained = True

    def predict(self, meta_vector: np.ndarray, config_vectors: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("MetaLearner not trained")
        X = np.hstack([np.repeat(meta_vector.reshape(1,-1), len(config_vectors), axis=0), config_vectors])
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)

    def save(self, path: str):
        joblib.dump((self.scaler, self.model), path)

    def load(self, path: str):
        self.scaler, self.model = joblib.load(path)
        self.trained = True