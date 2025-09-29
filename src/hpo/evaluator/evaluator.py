from typing import Dict, Any, Optional
import time
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluator.train_and_eval(model, dataset, budget) -> metrics dict"""
    @staticmethod
    def train_and_eval(model, dataset: Dict[str, Any], budget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_val = dataset.get('X_val')
        y_val = dataset.get('y_val')

        # apply sample fraction budget
        if budget and 'sample_fraction' in budget:
            frac = float(budget['sample_fraction'])
            n = max(1, int(len(X_train) * frac))
            idx = np.random.choice(len(X_train), n, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Fit (synchronous). For keras, use budget['epochs'] if provided
        model.fit(X_train, y_train)

        results = {}
        if X_val is not None and y_val is not None:
            preds = model.predict(X_val)
            try:
                acc = float(accuracy_score(y_val, preds))
                results['accuracy'] = acc
            except Exception:
                pass

        results['train_time_s'] = time.time() - t0
        # model size: best-effort
        try:
            import pickle
            results['model_size_bytes'] = len(pickle.dumps(model))
        except Exception:
            results['model_size_bytes'] = None

        return results