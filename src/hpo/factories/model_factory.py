from __future__ import annotations
from typing import Dict, Any

# Optional imports
try:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
except Exception:
    Pipeline = None

class ModelFactory:
    def __init__(self, template_name: str):
        self.template_name = template_name

    def build(self, params: Dict[str, Any]):
        if self.template_name == 'sklearn_rf_pipeline':
            if Pipeline is None:
                raise RuntimeError('sklearn not installed')
            rf = RandomForestClassifier(
                n_estimators=int(params.get('n_estimators', 200)),
                max_depth=None if params.get('max_depth') is None else int(params.get('max_depth')),
                min_samples_leaf=int(params.get('min_samples_leaf',1)),
                max_features=params.get('max_features','sqrt'),
                n_jobs=1
            )
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', rf)])
            return pipe
        if self.template_name == 'xgboost_tree':
            import xgboost as xgb
            return xgb.XGBClassifier(**params)
        if self.template_name == 'lightgbm':
            import lightgbm as lgb
            return lgb.LGBMClassifier(**params)
        if self.template_name == 'keras_mlp':
            # This template is a placeholder. The user is expected to implement
            # the Keras builder in their own environment to avoid making Keras
            # a hard dependency for this library.
            raise NotImplementedError('Implement Keras builder in your environment. TensorFlow is not a direct dependency.')
            # The following lines would only be reached if the above error is removed.
            # from tensorflow.keras import Model
            # ... actual implementation would go here
        raise NotImplementedError(f"Unknown template {self.template_name}")