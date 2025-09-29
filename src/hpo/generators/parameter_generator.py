from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ParamSpec:
    name: str
    ptype: str  # 'int','float','log_float','categorical','bool'
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    prior: Optional[str] = None  # 'uniform','log','normal'...
    conditional_on: Optional[Dict[str, Any]] = None

class ParameterGenerator:
    """Generate parameter search spaces from dataset metadata and model templates."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def _heuristic_for_template(template_name: str, dataset_meta: Dict[str, Any]) -> List[ParamSpec]:
        # Rules-based defaults (expandable)
        if template_name == "sklearn_rf_pipeline":
            return [
                ParamSpec('n_estimators','int',(50,1000),default=200,prior='uniform'),
                ParamSpec('max_depth','int',(3,50),default=None,prior='uniform'),
                ParamSpec('min_samples_leaf','int',(1,20),default=1,prior='uniform'),
                ParamSpec('max_features','categorical',choices=['sqrt','log2', None],default='sqrt')
            ]
        if template_name == 'keras_mlp':
            n_features = dataset_meta.get('n_features', 10)
            return [
                ParamSpec('hidden_layers','int',(1,5),default=2,prior='uniform'),
                ParamSpec('hidden_size','int',(max(8, n_features), min(2048, n_features*32)),default=64,prior='log'),
                ParamSpec('dropout','float',(0.0,0.6),default=0.1,prior='uniform'),
                ParamSpec('lr','log_float',(1e-5,1e-1),default=1e-3,prior='log')
            ]
        # fallback
        return []

    def suggest_search_space(self, template_name: str, dataset_meta: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        specs = self._heuristic_for_template(template_name, dataset_meta)
        out = {}
        for s in specs:
            item = { 'type': s.ptype }
            if s.bounds is not None:
                item['bounds'] = s.bounds
            if s.choices is not None:
                item['choices'] = s.choices
            if s.default is not None:
                item['default'] = s.default
            if s.prior is not None:
                item['prior'] = s.prior
            if s.conditional_on is not None:
                item['conditional_on'] = s.conditional_on
            out[s.name] = item
        return out

    def sample_from_spec(self, spec: Dict[str, Any]) -> Any:
        t = spec['type']
        if t == 'int':
            low, high = spec['bounds']
            return int(self.rng.randint(low, high+1))
        if t == 'float':
            low, high = spec['bounds']
            return float(self.rng.uniform(low, high))
        if t == 'log_float':
            low, high = spec['bounds']
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        if t == 'categorical':
            return self.rng.choice(spec['choices'])
        if t == 'bool':
            return bool(self.rng.choice([False, True]))
        raise ValueError(f"Unknown type {t}")