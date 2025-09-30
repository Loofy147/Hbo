from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


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