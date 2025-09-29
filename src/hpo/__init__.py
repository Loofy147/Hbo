# hpo/__init__.py

__version__ = "2.0.0"

# Imports from the original, modular structure
from .study import Study
from .space import SearchSpace, TrialResult

# Imports from the monolithic legacy file to support old tests and provide functionality
try:
    from .legacy import (
        Parameter,
        ParameterType,
        Trial,
        OptimizationConfig,
        BaseOptimizer,
        BayesianOptimizer,
        TPEOptimizer,
        HyperbandOptimizer,
    )
except ImportError:
    # This might happen if legacy dependencies are not installed.
    # We can allow the package to be imported but the legacy features will fail at runtime.
    pass

# New generator and factory classes, as per the new architecture
from .generators.parameter_generator import ParameterGenerator
from .factories.model_factory import ModelFactory


__all__ = [
    # Existing
    "Study",
    "SearchSpace",
    "TrialResult",
    # From legacy
    "Parameter",
    "ParameterType",
    "Trial",
    "OptimizationConfig",
    "BaseOptimizer",
    "BayesianOptimizer",
    "TPEOptimizer",
    "HyperbandOptimizer",
    # New
    "ParameterGenerator",
    "ModelFactory",
]