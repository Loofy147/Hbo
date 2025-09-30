# hpo/__init__.py

__version__ = "3.0.0"  # Bumping version due to major refactoring

# Expose the core user-facing classes from the new architecture
from .configuration import HyperParameter, ConfigurationSpace
from .database import ExperimentDatabase
from .orchestrator import EnhancedBayesianOptimizer, BOHBOptimizer
from .warmstart.warm_start_manager import WarmStartManager
from .meta.meta_learner import MetaLearner
from .optimizers.bohb_kde import BOHB_KDE

__all__ = [
    "HyperParameter",
    "ConfigurationSpace",
    "ExperimentDatabase",
    "EnhancedBayesianOptimizer",
    "BOHBOptimizer",
    "WarmStartManager",
    "MetaLearner",
    "BOHB_KDE",
]