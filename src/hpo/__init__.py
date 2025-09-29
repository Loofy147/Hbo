# hpo/__init__.py

__version__ = "2.0.0"

# Expose the core user-facing classes from the new architecture
from .core.study import Study
from .core.parameter import SearchSpace
from .core.trial import Trial, TrialState, TrialPruned

__all__ = [
    "Study",
    "SearchSpace",
    "Trial",
    "TrialState",
    "TrialPruned",
]