"""
A simple random sampling optimizer.
"""
from __future__ import annotations
from typing import Dict, Any, List

from .base import BaseOptimizer
from ..study import SearchSpace # Assuming SearchSpace will be in study.py

class RandomSampler(BaseOptimizer):
    """
    A simple sampler that suggests hyperparameters completely at random.

    This sampler is useful for establishing a baseline or for the initial
    startup phase of an optimization process.
    """
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def ask(self) -> Dict[str, Any]:
        """
        Suggests a new set of hyperparameters by sampling randomly from the search space.
        """
        return self.search_space.sample()

    def tell(self, config: Dict[str, Any], result: float) -> None:
        """
        This method is a no-op for the RandomSampler, as it does not learn
        from past results.
        """
        pass