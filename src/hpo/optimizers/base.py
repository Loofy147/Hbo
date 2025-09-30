"""
Defines the base interface for all HPO optimizers.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseOptimizer(ABC):
    """
    Abstract base class for all HPO optimizers.

    This class defines a standard interface for suggesting new hyperparameter
    configurations (`ask`) and reporting the results of completed trials (`tell`).
    """

    @abstractmethod
    def ask(self) -> Dict[str, Any]:
        """
        Suggest a new set of hyperparameters to evaluate.

        Returns:
            A dictionary representing the hyperparameter configuration to try.
        """
        pass

    @abstractmethod
    def tell(self, config: Dict[str, Any], result: float) -> None:
        """
        Report the result of a completed trial back to the optimizer.

        Args:
            config: The hyperparameter configuration that was evaluated.
            result: The value of the objective function for the given configuration.
        """
        pass