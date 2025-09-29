from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseStorage(ABC):
    """
    Abstract base class for storage backends.

    This class defines the interface for persisting and retrieving study data,
    including study details, trial information, and hyperparameters.
    """

    @abstractmethod
    def create_study(self, study_name: str, direction: str) -> int:
        """
        Creates a new study in the backend.

        Args:
            study_name: The name of the study.
            direction: The optimization direction ('minimize' or 'maximize').

        Returns:
            The unique ID of the newly created study.
        """
        pass

    @abstractmethod
    def get_study_id(self, study_name: str) -> Optional[int]:
        """
        Retrieves the ID of a study given its name.

        Args:
            study_name: The name of the study.

        Returns:
            The study ID if found, otherwise None.
        """
        pass

    @abstractmethod
    def get_study_direction(self, study_id: int) -> Optional[str]:
        """
        Retrieves the direction of a study given its ID.

        Args:
            study_id: The ID of the study.

        Returns:
            The study direction if found, otherwise None.
        """
        pass

    @abstractmethod
    def create_trial(self, study_id: int) -> int:
        """
        Creates a new trial for a given study.

        Args:
            study_id: The ID of the study to which the trial belongs.

        Returns:
            The unique ID of the newly created trial.
        """
        pass

    @abstractmethod
    def set_trial_params(self, trial_id: int, params: Dict[str, Any]) -> None:
        """
        Sets the hyperparameters for a given trial.

        Args:
            trial_id: The ID of the trial.
            params: The dictionary of hyperparameters to set.
        """
        pass

    @abstractmethod
    def update_trial(self, trial_id: int, state: str, value: Optional[float] = None) -> None:
        """
        Updates the state and objective value of a trial.

        Args:
            trial_id: The ID of the trial to update.
            state: The new state of the trial (e.g., 'COMPLETE', 'PRUNED').
            value: The final objective value of the trial, if applicable.
        """
        pass

    @abstractmethod
    def get_all_trials(self, study_id: int) -> List[Any]:
        """
        Retrieves all trials associated with a study.

        Args:
            study_id: The ID of the study.

        Returns:
            A list of trial objects.
        """
        pass