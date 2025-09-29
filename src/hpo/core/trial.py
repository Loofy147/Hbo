import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional

class TrialState(Enum):
    """
    Represents the state of a trial.
    """
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAILED = "FAILED"

@dataclass
class Trial:
    """
    A dataclass representing a single trial in an optimization study.

    Attributes:
        trial_id: The unique identifier for the trial.
        study_id: The ID of the study this trial belongs to.
        state: The current state of the trial (e.g., RUNNING, COMPLETE).
        value: The objective value obtained by the trial.
        params: A dictionary of hyperparameters being evaluated.
        start_time: The time when the trial started.
        end_time: The time when the trial completed.
    """
    trial_id: int
    study_id: int
    state: TrialState
    value: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None


class TrialPruned(Exception):
    """Exception to indicate that a trial was pruned."""
    pass