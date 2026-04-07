"""
Data models for the Smart Meeting Scheduling Environment.
Follows the OpenEnv standard API specification.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class MeetingRequest(BaseModel):
    """Represents a meeting that needs to be scheduled."""
    meeting_id: int
    title: str
    duration: int  # in slots (1 = 1 hour)
    priority: int  # 1=low, 2=medium, 3=high
    preferred_slot: Optional[str] = None
    participants: List[str] = []

    def to_dict(self):
        return self.model_dump()


class Observation(BaseModel):
    """The environment state returned after each step."""
    pending_meetings: List[MeetingRequest]
    scheduled_meetings: Dict[str, str]   # meeting_id -> time_slot
    available_slots: List[str]
    conflicts: int
    step_count: int

    def to_dict(self):
        return self.model_dump()


class Action(BaseModel):
    """An action the agent can take."""
    action_type: str              # "schedule" or "reject"
    meeting_id: int
    time_slot: Optional[str] = None

    def to_dict(self):
        return self.model_dump()


class StepResult(BaseModel):
    """Result returned by env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class StateResult(BaseModel):
    """Result returned by env.state() for API."""
    observation: Observation
    total_reward: float
    done: bool
    info: Dict[str, Any]
