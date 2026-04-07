"""
Smart Meeting Scheduling Environment – OpenEnv compliant.

Simulates realistic meeting scheduling with constraints:
  - Limited time slots (8 one-hour slots: 9 AM – 4 PM)
  - Meeting priorities (1-low, 2-medium, 3-high)
  - User preferred time-slot preferences
  - Conflict detection

API:
  reset()       -> Observation
  step(action)  -> StepResult(observation, reward, done, info)
  state()       -> Observation
"""

import random
from .models import MeetingRequest, Observation, Action, StepResult
from .reward import calculate_reward

# ── Available time slots ────────────────────────────────────────────
TIME_SLOTS = ["9AM", "10AM", "11AM", "12PM", "1PM", "2PM", "3PM", "4PM"]

# ── Pool of realistic meeting templates ─────────────────────────────
MEETING_POOL = [
    {"title": "Sprint Planning",    "duration": 1, "priority": 3, "preferred_slot": "9AM"},
    {"title": "Design Review",      "duration": 1, "priority": 2, "preferred_slot": "10AM"},
    {"title": "1:1 Standup",        "duration": 1, "priority": 1, "preferred_slot": "9AM"},
    {"title": "Client Call",        "duration": 1, "priority": 3, "preferred_slot": "2PM"},
    {"title": "Team Lunch",         "duration": 1, "priority": 1, "preferred_slot": "12PM"},
    {"title": "Code Review",        "duration": 1, "priority": 2, "preferred_slot": "3PM"},
    {"title": "Retrospective",      "duration": 1, "priority": 2, "preferred_slot": "4PM"},
    {"title": "Tech Talk",          "duration": 1, "priority": 1, "preferred_slot": "11AM"},
    {"title": "Product Demo",       "duration": 1, "priority": 3, "preferred_slot": "10AM"},
    {"title": "Budget Review",      "duration": 1, "priority": 2, "preferred_slot": "1PM"},
    {"title": "Onboarding",         "duration": 1, "priority": 1, "preferred_slot": "11AM"},
    {"title": "Architecture Sync",  "duration": 1, "priority": 3, "preferred_slot": "3PM"},
]


class MeetingSchedulingEnv:
    """OpenEnv-compliant meeting scheduling environment."""

    def __init__(self, num_meetings: int = 5):
        self.num_meetings = num_meetings
        self.time_slots = TIME_SLOTS[:]
        # These are set by reset()
        self.pending_meetings: list[MeetingRequest] = []
        self.scheduled_meetings: dict[int, str] = {}
        self.available_slots: list[str] = []
        self.conflicts: int = 0
        self.preference_matches: int = 0
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.history: list[dict] = []
        # Keep original preference map for grading
        self._original_preferences: dict[int, str] = {}

    # ── OpenEnv API ─────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset the environment and generate a new set of meeting requests."""
        selected = random.sample(MEETING_POOL, min(self.num_meetings, len(MEETING_POOL)))

        self.pending_meetings = []
        self._original_preferences = {}
        for i, m in enumerate(selected):
            meeting = MeetingRequest(
                meeting_id=i,
                title=m["title"],
                duration=m["duration"],
                priority=m["priority"],
                preferred_slot=m["preferred_slot"],
                participants=[f"User{j}" for j in range(random.randint(2, 5))],
            )
            self.pending_meetings.append(meeting)
            self._original_preferences[i] = m["preferred_slot"]

        self.scheduled_meetings = {}
        self.available_slots = self.time_slots[:]
        self.conflicts = 0
        self.preference_matches = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.history = []

        return self.state()

    def state(self) -> Observation:
        """Return the current observation."""
        return Observation(
            pending_meetings=list(self.pending_meetings),
            scheduled_meetings={str(k): v for k, v in self.scheduled_meetings.items()},
            available_slots=list(self.available_slots),
            conflicts=self.conflicts,
            step_count=self.step_count,
        )

    def step(self, action: Action) -> StepResult:
        """Execute one action and return (observation, reward, done, info)."""
        if self.done:
            return StepResult(observation=self.state(), reward=0.0, done=True, info={"error": "Episode finished"})

        self.step_count += 1
        info: dict = {}

        # Find the target meeting
        meeting = None
        for m in self.pending_meetings:
            if m.meeting_id == action.meeting_id:
                meeting = m
                break

        if meeting is None:
            reward = calculate_reward("schedule", conflict=True)
            info["error"] = f"Meeting {action.meeting_id} not found"
            self._record(action, meeting, reward, info)
            return StepResult(observation=self.state(), reward=reward, done=self.done, info=info)

        scheduled = False
        conflict = False
        preferred_match = False

        if action.action_type == "schedule":
            if action.time_slot not in self.available_slots:
                # Slot already taken → conflict
                conflict = True
                self.conflicts += 1
                info["conflict"] = True
            else:
                # Successful scheduling
                scheduled = True
                self.scheduled_meetings[meeting.meeting_id] = action.time_slot
                self.available_slots.remove(action.time_slot)
                info["scheduled"] = True

                if action.time_slot == meeting.preferred_slot:
                    preferred_match = True
                    self.preference_matches += 1
                    info["preferred_match"] = True

            self.pending_meetings.remove(meeting)

        elif action.action_type == "reject":
            self.pending_meetings.remove(meeting)
            info["rejected"] = True

        reward = calculate_reward(
            action.action_type,
            scheduled=scheduled,
            conflict=conflict,
            preferred_match=preferred_match,
        )
        self.total_reward += reward

        # Episode ends when:
        # 1. Task completed (all meetings scheduled/rejected)
        # OR 2. Max steps reached (safety stop)
        if len(self.pending_meetings) == 0 or self.step_count >= 20:
            self.done = True

        self._record(action, meeting, reward, info)
        return StepResult(observation=self.state(), reward=reward, done=self.done, info=info)

    # ── Helpers ─────────────────────────────────────────────────────

    def _record(self, action: Action, meeting, reward: float, info: dict):
        self.history.append({
            "step": self.step_count,
            "action_type": action.action_type,
            "meeting_id": action.meeting_id,
            "meeting_title": meeting.title if meeting else "Unknown",
            "time_slot": action.time_slot,
            "reward": reward,
            "done": self.done,
            "info": info,
        })
