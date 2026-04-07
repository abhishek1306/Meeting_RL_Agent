import random
import numpy as np
from collections import defaultdict
from typing import Callable, Optional

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv, TIME_SLOTS
from server.graders import grade_task1, grade_task2, grade_task3

NUM_ACTIONS = len(TIME_SLOTS) + 1  # 8 slot choices + 1 reject


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: dict = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.training_history: list[dict] = []

    # ── State encoding ──────────────────────────────────────────────

    @staticmethod
    def encode_state(obs: Observation) -> tuple:
        """Convert an Observation into a hashable state key."""
        num_pending = len(obs.pending_meetings)
        num_scheduled = len(obs.scheduled_meetings)
        num_available = len(obs.available_slots)
        conflicts = min(obs.conflicts, 5)

        if obs.pending_meetings:
            m = obs.pending_meetings[0]
            priority = m.priority
            pref_avail = 1 if m.preferred_slot in obs.available_slots else 0
            # Encode preferred slot index for richer state
            pref_idx = TIME_SLOTS.index(m.preferred_slot) if m.preferred_slot in TIME_SLOTS else -1
        else:
            priority = 0
            pref_avail = 0
            pref_idx = -1

        return (num_pending, num_scheduled, num_available, conflicts, priority, pref_avail, pref_idx)

    # ── Action selection ────────────────────────────────────────────

    def select_action(self, obs: Observation) -> int:
        """Epsilon-greedy action selection."""
        state = self.encode_state(obs)
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q_table[state]))

    @staticmethod
    def action_index_to_action(index: int, meeting_id: int) -> Action:
        """Convert a discrete action index to an Action object."""
        if index < len(TIME_SLOTS):
            return Action(action_type="schedule", meeting_id=meeting_id, time_slot=TIME_SLOTS[index])
        return Action(action_type="reject", meeting_id=meeting_id, time_slot=None)

    # ── Learning ────────────────────────────────────────────────────

    def update(self, state: tuple, action_idx: int, reward: float, next_state: tuple):
        """Standard Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        current = self.q_table[state][action_idx]
        self.q_table[state][action_idx] = current + self.lr * (
            reward + self.gamma * best_next - current
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Training loop ───────────────────────────────────────────────

    def train_episode(self, env: MeetingSchedulingEnv) -> dict:
        """Run one full episode and return episode metrics."""
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        episode_actions = []

        while not env.done and obs.pending_meetings:
            state = self.encode_state(obs)
            action_idx = self.select_action(obs)

            meeting = obs.pending_meetings[0]
            action = self.action_index_to_action(action_idx, meeting.meeting_id)

            result = env.step(action)
            next_state = self.encode_state(result.observation)

            # Q-learning update
            self.update(state, action_idx, result.reward, next_state)

            total_reward += result.reward
            steps += 1
            obs = result.observation

            episode_actions.append({
                "step": steps,
                "action_type": action.action_type,
                "meeting_id": action.meeting_id,
                "meeting_title": meeting.title,
                "time_slot": action.time_slot,
                "reward": result.reward,
                "info": result.info,
            })

        self.decay_epsilon()

        scheduled_count = len(env.scheduled_meetings)
        total = env.num_meetings

        # Build the scheduled meetings map with titles
        schedule_map = {}
        for mid, slot in env.scheduled_meetings.items():
            # Find the meeting title from history
            title = f"Meeting {mid}"
            for a in episode_actions:
                if a["meeting_id"] == mid and a["action_type"] == "schedule":
                    title = a["meeting_title"]
                    break
            schedule_map[slot] = {"meeting_id": mid, "title": title}

        episode_data = {
            "total_reward": round(total_reward, 3),
            "steps": steps,
            "scheduled": scheduled_count,
            "total_meetings": total,
            "conflicts": env.conflicts,
            "preference_matches": env.preference_matches,
            "epsilon": round(self.epsilon, 5),
            "task1_score": grade_task1(scheduled_count, total),
            "task2_score": grade_task2(env.conflicts, total),
            "task3_score": grade_task3(env.preference_matches, total),
            "actions": episode_actions,
            "schedule_map": schedule_map,
            "q_table_size": len(self.q_table),
        }
        return episode_data
