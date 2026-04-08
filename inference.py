"""
Meeting Scheduling Environment - Inference Script
Meta Scaler OpenEnv Hackathon

MANDATORY ENV VARIABLES (injected by grader):
    API_BASE_URL  - LiteLLM proxy endpoint
    API_KEY       - LiteLLM proxy key
    MODEL_NAME    - Model identifier

STDOUT FORMAT (STRICT):
    [START] task=<name> env=<env> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
from typing import List, Optional
from openai import OpenAI

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3

# ─────────────────────────────────────────────
# ENV CONFIG — strictly use grader-injected vars
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY      = os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
ENV_NAME     = "Meeting_RL_Agent"


# ─────────────────────────────────────────────
# LOGGING (strict format)
# ─────────────────────────────────────────────
def log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
# LLM AGENT (uses grader proxy — MANDATORY)
# ─────────────────────────────────────────────
class LLMAgent:
    def __init__(self):
        # MUST use grader-provided credentials — do not substitute
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )

    def _parse_action(self, raw: str, meeting_id: int, preferred_slot: str, available: list) -> Action:
        """Extract JSON from LLM response with robust fallback."""
        try:
            text = raw.strip()
            # Strip markdown code fences if present
            if "```" in text:
                for part in text.split("```"):
                    part = part.strip().lstrip("json").strip()
                    if part.startswith("{"):
                        text = part
                        break
            # Extract JSON object
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]
            data = json.loads(text)
            return Action(
                action_type=data.get("action_type", "schedule"),
                meeting_id=data.get("meeting_id", meeting_id),
                time_slot=data.get("time_slot", preferred_slot),
            )
        except Exception:
            # JSON parse failed — use preferred slot if free, else first available
            if preferred_slot in available:
                return Action(action_type="schedule", meeting_id=meeting_id, time_slot=preferred_slot)
            if available:
                return Action(action_type="schedule", meeting_id=meeting_id, time_slot=available[0])
            return Action(action_type="reject", meeting_id=meeting_id, time_slot=None)

    def select_action(self, obs: Observation, used_slots: set) -> Action:
        if not obs.pending_meetings:
            return Action(action_type="reject", meeting_id=0, time_slot=None)

        meeting = obs.pending_meetings[0]
        available = [s for s in obs.available_slots if s not in used_slots]

        prompt = f"""You are a meeting scheduling agent.

Meeting to schedule:
  ID: {meeting.meeting_id}
  Title: {meeting.title}
  Preferred Slot: {meeting.preferred_slot}

Available Slots: {available}

Rules:
  1. Only choose a slot from Available Slots.
  2. Prefer the preferred_slot if it appears in Available Slots.
  3. If no slots are available, reject.

Respond with ONLY a JSON object, no explanation:
{{
  "action_type": "schedule",
  "meeting_id": {meeting.meeting_id},
  "time_slot": "<chosen_slot>"
}}"""

        # Try LLM twice — this registers traffic through the grader proxy
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                raw = response.choices[0].message.content or ""
                return self._parse_action(raw, meeting.meeting_id, meeting.preferred_slot, available)
            except Exception:
                if attempt == 1:
                    break
                continue

        # Last-resort offline fallback (only if proxy is completely unreachable)
        if meeting.preferred_slot in available:
            return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=meeting.preferred_slot)
        if available:
            return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=available[0])
        return Action(action_type="reject", meeting_id=meeting.meeting_id, time_slot=None)


# ─────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────
def run_task(task_name: str, grader):
    log_start(task_name)

    env = MeetingSchedulingEnv(num_meetings=5)
    obs = env.reset()
    agent = LLMAgent()

    rewards: List[float] = []
    used_slots: set = set()
    steps = 0
    success = True

    try:
        while not env.done and obs.pending_meetings and steps < 20:
            steps += 1

            action = agent.select_action(obs, used_slots)
            if action.time_slot:
                used_slots.add(action.time_slot)

            result = env.step(action)
            reward = result.reward
            done = result.done
            action_str = f"{action.action_type}_{action.meeting_id}_{action.time_slot or 'none'}"

            log_step(steps, action_str, reward, done, None)
            rewards.append(reward)
            obs = result.observation

            if done:
                break

    except Exception as e:
        success = False
        log_step(steps + 1, "error", 0.0, True, str(e))

    # Score
    try:
        if task_name == "task1_basic_scheduling":
            score = grader(len(env.scheduled_meetings), 5)
        elif task_name == "task2_conflict_resolution":
            score = grader(env.conflicts, 5)
        else:
            score = grader(env.preference_matches, 5)
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.0
        success = False

    log_end(success, steps, score, rewards)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TASKS = [
        ("task1_basic_scheduling",       grade_task1),
        ("task2_conflict_resolution",    grade_task2),
        ("task3_preference_optimization", grade_task3),
    ]
    for task_name, grader in TASKS:
        run_task(task_name, grader)
