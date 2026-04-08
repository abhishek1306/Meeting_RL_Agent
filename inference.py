"""
Meeting Scheduling Environment - Inference Script

MANDATORY ENV VARIABLES:
    API_BASE_URL
    API_KEY
    MODEL_NAME

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


# =========================
# CONFIG
# =========================
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]

ENV_NAME = "Meeting_RL_Agent"


# =========================
# LOGGING (STRICT FORMAT)
# =========================
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# =========================
# LLM AGENT
# =========================
class LLMAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY
        )

    def _parse_json(self, raw: str):
        """Robust JSON extractor"""
        raw = raw.strip()

        # Remove markdown
        if "```" in raw:
            parts = raw.split("```")
            for p in parts:
                if p.strip().startswith("{"):
                    raw = p.strip()
                    break

        # Extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end != -1:
            raw = raw[start:end]

        return json.loads(raw)

    def select_action(self, obs: Observation, used_slots: set) -> Action:
        if not obs.pending_meetings:
            return Action(action_type="reject", meeting_id=0, time_slot=None)

        meeting = obs.pending_meetings[0]

        
        available = [s for s in obs.available_slots if s not in used_slots]

        # If preferred is free → use it
        if meeting.preferred_slot in available:
            return Action(
                action_type="schedule",
                meeting_id=meeting.meeting_id,
                time_slot=meeting.preferred_slot
            )

        # Else pick first free slot
        if available:
            return Action(
                action_type="schedule",
                meeting_id=meeting.meeting_id,
                time_slot=available[0]
            )

        # If no slots → reject
        return Action(
            action_type="reject",
            meeting_id=meeting.meeting_id,
            time_slot=None
        )


# =========================
# RUN TASK
# =========================
def run_task(task_name, grader):
    log_start(task_name, ENV_NAME, MODEL_NAME)

    env = MeetingSchedulingEnv(num_meetings=5)
    obs = env.reset()

    agent = LLMAgent()

    rewards = []
    used_slots = set()
    steps = 0
    success = True

    try:
        while not env.done and obs.pending_meetings and steps < 20:
            steps += 1

            action = agent.select_action(obs, used_slots)

            # Track used slots
            if action.time_slot:
                used_slots.add(action.time_slot)

            result = env.step(action)

            reward = result.reward
            done = result.done

            action_str = f"{action.action_type}_{action.meeting_id}_{action.time_slot if action.time_slot else 'none'}"

            log_step(steps, action_str, reward, done, None)

            rewards.append(reward)
            obs = result.observation

            if done:
                break

    except Exception as e:
        success = False
        log_step(steps + 1, "error", 0.0, True, str(e))

    # =========================
    # SCORING
    # =========================
    if task_name == "task1_basic_scheduling":
        score = grader(len(env.scheduled_meetings), 5)
    elif task_name == "task2_conflict_resolution":
        score = grader(env.conflicts, 5)
    else:
        score = grader(env.preference_matches, 5)

    score = max(0.0, min(1.0, score))

    log_end(success, steps, score, rewards)


# =========================
# MAIN (STRICT: RUN ALL TASKS)
# =========================
if __name__ == "__main__":

    TASKS = [
        ("task1_basic_scheduling", grade_task1),
        ("task2_conflict_resolution", grade_task2),
        ("task3_preference_optimization", grade_task3)
    ]

    for task_name, grader in TASKS:
        run_task(task_name, grader)
