"""
Meeting Scheduling Environment - Phase 2 Safe Inference
GOAL:
- Never crash
- Always produce valid logs
- Deterministic scheduling (no LLM dependency)
STDOUT FORMAT (STRICT):
[START] task=<name> env=<env> model=<model>
[STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
from typing import List, Optional

from server.models import Action
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3


# =========================
# SAFE ENV CONFIG
# =========================
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

ENV_NAME = "Meeting_RL_Agent"


# =========================
# LOGGING (STRICT)
# =========================
def log_start(task: str):
    print(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )


# =========================
# CORE SCHEDULER (NO LLM)
# =========================
def select_action(obs, used_slots):
    try:
        if not obs.pending_meetings:
            return Action(action_type="reject", meeting_id=0, time_slot=None)

        meeting = obs.pending_meetings[0]

        # Remove already used slots
        available = [s for s in obs.available_slots if s not in used_slots]

        # Prefer preferred slot
        if meeting.preferred_slot in available:
            return Action(
                action_type="schedule",
                meeting_id=meeting.meeting_id,
                time_slot=meeting.preferred_slot
            )

        # Otherwise take first available
        if available:
            return Action(
                action_type="schedule",
                meeting_id=meeting.meeting_id,
                time_slot=available[0]
            )

        # No slots → reject
        return Action(
            action_type="reject",
            meeting_id=meeting.meeting_id,
            time_slot=None
        )

    except Exception:
        # Ultimate fallback (never crash)
        return Action(
            action_type="schedule",
            meeting_id=0,
            time_slot=None
        )


# =========================
# RUN TASK
# =========================
def run_task(task_name, grader):
    log_start(task_name)

    rewards = []
    steps = 0
    success = True
    used_slots = set()

    try:
        env = MeetingSchedulingEnv(num_meetings=5)
        obs = env.reset()

        while not env.done and obs.pending_meetings and steps < 20:
            steps += 1

            action = select_action(obs, used_slots)

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
    try:
        if task_name == "task1_basic_scheduling":
            score = grader(len(env.scheduled_meetings), 5)
        elif task_name == "task2_conflict_resolution":
            score = grader(env.conflicts, 5)
        else:
            score = grader(env.preference_matches, 5)

        score = max(0.1, min(0.9, score))

    except Exception:
        score = 0.0
        success = False

    log_end(success, steps, score, rewards)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    TASKS = [
        ("task1_basic_scheduling", grade_task1),
        ("task2_conflict_resolution", grade_task2),
        ("task3_preference_optimization", grade_task3),
    ]

    for task_name, grader in TASKS:
        run_task(task_name, grader)
