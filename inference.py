"""
Meeting Scheduling Environment - Inference Script
Meta Scaler OpenEnv Hackathon

Required environment variables (injected by grader):
    API_BASE_URL  — The LiteLLM proxy endpoint
    API_KEY       — The LiteLLM proxy key
    MODEL_NAME    — The model identifier to use

Emits structured [START], [STEP], [END] logs to stdout.
"""

import os
import json
from typing import List, Optional

from openai import OpenAI

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3

# ─────────────────────────────────────────────
# CONFIG — use grader-injected variables
# ─────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_NAME     = "meeting_scheduling_env"

# openai>=1.x requires base_url to end with "/"
if API_BASE_URL and not API_BASE_URL.endswith("/"):
    API_BASE_URL = API_BASE_URL + "/"

TASKS = [
    ("task1_basic_scheduling",        grade_task1),
    ("task2_conflict_resolution",     grade_task2),
    ("task3_preference_optimization", grade_task3),
]

# ─────────────────────────────────────────────
# LOGGING
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
# LLM CALL — direct, no class wrapping
# ─────────────────────────────────────────────
def get_action(client: OpenAI, obs: Observation, used_slots: set) -> Action:
    """Query the LLM through the grader proxy and parse the response."""
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
  3. If no slots are available, use action_type "reject".

Respond with ONLY a JSON object, no explanation:
{{
  "action_type": "schedule",
  "meeting_id": {meeting.meeting_id},
  "time_slot": "<chosen_slot>"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Parse JSON from response
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    raw = part
                    break
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw[start:end])
            return Action(
                action_type=data.get("action_type", "schedule"),
                meeting_id=data.get("meeting_id", meeting.meeting_id),
                time_slot=data.get("time_slot", meeting.preferred_slot),
            )
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)

    # Fallback only if LLM call/parse fails
    if meeting.preferred_slot in available:
        return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=meeting.preferred_slot)
    if available:
        return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=available[0])
    return Action(action_type="reject", meeting_id=meeting.meeting_id, time_slot=None)


# ─────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str, grader):
    log_start(task_name)

    env = MeetingSchedulingEnv(num_meetings=5)
    obs = env.reset()

    rewards: List[float] = []
    used_slots: set = set()
    steps = 0
    success = True

    try:
        while not env.done and obs.pending_meetings and steps < 20:
            steps += 1

            action = get_action(client, obs, used_slots)
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
def main():
    # Single client — created once with grader-injected credentials
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )
    except Exception as e:
        print(f"[DEBUG] OpenAI client init error: {e}", flush=True)
        # Re-raise so grader sees explicit failure message instead of raw traceback
        raise RuntimeError(f"Failed to initialize OpenAI client with API_BASE_URL={API_BASE_URL!r}: {e}") from e

    for task_name, grader in TASKS:
        run_task(client, task_name, grader)


if __name__ == "__main__":
    main()
