"""
Meeting Scheduling Environment - Inference Script
Meta Scaler OpenEnv Hackathon

Environment variables:
    API_BASE_URL  default: https://router.huggingface.co/v1
    MODEL_NAME    default: meta-llama/Meta-Llama-3-8B-Instruct
    HF_TOKEN      required (your Hugging Face / API key)

Stdout format:
    [START] task=<name> env=<env> model=<model>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
from typing import List, Optional

from openai import OpenAI

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3

# ─────────────────────────────────────────────
# CONFIG — use `or` so empty-string env vars fall back to defaults
# ─────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
ENV_NAME     = "meeting_scheduling_env"

TASKS = [
    ("task1_basic_scheduling",        grade_task1),
    ("task2_conflict_resolution",     grade_task2),
    ("task3_preference_optimization", grade_task3),
]

SYSTEM_PROMPT = """You are a smart meeting scheduling agent.

Your job is to assign meetings to time slots in a daily calendar.
You will receive the current meeting details and available time slots.
Always prefer the meeting's preferred_slot if it is available.
Respond with ONLY a JSON object — no explanation, no markdown.

Example response:
{"action_type": "schedule", "meeting_id": 0, "time_slot": "10AM"}
"""


# ─────────────────────────────────────────────
# LOGGING (strict format — matches sample script)
# ─────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ─────────────────────────────────────────────
# LLM CALL (matches sample pattern exactly)
# ─────────────────────────────────────────────
def get_model_action(client: OpenAI, obs: Observation, used_slots: set) -> Action:
    """Query the LLM through the grader proxy to choose a scheduling action."""
    if not obs.pending_meetings:
        return Action(action_type="reject", meeting_id=0, time_slot=None)

    meeting = obs.pending_meetings[0]
    available = [s for s in obs.available_slots if s not in used_slots]

    user_prompt = f"""Schedule this meeting:
  ID: {meeting.meeting_id}
  Title: {meeting.title}
  Preferred Slot: {meeting.preferred_slot}

Available Slots: {available}

Return ONLY JSON:
{{"action_type": "schedule", "meeting_id": {meeting.meeting_id}, "time_slot": "<slot>"}}

If no slots available, return:
{{"action_type": "reject", "meeting_id": {meeting.meeting_id}, "time_slot": null}}"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        import json
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    text = part
                    break

        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(text[start:end])
            return Action(
                action_type=data.get("action_type", "schedule"),
                meeting_id=data.get("meeting_id", meeting.meeting_id),
                time_slot=data.get("time_slot") or meeting.preferred_slot,
            )
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)

    # Fallback — only reached if LLM call or parse fails
    if meeting.preferred_slot in available:
        return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=meeting.preferred_slot)
    if available:
        return Action(action_type="schedule", meeting_id=meeting.meeting_id, time_slot=available[0])
    return Action(action_type="reject", meeting_id=meeting.meeting_id, time_slot=None)


# ─────────────────────────────────────────────
# TASK RUNNER (mirrors sample script's finally pattern)
# ─────────────────────────────────────────────
def run_task(client: OpenAI, task_name: str, grader) -> None:
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    env = MeetingSchedulingEnv(num_meetings=5)
    obs = env.reset()

    rewards: List[float] = []
    used_slots: set = set()
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, 21):  # max 20 steps
            if env.done or not obs.pending_meetings:
                break

            action = get_model_action(client, obs, used_slots)
            if action.time_slot:
                used_slots.add(action.time_slot)

            result = env.step(action)
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action.action_type}_{action.meeting_id}_{action.time_slot or 'none'}"
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            obs = result.observation
            if done:
                break

        # Calculate score
        if task_name == "task1_basic_scheduling":
            score = grader(len(env.scheduled_meetings), 5)
        elif task_name == "task2_conflict_resolution":
            score = grader(env.conflicts, 5)
        else:
            score = grader(env.preference_matches, 5)

        score = min(max(score, 0.0), 1.0)
        success = score > 0.5

    finally:
        # [END] always emitted — even on exception (matches sample script)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ─────────────────────────────────────────────
# MAIN (matches sample script pattern exactly)
# ─────────────────────────────────────────────
def main() -> None:
    # Single client created with grader-injected credentials
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_name, grader in TASKS:
        run_task(client, task_name, grader)


if __name__ == "__main__":
    main()
