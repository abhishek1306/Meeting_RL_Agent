import asyncio
import os
import json
from typing import List, Optional

from openai import OpenAI

# Assuming these are provided by the environment package
from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_NAME   = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
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
Respond with ONLY a JSON object. No explanation.
Example: {"action_type": "schedule", "meeting_id": 0, "time_slot": "10AM"}"""

# ─────────────────────────────────────────────
# LOGGING (Strict STDOUT Format)
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
# LLM LOGIC
# ─────────────────────────────────────────────
def get_model_action(client: OpenAI, obs: Observation, used_slots: set) -> Action:
    if not obs.pending_meetings:
        return Action(action_type="reject", meeting_id=0, time_slot=None)

    meeting = obs.pending_meetings[0]
    available = [s for s in obs.available_slots if s not in used_slots]

    user_prompt = f"Schedule meeting {meeting.meeting_id} ({meeting.title}). Pref: {meeting.preferred_slot}. Available: {available}"

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
        text = (completion.choices[0].message.content or "").strip()
        
        # Clean potential markdown
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        data = json.loads(text[text.find("{"):text.rfind("}")+1])
        return Action(
            action_type=data.get("action_type", "schedule"),
            meeting_id=data.get("meeting_id", meeting.meeting_id),
            time_slot=data.get("time_slot")
        )
    except Exception as exc:
        # Fallback logic
        slot = meeting.preferred_slot if meeting.preferred_slot in available else (available[0] if available else None)
        return Action(action_type="schedule" if slot else "reject", meeting_id=meeting.meeting_id, time_slot=slot)

# ─────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────
async def run_task(client: OpenAI, task_name: str, grader) -> None:
    log_start(task=task_name, env=ENV_NAME, model=MODEL_NAME)

    # Initialize environment via Docker if IMAGE_NAME is present, otherwise direct init
    if IMAGE_NAME:
        env = await MeetingSchedulingEnv.from_docker_image(IMAGE_NAME)
    else:
        env = MeetingSchedulingEnv(num_meetings=5)

    rewards: List[float] = []
    used_slots: set = set()
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # Check if reset is async based on reference
        obs = await env.reset() if hasattr(env, 'from_docker_image') else env.reset()

        for step in range(1, 21):
            if env.done or not obs.pending_meetings:
                break

            action = get_model_action(client, obs, used_slots)
            if action.time_slot:
                used_slots.add(action.time_slot)

            # Step the environment
            result = await env.step(action) if hasattr(env, 'from_docker_image') else env.step(action)
            
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            
            action_str = f"{action.action_type}({action.meeting_id},{action.time_slot})"
            log_step(step=step, action=action_str, reward=reward, done=result.done, error=None)

            obs = result.observation
            if result.done:
                break

        # Calculate score using provided graders
        if task_name == "task1_basic_scheduling":
            score = grader(len(env.scheduled_meetings), 5)
        elif task_name == "task2_conflict_resolution":
            score = grader(env.conflicts, 5)
        else:
            score = grader(env.preference_matches, 5)

        score = min(max(score, 0.0), 1.0)
        success = score > 0.5

    finally:
        if hasattr(env, 'close'):
            if asyncio.iscoroutinefunction(env.close):
                await env.close()
            else:
                env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name, grader in TASKS:
        await run_task(client, task_name, grader)

if __name__ == "__main__":
    asyncio.run(main())
