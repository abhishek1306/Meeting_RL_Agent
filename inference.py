"""
Meeting Scheduling Environment Inference Script for Meta Scaler OpenEnv Hackathon.
STRICT COMPLIANCE:
- Uses ONLY API_BASE_URL, API_KEY, MODEL_NAME (no fallback)
- Uses OpenAI Client
- Uses responses.create() (MANDATORY for proxy tracking)
- Emits [START], [STEP], [END] logs exactly
"""

import os
import json
from openai import OpenAI

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv
from server.graders import grade_task1, grade_task2, grade_task3


# 🔥 LLM AGENT (STRICT PROXY USAGE)
class LLMAgent:
    def __init__(self):
    
        self.api_base = os.environ["API_BASE_URL"]
        self.api_key = os.environ["API_KEY"]
        self.model_name = os.environ["MODEL_NAME"]

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

    def select_action(self, obs: Observation) -> Action:
        if not obs.pending_meetings:
            return Action(action_type="reject", meeting_id=0, time_slot=None)

        meeting = obs.pending_meetings[0]

        prompt = f"""
You are a meeting scheduling agent.
Meeting:
ID: {meeting.meeting_id}
Preferred Slot: {meeting.preferred_slot}
Available Slots: {obs.available_slots}
Return ONLY JSON:
{{
  "action_type": "schedule",
  "meeting_id": {meeting.meeting_id},
  "time_slot": "<slot>"
}}
If no valid slot → reject.
"""

        try:
            # ✅ MANDATORY: responses API (proxy tracks this)
            response = self.client.responses.create(
                model=self.model_name,
                input=prompt,
                temperature=0.0
            )

            # ✅ SAFE extraction (robust across proxies)
            raw = ""
            try:
                raw = response.output[0].content[0].text
            except Exception:
                raw = str(response)

            raw = raw.strip()

            # clean markdown if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if "json" in raw:
                    raw = raw.replace("json", "", 1)

            data = json.loads(raw.strip())

            return Action(
                action_type=data.get("action_type", "schedule"),
                meeting_id=data.get("meeting_id", meeting.meeting_id),
                time_slot=data.get("time_slot", meeting.preferred_slot)
            )

        except Exception as e:
            
            print(f"[DEBUG] LLM error: {e}", flush=True)
            return Action(
                action_type="schedule",
                meeting_id=meeting.meeting_id,
                time_slot=meeting.preferred_slot
            )


# 🧾 LOGGING (STRICT FORMAT)

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True
    )




if __name__ == "__main__":
    ENV_NAME = "Meeting_RL_Agent"

   
    MODEL_NAME = os.environ["MODEL_NAME"]

    TASKS = [
        ("task1_basic_scheduling", grade_task1),
        ("task2_conflict_resolution", grade_task2),
        ("task3_preference_optimization", grade_task3)
    ]

    for task_name, grader in TASKS:

        # 1️⃣ START
        log_start(task_name, ENV_NAME, MODEL_NAME)

        env = MeetingSchedulingEnv(num_meetings=5)
        obs = env.reset()

        agent = LLMAgent()

        rewards = []
        steps = 0
        success = True

        try:
            while not env.done and obs.pending_meetings and steps < 20:
                steps += 1

                
                action = agent.select_action(obs)

                result = env.step(action)

                reward = result.reward
                done = result.done

                action_str = f"{action.action_type}_{action.meeting_id}_{action.time_slot if action.time_slot else 'none'}"

                # 2️⃣ STEP
                log_step(steps, action_str, reward, done)

                rewards.append(reward)
                obs = result.observation

                if done:
                    break

        except Exception as e:
            success = False
            print(f"[STEP] step={steps+1} action=error reward=0.00 done=true error={str(e)}", flush=True)

        
        if task_name == "task1_basic_scheduling":
            score = grader(len(env.scheduled_meetings), 5)
        elif task_name == "task2_conflict_resolution":
            score = grader(env.conflicts, 5)
        else:
            score = grader(env.preference_matches, 5)

        score = max(0.0, min(1.0, score))

        
        log_end(success, steps, score, rewards)
