"""
Meeting Scheduling Environment Inference Script for Meta Scaler OpenEnv Hackathon.

Strictly adheres to:
1. Pydantic Action/Observation typed models.
2. OpenAI Client usage (HF_TOKEN, MODEL_NAME, API_BASE_URL).
3. [START], [STEP], [END] logging formats.
"""

import os
import json
import random
from typing import Optional
from openai import OpenAI
import numpy as np

from server.models import Action, Observation
from server.environment import MeetingSchedulingEnv, TIME_SLOTS
from server.graders import grade_task1, grade_task2, grade_task3


class LLMAgent:
    """Agent that calls an LLM using the OpenAI Client."""

    def __init__(self):
        # 1. Strictly prioritize API_BASE_URL assigned by the LiteLLM Proxy grader
        self.api_base = os.environ.get("API_BASE_URL")
        if not self.api_base:
            self.api_base = "https://router.huggingface.co/v1"

        # 2. Strictly prioritize API_KEY assigned by the LiteLLM Proxy grader
        self.api_key = os.environ.get("API_KEY")
        if not self.api_key:
            self.api_key = os.environ.get("HF_TOKEN", "dummy-key")
            
        self.model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
        
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

    def select_action(self, obs: Observation) -> Action:
        """Query the LLM to choose an action based on the observation."""
        if not obs.pending_meetings:
            return Action(action_type="reject", meeting_id=0, time_slot=None)

        current_meeting = obs.pending_meetings[0]
        
        prompt = f"""You are a smart meeting scheduler AI agent.
        
You must schedule the following meeting:
- ID: {current_meeting.meeting_id}
- Title: {current_meeting.title}
- Preferred Slot: {current_meeting.preferred_slot}

Current Environment State:
- Available Slots: {obs.available_slots}

You must return a raw JSON object and nothing else.
Format:
{{
    "action_type": "schedule",
    "meeting_id": {current_meeting.meeting_id},
    "time_slot": "<CHOOSE FROM AVAILABLE SLOTS>"
}}
If no slots are available, set action_type to "reject" and time_slot to null.
"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw = response.choices[0].message.content
        
        # Extract JSON block if surrounded by markdown
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
            
        data = json.loads(raw.strip())
        return Action(
            action_type=data.get("action_type", "reject"),
            meeting_id=data.get("meeting_id", current_meeting.meeting_id),
            time_slot=data.get("time_slot")
        )


if __name__ == "__main__":
    ENV_NAME = "meeting_scheduling_env"
    
    # Extract Model Name for Logging
    ACTIVE_MODEL_NAME = os.getenv("MODEL_NAME", "LLMAgent_Fallback")

    # The 3 mandatory evaluation tasks
    TASKS = [
        {"id": 1, "name": "task1_basic_scheduling", "grader": grade_task1},
        {"id": 2, "name": "task2_conflict_resolution", "grader": grade_task2},
        {"id": 3, "name": "task3_preference_optimization", "grader": grade_task3}
    ]

    # Iterate through Hackathon Tasks
    for t_info in TASKS:
        TASK_NAME = t_info["name"]
        grader_func = t_info["grader"]
        
        # 1. [START] Log Format
        print(f"[START] task={TASK_NAME} env={ENV_NAME} model={ACTIVE_MODEL_NAME}", flush=True)

        num_meetings = 5
        env = MeetingSchedulingEnv(num_meetings=num_meetings)
        obs = env.reset()
        
        total_reward = 0.0
        steps_count = 0
        rewards_list = []
        success = True

        try:
            # Initialize the LLM API Client Agent dynamically to catch grader proxy exceptions
            agent = LLMAgent()
            
            # Safety loop max 20 steps
            while not env.done and obs.pending_meetings and steps_count < 20:
                steps_count += 1
                
                # Model Inference
                action = agent.select_action(obs)

                # Step Environment
                result = env.step(action)
                
                # Gather variables
                reward = result.reward
                is_done = env.done
                action_desc = f"{action.action_type}_{action.meeting_id}_{action.time_slot if action.time_slot else 'none'}"
                
                # 2. [STEP] Log Format
                print(f"[STEP] step={steps_count} action={action_desc} reward={reward:.2f} done={str(is_done).lower()} error=null", flush=True)

                total_reward += reward
                rewards_list.append(f"{reward:.2f}")
                obs = result.observation

        except Exception as e:
            success = False
            print(f"[STEP] step={steps_count + 1} action=error reward=0.00 done=true error={str(e)}", flush=True)

        # Calculate Grade strictly bounded 0.0 - 1.0 (Hackathon Req)
        if TASK_NAME == "task1_basic_scheduling":
            score = grade_task1(len(env.scheduled_meetings), num_meetings)
        elif TASK_NAME == "task2_conflict_resolution":
            score = grade_task2(env.conflicts, num_meetings)
        else:
            score = grade_task3(env.preference_matches, num_meetings)

        # Ensure score bounds
        final_score = max(0.0, min(1.0, score))

        # 3. [END] Log Format 
        # (Included calculated grader score as an integrated metric exactly mapping the sample)
        rewards_str = ",".join(rewards_list) if rewards_list else "0.00"
        print(f"[END] success={str(success).lower()} steps={steps_count} score={final_score:.3f} rewards={rewards_str}", flush=True)
