# Smart Meeting Scheduler (OpenEnv Hackathon)

## Environment Description and Motivation
This is a stringent test-bed of reinforcement learning agents and LLM planners. It is a well-known, combinatorical optimization issue to plan meetings between cross-functional teams whose schedules are overlapping and who have constrained resources and user preferences that are complex.

That is precisely the constraint that is simulated in this environment: the agent has to sift a dynamic queue of incoming meetings and allocate them safely to an 8-slot per, day calendar. Planners are strongly suspected, in regard, to their ability of distinguishing the phenomenon of double-booking, beautiful cancellation of schedules as well overriding user satisfaction by a respectful practice of making assignments to the most suitable moments.

---

## Action and Observation Space Definitions
The environment interfaces rigidly via standardized JSON schemas mapped internally.

**Observation Space**: Discrete State Dictionary
*   `pending_meetings` (List): A real-time queue of meetings yet to be assigned.
*   `scheduled_meetings` (Dict): The current mapping of Time Slots -> Meeting IDs.
*   `available_slots` (List): Time slots strictly available for scheduling.
*   `conflicts` (int): A tracking counter of all double-booked errors made.
*   `step_count` (int): Current operation marker within the episode bounds.

**Action Space**: Discrete (9 Possible Triggers)
*   **Schedule (Actions 0-7)**: Attempt to bind the current meeting directly to an available time slot (9AM to 4PM).
*   **Reject (Action 8)**: Deliberately skip the incoming meeting query entirely (binds a null payload).

---

## Task Descriptions with Expected Difficulty

| Task | Description | Difficulty |
| :--- | :--- | :--- |
| **Task 1: Basic Scheduling** | The agent must successfully schedule as many meetings as possible securely into the calendar without failing constraints. | **Easy** |
| **Task 2: Conflict Resolution** | The agent must recognize collision patterns and gracefully ignore overlapping meeting requests to mathematically minimize total schedule conflicts. | **Medium** |
| **Task 3: Preference Optimization** | The highest-level constraint where the agent must successfully schedule meetings *while actively prioritizing* each user's unique favorite time slot. | **Hard** |

---

## Setup and Usage Instructions

### 1. Requirements
Ensure you are using Python 3.10+ and install the strict environment dependencies natively.
```bash
pip install -r requirements.txt
```

### 2. Hugging Face Spaces App (FastAPI)
Developers can dynamically evaluate and inspect the REST API structure and environment constraints visually by triggering the OpenEnv dashboard:
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Navigate to `http://localhost:7860` in any web-browser.

### 3. OpenEnv Execution (Offline Benchmark script)
The native LLM offline execution agent maps directly to HuggingFace space routers locally. To test standard scoring metrics via `inference.py`:
```bash
export HF_TOKEN="your_hf_token_here"
python inference.py
```

---

## Baseline Scores
Scores are strictly within `(0.01, 0.99)` — values of exactly `0.0` or `1.0` are clamped by the grader. The table below shows benchmark metrics from the bundled zero-shot LLM baseline and the pre-trained Q-Learning agent.

| Agent | Task 1 (Scheduling) | Task 2 (Conflict Res) | Task 3 (Optimization) |
| :--- | :--- | :--- | :--- |
| **meta-llama/Meta-Llama-3-8B-Instruct** | `0.40` | `0.80` | `0.40` |
| **Qwen/Qwen2.5-72B-Instruct** | `0.80` | `0.60` | `0.80` |
| **Tabular Q-Learning (Pre-Trained 500 Episodes)** | `0.99` | `0.99` | `0.99` |
