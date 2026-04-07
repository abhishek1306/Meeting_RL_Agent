
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.environment import MeetingSchedulingEnv
from server.agent import QLearningAgent
from server.graders import grade_task1, grade_task2, grade_task3
from server.models import Action, StepResult, StateResult

app = FastAPI(title="Meeting RL Agent")

FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Global state ────────────────────────────────────────
current_agent: QLearningAgent | None = None
is_training = False
stop_requested = False
training_history: list[dict] = []
best_episode_data: dict | None = None
selected_task: int = 1
current_env: MeetingSchedulingEnv | None = None


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global current_agent, is_training, stop_requested, training_history, selected_task, best_episode_data

    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg["type"] == "train":
                if is_training:
                    await ws.send_text(json.dumps({"type": "error", "message": "Training already running"}))
                    continue

                stop_requested = False
                is_training = True
                selected_task = msg.get("task", 1)
                episodes = msg.get("episodes", 300)
                num_meetings = msg.get("num_meetings", 5)
                lr = msg.get("lr", 0.15)
                gamma = msg.get("gamma", 0.95)
                epsilon = msg.get("epsilon", 1.0)
                epsilon_decay = msg.get("epsilon_decay", 0.995)

                current_agent = QLearningAgent(
                    learning_rate=lr,
                    discount_factor=gamma,
                    epsilon=epsilon,
                    epsilon_decay=epsilon_decay,
                )
                env = MeetingSchedulingEnv(num_meetings=num_meetings)
                training_history = []
                best_episode_data = None

                try:
                    # 1. Send [START] globally for the session
                    await ws.send_text(json.dumps({
                        "type": "log",
                        "line": f"[START] task=task{selected_task} env=meeting_scheduling_env model=QLearningAgent_Tabular",
                        "cls": "start"
                    }))

                    # ── Pre-training (builds the Reward Curve) ──────────────────
                    pre_train_episodes = 500
                    for pe in range(1, pre_train_episodes + 1):
                        ep_data = current_agent.train_episode(env)
                        
                        chart_data = {
                            "episode": pe,
                            "total_episodes": pre_train_episodes,
                            "total_steps": ep_data["steps"], # DO NOT ACCUMULATE
                            "total_reward": ep_data["total_reward"],
                            "scheduled": ep_data["scheduled"],
                            "total_meetings": ep_data["total_meetings"],
                            "conflicts": ep_data["conflicts"],
                            "preference_matches": ep_data["preference_matches"],
                            "epsilon": ep_data["epsilon"],
                            "task1_score": grade_task1(ep_data["scheduled"], ep_data["total_meetings"]),
                            "task2_score": grade_task2(ep_data["conflicts"], ep_data["total_meetings"]),
                            "task3_score": grade_task3(ep_data["preference_matches"], ep_data["total_meetings"])
                        }
                        training_history.append(chart_data)
                        
                        # Stream every 10th episode to build the chart
                        if pe % 10 == 0:
                            await ws.send_text(json.dumps({"type": "episode", "data": chart_data}))
                            await asyncio.sleep(0.001)

                    # Force expert mode (no randomness) for the final visible run
                    current_agent.epsilon = 0.0 
                    
                    obs = env.reset()
                    ep_reward = 0
                    ep_steps = 0
                    ep_rewards = []
                    
                    while not env.done and ep_steps < 20:
                        if stop_requested: break

                        state = current_agent.encode_state(obs)
                        action_idx = current_agent.select_action(obs)
                        
                        # Get current meeting safely
                        meeting = obs.pending_meetings[0] if obs.pending_meetings else None
                        if not meeting: break
                        
                        action = current_agent.action_index_to_action(action_idx, meeting.meeting_id)
                        result = env.step(action)
                        next_state = current_agent.encode_state(result.observation)
                        current_agent.update(state, action_idx, result.reward, next_state)

                        ep_reward += result.reward
                        ep_steps += 1
                        ep_rewards.append(f"{result.reward:.2f}")

                        # Stream every single step to the UI
                        action_desc = f"{action.action_type}_{action.meeting_id}_{action.time_slot if action.time_slot else 'none'}"
                        await ws.send_text(json.dumps({
                            "type": "log",
                            "line": f"[STEP] step={ep_steps} action={action_desc} reward={result.reward:.2f} done={str(env.done).lower()} error=null",
                            "cls": "step"
                        }))
                        
                        obs = result.observation
                        await asyncio.sleep(0.3) # Wait for half a second so user can watch

                    current_agent.decay_epsilon()
                    
                    # Final [END]
                    await ws.send_text(json.dumps({
                        "type": "log",
                        "line": f"[END] success=true steps={ep_steps} rewards={','.join(ep_rewards)}",
                        "cls": "end"
                    }))

                    # Prepare final summary data
                    data = {
                        "episode": pre_train_episodes + 1,
                        "total_episodes": pre_train_episodes + 1,
                        "total_steps": ep_steps, # REVERTED
                        "total_reward": ep_reward,
                        "scheduled": len(env.scheduled_meetings),
                        "total_meetings": env.num_meetings,
                        "conflicts": env.conflicts,
                        "preference_matches": env.preference_matches,
                        "epsilon": current_agent.epsilon,
                        "task1_score": grade_task1(len(env.scheduled_meetings), env.num_meetings),
                        "task2_score": grade_task2(env.conflicts, env.num_meetings),
                        "task3_score": grade_task3(env.preference_matches, env.num_meetings),
                        "actions": env.history,
                        "schedule_map": {slot: {"meeting_id": mid, "title": f"M{mid}"} for mid, slot in env.scheduled_meetings.items()}
                    }
                    training_history.append(data)
                    best_episode_data = data

                    await ws.send_text(json.dumps({"type": "episode", "data": data}))
                    await ws.send_text(json.dumps({"type": "training_complete", "total_episodes": 1}))
                finally:
                    is_training = False

            elif msg["type"] == "stop":
                stop_requested = True

            elif msg["type"] == "reset":
                current_agent = None
                training_history = []
                best_episode_data = None
                selected_task = 1
                is_training = False
                stop_requested = False
                await ws.send_text(json.dumps({"type": "reset_done"}))
            elif msg["type"] == "state":
                if not training_history or current_agent is None or best_episode_data is None:
                    await ws.send_text(json.dumps({
                        "type": "state_result",
                        "data": None,
                        "message": "No training data. Select a task and click Start first."
                    }))
                    continue

                # Build comprehensive state report
                total_eps = len(training_history)
                rewards = [d["total_reward"] for d in training_history]
                scheduled_list = [d["scheduled"] for d in training_history]
                conflicts_list = [d["conflicts"] for d in training_history]
                pref_list = [d["preference_matches"] for d in training_history]
                t1_scores = [d["task1_score"] for d in training_history]
                t2_scores = [d["task2_score"] for d in training_history]
                t3_scores = [d["task3_score"] for d in training_history]

                last = training_history[-1]
                total_meetings = last["total_meetings"]

                # Averages over last 20 episodes
                w = min(20, total_eps)
                avg = lambda lst: sum(lst[-w:]) / w

                # Pick score based on selected task
                task_scores = {1: t1_scores, 2: t2_scores, 3: t3_scores}
                task_names = {1: "Basic Scheduling", 2: "Conflict Resolution", 3: "Preference Optimization"}

                # First/last 20 avg reward to show improvement
                first_w = min(20, total_eps)
                first_avg = sum(rewards[:first_w]) / first_w
                last_avg = avg(rewards)

                state_data = {
                    "task": selected_task,
                    "task_name": task_names[selected_task],
                    "total_episodes": total_eps,
                    "total_meetings_per_ep": total_meetings,

                    # Reward
                    "avg_reward_first20": round(first_avg, 3),
                    "avg_reward_last20": round(last_avg, 3),
                    "best_reward": round(max(rewards), 3),
                    "worst_reward": round(min(rewards), 3),
                    "reward_improvement": round(last_avg - first_avg, 3),

                    # Scheduling
                    "avg_scheduled_last20": round(avg(scheduled_list), 2),
                    "avg_conflicts_last20": round(avg(conflicts_list), 2),
                    "avg_pref_matches_last20": round(avg(pref_list), 2),

                    # Task score
                    "task_score_first20": round(sum(task_scores[selected_task][:first_w]) / first_w, 4),
                    "task_score_last20": round(avg(task_scores[selected_task]), 4),
                    "task_score_best": round(max(task_scores[selected_task]), 4),

                    # Agent
                    "epsilon_final": round(current_agent.epsilon, 5),
                    "q_table_size": len(current_agent.q_table),

                    # All scores for charts
                    "all_task1": t1_scores,
                    "all_task2": t2_scores,
                    "all_task3": t3_scores,
                    "all_rewards": rewards,
                    "all_scheduled": scheduled_list,
                    "all_conflicts": conflicts_list,

                    # Best episode details (what we now show on the grid)
                    "best_episode": best_episode_data,
                }

                await ws.send_text(json.dumps({"type": "state_result", "data": state_data}))

    except WebSocketDisconnect:
        pass

@app.api_route("/reset", methods=["GET", "POST"], response_model=StepResult)
async def http_reset():
    global current_agent, training_history, best_episode_data, selected_task, is_training, stop_requested, current_env
    current_agent = None
    training_history = []
    best_episode_data = None
    selected_task = 1
    is_training = False
    stop_requested = False
    
    # Generate initial observation from a fresh environment and store it globally
    current_env = MeetingSchedulingEnv(num_meetings=5)
    obs = current_env.reset()
    
    return StepResult(
        observation=obs,
        reward=0.0,
        done=False,
        info={}
    )


@app.post("/step", response_model=StepResult)
async def http_step_post(action: Action):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    result = current_env.step(action)
    return result


@app.get("/step", response_model=StepResult)
async def http_step_get(
    action_type: str | None = None,
    meeting_id: int | None = None,
    time_slot: str | None = None
):
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    if action_type is None or meeting_id is None:
        raise HTTPException(status_code=400, detail="Missing action parameters (action_type, meeting_id)")
        
    action = Action(action_type=action_type, meeting_id=meeting_id, time_slot=time_slot)
    result = current_env.step(action)
    return result


@app.api_route("/state", methods=["GET", "POST"], response_model=StateResult)
async def http_state():
    global current_env
    if current_env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    obs = current_env.state()
    
    return StateResult(
        observation=obs,
        total_reward=current_env.total_reward,
        done=current_env.done,
        info={}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)
