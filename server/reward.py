"""
Reward function for the Meeting Scheduling Environment.

Reward Table:
  +1.0  Successful scheduling
  +0.5  Preferred slot match
  -1.0  Conflict (slot already taken or unavailable)
  -0.2  Reject a meeting
  -0.1  Step penalty (per action)
"""


def calculate_reward(
    action_type: str,
    scheduled: bool = False,
    conflict: bool = False,
    preferred_match: bool = False,
) -> float:
    """Calculate the reward for a single action."""
    reward = -0.1  # step penalty

    if action_type == "schedule":
        if conflict:
            reward += -1.0
        elif scheduled:
            reward += 1.0
            if preferred_match:
                reward += 0.5
    elif action_type == "reject":
        reward += -0.2

    return round(reward, 2)
