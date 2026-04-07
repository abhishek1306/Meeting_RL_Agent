"""
Deterministic graders for the three task difficulty levels.
"""


def grade_task1(scheduled_count: int, total_meetings: int) -> float:
    """
    Task 1 – Basic Scheduling (Easy)
    Score = scheduled_meetings / total_meetings
    """
    if total_meetings == 0:
        return 0.0
    return round(scheduled_count / total_meetings, 4)


def grade_task2(conflicts: int, total_meetings: int) -> float:
    """
    Task 2 – Conflict Resolution (Medium)
    Score = 1 - (conflicts / total_meetings)
    """
    if total_meetings == 0:
        return 0.0
    return round(max(0.0, 1.0 - (conflicts / total_meetings)), 4)


def grade_task3(preference_matches: int, total_meetings: int) -> float:
    """
    Task 3 – Preference Optimization (Hard)
    Score = preference_satisfaction / total_meetings
    """
    if total_meetings == 0:
        return 0.0
    return round(preference_matches / total_meetings, 4)
