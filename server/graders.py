"""
Deterministic graders for the three task difficulty levels.
Scores are strictly in (0.01, 0.99) as required by the OpenEnv grader.
"""


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.01, 0.99)."""
    return round(max(0.01, min(0.99, score)), 4)


def grade_task1(scheduled_count: int, total_meetings: int) -> float:
    """
    Task 1 – Basic Scheduling (Easy)
    Score = scheduled_meetings / total_meetings
    """
    if total_meetings == 0:
        return 0.01
    return _clamp(scheduled_count / total_meetings)


def grade_task2(conflicts: int, total_meetings: int) -> float:
    """
    Task 2 – Conflict Resolution (Medium)
    Score = 1 - (conflicts / total_meetings)
    """
    if total_meetings == 0:
        return 0.01
    return _clamp(1.0 - (conflicts / total_meetings))


def grade_task3(preference_matches: int, total_meetings: int) -> float:
    """
    Task 3 – Preference Optimization (Hard)
    Score = preference_satisfaction / total_meetings
    """
    if total_meetings == 0:
        return 0.01
    return _clamp(preference_matches / total_meetings)
