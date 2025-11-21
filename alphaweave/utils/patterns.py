"""Pattern utilities for alphaweave."""

class ConditionStreak:
    """Tracks the number of consecutive times a condition has been True."""

    def __init__(self) -> None:
        self.streak = 0

    def update(self, condition: bool) -> int:
        """Update streak with the latest condition value and return streak length."""
        if condition:
            self.streak += 1
        else:
            self.streak = 0
        return self.streak
