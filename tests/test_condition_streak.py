"""Tests for ConditionStreak utility."""

from alphaweave.utils import ConditionStreak


def test_condition_streak_basic():
    streak = ConditionStreak()
    seq = [False, True, True, False, True, True, True]
    values = [streak.update(cond) for cond in seq]
    assert values == [0, 1, 2, 0, 1, 2, 3]
    assert streak.streak == 3
