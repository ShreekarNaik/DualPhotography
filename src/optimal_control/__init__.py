"""
Optimal Control Module

Implements the core optimal control algorithms for adaptive measurement selection.
Includes Kalman filtering, greedy pattern selection, and value function approximation.
"""

from .state_estimator import StateEstimator
from .pattern_selector import GreedyPatternSelector

__all__ = ['StateEstimator', 'GreedyPatternSelector']
