"""Utilities for building a Qwen/Jev-style evacuation decision model safely.

This package is deliberately independent of EMBERBench evaluator internals.
"""
from .contract import validate_observation, validate_actions
from .encoding import encode_observation
from .dataset import build_decision_row

__all__ = ["validate_observation", "validate_actions", "encode_observation", "build_decision_row"]
