"""Independent, deterministic CPU simulator for the public EMBERBench V1 rules (P1-01).

Standard library only. No import of EMBERBench, the decision model or Modal.
Rule provenance and open questions: docs/P1_SIMULATOR_PARITY.md.
"""
from .movement import ACTIONS, OUTCOMES, resolve_moves
from .rng import SplitMix64, stream_seed
from .scenario import Scenario
from .simulator import (InvalidActionError, ReplayVerdict, SimulationError, Simulator, first_divergence,
                        replay, run_episode, verify_replay)

__all__ = ["ACTIONS", "OUTCOMES", "resolve_moves", "SplitMix64", "stream_seed", "Scenario",
           "Simulator", "InvalidActionError", "SimulationError", "run_episode", "replay",
           "first_divergence", "verify_replay", "ReplayVerdict"]
