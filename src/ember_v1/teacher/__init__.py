"""Observation-only reference policies (P1): T0 (issue #7) and T1 (issue #8).

`interface.ReferencePolicy` is the versioned interface that T0-T4 implement.
`controls` holds simple baseline controls; they are not teachers.
Design and evidence: docs/P1_T0_BASELINE.md, docs/P1_T1_PLANNER.md.
"""
from .controls import GreedyNearestExitControl, RandomControl, StayControl
from .grid import GRID_UTILS_VERSION, PublicGrid, exit_distance_field
from .interface import POLICY_INTERFACE_VERSION, PolicyDecision, PolicyIdentity, ReferencePolicy
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware

__all__ = ["GreedyNearestExitControl", "RandomControl", "StayControl", "GRID_UTILS_VERSION", "PublicGrid",
           "exit_distance_field", "POLICY_INTERFACE_VERSION", "PolicyDecision", "PolicyIdentity",
           "ReferencePolicy", "T0ShortestPath", "T1CongestionAware"]
