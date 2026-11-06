"""Observation-only reference policies (P1). T0 only in this revision (issue #7).

`interface.ReferencePolicy` is the versioned interface that T1-T4 will also
implement. `controls` holds simple baseline controls; they are not teachers.
Design and evidence: docs/P1_T0_BASELINE.md.
"""
from .controls import GreedyNearestExitControl, RandomControl, StayControl
from .grid import GRID_UTILS_VERSION, PublicGrid, exit_distance_field
from .interface import POLICY_INTERFACE_VERSION, PolicyDecision, PolicyIdentity, ReferencePolicy
from .t0 import T0ShortestPath

__all__ = ["GreedyNearestExitControl", "RandomControl", "StayControl", "GRID_UTILS_VERSION", "PublicGrid",
           "exit_distance_field", "POLICY_INTERFACE_VERSION", "PolicyDecision", "PolicyIdentity",
           "ReferencePolicy", "T0ShortestPath"]
