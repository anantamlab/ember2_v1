"""Observation-only reference policies (P1): T0 (issue #7), T1 (issue #8), T2 (issue #9), T3 (issue #23)
and T4 with its A0/A1 ablations (issue #24; v1.0.0 frozen in `t4`, v1.1.0 transactional planner in `t4_v11`).

`interface.ReferencePolicy` is the versioned interface that T0-T4 implement.
`controls` holds simple baseline controls; they are not teachers.
Design and evidence: docs/P1_T0_BASELINE.md, docs/P1_T1_PLANNER.md, docs/P1_T2_ROLLOUT.md, docs/P1_T3_ACO.md,
docs/P1_T4_RISK_PRIORITY.md.
"""
from .controls import GreedyNearestExitControl, RandomControl, StayControl
from .grid import GRID_UTILS_VERSION, PublicGrid, exit_distance_field
from .interface import POLICY_INTERFACE_VERSION, PolicyDecision, PolicyIdentity, ReferencePolicy
from .t0 import T0ShortestPath
from .t1 import T1CongestionAware
from .t2 import T2StochasticRollout
from .t3 import T3DynamicACO
from .t4 import T4DistancePriority, T4NoPriority, T4RiskPriority
from .t4_v11 import T4DistancePriorityV11, T4NoPriorityV11, T4RiskPriorityV11

__all__ = ["GreedyNearestExitControl", "RandomControl", "StayControl", "GRID_UTILS_VERSION", "PublicGrid",
           "exit_distance_field", "POLICY_INTERFACE_VERSION", "PolicyDecision", "PolicyIdentity",
           "ReferencePolicy", "T0ShortestPath", "T1CongestionAware", "T2StochasticRollout",
           "T3DynamicACO", "T4RiskPriority", "T4NoPriority", "T4DistancePriority", "T4RiskPriorityV11",
           "T4NoPriorityV11", "T4DistancePriorityV11"]
