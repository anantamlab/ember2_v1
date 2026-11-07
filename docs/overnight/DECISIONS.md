# Overnight decisions log

Each entry: decision, reason, alternatives rejected. Newest last.

## D1 — One stacked PR for P0–P6

The T4 correction, its protocol, pilot and verdict are one evidence chain. One stacked draft PR on top of PR #32 keeps the chain reviewable. P7 (#25) and P8 (#10) are independent of T4 code, so they branch from `main`.

## D2 — v1.1 is a subclass of the v1.0.0 planner in a new module

`t4.py` (v1.0.0) is not edited. `t4_v11.py` subclasses `_PriorityPlanner` and replaces only the reservation, vacate and repair methods. Urgency, forecast, exit assignment, A* cost and diagnostics code stay shared. Reason: v1.0.0 stays importable and bit-reproducible, and the shared code proves that only the repair semantics changed. Rejected: editing `t4.py` in place (breaks the frozen v1 record); a full copy of the planner (parallel implementation, drift risk).
