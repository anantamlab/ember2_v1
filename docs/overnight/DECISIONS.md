# Overnight decisions log

Each entry: decision, reason, alternatives rejected. Newest last.

## D1 — One stacked PR for P0–P6

The T4 correction, its protocol, pilot and verdict are one evidence chain. One stacked draft PR on top of PR #32 keeps the chain reviewable. P7 (#25) and P8 (#10) are independent of T4 code, so they branch from `main`.

## D2 — v1.1 is a subclass of the v1.0.0 planner in a new module

`t4.py` (v1.0.0) is not edited. `t4_v11.py` subclasses `_PriorityPlanner` and replaces only the reservation, vacate and repair methods. Urgency, forecast, exit assignment, A* cost and diagnostics code stay shared. Reason: v1.0.0 stays importable and bit-reproducible, and the shared code proves that only the repair semantics changed. Rejected: editing `t4.py` in place (breaks the frozen v1 record); a full copy of the planner (parallel implementation, drift risk).

## D3 — v1.1 plans each person once (push / priority inheritance), no repair loop

v1.0.0 repair re-plans roots against stale later-ranked reservations (D2 defect) and keeps dependent plans (D3 defect). v1.1.0 plans every person once in rank order. An unplanned occupant of a τ = 1 target is pushed at once and inherits the root's origin. A failed push rolls back the linear chain, forbids the cell and retries. Reason: precedence (I1) then holds by construction and is audited. Rejected: more repair rounds (does not remove stale reservations); a global solver (new algorithm family, CPU risk). `repair_rounds` leaves the config; `push_depth_limit: 256` is a resource bound, not a tuned value.

## D4 — G5c is stricter than "any link" and adds a non-inferiority clause

Before the freeze, G5c(i) counts only direct yields to an at-risk person (exit reassignment is reported only). G5c(ii) requires total at-risk immediate progress under A2 ≥ A0 on neutral T1 states. Reason: the v1 failure was that priority stalled the people it meant to protect; one favourable case alone would not detect that. The clause has a natural threshold (0) and was set without v2 pilot data. A smoke run on test-only dense rooms (`sim.bench.room`, not pilot cells) checked that the detector code works; no threshold was changed after it.

## D5 — Neutral paired states use frozen T1 trajectories

A0 and A2 decide on the same T1-created observations and do not act. Reason: closed-loop A0/A2 traces diverge after step 0, so their differences mix mechanism and state (v1 diagnosis limitation). T1 is frozen and is neither arm.
