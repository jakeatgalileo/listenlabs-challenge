# Scenario 1 — Spec: Simple Greedy Strategy

Goal: With two symmetric min‑count attributes (e.g., `young`, `well_dressed`),
apply a simple greedy policy:
- Accept candidates who have at least one of the constrained attributes
  (including both-true).
- Reject candidates with neither attribute.
- After both minima are met, accept everyone to minimize rejections.
- Always enforce feasibility: never accept a candidate if taking their seat
  would make any minimum impossible to reach with the remaining seats.

Inputs/State
- `person`: map of attributes → bool.
- `state`: `N`, `admitted_count`, `counts[a]`, `constraints[a]`, `freqs[a]`.
- `rejection_history`: recent 0/1 rejects (not used by this rule).

Guards
- Feasibility guard: let `R = N - admitted`, `need[a] = max(0, min[a] - counts[a])`.
  - If the candidate lacks some `a` and `R - 1 < need[a]`, reject (accepting
    would make meeting `a` impossible).

Decision Flow
1) If both minima are met → accept all.
2) If feasibility guard triggers for any missing attribute → reject.
3) Otherwise, accept if the candidate has any constrained attribute; else reject.

Empirical Frequency
- A small rolling window may be tracked for analysis, but the greedy rule does
  not depend on it.

Tunables
- None required for the greedy policy beyond feasibility.

Edge Cases
- If only one constraint exists, accept if the person has it; once the minimum
  is met, accept all.

Code
- Source: `scenario1.py` (`decide`).
