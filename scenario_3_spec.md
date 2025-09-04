# Scenario 3 — Spec: 64‑Combo Scoring + Dynamic Threshold

Goal: Precompute a score for every combination of the six attributes, then accept using a dynamic threshold that adapts to constraint health and progress, while preserving hard safety and conservative endgame feasibility.

Attributes and Order

- Order (bit/tuple): `(underground_veteran, international, fashion_forward, queer_friendly, vinyl_collector, german_speaker)`.

Inputs/State

- `person`, `state` (`N`, `admitted_count`, `counts`, `constraints`, `freqs`, `corr`).

Invariants & Guards

- Hard safety: if the candidate lacks a currently critical attribute and `R-1 < need[a]` for any `a`, reject.
- Accept‑all once all minima are satisfied (minimize rejections).
- Endgame feasibility (R ≤ 120): reject if `cur + (R-1)*p[a] < M[a]` for any `a` after accepting the candidate.

Precomputed Scores (64 combos)

- Base weight per attribute: `W[a] = clamp((minRate[a] / baseRate[a]), 0.8, 6.0)` computed from `constraints` and `freqs`.
- Combination bonuses/penalties:
  - `+6.0 + 2.0*max(0,c_qf,vc)` if QF∧VC (rare, positively correlated).
  - `+1.5 + 1.0*max(0, -c_intl,gs)` if GS∧¬INTL (exploit strong negative INTL–GS correlation).
  - `+0.5 + 0.5*max(0, c_intl,ff)` if INTL∧FF.
  - `−1.0 − 0.5*max(0, -c_intl,gs)` if INTL∧GS together (competing signals).
  - `+1.0` if ≥4 attributes true and includes at least one rare (QF or VC).
- Always‑accept tier: any combo with QF∧VC is clamped to score ≥ 10.0.

Dynamic Threshold

- `threshold = base_threshold * f(constraint_health) * g(progress)` where:
  - `constraint_health`: For each `a`, project `expected_final = (count[a]/admitted)*N`, then `health[a] = expected_final/min[a]`. If `min(health) < 0.9`, scale threshold by `(0.7 + 0.3*min_health)`.
  - `progress`: If progress > 0.8, multiply threshold by 1.2; if < 0.3, multiply by 0.9.
- Default `base_threshold = 3.0`.
- Early strict phase: until either QF or VC reaches ~80% of its minimum, clamp threshold to ≥ 8.5 for non‑QF/VC sets so we strongly bias toward QF or VC early. Once `max(QF_fill, VC_fill) ≥ 0.8`, normal thresholding resumes.

Decision Flow

1. Hard safety check → reject if violated.
2. If all minima met → accept.
3. If `R ≤ 120` and accepting breaks feasibility → reject.
4. Compute combo key and lookup score.
5. If candidate has QF or VC → accept (after safety/feasibility guards).
6. Else, if `score ≥ 8.5` (always‑accept tier) → accept.
7. Else, compute dynamic threshold; accept iff `score ≥ threshold`.

Tuning Guidance

- If rejections are high and minima are met early, raise `base_threshold` slightly (more selective late).
- If critical constraints lag, lower `base_threshold` or strengthen specific bonuses (e.g., GS∧¬INTL).
- To force more GS, add a small global `+0.3` to any combo with GS.

Code

- Source: `scenario3.py` (`decide`), with helpers for keys, scores, and dynamic threshold.
