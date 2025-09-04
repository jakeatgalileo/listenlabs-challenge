# Scenario 3 — Spec: 64‑Combo Scoring + Need‑Ratio Threshold

Goal: Use a curated score table over all 64 combinations of six attributes and accept based on a dynamic threshold driven by need ratios for key constraints, while preserving hard safety and a conservative endgame feasibility check.

Attributes and Order
- Bit/tuple order: `(underground_veteran, international, fashion_forward, queer_friendly, vinyl_collector, german_speaker)`.

Inputs/State
- `person`, `state` (`N`, `admitted_count`, `counts`, `constraints`, `freqs`).

Invariants & Guards
- Hard safety: if the candidate lacks a critical attribute and `R-1 < need[a]` for any `a`, reject.
- Accept‑all once all minima are satisfied.
- Endgame feasibility (R ≤ 120): reject if, after accepting, `cur + (R-1)*p[a] < M[a]` for any `a`.

Precomputed Scores (64 combos)
- `SCORING_MAP` assigns explicit scores on a 0–100 scale for many meaningful combinations; unspecified combos default to `0.0`.
- Emphasis patterns in the current map (see `scenario3.py`):
  - High scores for sets combining `german_speaker` and `international`, especially when paired with `queer_friendly` and other supportive traits.
  - QF with either GS or INT receives solid scores; GS‑only or INT‑only also score moderately to recover lagging constraints.
  - Low scores for already‑met/low‑value traits and for “no useful attributes”.

Dynamic Threshold (0–100 scale)
- Compute need ratios over remaining seats for `german_speaker`, `international`, and a discounted `queer_friendly`:
  - `ratio = need[a] / remaining` (QF ratio divided by 3 to account for heavier scoring weight).
- Threshold levels by max ratio:
  - ≥ 0.95 → 35 (super desperate)
  - ≥ 0.85 → 40 (very desperate)
  - ≥ 0.75 → 45 (desperate)
  - ≥ 0.60 → 50 (concerned)
  - else → 55 (comfortable)

Decision Flow
1. Hard safety → reject if violated.
2. All minima met → accept all.
3. If `R ≤ 120` and accepting breaks feasibility → reject.
4. Lookup the 6‑bit combo key and get the precomputed score.
5. Accept if `score ≥ threshold` (no special auto‑accept tiers in code).

Tuning Guidance
- Adjust `SCORING_MAP` entries to reprioritize combinations.
- If too many rejects late, lower the threshold breakpoints (e.g., −5 across tiers).
- If a constraint lags, increase scores for combos that include that attribute (e.g., add +3–5 to all GS combos).

Code
- Source: `scenario3.py` (`decide`), helpers: `_person_to_key`, `_tuple_to_key`, `_build_scores`, `_dynamic_threshold`, plus safety/feasibility utilities.
