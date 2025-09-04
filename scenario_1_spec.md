# Scenario 1 — Spec: Two Symmetric Min-Count Attributes

Goal: Meet both minimums with minimal rejections using a feasibility‑guarded, phase‑based policy for two symmetric attributes: `young` and `well_dressed`.

Inputs/State
- `person`: map of attributes → bool.
- `state`: `N`, `admitted_count`, `counts[a]`, `constraints[a]`, `freqs[a]`.
- `rejection_history`: recent 0/1 rejects (not central here).

Guards
- Feasibility force: let `R = N - admitted`, `need[a] = max(0, min[a] - counts[a])`.
  - If `R <= need[young]` → must accept iff person has `young`.
  - If `R <= need[well_dressed]` → must accept iff person has `well_dressed`.

Empirical Frequency (for neither safety)
- Maintain last 100 accepted (`_ACCEPTED_WINDOW`).
- `_adjusted_p(a) = w_prior*prior + w_obs*observed` with `w_obs` up to 0.3 by 50 samples; clamp to `[0.01, 0.99]`.

LCB Feasibility for Neither
- For a candidate with neither attribute, accept only if for each `a ∈ {young, well_dressed}`:
  - `cur[a] + LCB(n=R-1, p=_adjusted_p[a]) ≥ min[a]`, where `LCB = n*p − z*sqrt(n*p*(1-p))` and `z = 0.3 + 0.5 * ((N - admitted)/N)`.

Decision Flow
1) If both minima met → accept all (minimize rejections).
2) Apply feasibility force rules (above) if triggered.
3) Phase A (both unmet):
   - If both attributes true → accept.
   - If exactly one true → accept with probability based on safety margin for that attribute:
     - `expected_other = (R-1)*p_other` with calibrated priors `p ≈ 0.3225`.
     - If margin < −5 → accept; < 10 → 70%; < 20 → 40%; else → 20%.
   - If neither and `_safe_accept_neither()` → accept with dynamic probability schedule: 60% (<300 admitted), 40% (<700), 25% (<900), else 12%.
4) Phase B (one met):
   - If person has the unmet attribute (including both) → accept.
   - If neither and `_safe_accept_neither()` → accept with the same schedule as above.
   - If only the met attribute: accept based on unmet side safety margin: > 20 → accept; > 10 → 50%; > 5 → 20%; else reject.
5) Default fallback: accept (should be unreachable under normal states).

Tunables
- Probabilities for neither: 60%/40%/25%/12% by admitted count thresholds (300/700/900).
- Single‑attr safety margin cutoffs: −5, 10, 20.
- Priors: `p_young = p_wd = 0.3225`.
- LCB `z` schedule: `0.3 + 0.5 * remaining_fraction`.

Edge Cases
- If only one constraint exists, accept if the person has it or it’s already met.
- When `R <= 1`, `_safe_accept_neither` returns False.

Code
- Source: `scenario1.py` (`decide`, `_adjusted_p`, `_safe_accept_neither`).
