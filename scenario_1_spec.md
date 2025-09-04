# Scenario 1 — Spec: Two Symmetric Min-Count Attributes

Goal: Meet both minimums with minimal rejections using a feasibility‑guarded greedy policy for two symmetric attributes: `young` and `well_dressed`.

Inputs/State
- `person`: map of attributes → bool.
- `state`: `N`, `admitted_count`, `counts[a]`, `constraints[a]`, `freqs[a]`, `corr[a][b]`.
- `rejection_history`: recent 0/1 rejects (not central here).

Key Invariants & Guards
- Hard feasibility (per‑person): if `(R-1) < need[a]` then:
  - If `person[a]` → force accept; else → force reject.
- LCB safety for risky admits: for wrong‑side or neither admits, require for each affected `a`:
  - `cur[a] + LCB(n=R-1, p=adjusted_p[a]) >= minCount[a]`.

Frequency Estimate (adjusted_p)
- Blend prior and empirical from last 100 accepted: `p = w_prior*prior + w_obs*observed`.
- `w_obs` scales to 0.3 by 50 samples; clamp `p ∈ [0.01, 0.99]`.

LCB Formula
- `n = R - 1`, `mu = n*p`, `var = n*p*(1-p)`, `z = 0.3 + 0.5 * ((N - admitted)/N)`.
- `LCB = mu - z*sqrt(var)`.

Decision Flow (high level)
1) If both minima already met → accept all (minimize rejections).
2) Apply feasibility force (accept/reject) if triggered.
3) Early fill (admitted < 90% of `N`):
   - Accept if person has at least one attr.
   - If neither, accept only if LCB says both constraints remain feasible.
4) Sprint finish (> 900 admitted): if both are within 20 of min and person has any attr → accept.
5) Phase A (both unmet): accept any with ≥1 attr; allow neither only if LCB‑safe for both.
6) Phase B (one met, one unmet):
   - If person has unmet attr (including both) → accept.
   - Else, accept only if `_safe_wrong_side_accept(unmet)` LCB check passes.
7) Fallback: accept if any attr; else accept neither only if `_safe_accept_neither` passes.

Pseudocode
- R = N - admitted; need[a] = max(0, min[a] - counts[a])
- If all need == 0: return True
- forced = feasibility_guard(person)
- If forced: return forced
- If pre90 and (has1 or has2): return True; elif pre90 and LCB_safe_both(): return True
- If admitted >= 900 and nearly_met_both and (has1 or has2): return True
- If need[a1] > 0 and need[a2] > 0: if (has1 or has2) return True; elif LCB_safe_both() return True
- If need[a1] <= 0 < need[a2]: if has2 return True; elif safe_wrong_side(a2) return True (symmetrically for a1)
- Else: if (has1 or has2) return True; elif LCB_safe_both() return True; else False

Tunables
- Accept window: 100; pre‑90% threshold: 0.9N; sprint cutoff: 900.
- LCB `z`: 0.3 → 0.8 as capacity remains high.
- “Nearly met” buffer: 20.

Edge Cases
- If only one constraint present, accept if person has it or it’s already met.
- When `R <= 1`, neither/wrong‑side safety checks return False (don’t risk infeasibility).

Code
- Source: `scenario1.py` (`decide`, `_adjusted_p`, `_safe_wrong_side_accept`, `_safe_accept_neither`).

