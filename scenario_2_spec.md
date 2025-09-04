# Scenario 2 — Spec: 16‑Combo Scoring + Dynamic Adjustments

Goal: Meet minima with minimal rejections by strongly valuing `creative` (rare, 6.23%) and efficiently filling `berlin_local` (critical, 75% required) while mitigating the negative `berlin_local ↔ techno_lover` correlation (−0.655) and leveraging the positive `berlin_local ↔ well_connected` correlation (+0.572).

Attributes
- TL: `techno_lover`, WC: `well_connected`, CR: `creative`, BL: `berlin_local`.

Inputs/State
- `person`, `state` (`N`, `admitted_count`, `counts`, `constraints`, `freqs`, `corr`).

Guards
- Hard safety: if a person lacks some `a` and `R-1` seats is fewer than the remaining need for `a`, reject.
- Endgame guard: when `R ≤ 80`, reject if accepting would make expected max fall below any minimum: `cur + (R-1)*p[a] < M[a]`.
- All‑minima‑met: once all minima are satisfied, accept everyone (finish with minimal rejections).

Curated Scoring Map (all 16 combos)
- Key order: `(TL, WC, CR, BL)` → float score.
- Tiers and examples (full map in `scenario2.py`):
  - Tier 1 (10.0, always accept): any `CR ∧ BL` combo.
  - Tier 2 (8.0–9.0): `CR` without `BL`; `BL ∧ TL` (fight −0.655).
  - Tier 3 (5.5–7.0): `BL` combos (exploit +0.572 with `WC`).
  - Tier 4 (3.0–3.5): `TL` combos (common but needed; `TL+WC` is only 3.0 due to −0.470).
  - Tier 5 (2.0): `WC` only.
  - Tier 6 (0.0): no attributes.

Dynamic Score Adjustment
- Progress‑aware boosts/penalties:
  - If `person.BL` and `BL_progress < 0.7`: `+1.5`.
  - If `person.CR` and `CR_progress < 0.7`: `+2.0`.
  - If `person.BL ∧ person.TL`: `+1.0` (rare due to −0.655; accept almost all).
  - If `person.TL ∧ ¬person.BL` and `TL_progress > 0.9`: `−1.0`.

Threshold
- Base threshold: `BASE_THRESHOLD_SCENARIO2 = 2.5`.
- Light modulation: if both `BL` and `CR` are < 70% to their minima, lower to `≈ 2.2`.
- After `creative` reaches 95% of its minimum, lower the threshold by `0.5` for creative candidates to continue preferring them without auto‑accepting.
- Decision: `score + adj ≥ threshold` → accept.

Key Strategy Points
- Creative is gold: accept almost anyone with `creative` (scores 8–10).
- Berlin_local is critical: need 75% but only 39.8% frequency → efficient admissions.
- Correlation trap: `BL` avoids `TL` (−0.655) yet both are needed → explicitly boost/accept `BL ∧ TL`.
- Exploit positive correlation: `BL ∧ WC` is favored.

Early Creative Auto‑Accept
- Auto‑accept `creative` candidates until they reach 95% of their minimum (still respecting hard safety and endgame guards).
- After 95%, `creative` candidates are no longer auto‑accepted, but get a lower decision threshold (−0.5) to keep them slightly favored.

Expected Performance
- With reasonable thresholds, should achieve ~3500–4000 rejections.
- Bottleneck: `CR ∧ BL` combinations.

Tuning Guidance
- If rejections > 4000: lower `BASE_THRESHOLD_SCENARIO2` to `2.0`.
- If `berlin_local` fails: add `+1.0` to all `BL` combos.
- If `creative` fails: make all `CR` scores ≥ `9.0`.
- If `techno_lover` fails: increase scores for `TL ∧ BL` combinations.

Code
- Source: `scenario2.py` (`decide`, `_hard_safety`, `_feasible_if_accept`, `_dynamic_threshold`, `get_dynamic_score_adjustment`).
