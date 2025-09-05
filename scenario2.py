from typing import Dict, List, Tuple


"""
Scenario 2: Curated 16‑combo scoring + dynamic adjustments.

Goal: Prioritize `creative` (rare, critical) and `berlin_local` (most constrained),
while respecting the strong negative correlation between `berlin_local` and
`techno_lover`, and exploiting the positive correlation between `berlin_local`
and `well_connected`.

Approach:
- Hard safety guard to prevent infeasibility.
- Curated score for all 16 (TL, WC, CR, BL) combinations.
- Dynamic score adjustments based on current progress toward minima.
- Simple dynamic threshold (base 2.5), optionally tunable.
- Endgame feasibility guard when nearing capacity.

Tuning guidance is documented in scenario_2_spec.md and mirrored in comments below.
"""

# Attribute ids used by the API
A_T = "techno_lover"
A_W = "well_connected"
A_C = "creative"
A_B = "berlin_local"

# Attribute order for keys in the curated map (TL, WC, CR, BL)
ATTR_ORDER: Tuple[str, str, str, str] = (A_T, A_W, A_C, A_B)

# Thresholds / guards
BASE_THRESHOLD_SCENARIO2 = 2.5
ALWAYS_ACCEPT_CUTOFF = 9.5  # Tier 1 gets 10.0; accept immediately
ENDGAME_REMAINING = 80      # conservative finishing window


# Curated scoring map (current state)
# Attribute order: (techno_lover, well_connected, creative, berlin_local)
# Format: (TL, WC, CR, BL): score
SCORING_MAP_SCENARIO2: Dict[Tuple[bool, bool, bool, bool], float] = {
    # TIER 1: ALWAYS ACCEPT (Score 10.0)
    # Has creative (rare) + berlin_local (critical)
    (True, True, True, True): 10.0,   # All attributes (ultra rare due to correlations)
    (True, False, True, True): 10.0,  # CR+BL+TL (rare combo)
    (False, True, True, True): 10.0,  # CR+BL+WC (excellent combo)
    (False, False, True, True): 10.0, # CR+BL only

    # TIER 2: VERY HIGH PRIORITY (Score 8.0-9.0)
    # Creative with other helpful attributes
    (True, True, True, False): 8.5,   # CR+TL+WC (no BL) — WC neutralized
    (True, False, True, False): 8.5,  # CR+TL only
    (False, True, True, False): 8.0,  # CR+WC only — WC neutralized
    (False, False, True, False): 8.0, # CR only (still very valuable)

    # Berlin local + techno_lover (fighting the negative correlation)
    (True, True, False, True): 9.0,   # BL+TL+WC (all constraints) — bumped
    (True, False, False, True): 9.0,  # BL+TL (rare due to -0.655 correlation) — bumped

    # TIER 3: HIGH PRIORITY (Score 5.0-7.0)
    # Berlin local combinations (exploiting positive correlations)
    (False, True, False, True): 7.0,  # BL+WC — WC neutralized to BL only
    (False, False, False, True): 7.0, # BL only — further bump

    # TIER 4: MEDIUM PRIORITY (Score 2.5-4.0)
    # Techno_lover combinations (common but needed)
    (True, True, False, False): 4.5,  # TL+WC — WC neutralized to TL only
    (True, False, False, False): 4.5, # TL only — bump

    # TIER 5: LOW PRIORITY (Score 1.0-2.0)
    # Well_connected only
    (False, True, False, False): 0.0, # WC only — ignored

    # TIER 6: REJECT (Score 0.0)
    (False, False, False, False): 0.0, # No attributes
}


def _remaining(state) -> int:
    return max(0, state.N - state.admitted_count)


def _hard_safety(person: Dict[str, bool], state) -> bool:
    """If this person lacks some attribute `a` and admitting them would
    leave fewer than the remaining required seats to still satisfy `a`,
    then we must reject them (unsafe)."""
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        if not person.get(a, False) and (R - 1) < max(0, M_a - state.counts.get(a, 0)):
            return False
    return True


def _feasible_if_accept(person: Dict[str, bool], state) -> bool:
    """Conservative feasibility check: after accepting this person, can we
    still reach each minimum under expected frequencies?"""
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        cur = state.counts.get(a, 0) + (1 if person.get(a, False) else 0)
        exp_max = cur + max(0, R - 1) * max(0.0, state.freqs.get(a, 0.0))
        if exp_max < M_a:
            return False
    return True


def _all_minima_met(state) -> bool:
    for a, m in state.constraints.items():
        if state.counts.get(a, 0) < m:
            return False
    return True


def _person_to_key(person: Dict[str, bool]) -> Tuple[bool, bool, bool, bool]:
    return tuple(bool(person.get(a, False)) for a in ATTR_ORDER)  # type: ignore[return-value]


def _progress(state, a: str) -> float:
    m = max(1, int(state.constraints.get(a, 1)))
    return state.counts.get(a, 0) / m


def get_dynamic_score_adjustment(person: Dict[str, bool], state) -> float:
    """Adjust scores based on current progress toward constraints.

    Spec highlights:
    - Boost lagging constraints: BL (+1.5) and CR (+2.0) when < 70% to min.
    - Special boost for rare BL∧TL (+1.0) to fight negative correlation.
    - Penalty for over-represented TL‑only when TL progress > 90%.
    """
    adj = 0.0

    bl_prog = _progress(state, A_B)
    cr_prog = _progress(state, A_C)
    tl_prog = _progress(state, A_T)
    # wc_prog intentionally unused: WC treated as neutral in scoring

    if person.get(A_B) and bl_prog < 0.85:
        adj += 1.5
    if person.get(A_C) and cr_prog < 0.7:
        adj += 2.0

    # Boost TL when lagging toward its minimum
    if person.get(A_T) and tl_prog < 0.95:
        adj += 1.0

    # Rare BL ∧ TL combo boost
    if person.get(A_B) and person.get(A_T):
        adj += 1.0

    # TL‑only penalty when TL is already well on track and no BL
    if person.get(A_T) and not person.get(A_B) and tl_prog > 0.98:
        adj -= 1.0

    # No explicit WC adjustments: we treat WC as neutral unless required by safety

    return adj


def _dynamic_threshold(state, base: float = BASE_THRESHOLD_SCENARIO2) -> float:
    """Base threshold modulation from state only.

    Keep neutral thresholding by default; person-aware shaping below will
    implement the aggressive early behavior (creative-only until 95%).
    """
    if _all_minima_met(state):
        return 0.0
    bl_prog = _progress(state, A_B)
    cr_prog = _progress(state, A_C)
    if bl_prog < 0.7 and cr_prog < 0.7:
        return max(2.0, base - 0.3)
    return base


def _threshold_for(person: Dict[str, bool], state) -> float:
    """Person-aware threshold shaping:
    - Auto-accept creatives until they hit 95% of their minimum (handled in decide).
    - After creatives reach 95%, ease threshold by 0.5 for creative candidates
      to continue favoring them without auto-accepting.
    """
    thr = _dynamic_threshold(state)
    cr_prog = _progress(state, A_C)

    # Until creative progress reaches 95%, effectively block non-creative candidates
    if cr_prog < 0.95 and not person.get(A_C, False):
        return 1e9  # impossible bar for non-creative early

    # After creatives 95%+, still give them a small threshold easing
    if person.get(A_C, False) and cr_prog >= 0.95:
        thr = max(1.5, thr - 0.5)
    return thr


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    # 1) Hard safety guard
    if not _hard_safety(person, state):
        return False

    # 2) If all minima already met, accept everyone to finish with minimal rejections
    if _all_minima_met(state):
        return True

    # 3) Endgame feasibility guard
    if _remaining(state) <= ENDGAME_REMAINING and not _feasible_if_accept(person, state):
        return False

    # 3.5) Creative auto-accept until 95% of minimum is reached
    if person.get(A_C, False):
        if _progress(state, A_C) < 0.95:
            return True

    # 4) Curated 16-combo score + dynamic adjustment
    key = _person_to_key(person)
    base_score = SCORING_MAP_SCENARIO2.get(key, 0.0)

    # Tier-1: accept immediately
    if base_score >= ALWAYS_ACCEPT_CUTOFF:
        return True

    score = base_score + get_dynamic_score_adjustment(person, state)
    threshold = _threshold_for(person, state)
    return score >= threshold
