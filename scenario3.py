from typing import Any, Dict, List, Tuple, Optional


# Scenario 3: Precomputed 64-combo scoring with dynamic thresholding.
# Standalone (no imports from bouncer).

ATTR_ORDER: Tuple[str, ...] = (
    "underground_veteran",
    "international",
    "fashion_forward",
    "queer_friendly",
    "vinyl_collector",
    "german_speaker",
)

_ATTR_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ATTR_ORDER)}

# Cached scores per game (lazy init)
_SCORES: Optional[List[float]] = None

# Global per-attribute adjustments applied when building the score array
# Ignore underground_veteran and fashion_forward in acceptance scoring.
# Emphasize vinyl_collector; slightly boost german_speaker and international.
VC_SCORE_BONUS: float = 5.0
GS_SCORE_BONUS: float = 2.0
INT_SCORE_BONUS: float = 2.0

"""
Granular scoring map (0–100 scale), keyed by
(UV, INT, FF, QF, VC, GS) in that order, matching ATTR_ORDER.
"""
SCORING_MAP: Dict[Tuple[bool, bool, bool, bool, bool, bool], float] = {
    # CRITICAL: Must have both INT and GS (fight -0.72 correlation)
    (True, True, True, True, True, True): 100.0,   # Perfect person
    (False, True, False, True, False, True): 95.2, # QF+GS+INT (helps all 3 critical)
    (False, True, True, True, False, True): 94.8,  # QF+GS+INT+FF (helps all 4 remaining)
    (True, True, False, True, False, True): 93.6,  # QF+GS+INT+UV
    (True, True, True, True, False, True): 93.1,   # QF+GS+INT+UV+FF

    # Has both GS+INT but no QF (still critical for the correlation fight)
    (True, True, True, False, False, True): 82.3,  # GS+INT+UV+FF
    (True, True, False, False, False, True): 78.7, # GS+INT+UV
    (False, True, True, False, False, True): 79.4, # GS+INT+FF
    (False, True, False, False, False, True): 76.2,# GS+INT only

    # QF with either GS or INT (not both due to correlation)
    (True, False, True, True, True, True): 71.3,   # QF+GS+others (no INT)
    (False, False, True, True, True, True): 70.8,  # QF+GS+FF+VC
    (True, False, False, True, True, True): 68.9,  # QF+GS+UV+VC
    (True, False, True, True, False, True): 67.4,  # QF+GS+UV+FF
    (True, False, False, True, False, True): 64.2, # QF+GS+UV
    (False, False, True, True, False, True): 63.7, # QF+GS+FF
    (False, False, False, True, True, True): 62.1, # QF+GS+VC
    (False, False, False, True, False, True): 58.9,# QF+GS only

    (True, True, True, True, True, False): 69.8,   # QF+INT+others (no GS)
    (False, True, True, True, True, False): 69.2,  # QF+INT+FF+VC
    (True, True, False, True, True, False): 67.6,  # QF+INT+UV+VC
    (True, True, True, True, False, False): 66.1,  # QF+INT+UV+FF
    (True, True, False, True, False, False): 62.8, # QF+INT+UV
    (False, True, True, True, False, False): 62.3, # QF+INT+FF
    (False, True, False, True, True, False): 60.4, # QF+INT+VC
    (False, True, False, True, False, False): 57.2,# QF+INT only

    # GS without QF or INT (still critical - need 100% of remaining!)
    (True, False, True, False, True, True): 48.6,  # GS+UV+FF+VC
    (True, False, True, False, False, True): 46.3,  # GS+UV+FF
    (True, False, False, False, True, True): 44.7,  # GS+UV+VC
    (True, False, False, False, False, True): 42.1, # GS+UV
    (False, False, True, False, True, True): 43.8,  # GS+FF+VC
    (False, False, True, False, False, True): 41.4, # GS+FF
    (False, False, False, False, True, True): 39.2, # GS+VC
    (False, False, False, False, False, True): 36.8,# GS only

    # INT without QF or GS (still critical - need 100% of remaining!)
    (True, True, True, False, True, False): 47.9,   # INT+UV+FF+VC
    (True, True, True, False, False, False): 45.7,  # INT+UV+FF
    (True, True, False, False, True, False): 43.2,  # INT+UV+VC
    (True, True, False, False, False, False): 41.3, # INT+UV
    (False, True, True, False, True, False): 42.6,  # INT+FF+VC
    (False, True, True, False, False, False): 40.9, # INT+FF
    (False, True, False, False, True, False): 37.4, # INT+VC
    (False, True, False, False, False, False): 35.1,# INT only

    # QF without GS or INT (still valuable - need 18% of remaining)
    (True, False, True, True, True, False): 28.7,   # QF+UV+FF+VC
    (True, False, True, True, False, False): 26.3,  # QF+UV+FF
    (True, False, False, True, True, False): 24.1,  # QF+UV+VC
    (True, False, False, True, False, False): 21.8, # QF+UV
    (False, False, True, True, True, False): 23.4,  # QF+FF+VC
    (False, False, True, True, False, False): 20.9, # QF+FF
    (False, False, False, True, True, False): 18.6, # QF+VC
    (False, False, False, True, False, False): 16.2,# QF only

    # Fashion forward combinations (need 8% of remaining)
    (True, False, True, False, True, False): 11.3,  # FF+UV+VC
    (True, False, True, False, False, False): 8.7,  # FF+UV
    (False, False, True, False, True, False): 7.2,  # FF+VC
    (False, False, True, False, False, False): 4.9, # FF only

    # Already met constraints (UV, VC) - low priority
    (True, False, False, False, True, False): 3.8,  # UV+VC
    (True, False, False, False, False, False): 2.1, # UV only
    (False, False, False, False, True, False): 1.4, # VC only

    # No useful attributes
    (False, False, False, False, False, False): 0.0, # Nothing
}


def _remaining(state: Any) -> int:
    return max(0, state.N - state.admitted_count)


def _need_map(state: Any) -> Dict[str, int]:
    return {a: max(0, state.constraints[a] - state.counts.get(a, 0)) for a in state.constraints}


def _progress(state: Any, attr: str) -> float:
    """Progress toward meeting the minimum for a given attribute.
    Returns ratio in [0, inf) where 1.0 means the minimum is met.
    """
    m = int(state.constraints.get(attr, 0))
    if m <= 0:
        return 1.0
    return float(state.counts.get(attr, 0)) / float(m)


def _all_constraints_met(state: Any) -> bool:
    for a, M_a in state.constraints.items():
        if state.counts.get(a, 0) < M_a:
            return False
    return True


def _hard_safety(person: Dict[str, bool], state: Any) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        if not person.get(a, False) and (R - 1) < max(0, M_a - state.counts.get(a, 0)):
            return False
    return True


def _feasible_if_accept(person: Dict[str, bool], state: Any) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        cur = state.counts.get(a, 0) + (1 if person.get(a, False) else 0)
        exp_max = cur + max(0, R - 1) * max(0.0, state.freqs.get(a, 0.0))
        if exp_max < M_a:
            return False
    return True


def _person_to_key(person: Dict[str, bool]) -> int:
    key = 0
    for i, attr in enumerate(ATTR_ORDER):
        if person.get(attr, False):
            key |= (1 << i)
    return key

def _tuple_to_key(t: Tuple[bool, bool, bool, bool, bool, bool]) -> int:
    key = 0
    for i, v in enumerate(t):
        if v:
            key |= (1 << i)
    return key


def _weights_required_over_base(state: Any) -> Dict[str, float]:
    W: Dict[str, float] = {}
    for a, M_a in state.constraints.items():
        req = (M_a / state.N) if state.N else 0.0
        p = max(1e-6, float(state.freqs.get(a, 0.0)))
        w = (req / p) if p > 0 else 6.0
        W[a] = max(0.8, min(6.0, w))
    return W


def _build_scores(state: Any) -> List[float]:
    # Build a 64-length score table while ignoring UV and FF.
    # Collapse the curated map across UV/FF and keep the max score
    # for each core configuration of (INT, QF, VC, GS).
    scores = [0.0] * 64

    core_max: Dict[Tuple[bool, bool, bool, bool], float] = {}
    for combo, val in SCORING_MAP.items():
        core = (combo[1], combo[3], combo[4], combo[5])  # (INT, QF, VC, GS)
        core_max[core] = max(core_max.get(core, 0.0), float(val))

    # Fill all 64 combos using the core score, ignoring UV and FF
    for uv in (False, True):
        for intl in (False, True):
            for ff in (False, True):
                for qf in (False, True):
                    for vc in (False, True):
                        for gs in (False, True):
                            core = (intl, qf, vc, gs)
                            base = core_max.get(core, 0.0)

                            bumped = base
                            if vc:
                                bumped += VC_SCORE_BONUS
                            if gs:
                                bumped += GS_SCORE_BONUS
                            if intl:
                                bumped += INT_SCORE_BONUS

                            key_tuple = (uv, intl, ff, qf, vc, gs)
                            scores[_tuple_to_key(key_tuple)] = max(0.0, min(100.0, bumped))

    return scores


def _dynamic_threshold(state: Any) -> float:
    """Granular dynamic threshold on 0–100 scale.
    Optimizes for INT, GS, QF, and VC only (ignores UV/FF).
    """
    remaining = max(1, state.N - state.admitted_count)

    gs_need = max(0, int(state.constraints.get("german_speaker", 0)) - int(state.counts.get("german_speaker", 0)))
    int_need = max(0, int(state.constraints.get("international", 0)) - int(state.counts.get("international", 0)))
    qf_need = max(0, int(state.constraints.get("queer_friendly", 0)) - int(state.counts.get("queer_friendly", 0)))
    vc_need = max(0, int(state.constraints.get("vinyl_collector", 0)) - int(state.counts.get("vinyl_collector", 0)))

    gs_ratio = gs_need / remaining
    int_ratio = int_need / remaining
    qf_ratio = (qf_need / remaining) / 3.0  # QF discounted due to strong gating
    vc_ratio = (vc_need / remaining) / 2.0  # VC moderate weight

    max_ratio = max(gs_ratio, int_ratio, qf_ratio, vc_ratio)

    if max_ratio >= 0.95:
        thr = 35.0  # Super desperate
    elif max_ratio >= 0.85:
        thr = 40.0  # Very desperate
    elif max_ratio >= 0.75:
        thr = 45.0  # Desperate
    elif max_ratio >= 0.60:
        thr = 50.0  # Concerned
    else:
        thr = 55.0  # Comfortable

    # Person-aware gating will enforce QF-only until 95%.
    return thr


def decide(person: Dict[str, bool], state: Any, rejection_history: List[int]) -> bool:
    # Hard safety: if taking this seat without a critical attribute would make minima infeasible later, reject
    if not _hard_safety(person, state):
        return False

    # If all constraints are already met, accept-all to minimize rejections
    if _all_constraints_met(state):
        return True

    # Endgame feasibility check: if accepting makes any target unattainable in expectation, reject
    R = _remaining(state)
    if R <= 120 and not _feasible_if_accept(person, state):
        return False

    # Lazy-init static scores (once per game) on granular 0–100 scale
    global _SCORES
    if _SCORES is None:
        _SCORES = _build_scores(state)

    # Score lookup
    key = _person_to_key(person)
    score = _SCORES[key]

    # Early auto-accept: QF-only until reaching 95% of its minimum
    if person.get("queer_friendly", False) and _progress(state, "queer_friendly") < 0.95:
        return True

    # Until QF progress reaches 95%, block non-QF candidates
    if _progress(state, "queer_friendly") < 0.95 and not person.get("queer_friendly", False):
        return False

    # Dynamic thresholding (0–100 scale)
    threshold = _dynamic_threshold(state)

    # Slightly favor vinyl collectors post-QF phase by easing threshold
    if _progress(state, "queer_friendly") >= 0.95 and person.get("vinyl_collector", False):
        threshold = max(30.0, threshold - 3.0)

    # Final decision
    return score >= threshold
