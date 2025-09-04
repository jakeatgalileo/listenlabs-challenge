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
    # With the granular map we directly fill a 64-size array;
    # unspecified combos default to 0.0 (very low priority).
    scores = [0.0] * 64
    for combo, val in SCORING_MAP.items():
        scores[_tuple_to_key(combo)] = float(val)
    return scores


def _dynamic_threshold(state: Any) -> float:
    """Granular dynamic threshold on 0–100 scale.
    Uses desperation ratios of GS/INT/QF relative to remaining slots.
    """
    remaining = max(1, state.N - state.admitted_count)

    gs_need = max(0, int(state.constraints.get("german_speaker", 0)) - int(state.counts.get("german_speaker", 0)))
    int_need = max(0, int(state.constraints.get("international", 0)) - int(state.counts.get("international", 0)))
    qf_need = max(0, int(state.constraints.get("queer_friendly", 0)) - int(state.counts.get("queer_friendly", 0)))

    gs_ratio = gs_need / remaining
    int_ratio = int_need / remaining
    qf_ratio = (qf_need / remaining) / 3.0  # QF 3x discounted due to rarity weighting

    max_ratio = max(gs_ratio, int_ratio, qf_ratio)

    if max_ratio >= 0.95:
        return 35.0  # Super desperate
    elif max_ratio >= 0.85:
        return 40.0  # Very desperate
    elif max_ratio >= 0.75:
        return 45.0  # Desperate
    elif max_ratio >= 0.60:
        return 50.0  # Concerned
    else:
        return 55.0  # Comfortable


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

    # Dynamic thresholding (0–100 scale)
    threshold = _dynamic_threshold(state)

    # Final decision
    return score >= threshold
