from typing import Dict, List


"""
Scenario 2: Creative-first with feasibility guards.

Implements:
- Hard safety to avoid infeasibility.
- Creative reservation: reject non-creative when remaining capacity cannot safely
  cover remaining creative need.
- Priority ordering: accept creative; then accept B∧T when required overlap is
  positive; accept single B/T only if the other remains feasible.
- Moderate endgame guard at R ≤ 80 using a conservative feasibility check.
- Scoring fallback: scarcity-weighted with explicit B∧T synergy and light
  correlation bonus; adaptive threshold from recent rejection rate and deficits.
"""

# Attribute ids used by the API
A_T = "techno_lover"
A_W = "well_connected"
A_C = "creative"
A_B = "berlin_local"

# Tunables (chosen per analysis):
ALPHA_CREATIVE_RESERVE = 1.1  # slightly lower to avoid over-reserving for C
ENDGAME_REMAINING = 80        # conservative finishing window
SCARCITY_CLIP = 5.0
UNION_MARGIN_FRAC = 0.10      # slack fraction for B∧T union feasibility
UNION_MARGIN_MIN = 8


def _remaining(state) -> int:
    return max(0, state.N - state.admitted_count)


def _need_map(state) -> Dict[str, int]:
    return {a: max(0, state.constraints[a] - state.counts.get(a, 0)) for a in state.constraints}


def _hard_safety(person: Dict[str, bool], state) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        if not person.get(a, False) and (R - 1) < max(0, M_a - state.counts.get(a, 0)):
            return False
    return True


def _endgame_feasible(person: Dict[str, bool], state) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        cur = state.counts.get(a, 0) + (1 if person.get(a, False) else 0)
        exp_max = cur + max(0, R - 1) * max(0.0, state.freqs.get(a, 0.0))
        if exp_max < M_a:
            return False
    return True


def _adaptive_threshold(state, rejection_history: List[int]) -> float:
    progress = (state.admitted_count / state.N) if state.N else 0.0
    base = 0.45 * (1.0 - progress)
    # Pressure from deficits
    pressure = 0.0
    for a, M_a in state.constraints.items():
        req = (M_a / state.N) if state.N else 0.0
        cur_rate = state.counts.get(a, 0) / max(1, state.admitted_count)
        pressure = max(pressure, req - cur_rate)
    # Recent rejection rate modulation
    if rejection_history:
        window = rejection_history[-min(100, len(rejection_history)) :]
        rej_rate = sum(window) / len(window)
        if rej_rate > 0.85:
            base *= 0.8
        elif rej_rate > 0.7:
            base *= 0.9
        elif rej_rate < 0.25 and pressure > 0.1:
            base *= 1.15
    return base + pressure


def _overlap_needed(need: Dict[str, int], R: int) -> int:
    # Required overlap between B and T to fit both in remaining R seats
    b = need.get(A_B, 0)
    t = need.get(A_T, 0)
    return max(0, b + t - R)


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    # 1) Hard safety
    if not _hard_safety(person, state):
        return False

    R = _remaining(state)
    need = _need_map(state)

    # 2) Moderate endgame guard
    if R <= ENDGAME_REMAINING:
        return _endgame_feasible(person, state)

    # 3) Reconcile B vs T via overlap priority and union control
    need_B = need.get(A_B, 0)
    need_T = need.get(A_T, 0)
    need_W = need.get(A_W, 0)
    has_B = bool(person.get(A_B, False))
    has_T = bool(person.get(A_T, False))
    has_W = bool(person.get(A_W, False))

    # Union feasibility metric U = (need_B + need_T - R)
    U = (need_B + need_T) - R
    union_margin = max(UNION_MARGIN_MIN, int(UNION_MARGIN_FRAC * R))

    # If union pressure is high (U >= margin), only overlapped B∧T helps; reject others
    if U >= union_margin:
        return bool(has_B and has_T)

    overlap_req = _overlap_needed(need, R)
    if has_B and has_T and overlap_req > 0:
        return True

    # 4) Single-attribute admits only if the other remains feasible
    pB = max(0.0, state.freqs.get(A_B, 0.0))
    pT = max(0.0, state.freqs.get(A_T, 0.0))

    if has_B and not has_T and need_B > 0:
        if (R - 1) * pT >= need_T and U < 0:
            return True
        if (
            U < 0
            and need_W > 0
            and has_W
            and (R - 1) * pT >= need_T
            and (R - 1) * pB >= need_B
        ):
            return True

    if has_T and not has_B and need_T > 0:
        if (R - 1) * pB >= need_B and U < 0:
            return True
        if (
            U < 0
            and need_W > 0
            and has_W
            and (R - 1) * pT >= need_T
            and (R - 1) * pB >= need_B
        ):
            return True

    # 5) If union is tight but not critical (-margin <= U < margin), avoid C-only admits
    if -union_margin <= U < union_margin and person.get(A_C, False) and not (has_B or has_T):
        # allow only if creative need is still pressing
        if need_C > 0 and (R - 1) * pC < 1.15 * need_C:
            pass
        else:
            return False

    # 6) Creative reservation (after union handling): preserve capacity for C except when B∧T present
    need_C = need.get(A_C, 0)
    pC = max(1e-6, state.freqs.get(A_C, 0.0))
    if not (has_B and has_T):
        if not person.get(A_C, False) and need_C > 0:
            if (R - 1) * pC < ALPHA_CREATIVE_RESERVE * need_C:
                return False
        # Always take creatives while creative minimum unmet (unless union is critical handled above)
        if person.get(A_C, False) and need_C > 0:
            return True

    # 6.5) Auto-accept struggling attributes until 90% (with guards)
    ratios: Dict[str, float] = {a: (state.counts.get(a, 0) / max(1, state.constraints.get(a, 1))) for a in state.constraints}
    struggling = {a for a in state.constraints if ratios[a] < 0.90}
    if U < union_margin and struggling:
        # Priority: C, then B/T (respect other-attr feasibility), then W
        if person.get(A_C, False) and A_C in struggling:
            return True
        # B-only struggling
        if (A_B in struggling) and has_B and not has_T:
            if (R - 1) * pT >= need_T:
                return True
        # T-only struggling
        if (A_T in struggling) and has_T and not has_B:
            if (R - 1) * pB >= need_B:
                return True
        # B∧T always helps when struggling and not union-critical
        if (A_B in struggling or A_T in struggling) and has_B and has_T:
            return True
        # W when struggling and not union-critical
        if (A_W in struggling) and has_W:
            return True

    # 7) Scoring fallback with scarcity and synergy
    scarcity: Dict[str, float] = {}
    for a in state.constraints:
        p = max(1e-6, state.freqs.get(a, 0.0))
        scarcity[a] = min(SCARCITY_CLIP, need[a] / max(1.0, R * p)) if R > 0 else (10.0 if need[a] > 0 else 0.0)

    s = 0.0
    # Heavy weight for creative, stronger for uncommon B/T under scarcity, light for W
    if person.get(A_C, False):
        s += 1.0 * (1.0 + 3.0 * scarcity.get(A_C, 0.0))
    if has_B:
        s += 1.0 * (1.0 + 1.8 * scarcity.get(A_B, 0.0))
    else:
        if need_B > 0:
            s -= 0.5 * (1.0 + 1.2 * scarcity.get(A_B, 0.0))
    if has_T:
        s += 1.0 * (1.0 + 1.5 * scarcity.get(A_T, 0.0))
    else:
        if need_T > 0:
            s -= 0.5 * (1.0 + 1.1 * scarcity.get(A_T, 0.0))
    if has_W:
        s += 0.3 * (1.0 + 0.5 * scarcity.get(A_W, 0.0))

    # Explicit synergy bonus when both B and T present
    if has_B and has_T:
        s += 2.0

    # Light correlation bonus toward unmet, scarcity-weighted needs
    for a_true, v in person.items():
        if not v:
            continue
        for a_need in state.constraints:
            if need.get(a_need, 0) <= 0:
                continue
            c = float(state.corr.get(a_true, {}).get(a_need, 0.0))
            if c > 0.0:
                s += 0.12 * c * (1.0 + 0.6 * scarcity.get(a_need, 0.0))

    # If union is tight, penalize candidates lacking both B and T
    if -union_margin <= U < union_margin and not (has_B or has_T):
        s -= 0.5

    threshold = _adaptive_threshold(state, rejection_history)
    return s >= threshold
