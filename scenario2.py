from typing import Dict, List


"""
Scenario 2: Creative-first with B emphasis.

Implements:
- Hard safety to avoid infeasibility.
- Priority ordering: accept creative (always); then accept B∧T when required overlap is
  positive; accept single B/T only if the other remains feasible.
- Moderate endgame guard at R ≤ 80 using a conservative feasibility check.
- Scoring fallback: scarcity-weighted with explicit B∧T synergy and light
  correlation bonus; adaptive threshold from recent rejection rate and deficits.
 - Strict W gating: reject W unless paired with B or C (i.e., reject W-only and W+T-only).
"""

# Attribute ids used by the API
A_T = "techno_lover"
A_W = "well_connected"
A_C = "creative"
A_B = "berlin_local"

# Tunables (chosen per analysis):
ENDGAME_REMAINING = 80        # conservative finishing window
SCARCITY_CLIP = 5.0
UNION_MARGIN_FRAC = 0.10      # slack fraction for B∧T union feasibility
UNION_MARGIN_MIN = 8
B_STRUGGLE_RATIO = 0.95       # treat B as struggling until 95%
T_STRUGGLE_RATIO = 0.93       # slightly higher to accept more T
SINGLE_ACCEPT_SLACK = 0.05    # 5% slack on feasibility for single-attr B/T admits


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
        if a == A_W:
            continue
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

    # 2) Unconditional creative acceptance (per strategy preference)
    if person.get(A_C, False):
        return True

    # 2.5) Strict W gating per strategy: only accept W when paired with B or C
    has_B = bool(person.get(A_B, False))
    has_T = bool(person.get(A_T, False))
    has_W = bool(person.get(A_W, False))
    if has_W and not (has_B or person.get(A_C, False)):
        return False

    # 3) Moderate endgame guard
    if R <= ENDGAME_REMAINING:
        return _endgame_feasible(person, state)

    # 3) Reconcile B vs T via overlap priority and union control
    need_B = need.get(A_B, 0)
    need_T = need.get(A_T, 0)

    # Current fulfillment ratios
    minB = max(1, int(state.constraints.get(A_B, 1)))
    minT = max(1, int(state.constraints.get(A_T, 1)))
    ratioB = (state.counts.get(A_B, 0) / minB)
    ratioT = (state.counts.get(A_T, 0) / minT)

    # Union feasibility metric U = (need_B + need_T - R)
    U = (need_B + need_T) - R
    union_margin = max(UNION_MARGIN_MIN, int(UNION_MARGIN_FRAC * R))

    # If union pressure is high (U >= margin), only overlapped B∧T helps; reject others
    if U >= union_margin:
        return bool(has_B and has_T)

    overlap_req = _overlap_needed(need, R)
    if has_B and has_T and (overlap_req > 0 or (U < union_margin and (ratioB < 1.0 or ratioT < 1.0))):
        return True

    # W-only and W+T-only were already rejected by strict W gating above

    # 4) Single-attribute admits only if the other remains feasible
    pB = max(0.0, state.freqs.get(A_B, 0.0))
    pT = max(0.0, state.freqs.get(A_T, 0.0))

    if has_B and not has_T and need_B > 0:
        # If T is already safe or fulfilled, lean into B
        if ratioT >= 0.95 and U < union_margin:
            return True
        # Allow slight slack on T feasibility to keep B moving
        if ((R - 1) * pT) >= (need_T * (1.0 - SINGLE_ACCEPT_SLACK)) and U <= 0:
            return True

    if has_T and not has_B and need_T > 0:
        # If B is already safe or fulfilled, lean into T
        if ratioB >= 0.95 and U < union_margin:
            return True
        if ((R - 1) * pB) >= (need_B * (1.0 - SINGLE_ACCEPT_SLACK)) and U <= 0:
            return True

    # (Removed) C-only avoidance and creative reservation – creatives are always accepted above

    # 6.5) Auto-accept struggling attributes for B/T only (with guards)
    ratios: Dict[str, float] = {
        A_B: (state.counts.get(A_B, 0) / max(1, state.constraints.get(A_B, 1))),
        A_T: (state.counts.get(A_T, 0) / max(1, state.constraints.get(A_T, 1))),
    }
    struggling = set()
    for a, r in ratios.items():
        thr = B_STRUGGLE_RATIO if a == A_B else T_STRUGGLE_RATIO
        if r < thr:
            struggling.add(a)
    if U < union_margin and struggling:
        # Priority: B/T (respect other-attr feasibility). C is handled earlier.
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
        # Note: Do not auto-accept W on struggle; it fills naturally.

    # 6) Scoring fallback with scarcity and synergy (B and T weighted higher)
    scarcity: Dict[str, float] = {}
    for a in state.constraints:
        p = max(1e-6, state.freqs.get(a, 0.0))
        scarcity[a] = min(SCARCITY_CLIP, need[a] / max(1.0, R * p)) if R > 0 else (10.0 if need[a] > 0 else 0.0)

    s = 0.0
    # B weighted highest, then T, then W. C is auto-accepted above.
    if has_B:
        s += 1.8 * (1.0 + 2.6 * scarcity.get(A_B, 0.0))
    else:
        if need_B > 0:
            s -= 0.7 * (1.0 + 1.3 * scarcity.get(A_B, 0.0))
    if has_T:
        s += 1.0 * (1.0 + 1.4 * scarcity.get(A_T, 0.0))
    else:
        if need_T > 0:
            s -= 0.35 * (1.0 + 1.0 * scarcity.get(A_T, 0.0))
    # Ignore W in scoring entirely; gating handles its effect.

    # Explicit synergy bonus when both B and T present
    if has_B and has_T:
        s += 2.3

    # Light correlation bonus toward unmet, scarcity-weighted needs
    for a_true, v in person.items():
        if not v:
            continue
        # Only consider correlation toward unmet B/T
        for a_need in (A_B, A_T):
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
