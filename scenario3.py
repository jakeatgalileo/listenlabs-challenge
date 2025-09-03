from typing import Dict, List


# Scenario 3: correlation-aware strategy with scarcity-weighted scoring
# and dynamic thresholding. Standalone (no imports from bouncer).


def _remaining(state) -> int:
    return max(0, state.N - state.admitted_count)


def _need_map(state) -> Dict[str, int]:
    return {a: max(0, state.constraints[a] - state.counts.get(a, 0)) for a in state.constraints}


def _all_constraints_met(state) -> bool:
    for a, M_a in state.constraints.items():
        if state.counts.get(a, 0) < M_a:
            return False
    return True


def _hard_safety(person: Dict[str, bool], state) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        if not person.get(a, False) and (R - 1) < max(0, M_a - state.counts.get(a, 0)):
            return False
    return True


def _feasible(person: Dict[str, bool], state) -> bool:
    R = _remaining(state)
    for a, M_a in state.constraints.items():
        cur = state.counts.get(a, 0) + (1 if person.get(a, False) else 0)
        exp_max = cur + max(0, R - 1) * max(0.0, state.freqs.get(a, 0.0))
        if exp_max < M_a:
            return False
    return True


def _expected_value(person: Dict[str, bool], state, horizon: int) -> float:
    R = _remaining(state)
    H = min(horizon, R)
    need = _need_map(state)
    direct = 0.0
    for a in state.constraints:
        if person.get(a, False) and need[a] > 0:
            direct += 1.0 / max(1, need[a])

    opp_cost = 0.0
    for a in state.constraints:
        if not person.get(a, False) and need[a] > 0:
            scarcity = 1.0 - max(0.0, min(1.0, state.freqs.get(a, 0.5)))
            opp_cost += scarcity * (need[a] / max(1, R))

    # Correlation lookahead (small weight)
    corr_bonus = 0.0
    for at, v in person.items():
        if not v:
            continue
        for an in state.constraints:
            if need[an] <= 0:
                continue
            c = state.corr.get(at, {}).get(an, 0.0)
            if c > 0:
                corr_bonus += 0.5 * c * max(0.0, state.freqs.get(an, 0.5)) * (need[an] / max(1, R))
    return direct - opp_cost + 0.3 * corr_bonus


def _scarcity_map(state) -> Dict[str, float]:
    R = max(1, _remaining(state))
    need = _need_map(state)
    S: Dict[str, float] = {}
    for a in state.constraints:
        p = max(1e-6, state.freqs.get(a, 0.0))
        S[a] = min(6.0, need[a] / max(1.0, R * p))
    return S


def _weights_required_over_base(state) -> Dict[str, float]:
    """
    Attribute weights ~ required rate / base rate, clipped.
    Encourages rare-but-required attributes (qf, vc, gs, intl).
    """
    W: Dict[str, float] = {}
    for a, M_a in state.constraints.items():
        req = (M_a / state.N) if state.N else 0.0
        p = max(1e-6, state.freqs.get(a, 0.0))
        w = req / p if p > 0 else 5.0
        W[a] = max(0.8, min(6.0, w))
    return W


def _adaptive_threshold(state, rejection_history: List[int], base_start: float = 0.7, window: int = 100) -> float:
    progress = (state.admitted_count / state.N) if state.N else 0.0
    base = base_start * (1.0 - progress)

    # Pressure from most lagging constraint
    max_pressure = 0.0
    for a, M_a in state.constraints.items():
        req = (M_a / state.N) if state.N else 0.0
        cur = state.counts.get(a, 0) / max(1, state.admitted_count)
        max_pressure = max(max_pressure, req - cur)

    if window > 0 and len(rejection_history) >= window:
        recent = rejection_history[-window:]
        rate = sum(recent) / window
        if rate > 0.9:
            base *= 0.8
        elif rate > 0.8:
            base *= 0.9
        elif rate < 0.3 and max_pressure > 0.1:
            base *= 1.15

    return base + max_pressure


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    # 1) Hard safety: never make feasibility impossible with one rejection
    if not _hard_safety(person, state):
        return False

    # 2) If all constraints are already met, accept-all to minimize rejections
    if _all_constraints_met(state):
        return True

    R = _remaining(state)
    need = _need_map(state)
    scarcity = _scarcity_map(state)

    # Promote rare categories while under minimum (subject to endgame feasibility)
    qf = person.get("queer_friendly", False)
    vc = person.get("vinyl_collector", False)
    need_qf = need.get("queer_friendly", 0)
    need_vc = need.get("vinyl_collector", 0)
    if (qf and need_qf > 0) or (vc and need_vc > 0):
        if R <= 120 and not _feasible(person, state):
            return False
        return True

    # 3) Reservation mode phases while qf/vc still below target for non-qf/vc
    qf_under = state.counts.get("queer_friendly", 0) < state.constraints.get("queer_friendly", 0)
    vc_under = state.counts.get("vinyl_collector", 0) < state.constraints.get("vinyl_collector", 0)
    if (qf_under or vc_under) and not (qf or vc):
        # Compute fill ratios for qf/vc relative to their minima
        def _ratio(attr: str) -> float:
            M = max(1, state.constraints.get(attr, 0))
            return state.counts.get(attr, 0) / M

        qf_ratio = _ratio("queer_friendly")
        vc_ratio = _ratio("vinyl_collector")
        focus_ratio = min(qf_ratio, vc_ratio)
        # Stage 1: if focus_ratio < 0.5, only allow non-qf/vc if they have
        # BOTH german_speaker AND international (very selective early)
        if focus_ratio < 0.5:
            # Be stricter with underground_veteran; modestly stricter with fashion_forward.
            gs_ok = person.get("german_speaker", False)
            intl_ok = person.get("international", False)
            ugv = person.get("underground_veteran", False)
            ff = person.get("fashion_forward", False)
            if ugv:
                # Require BOTH gs and intl to admit an underground veteran here
                if not (gs_ok and intl_ok):
                    return False
            elif ff:
                # Require at least one of gs or intl for fashion_forward here
                if not (gs_ok or intl_ok):
                    return False
            # General gate for all other non-qf/vc
            if not (gs_ok and intl_ok):
                return False
        # Stage 2: if 0.5 <= focus_ratio < 0.9, allow non-qf/vc only if they
        # help any attribute that is < 90% of its minimum
        elif focus_ratio < 0.9:
            underfilled_90 = [
                a for a, M in state.constraints.items()
                if M > 0 and state.counts.get(a, 0) < 0.9 * M
            ]
            if not any(person.get(a, False) for a in underfilled_90):
                return False
            # If candidate is underground_veteran but doesn't help hard ones, reject
            if person.get("underground_veteran", False):
                if not (person.get("german_speaker", False) or person.get("international", False)):
                    return False
        # Stage 3: focus_ratio >= 0.9, fall through to normal scoring

    # 4) Scarcity-aware early accept: take highly scarce contributors
    # But when focus is strongly on qf/vc (<0.5), skip non-qf/vc here
    general_criticality = 0.7
    if not ((qf_under or vc_under) and not (qf or vc) and (min(
        state.counts.get("queer_friendly", 0) / max(1, state.constraints.get("queer_friendly", 0)),
        state.counts.get("vinyl_collector", 0) / max(1, state.constraints.get("vinyl_collector", 0)),
    ) < 0.5)):
        scarce_hits = [
            scarcity[a]
            for a in state.constraints
            if person.get(a, False) and need[a] > 0
        ]
        if scarce_hits and max(scarce_hits) >= general_criticality:
            return True

    # 5) High-value correlation rules (qf-vc-gs synergy)
    gs = person.get("german_speaker", False)

    # Accept qf+vc combo if either is still needed (leverages 0.48 correlation)
    if qf and vc and (need_qf > 0 or need_vc > 0):
        return True

    # Strong combos with german speaker when still needed
    if (qf and gs) and (need_qf > 0 or need.get("german_speaker", 0) > 0):
        return True
    if (vc and gs) and (need_vc > 0 or need.get("german_speaker", 0) > 0):
        return True

    # Scarcity-priority for rare attrs
    criticality = 0.75
    if (qf and scarcity.get("queer_friendly", 0.0) >= criticality) or (
        vc and scarcity.get("vinyl_collector", 0.0) >= criticality
    ):
        return True

    # 6) Prioritized deficit-first for the hard attributes (<100%)
    # Keep a slightly stricter gate: only prioritize if < 90% filled or scarce
    for a in ("queer_friendly", "vinyl_collector", "german_speaker", "international"):
        M = state.constraints.get(a, 0)
        c = state.counts.get(a, 0)
        if person.get(a, False) and M > 0:
            if (c < 0.9 * M) or (scarcity.get(a, 0.0) >= 0.7):
                return True

    # 5) Endgame: conservative feasibility + EV
    if R <= 120:
        if not _feasible(person, state):
            return False
        return _expected_value(person, state, horizon=R) > 0.0

    # 6) Scarcity-weighted scoring with required/base weights and correlation bonus
    W = _weights_required_over_base(state)
    s = 0.0

    # Optional per-attribute weight boost/dampen to emphasize uncommon ones
    boost = {
        "queer_friendly": 2.0,
        "vinyl_collector": 2.0,
        "german_speaker": 1.25,
        "international": 1.15,
    }
    dampen = {
        # Stronger dampening on underground_veteran so they contribute less to score
        "underground_veteran": 0.35,
        # Softer dampening on fashion_forward
        "fashion_forward": 0.5,
    }

    # Direct contribution and opportunity penalty
    for a in state.constraints:
        Sa = scarcity[a]
        Wa = W[a] * boost.get(a, 1.0) * dampen.get(a, 1.0)
        if person.get(a, False) and need[a] > 0:
            s += 1.0 * Wa * (1.0 + 0.85 * Sa)
        else:
            if need[a] > 0:
                s -= 0.6 * Wa * (1.0 + 0.7 * Sa)

    # Additional negative bias: if ugv/ff are already at or above min, penalize
    # to reduce their acceptance when they don't help deficits.
    if person.get("underground_veteran", False) and need.get("underground_veteran", 0) <= 0:
        s -= 0.6
    if person.get("fashion_forward", False) and need.get("fashion_forward", 0) <= 0:
        s -= 0.25

    # Correlation-aware bonus toward needed, scarce targets
    for at, v in person.items():
        if not v:
            continue
        for an in state.constraints:
            if need[an] <= 0:
                continue
            c = state.corr.get(at, {}).get(an, 0.0)
            if c <= 0:
                continue
            Sa = scarcity[an]
            Wa = W[an] * boost.get(an, 1.0)
            s += 0.5 * c * Wa * (1.0 + 0.9 * Sa)

    # 7) Adaptive threshold with recent rejection rate and pressure
    threshold = _adaptive_threshold(
        state,
        rejection_history,
        base_start=0.6,
        window=min(100, max(10, state.total_seen if hasattr(state, "total_seen") else 100)),
    )

    # Slight easing when max scarcity is high
    try:
        max_s = max(scarcity.values()) if scarcity else 0.0
        threshold *= (1.0 - min(0.2, 0.05 * max_s))
    except Exception:
        pass

    # Make threshold a bit stricter for ugv/ff when they don't help deficits
    if (person.get("underground_veteran", False) or person.get("fashion_forward", False)) and not (
        (qf and need_qf > 0) or (vc and need_vc > 0) or any(
            person.get(a, False) and need.get(a, 0) > 0 for a in ("german_speaker", "international")
        )
    ):
        threshold *= 1.1

    return s >= threshold
