from typing import Dict, List


# Scenario 3: stricter policy with larger endgame and a forward-looking
# expected value check. No imports from bouncer.


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


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    if not _hard_safety(person, state):
        return False

    R = _remaining(state)
    need = _need_map(state)

    # Strong deficit-first
    for a in state.constraints:
        if need[a] > 0 and person.get(a, False):
            return True

    # Larger endgame window with feasibility and EV check
    if R <= 120:
        if not _feasible(person, state):
            return False
        return _expected_value(person, state, horizon=R) > 0.0

    # Baseline scoring: stricter penalties
    s = 0.0
    # Scarcity
    scarcity: Dict[str, float] = {}
    for a in state.constraints:
        p = max(1e-6, state.freqs.get(a, 0.0))
        scarcity[a] = min(6.0, need[a] / max(1.0, R * p)) if R > 0 else (10.0 if need[a] > 0 else 0.0)

    for a in state.constraints:
        Sa = scarcity[a]
        if person.get(a, False):
            s += 0.85 * (1.0 + 0.6 * Sa)
        else:
            if need[a] > 0:
                s -= 0.55 * (1.0 + 0.6 * Sa)

    # Correlation bonus smaller
    for at, v in person.items():
        if not v:
            continue
        for an in state.constraints:
            if need[an] <= 0:
                continue
            c = state.corr.get(at, {}).get(an, 0.0)
            if c > 0:
                s += 0.12 * c * (1.0 + 0.5 * scarcity[an])

    # Dynamic threshold slightly higher baseline
    progress = (state.admitted_count / state.N) if state.N else 0.0
    base = 0.65 * (1.0 - progress)
    pressure = 0.0
    for a, M_a in state.constraints.items():
        req = (M_a / state.N) if state.N else 0.0
        cur = state.counts.get(a, 0) / max(1, state.admitted_count)
        pressure = max(pressure, req - cur)
    threshold = base + pressure
    return s >= threshold
