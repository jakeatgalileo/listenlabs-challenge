from typing import Dict, List


# Scenario 2: balanced strategy with adaptive thresholding from recent rejection rate
# and a moderate endgame guard. Does not import from bouncer.


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
    base = 0.5 * (1.0 - progress)
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


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    if not _hard_safety(person, state):
        return False

    R = _remaining(state)
    need = _need_map(state)

    # Strong deficit-first
    for a in state.constraints:
        if need[a] > 0 and person.get(a, False):
            return True

    # Moderate endgame guard
    if R <= 80:
        return _endgame_feasible(person, state)

    # Scored decision with milder scarcity weights
    s = 0.0
    # Scarcity
    scarcity: Dict[str, float] = {}
    for a in state.constraints:
        p = max(1e-6, state.freqs.get(a, 0.0))
        scarcity[a] = min(5.0, need[a] / max(1.0, R * p)) if R > 0 else (10.0 if need[a] > 0 else 0.0)

    for a in state.constraints:
        Sa = scarcity[a]
        if person.get(a, False):
            s += 0.9 * (1.0 + 0.4 * Sa)
        else:
            if need[a] > 0:
                s -= 0.4 * (1.0 + 0.4 * Sa)

    # Correlation bonus but slightly reduced
    for a_true, v in person.items():
        if not v:
            continue
        for a_need in state.constraints:
            if need[a_need] <= 0:
                continue
            c = state.corr.get(a_true, {}).get(a_need, 0.0)
            if c > 0.0:
                s += 0.15 * c * (1.0 + 0.4 * scarcity[a_need])

    threshold = _adaptive_threshold(state, rejection_history)
    return s >= threshold
