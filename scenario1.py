from typing import Dict, List


# Greedy approach
# Scenario 1: Two symmetric attributes with equal minCounts (e.g., young and well_dressed).
# Requested policy:
# - Phase A (both below min): accept anyone with at least one attribute; reject neither.
# - Phase B (one met, other not): accept only candidates who have the unmet attribute (including both-true);
#   reject single-attribute candidates from the group that has already met its minimum, and reject neither.
# - Phase C (both met): accept all to minimize rejections.
# - Always enforce feasibility: if rejecting would make it impossible to reach a min, force-accept; if a person
#   does not have a critical attr and rejecting them would make it impossible, force-reject.


def _remaining(state) -> int:
    return max(0, state.N - state.admitted_count)


def _need_map(state) -> Dict[str, int]:
    return {a: max(0, state.constraints[a] - state.counts.get(a, 0)) for a in state.constraints}


def _must_accept_or_reject_by_feasibility(person: Dict[str, bool], state) -> Dict[str, bool]:
    """Return dict with optional forced decisions: {"force": True/False} if applicable.
    - If rejecting this person would make it impossible to reach M[a] (R-1 < need[a])
      and the person HAS attribute a, we must accept.
    - If rejecting this person would make it impossible to reach M[a] and the person
      does NOT have attribute a, we must reject.
    If multiple attributes trigger conflicting decisions, accepting wins (since it helps).
    """
    R = _remaining(state)
    need = _need_map(state)
    must_accept = False
    must_reject = False
    for a, n in need.items():
        if (R - 1) < n:
            if person.get(a, False):
                must_accept = True
            else:
                must_reject = True
    if must_accept:
        return {"force": True}
    if must_reject:
        return {"force": False}
    return {}


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    # Fast accept-all once both minimums are met
    need = _need_map(state)
    if all(n <= 0 for n in need.values()):
        return True

    # Strong feasibility guard (must accept/reject cases)
    forced = _must_accept_or_reject_by_feasibility(person, state)
    if "force" in forced:
        return bool(forced["force"])

    # Expect exactly two attributes in Scenario 1
    attrs = list(state.constraints.keys())
    if len(attrs) < 2:
        # Fallback: if the single attr is present, accept, else reject unless already met
        a = attrs[0]
        return person.get(a, False) or (need[a] <= 0)
    a1, a2 = attrs[0], attrs[1]

    has1 = bool(person.get(a1, False))
    has2 = bool(person.get(a2, False))

    # Phase logic
    n1 = need[a1]
    n2 = need[a2]

    # Phase A: both below min -> accept if at least one attr
    if n1 > 0 and n2 > 0:
        return has1 or has2

    # Phase B: one met, the other not -> primarily accept candidates with the unmet attr
    # Additionally, near the end, allow safe "wrong-side" acceptance if stats suggest
    # we can still reach the unmet minimum.
    if n1 <= 0 and n2 > 0:
        if has2:
            return True
        return _safe_wrong_side_accept(unmet_attr=a2, state=state)
    if n2 <= 0 and n1 > 0:
        if has1:
            return True
        return _safe_wrong_side_accept(unmet_attr=a1, state=state)

    # Fallback (should be Phase C handled above): reject neither, else accept
    return has1 or has2


def _safe_wrong_side_accept(unmet_attr: str, state) -> bool:
    """
    Allow accepting a candidate who doesn't help the unmet attribute when remaining
    capacity and estimated frequency still make meeting the minimum highly likely.

    Normal-approximation lower confidence bound on future arrivals with `unmet_attr`
    among the remaining R-1 slots. Accept if current + LCB >= minCount.
    """
    R = max(0, state.N - state.admitted_count)
    if R <= 1:
        return False
    cur = int(state.counts.get(unmet_attr, 0))
    M = int(state.constraints.get(unmet_attr, 0))
    need = max(0, M - cur)
    if need <= 0:
        return True
    p = float(state.freqs.get(unmet_attr, 0.2))
    n = R - 1
    mu = n * p
    var = max(1e-6, n * p * (1 - p))
    # z decays as we approach the end: early ~1.5, late ~0.5
    z = 0.5 + 1.0 * min(1.0, max(0.0, (n / 200.0)))
    lcb = mu - z * (var ** 0.5)
    return (cur + lcb) >= M
