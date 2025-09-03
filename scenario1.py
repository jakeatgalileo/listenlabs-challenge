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

    # Phase B: one met, the other not -> accept only if candidate has the unmet attr
    if n1 <= 0 and n2 > 0:
        return has2  # includes both-true
    if n2 <= 0 and n1 > 0:
        return has1  # includes both-true

    # Fallback (should be Phase C handled above): reject neither, else accept
    return has1 or has2
