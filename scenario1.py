from typing import Dict, List
from collections import deque

# Track last accepted candidates' attributes for empirical frequency estimates
_ACCEPTED_WINDOW = deque(maxlen=100)


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


def _record_accept(person: Dict[str, bool]) -> None:
    _ACCEPTED_WINDOW.append(person)


def _observed_freq(attr: str) -> float:
    if not _ACCEPTED_WINDOW:
        return 0.0
    cnt = 0
    for p in _ACCEPTED_WINDOW:
        if p.get(attr, False):
            cnt += 1
    return cnt / len(_ACCEPTED_WINDOW)


def _adjusted_p(attr: str, state) -> float:
    """Weighted prior + observed frequency over last accepted window.
    Target weights: 0.7 prior, 0.3 observed when window >= 50.
    Scale observed weight linearly for smaller samples.
    """
    prior_p = float(state.freqs.get(attr, 0.3225))
    obs_p = _observed_freq(attr)
    # Scale observed weight up to 0.3 by 50 samples
    w_obs = 0.3 * min(1.0, (len(_ACCEPTED_WINDOW) / 50.0) if 50.0 > 0 else 1.0)
    w_prior = 1.0 - w_obs
    p = w_prior * prior_p + w_obs * obs_p
    # Clamp to [0.01, 0.99] to avoid degenerate bounds
    return max(0.01, min(0.99, p))


def decide(person: Dict[str, bool], state, rejection_history: List[int]) -> bool:
    accept = False
    # Fast accept-all once both minimums are met
    need = _need_map(state)
    if all(n <= 0 for n in need.values()):
        accept = True
    else:
        # Strong feasibility guard (must accept/reject cases)
        forced = _must_accept_or_reject_by_feasibility(person, state)
        if "force" in forced:
            accept = bool(forced["force"])  # True: must accept, False: must reject
        else:
            # Expect exactly two attributes in Scenario 1
            attrs = list(state.constraints.keys())
            if len(attrs) < 2:
                # Fallback: if the single attr is present, accept, else reject unless already met
                a = attrs[0]
                accept = person.get(a, False) or (need[a] <= 0)
            else:
                a1, a2 = attrs[0], attrs[1]

                has1 = bool(person.get(a1, False))
                has2 = bool(person.get(a2, False))

                # Simple early policy: until 90% capacity, accept anyone
                # with at least one attribute; accept neither only if LCB
                # feasibility holds for both constraints.
                pre90 = state.admitted_count < int(0.9 * state.N)
                if not accept and pre90:
                    if has1 or has2:
                        accept = True
                    else:
                        if _safe_accept_neither(state):
                            accept = True

                # Sprint-to-finish: if > 90% full and both nearly met (<=20 short), accept anyone with at least one attr
                if state.admitted_count >= 900:
                    nearly1 = (state.constraints[a1] - state.counts.get(a1, 0)) <= 20
                    nearly2 = (state.constraints[a2] - state.counts.get(a2, 0)) <= 20
                    if nearly1 and nearly2 and (has1 or has2):
                        accept = True
                    # else fall through to standard logic

                # Phase logic
                n1 = need[a1]
                n2 = need[a2]

                # Phase A: both below min
                if not accept and n1 > 0 and n2 > 0:
                    if has1 or has2:
                        accept = True
                    else:
                        # Early capacity soak: accept neither if both constraints remain safely feasible
                        if _safe_accept_neither(state):
                            accept = True

                # Phase B: one met, the other not -> primarily accept candidates with the unmet attr
                if not accept and n1 <= 0 and n2 > 0:
                    if has2:
                        accept = True
                    else:
                        accept = _safe_wrong_side_accept(unmet_attr=a2, state=state)
                if not accept and n2 <= 0 and n1 > 0:
                    if has1:
                        accept = True
                    else:
                        accept = _safe_wrong_side_accept(unmet_attr=a1, state=state)

                # Fallback (should be Phase C handled above): reject neither, else accept if any attr
                if not accept:
                    if has1 or has2:
                        accept = True
                    else:
                        if _safe_accept_neither(state):
                            accept = True

    # Record accepted candidate for empirical frequencies
    if accept:
        _record_accept(person)
    return accept


def _safe_wrong_side_accept(unmet_attr: str, state) -> bool:
    """
    Accept a candidate who doesn't help the unmet attribute if, after this
    acceptance, meeting the minimum is still feasible at a conservative LCB.

    Uses a normal-approximation lower confidence bound on future arrivals with
    `unmet_attr` among the remaining R-1 slots. Accept if current + LCB >= minCount.
    Removes margin/schedule gates to reduce unnecessary rejections while preserving
    feasibility via the LCB check.
    """
    R = max(0, state.N - state.admitted_count)
    if R <= 1:
        return False
    cur = int(state.counts.get(unmet_attr, 0))
    M = int(state.constraints.get(unmet_attr, 0))
    need = max(0, M - cur)
    if need <= 0:
        return True
    # Use adjusted empirical frequency
    p = float(_adjusted_p(unmet_attr, state))
    n = R - 1
    mu = n * p
    var = max(1e-6, n * p * (1 - p))
    # More aggressive z: early lower, rises slightly as buffer shrinks
    # z = 0.3 + 0.5 * clamp((N - admitted)/N, 0, 1)
    rem_frac = 0.0
    if state.N > 0:
        rem_frac = max(0.0, min(1.0, (state.N - state.admitted_count) / state.N))
    z = 0.3 + 0.5 * rem_frac
    lcb = mu - z * (var ** 0.5)
    # Require feasibility via LCB (necessary condition)
    if (cur + lcb) < M:
        return False

    # If LCB feasibility holds, accept; drop margin-based gating for responsiveness
    return True


def _safe_accept_neither(state) -> bool:
    """
    Decide whether it's safe to accept a candidate with neither attribute.
    Uses the same lower-confidence-bound feasibility check for each constraint,
    computed over the remaining R-1 spots after taking this neither candidate.

    Margin/schedule gates are removed; feasibility is enforced solely via the LCB
    checks to avoid rejecting safe neither candidates.
    """
    R = max(0, state.N - state.admitted_count)
    if R <= 1:
        return False
    n = R - 1

    attrs = list(state.constraints.keys())
    if len(attrs) < 2:
        return False

    for a in attrs:
        cur = int(state.counts.get(a, 0))
        M = int(state.constraints.get(a, 0))
        p = float(_adjusted_p(a, state))
        mu = n * p
        var = max(1e-6, n * p * (1 - p))
        # same z policy as wrong-side acceptance
        rem_frac = 0.0
        if state.N > 0:
            rem_frac = max(0.0, min(1.0, (state.N - state.admitted_count) / state.N))
        z = 0.3 + 0.5 * rem_frac
        lcb = mu - z * (var ** 0.5)
        if (cur + lcb) < M:
            return False
    # LCB-feasible for both constraints; accept the neither candidate
    return True
