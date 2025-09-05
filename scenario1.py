from typing import Dict, List
from collections import deque
import random

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


# Removed older feasibility helpers that are no longer referenced by the
# current phase-based strategy to keep the module lean.


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
    """
    Greedy strategy (per request):
    - Accept any candidate who has at least one of the constrained attributes.
    - Reject candidates with neither attribute.
    - Once both minima are met, accept everyone.
    - Always enforce feasibility: if accepting would make it impossible to
      meet any remaining minimum (because the candidate lacks that attribute
      and there aren’t enough seats left), reject.
    """

    # If both minima are already met, accept-all to minimize rejections
    needs = {a: max(0, int(state.constraints.get(a, 0)) - int(state.counts.get(a, 0))) for a in state.constraints}
    if all(n == 0 for n in needs.values()):
        return True

    # Hard feasibility: if candidate lacks some attr `a` and taking this seat
    # would leave fewer than `need[a]` seats, we must reject.
    R = max(0, int(state.N) - int(state.admitted_count))
    for a, need in needs.items():
        if not person.get(a, False) and (R - 1) < need:
            return False

    # Greedy accept if any constrained attribute is present; else reject
    has_any = any(person.get(a, False) for a in state.constraints)
    decision = bool(has_any)
    if decision:
        _record_accept(person)
    return decision


# Removed older wrong-side acceptance helper that is not referenced anymore.


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
