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
    """
    Phase-based strategy tuned for Scenario 1 (two symmetric attributes).

    Laxer on accepting (0,0) when safe to push meeting 600/600 closer to the end,
    guarded by an LCB feasibility check.
    """

    def _impl() -> bool:
        has_young = bool(person.get("young", False))
        has_wd = bool(person.get("well_dressed", False))

        M_young = int(state.constraints.get("young", 0))
        M_wd = int(state.constraints.get("well_dressed", 0))
        cur_young = int(state.counts.get("young", 0))
        cur_wd = int(state.counts.get("well_dressed", 0))

        young_need = max(0, M_young - cur_young)
        wd_need = max(0, M_wd - cur_wd)
        remaining = max(0, state.N - state.admitted_count)

        # Phase C: both constraints met -> accept everyone
        if young_need == 0 and wd_need == 0:
            return True

        # Feasibility guard: must accept if rejecting would make it impossible to meet a min
        if remaining <= young_need:
            return has_young
        if remaining <= wd_need:
            return has_wd

        # Calibrated base rates and correlation insight
        p_young = 0.3225
        p_wd = 0.3225
        correlation = 0.183  # Positive correlation; (1,1) occurs more than independence

        # Expected future arrivals for each attribute among remaining-1 after this decision
        expected_young = max(0.0, (remaining - 1) * p_young)
        expected_wd = max(0.0, (remaining - 1) * p_wd)

        # Helper: dynamic probability for accepting a (neither) candidate when safe
        def _neither_accept_prob() -> float:
            a = state.admitted_count
            if a < 300:
                return 0.60
            if a < 700:
                return 0.40
            if a < 900:
                return 0.25
            return 0.12

        # Phase A: Both constraints unmet
        if young_need > 0 and wd_need > 0:
            # (1,1): always good
            if has_young and has_wd:
                return True
            # Single-attribute: be selective using safety margin for that attribute
            if has_young ^ has_wd:
                safety_margin = (expected_young - young_need) if has_young else (expected_wd - wd_need)
                if safety_margin < -5:
                    return True  # behind; take it
                elif safety_margin < 10:
                    return random.random() < 0.7
                elif safety_margin < 20:
                    return random.random() < 0.4
                else:
                    return random.random() < 0.2
            # Neither: if safe, accept with dynamic probability to delay hitting mins
            if _safe_accept_neither(state):
                return random.random() < _neither_accept_prob()
            return False

        # Phase B: young met, need well_dressed
        if young_need == 0 and wd_need > 0:
            if has_wd:
                return True
            if not has_young and not has_wd:
                if _safe_accept_neither(state):
                    return random.random() < _neither_accept_prob()
            if has_young:
                safety_margin = expected_wd - wd_need
                if safety_margin > 20:
                    return True
                elif safety_margin > 10:
                    return random.random() < 0.5
                elif safety_margin > 5:
                    return random.random() < 0.2
                return False

        # Phase B: well_dressed met, need young
        if wd_need == 0 and young_need > 0:
            if has_young:
                return True
            if not has_young and not has_wd:
                if _safe_accept_neither(state):
                    return random.random() < _neither_accept_prob()
            if has_wd:
                safety_margin = expected_young - young_need
                if safety_margin > 20:
                    return True
                elif safety_margin > 10:
                    return random.random() < 0.5
                elif safety_margin > 5:
                    return random.random() < 0.2
                return False

        # Default fallback (should be unreachable)
        return True

    decision = _impl()
    if decision:
        _record_accept(person)
    return decision


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
