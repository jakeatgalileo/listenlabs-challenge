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

        # Expected future arrivals for each attribute among remaining-1 after this decision
        expected_young = max(0.0, (remaining - 1) * p_young)
        expected_wd = max(0.0, (remaining - 1) * p_wd)

        # Helper: dynamic probability for accepting a (neither) candidate when safe
        def _neither_accept_prob() -> float:
            """More aggressive acceptance for (neither) — especially near end.

            Base schedule is bumped up overall, and we apply an endgame ramp
            based on remaining spots to be deliberately more permissive while the
            LCB feasibility guard continues to ensure we can still reach 600/600.
            """
            a = state.admitted_count
            R = max(0, state.N - state.admitted_count)

            # Bumped base schedule by admitted progress
            if a < 300:
                p = 0.70  # was 0.60
            elif a < 700:
                p = 0.55  # was 0.40
            elif a < 900:
                p = 0.40  # was 0.25
            else:
                p = 0.25  # was 0.12

            # Endgame ramp: as remaining shrinks, be even more permissive
            if R <= 30:
                p = max(p, 0.95)
            elif R <= 50:
                p = max(p, 0.85)
            elif R <= 80:
                p = max(p, 0.70)
            elif R <= 120:
                p = max(p, 0.50)

            return p

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
