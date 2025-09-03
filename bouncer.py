import argparse
import math
import sys
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import json

"""
Core Strategy Highlights:

1. Dynamic Threshold System: Start selective when you have many spots, become less selective as the venue fills up
2. Constraint Scoring: Score each person based on how much they help meet unfulfilled constraints
3. Adaptive Learning: Update your estimates of attribute frequencies based on who you've seen so far
4. Conservative Endgame: When almost full (last ~50 spots), only accept people who don't make constraints impossible to meet
5. Correlation Exploitation: Use the correlation matrix to predict future arrivals - if certain attributes tend to come together, you can plan ahead
"""

def get(url: str, path: str, params: Dict[str, Any]) -> Any:
    global SESSION
    if SESSION is None:
        retry = Retry(
            total=DEFAULT_RETRIES,
            connect=DEFAULT_RETRIES,
            read=DEFAULT_RETRIES,
            status=DEFAULT_RETRIES,
            backoff_factor=DEFAULT_BACKOFF,
            allowed_methods=frozenset(["GET"]),
            status_forcelist=[429, 500, 502, 503, 504],
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
        s = requests.Session()
        s.headers.update({"User-Agent": "listenlabs-bouncer/1.0"})
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        SESSION = s

    r = SESSION.get(url.rstrip("/") + path, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def start_game(base_url: str, scenario: int, player_id: str) -> Any:
    return get(base_url, "/new-game", {"scenario": scenario, "playerId": player_id})


def decide_and_next(base_url: str, game_id: str, person_index: int, accept: Optional[bool]) -> Any:
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        params["accept"] = "true" if accept else "false"
    return get(base_url, "/decide-and-next", params)


def decide(
    attrs: Dict[str, bool],
    k: int,
    N: int,
    M: Dict[str, int],
    counts: Dict[str, int],
    lambdas: Dict[str, float],
    r_targets: Dict[str, float],
    tau: float,
) -> bool:
    R = N - k
    need = {a: max(0, M[a] - counts.get(a, 0)) for a in M}

    # Safety
    for a, na in need.items():
        ya = 1 if attrs.get(a, False) else 0
        if ya == 0 and R - 1 < na:
            return False

    # Deficit-first
    for a, na in need.items():
        if na > 0 and attrs.get(a, False):
            return True

    # Dual score
    s = 0.0
    for a, ra in r_targets.items():
        ya = 1.0 if attrs.get(a, False) else 0.0
        s += lambdas.get(a, 0.0) * (ra - ya)
    return s <= tau


def update_on_accept(
    attrs: Dict[str, bool],
    counts: Dict[str, int],
    lambdas: Dict[str, float],
    r_targets: Dict[str, float],
    eta: float,
):
    for a in r_targets:
        if attrs.get(a, False):
            counts[a] = counts.get(a, 0) + 1
    for a, ra in r_targets.items():
        ya = 1.0 if attrs.get(a, False) else 0.0
        lambdas[a] = max(0.0, lambdas.get(a, 0.0) + eta * (ra - ya))

DEFAULT_N = 1000
DEFAULT_ENDGAME_SIZE = 50
DEFAULT_W_HELP = 2.0
DEFAULT_W_PENALTY = 0.5
DEFAULT_W_CORR = 0.1
DEFAULT_BASE_START = 0.5
DEFAULT_K0 = 100
DEFAULT_BETA = 0.0
VERBOSE = False

# Networking defaults
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 0.5
SESSION: Optional[requests.Session] = None
TIMEOUT: Tuple[float, float] = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)

@dataclass
class GameState:
    N: int
    constraints: Dict[str, int]               # target counts M[a]
    counts: Dict[str, int]                    # admitted counts per attribute
    freqs: Dict[str, float]                   # relative frequencies P(attr=True)
    corr: Dict[str, Dict[str, float]]         # correlations between attributes
    admitted_count: int = 0
    rejected_local: int = 0
    seen_counts: Dict[str, int] = field(default_factory=dict)  # observed True counts
    total_seen: int = 0
    accept_all_after_min_met: bool = False


def hard_safety(person: Dict[str, bool], state: GameState) -> bool:
    R = state.N - state.admitted_count
    for a, M_a in state.constraints.items():
        need = max(0, M_a - state.counts.get(a, 0))
        if not person.get(a, False) and (R - 1) < need:
            return False
    return True


def deficit_first(person: Dict[str, bool], state: GameState) -> bool:
    for a, M_a in state.constraints.items():
        if (M_a - state.counts.get(a, 0)) > 0 and person.get(a, False):
            return True
    return False


def constraint_score(
    person: Dict[str, bool],
    state: GameState,
    w_help: float,
    w_penalty: float,
    w_corr: float,
) -> float:
    R = max(1, state.N - state.admitted_count)
    s = 0.0
    need = {a: max(0, state.constraints[a] - state.counts.get(a, 0)) for a in state.constraints}
    urgency = {a: need[a] / R for a in state.constraints}

    # Direct contribution
    for a in state.constraints:
        if person.get(a, False):
            s += w_help * urgency[a]
        else:
            s -= w_penalty * urgency[a]

    # Correlation bonus
    for a1, v in person.items():
        if not v:
            continue
        for a2, u in urgency.items():
            if need[a2] <= 0:
                continue
            c = state.corr.get(a1, {}).get(a2, 0.0)
            if c > 0:
                s += w_corr * c * u
    return s


def dynamic_threshold(state: GameState, base_start: float) -> float:
    progress = state.admitted_count / state.N if state.N > 0 else 0.0
    base = base_start * (1.0 - progress)

    max_pressure = 0.0
    for a, M_a in state.constraints.items():
        required_rate = M_a / state.N if state.N > 0 else 0.0
        current_rate = state.counts.get(a, 0) / max(1, state.admitted_count)
        pressure = required_rate - current_rate
        if pressure > max_pressure:
            max_pressure = pressure
    return base + max_pressure


def conservative_endgame(person: Dict[str, bool], state: GameState) -> bool:
    R = state.N - state.admitted_count
    for a, M_a in state.constraints.items():
        cur = state.counts.get(a, 0) + (1 if person.get(a, False) else 0)
        exp_max = cur + (R - 1) * state.freqs.get(a, 0.0)
        if exp_max < M_a:
            return False
    return True


def update_adaptive_freqs(state: GameState, k0: int = 100) -> None:
    if state.total_seen <= 0:
        return
    w = min(0.9, k0 / max(1, state.total_seen))
    for a in list(state.freqs.keys()):
        obs = state.seen_counts.get(a, 0) / state.total_seen
        state.freqs[a] = w * state.freqs[a] + (1.0 - w) * obs


# =====================
# Enhanced optimizations
# =====================

def all_constraints_met(state: GameState) -> bool:
    for a, M_a in state.constraints.items():
        if state.counts.get(a, 0) < M_a:
            return False
    return True

def enhanced_correlation_score(
    person: Dict[str, bool],
    state: GameState,
    lookahead_depth: int = 3,
) -> float:
    """
    Better use of correlations to predict future value
    """
    score = 0.0
    R = state.N - state.admitted_count

    for attr1, has_attr in person.items():
        if not has_attr:
            continue
        for attr2 in state.constraints:
            if attr2 == attr1:
                continue
            correlation = state.corr.get(attr1, {}).get(attr2, 0.0)
            if correlation <= 0:
                continue
            need = max(0, state.constraints[attr2] - state.counts.get(attr2, 0))
            if need <= 0:
                continue
            expected_future = correlation * state.freqs.get(attr2, 0.5)
            discount = 1.0 / (1.0 + lookahead_depth * (1.0 - correlation))
            score += expected_future * discount * (need / R)

    return score


@dataclass
class ScenarioConfig:
    """Tuned parameters for each scenario"""
    endgame_size: int
    w_help: float
    w_penalty: float
    w_corr: float
    base_start: float
    buffer_multiplier: float  # For constraint buffering


SCENARIO_CONFIGS: Dict[int, ScenarioConfig] = {
    1: ScenarioConfig(
        endgame_size=30,
        w_help=1.5,
        w_penalty=0.3,
        w_corr=0.15,
        base_start=0.3,
        buffer_multiplier=0.0,
    ),
    2: ScenarioConfig(
        endgame_size=50,
        w_help=2.0,
        w_penalty=0.5,
        w_corr=0.2,
        base_start=0.5,
        buffer_multiplier=0.05,
    ),
    3: ScenarioConfig(
        endgame_size=100,
        w_help=3.0,
        w_penalty=0.8,
        w_corr=0.3,
        base_start=0.7,
        buffer_multiplier=0.1,
    ),
}


def adaptive_threshold_with_history(
    state: GameState,
    base_start: float,
    rejection_history: List[int],
    window_size: int = 100,
) -> float:
    """
    Adjust threshold based on recent rejection rate
    """
    progress = state.admitted_count / state.N
    base = base_start * (1.0 - progress)

    max_pressure = 0.0
    for a, M_a in state.constraints.items():
        required_rate = M_a / state.N
        current_rate = state.counts.get(a, 0) / max(1, state.admitted_count)
        pressure = required_rate - current_rate
        max_pressure = max(max_pressure, pressure)

    if len(rejection_history) >= window_size:
        recent_rejections = sum(rejection_history[-window_size:])
        recent_rate = recent_rejections / window_size
        if recent_rate > 0.9:
            base *= 0.8
        elif recent_rate > 0.8:
            base *= 0.9
        elif recent_rate < 0.3 and max_pressure > 0.1:
            base *= 1.2

    return base + max_pressure


def prioritized_deficit_check(
    person: Dict[str, bool],
    state: GameState,
) -> Tuple[bool, float]:
    """
    Prioritize constraints by criticality
    """
    R = state.N - state.admitted_count

    critical_attrs: List[Tuple[str, float]] = []
    for a, M_a in state.constraints.items():
        need = max(0, M_a - state.counts.get(a, 0))
        if need <= 0:
            continue
        max_possible = need + R * state.freqs.get(a, 1.0)
        criticality = need / max(0.01, max_possible - need)
        if person.get(a, False):
            critical_attrs.append((a, criticality))

    if critical_attrs:
        max_criticality = max(c for _, c in critical_attrs)
        return True, max_criticality

    return False, 0.0


def calculate_expected_value(
    person: Dict[str, bool],
    state: GameState,
    future_horizon: int = 50,
) -> float:
    """
    Calculate expected value of accepting this person considering opportunity cost
    """
    R = state.N - state.admitted_count
    horizon = min(future_horizon, R)

    direct_value = 0.0
    for a, M_a in state.constraints.items():
        if not person.get(a, False):
            continue
        need = max(0, M_a - state.counts.get(a, 0))
        if need > 0:
            direct_value += 1.0 / max(1, need)

    opportunity_cost = 0.0
    for a in state.constraints:
        if person.get(a, False):
            continue
        need = max(0, state.constraints[a] - state.counts.get(a, 0))
        if need > 0:
            scarcity = 1.0 - state.freqs.get(a, 0.5)
            opportunity_cost += scarcity * (need / R)

    future_value = enhanced_correlation_score(person, state, lookahead_depth=3)

    return direct_value - opportunity_cost + 0.5 * future_value


def decide_enhanced(
    person: Dict[str, bool],
    state: GameState,
    scenario: int,
    rejection_history: List[int],
) -> bool:
    """
    Enhanced decision function with all optimizations
    """
    config = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS[2])

    if not hard_safety(person, state):
        return False

    # If we've already satisfied all minimum constraints, accept everyone to minimize rejections
    if all_constraints_met(state):
        if not state.accept_all_after_min_met:
            sys.stderr.write("all minimum constraints met; switching to accept-all mode\n")
            sys.stderr.flush()
            state.accept_all_after_min_met = True
        return True

    accept, criticality = prioritized_deficit_check(person, state)
    if accept and criticality > 0.5:
        return True

    R = state.N - state.admitted_count
    if R <= config.endgame_size:
        if conservative_endgame(person, state):
            ev = calculate_expected_value(person, state, future_horizon=R)
            return ev > 0
        return False

    base_score = constraint_score(
        person,
        state,
        config.w_help,
        config.w_penalty,
        config.w_corr,
    )

    correlation_bonus = enhanced_correlation_score(person, state)
    total_score = base_score + 0.3 * correlation_bonus

    threshold = adaptive_threshold_with_history(
        state,
        config.base_start,
        rejection_history,
        window_size=min(100, state.total_seen),
    )

    return total_score >= threshold


def tune_parameters_via_simulation(
    scenario: int,
    sample_data: List[Dict[str, Any]],
    n_simulations: int = 100,
) -> ScenarioConfig:
    """
    Run quick simulations to tune parameters (placeholder)
    """
    best_config = SCENARIO_CONFIGS[scenario]
    best_score = float("inf")
    for w_help in [1.5, 2.0, 2.5, 3.0]:
        for base_start in [0.3, 0.5, 0.7]:
            for endgame in [30, 50, 70, 100]:
                config = ScenarioConfig(
                    endgame_size=endgame,
                    w_help=w_help,
                    w_penalty=w_help * 0.25,
                    w_corr=0.2,
                    base_start=base_start,
                    buffer_multiplier=0.05,
                )
                avg_rejections = 0
                if avg_rejections < best_score:
                    best_score = avg_rejections
                    best_config = config
    return best_config


def decide_dyn(
    person: Dict[str, bool],
    state: GameState,
    endgame_size: int,
    w_help: float,
    w_penalty: float,
    w_corr: float,
    base_start: float,
) -> bool:
    # Hard feasibility must hold
    if not hard_safety(person, state):
        return False

    # Deficit-first: greedily take contributors to unmet constraints
    if deficit_first(person, state):
        return True

    # Endgame: be conservative to avoid infeasibility
    if (state.N - state.admitted_count) <= endgame_size:
        return conservative_endgame(person, state)

    # Score vs threshold
    s = constraint_score(person, state, w_help, w_penalty, w_corr)
    thr = dynamic_threshold(state, base_start)
    return s >= thr


def main():
    global TIMEOUT, DEFAULT_RETRIES
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--player-id", required=True)
    # Networking knobs
    p.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    p.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT)
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    p.add_argument("--checkpoint-dir", default=".")
    p.add_argument("--no-auto-resume", action="store_true")
    args = p.parse_args()

    # Apply networking config before first request
    TIMEOUT = (args.connect_timeout, args.read_timeout)
    DEFAULT_RETRIES = args.retries

    # Scenario config
    config = SCENARIO_CONFIGS.get(args.scenario, SCENARIO_CONFIGS[2])
    
    def _checkpoint_path(player_id: str, scenario: int, directory: str) -> str:
        safe_player = "".join(ch for ch in player_id if ch.isalnum() or ch in ("-", "_"))
        return os.path.join(directory, f".bouncer_{safe_player}_s{scenario}.json")

    def _save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            tmp_path = path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, path)
        except Exception:
            pass

    def _load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    cp_path = _checkpoint_path(args.player_id, args.scenario, args.checkpoint_dir)
    resume_payload = None if args.no_auto_resume else _load_checkpoint(cp_path)

    if resume_payload:
        game_id = resume_payload.get("gameId")
        N = int(resume_payload.get("N", DEFAULT_N))
        M = {str(k): int(v) for k, v in resume_payload.get("constraints", {}).items()}
        counts = {str(k): int(v) for k, v in resume_payload.get("counts", {}).items()}
        k = int(resume_payload.get("admitted", 0))
        rejected_local = int(resume_payload.get("rejected_local", 0))
        freqs = {str(k): float(v) for k, v in resume_payload.get("freqs", {}).items()}
        correlations = {str(a1): {str(a2): float(c) for a2, c in resume_payload.get("corr", {}).get(a1, {}).items()} for a1 in resume_payload.get("corr", {})}
        seen_counts_loaded = {str(k): int(v) for k, v in resume_payload.get("seen_counts", {}).items()}
        total_seen_loaded = int(resume_payload.get("total_seen", 0))
        accept_all_flag = bool(resume_payload.get("accept_all_after_min_met", False))
        last_next_person = resume_payload.get("nextPerson")
        admitted_server = resume_payload.get("admitted_server", k)
        rejected_server = resume_payload.get("rejected_server", rejected_local)

        state = GameState(
            N=N,
            constraints=M,
            counts=counts,
            freqs=freqs,
            corr=correlations,
            admitted_count=k,
            rejected_local=rejected_local,
            seen_counts=seen_counts_loaded if seen_counts_loaded else {a: 0 for a in freqs},
            total_seen=total_seen_loaded,
            accept_all_after_min_met=accept_all_flag,
        )

        # Synthesize a response so the loop can continue
        res = {
            "status": "running",
            "nextPerson": last_next_person,
            "admittedCount": admitted_server,
            "rejectedCount": rejected_server,
        }
        last_progress_admitted = int(admitted_server) if isinstance(admitted_server, int) else 0
        rejection_history = list(resume_payload.get("rejection_history", []))
    else:
        init = start_game(args.base_url, args.scenario, args.player_id)
        game_id = init["gameId"]
        constraints = init["constraints"]  # list of {attribute, minCount}
        attr_stats = init.get("attributeStatistics", {})
        rel_freqs = attr_stats.get("relativeFrequencies", {})
        correlations = attr_stats.get("correlations", {})

        N = DEFAULT_N
        # Required counts (optionally buffered)
        M: Dict[str, int] = {}
        for c in constraints:
            base = int(c["minCount"])
            buff = int(math.ceil(config.buffer_multiplier * math.sqrt(N))) if config.buffer_multiplier > 0 else 0
            M[c["attribute"]] = min(N, base + buff)

        counts: Dict[str, int] = {a: 0 for a in M}
        k = 0  # admitted
        rejected_local = 0

        # Initialize dynamic state
        # Ensure seen_counts covers all attributes (stats + constraints)
        freq_keys = {str(a) for a in rel_freqs.keys()} | {str(a) for a in M.keys()}
        seeded_freqs = {a: float(rel_freqs.get(a, 0.2)) for a in freq_keys}
        state = GameState(
            N=N,
            constraints=M,
            counts=counts,
            freqs=seeded_freqs,
            corr={str(a1): {str(a2): float(c) for a2, c in correlations.get(a1, {}).items()} for a1 in correlations},
            admitted_count=0,
            rejected_local=0,
            seen_counts={a: 0 for a in freq_keys},
            total_seen=0,
        )

        # First person: no decision parameter
        res = decide_and_next(args.base_url, game_id, 0, None)
        last_progress_admitted = 0
        rejection_history = []
        # Persist initial checkpoint
        _save_checkpoint(
            cp_path,
            {
                "gameId": game_id,
                "scenario": args.scenario,
                "playerId": args.player_id,
                "N": N,
                "constraints": M,
                "counts": counts,
                "freqs": state.freqs,
                "corr": state.corr,
                "admitted": k,
                "rejected_local": rejected_local,
                "total_seen": state.total_seen,
                "seen_counts": state.seen_counts,
                "rejection_history": rejection_history,
                "accept_all_after_min_met": state.accept_all_after_min_met,
                "nextPerson": res.get("nextPerson"),
                "admitted_server": res.get("admittedCount"),
                "rejected_server": res.get("rejectedCount"),
            },
        )
    while res.get("status") == "running" and res.get("nextPerson"):
        person = res["nextPerson"]
        idx = person["personIndex"]
        attrs = person["attributes"]

        # Update dynamic state observations
        state.admitted_count = k
        state.rejected_local = rejected_local
        state.total_seen += 1
        for a in state.freqs:
            if attrs.get(a, False):
                state.seen_counts[a] = state.seen_counts.get(a, 0) + 1
        # Blend frequencies online (lightweight)
        update_adaptive_freqs(state, k0=DEFAULT_K0)

        # Enhanced decision algorithm
        accept = decide_enhanced(
            attrs,
            state,
            scenario=args.scenario,
            rejection_history=rejection_history,
        )

        res = decide_and_next(args.base_url, game_id, idx, accept)
        if accept:
            k += 1
            # Update counts for admitted person (only attributes we track in constraints)
            for a in counts:
                if attrs.get(a, False):
                    counts[a] = counts.get(a, 0) + 1
            rejection_history.append(0)
        else:
            rejected_local += 1
            rejection_history.append(1)

        # Persist checkpoint after each decision
        _save_checkpoint(
            cp_path,
            {
                "gameId": game_id,
                "scenario": args.scenario,
                "playerId": args.player_id,
                "N": N,
                "constraints": M,
                "counts": counts,
                "freqs": state.freqs,
                "corr": state.corr,
                "admitted": k,
                "rejected_local": rejected_local,
                "total_seen": state.total_seen,
                "seen_counts": state.seen_counts,
                "rejection_history": rejection_history[-500:],
                "accept_all_after_min_met": state.accept_all_after_min_met,
                "nextPerson": res.get("nextPerson"),
                "admitted_server": res.get("admittedCount"),
                "rejected_server": res.get("rejectedCount"),
            },
        )

        # Print status every 10 people analyzed
        if state.total_seen % 10 == 0:
            sys.stderr.write(
                f"analyzed: seen={state.total_seen} admitted={k} rejected={rejected_local}\n"
            )
            # Also show current quota progress per constraint (worst first)
            try:
                quota_items = []
                for a, M_a in state.constraints.items():
                    c = counts.get(a, 0)
                    ratio = (c / M_a) if M_a else 1.0
                    quota_items.append((ratio, a, c, M_a))
                quota_items.sort(key=lambda x: x[0])  # lowest fulfillment first
                preview = " ".join(f"{a}={c}/{M_a}" for _, a, c, M_a in quota_items)
                if preview:
                    sys.stderr.write(f"quota: {preview}\n")
            except Exception:
                # Non-fatal if formatting fails
                pass
            sys.stderr.flush()

        # Progress every 100 admissions (server-reported counts)
        admitted_server = res.get("admittedCount")
        rejected_server = res.get("rejectedCount")
        if isinstance(admitted_server, int) and (admitted_server // 100) > (last_progress_admitted // 100):
            sys.stderr.write(f"progress: admitted={admitted_server}/{N} rejected={rejected_server}\n")
            sys.stderr.flush()
            last_progress_admitted = admitted_server

    status = res.get("status")
    if status == "completed":
        print(f"completed: rejected={res.get('rejectedCount')}")
        # Clean up checkpoint on success
        try:
            if os.path.exists(cp_path):
                os.remove(cp_path)
        except Exception:
            pass
        return
    if status == "failed":
        print(f"failed: reason={res.get('reason')}")
        # Keep checkpoint for possible resume
        return

    # Fallback
    print(f"stopped: status={status}, admitted={k}, rejected(local)={rejected_local}")


if __name__ == "__main__":
    main()
