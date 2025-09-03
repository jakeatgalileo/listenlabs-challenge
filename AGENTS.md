# Repository Guidelines

## Project Structure & Module Organization
- `bouncer.py`: Main CLI and decision engine (networking, state, strategy).
- `requirements.txt`: Runtime dependencies.
- `README.md`: Setup and run instructions with example command.
- `PROBLEM_STATEMENT.md`: Problem and API description.
- `venv/`: Local virtualenv (do not rely on it in CI; create your own).

## Build, Test, and Development Commands
- Create venv: `python -m venv venv && source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Run locally:
  ```bash
  python bouncer.py \
    --base-url https://berghain.challenges.listenlabs.ai/ \
    --scenario 1 \
    --player-id <your-id> \
    --connect-timeout 5 --read-timeout 30 --retries 5
  ```
- Lint/format (optional): `ruff check .` and `black .` if you use those tools.

## Coding Style & Naming Conventions
- Python, 4‑space indentation, PEP 8, type hints required for new functions.
- Use snake_case for functions/variables; PascalCase for dataclasses.
- Keep pure logic separate from I/O; prefer small, testable functions.
- If adding formatters/linters, prefer `black` and `ruff` with sensible defaults.

## Testing Guidelines
- No tests exist yet. If adding tests:
  - Framework: `pytest`.
  - Layout: `tests/` directory; files named `test_*.py`.
  - Run: `pytest -q`.
  - Aim for coverage of decision helpers (e.g., `hard_safety`, `conservative_endgame`, `decide_enhanced`). Use small fixtures for `GameState`.

## Commit & Pull Request Guidelines
- Commits: Imperative mood, scoped, concise. Example: `feat: add adaptive thresholding`, `fix: correct endgame feasibility check`.
- PRs: Include description, rationale, and before/after behavior. Add sample command and expected log snippet. Link related issues.
- Keep diffs focused; avoid unrelated refactors. Include docs updates when behavior changes.

## Architecture Notes
- Entry point: `main()` in `bouncer.py` orchestrates API calls and decision flow.
- Core components: `GameState` (state), constraint/threshold calculators, scenario configs (`SCENARIO_CONFIGS`).
- Networking: `get()` centralizes retries/timeouts via `requests` session. Avoid hardcoding URLs; use `--base-url`.
- Extending: Add new strategies as pure helpers and switch from `decide_enhanced` via a flag or parameter to compare approaches.
