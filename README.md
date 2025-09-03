## Berghain Bouncer Challenge

Logged in as jake (ID: d79fdbcc-46bd-4b65-b232-bc2c1adc2114)

# Setup

1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt

```bash
python bouncer.py \
  --base-url https://berghain.challenges.listenlabs.ai/ \
  --scenario 1 \
  --player-id d79fdbcc-46bd-4b65-b232-bc2c1adc2114 \
  --connect-timeout 5 --read-timeout 120 --retries 1
    --checkpoint-dir ./.checkpoints
```

```bash
python bouncer.py \
  --base-url https://berghain.challenges.listenlabs.ai/ \
  --scenario 2 \
  --player-id d79fdbcc-46bd-4b65-b232-bc2c1adc2114 \
  --connect-timeout 5 --read-timeout 120 --retries 1
    --checkpoint-dir ./.checkpoints
```

```bash
python bouncer.py \
  --base-url https://berghain.challenges.listenlabs.ai/ \
  --scenario 3 \
  --player-id d79fdbcc-46bd-4b65-b232-bc2c1adc2114 \
  --connect-timeout 5 --read-timeout 120 --retries 1
    --checkpoint-dir ./.checkpoints
```

## How it works

- Dynamic threshold: becomes more permissive as the venue fills.
- Constraint scoring: weighs how each person helps unmet minimums.
- Endgame feasibility: avoids choices that could make constraints impossible.
- Scarcity-weighted prioritization: prioritizes rare-but-required attributes using a per-attribute scarcity S[a] = need[a] / (R \* p[a]) with clipping. Direct help, missing penalties, and correlation bonuses are scaled by scarcity, and an early accept triggers if a candidate contributes to highly scarce requirements. Tune via scenario configs or `--scarcity-mult`.

## Scenario-specific strategies

- Files: `scenario1.py`, `scenario2.py`, `scenario3.py` implement per-scenario decision functions.
- Selection: `bouncer.py` dynamically imports `scenario{scenario}.py` and calls its `decide(...)`.
- Independence: These modules are self-contained (no imports from `bouncer.py`). Each implements its own heuristics and feasibility checks.
- Customizing: Tweak per-scenario heuristics directly in those files. The CLI continues to manage networking and passes minimal state to each strategy.

Environment knobs:
(none specific to Scenario 1 at this time)
