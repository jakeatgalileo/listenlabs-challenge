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
  --connect-timeout 5 --read-timeout 30 --retries 5
```

If the script errors out, you can resume:

```
python bouncer.py \
  --base-url https://berghain.challenges.listenlabs.ai/ \
  --scenario 1 \
  --player-id d79fdbcc-46bd-4b65-b232-bc2c1adc2114 \
  --checkpoint-dir ./.checkpoints
```
