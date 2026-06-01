# catan-brain

Stateless **Catanatron**-powered decision service for **Topographia** bot seats. Given a
Topographia `GameState` + seat + difficulty, it returns one move.

## Run locally

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn catan_brain.server:app --port 8001
# health:  curl localhost:8001/health
```

Or via docker-compose from the parent repo: `docker compose up brain`.

## How it works

- `geometry.py` — static, verified Topographia↔Catanatron board bijection (19 hexes / 54
  nodes / 72 edges). Coordinate systems align (`q=x, r=z`); proven in
  `scripts/verify_bijection.py`.
- `translate.py` — rebuilds a Catanatron `State` from a Topographia `GameState` (random
  board ⇒ resources/numbers/ports are read from the request), and maps a chosen Catanatron
  action back to a compact Topographia "intent" (the TS side fills in structural ids).
- `bot.py` — expectiminimax + alpha-beta + anytime iterative-deepening over Catanatron's
  engine, evaluated with **Catanatron's real tuned value function** (`vendor/value.py` +
  `vendor/features.py`, vendored verbatim from a pinned master commit — see
  `vendor/CATANATRON_COMMIT.txt`; `BRAIN_VALUE_FN=C` for the optimizer-tuned weights).
  Difficulty = (target depth, epsilon blunder rate, time cap):
  Easy `(1, 0.30, 0.10s)`, Medium `(2, 0.10, 0.40s)`, Hard `(3, 0.00, 1.50s)`.
- `vendor/` — GPL files copied from Catanatron unmodified (one import made relative). They
  use only stable 3.2.1 engine APIs, so they run against our reconstructed `Game`.
- `server.py` — FastAPI + `ProcessPoolExecutor` (one move per worker; workers ≈ vCPUs).

## Known divergences from the plan (catanatron 3.2.1)

1. **No domestic/player trading** in catanatron 3.2.1 (only `MARITIME_TRADE`). Bots never
   propose/accept player trades — exactly like Topographia's old local bot.
2. **`AlphaBetaPlayer` is not in the PyPI package** (only on a refactored GitHub master that
   is incompatible with the stable engine). We implement the *search* here instead, on
   catanatron's stable rules engine — but the *evaluation* is catanatron's real tuned value
   function, vendored verbatim into `vendor/` (see above).
3. **Robber move + steal are one action in catanatron**, two in Topographia. The 'move'
   step uses catanatron; the standalone 'steal' step is a heuristic here (rob the richest).
4. **Discards** are a single random `DISCARD` action in catanatron, so strategic discards
   stay on Topographia's local heuristic; the brain has a greedy fallback only.
