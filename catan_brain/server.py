"""FastAPI decision service.

Concurrency: AlphaBeta search is pure-Python CPU work that holds the GIL, so a single
process computes one move at a time. We use an async front door + a ProcessPoolExecutor
(workers ~= vCPUs) so the server computes N moves in parallel, one per worker. The event
loop only awaits, staying responsive and naturally queueing when all workers are busy.
See docs/bot.md §6.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .bot import TIERS, decide
from .translate import build_game, action_to_intent, special_intent

# Bound the wait a touch above the tier cap (transport backstop; see docs/bot.md §7).
RTT_MARGIN_S = 1.0
_DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _decide_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure CPU work, runs in a pool process: DTO -> chosen action intent."""
    gs = payload["state"]
    seat = payload["seat"]
    difficulty = payload.get("difficulty", "medium")
    seed = payload.get("seed", 0)

    # Decisions Catanatron's engine can't model (e.g. the standalone steal step).
    special = special_intent(gs, seat)
    if special is not None:
        return {"intent": special, "info": {"special": True}}

    game = build_game(gs, seat)
    seat_color = game.state.colors[seat]
    action, info = decide(game, seat_color, difficulty, seed)
    intent = action_to_intent(action, game.state)
    return {"intent": intent, "info": info}


def _warm_up():
    # Each worker pays one-time costs here, not on its first request: build the geometry
    # bijection and prime Catanatron's cached all-pairs node distances (floyd-warshall),
    # which the tuned value function's reachability features need.
    from . import geometry  # noqa: F401
    from catanatron.models.board import get_node_distances

    get_node_distances()


POOL: Optional[ProcessPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global POOL
    workers = int(os.environ.get("BRAIN_WORKERS", _DEFAULT_WORKERS))
    POOL = ProcessPoolExecutor(max_workers=workers, initializer=_warm_up)
    # Warm the parent process too.
    _warm_up()
    yield
    POOL.shutdown(cancel_futures=True)


app = FastAPI(title="catan-brain", lifespan=lifespan)


class DecideRequest(BaseModel):
    state: Dict[str, Any]
    seat: int
    difficulty: str = "medium"
    seed: int = 0


class DecideResponse(BaseModel):
    intent: Dict[str, Any]
    info: Dict[str, Any]


@app.get("/health")
async def health():
    return {"ok": True, "tiers": list(TIERS.keys()), "workers": _DEFAULT_WORKERS}


@app.post("/decide", response_model=DecideResponse)
async def decide_endpoint(req: DecideRequest) -> DecideResponse:
    assert POOL is not None
    cap_s = TIERS.get(req.difficulty, TIERS["medium"])[2]
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(POOL, _decide_worker, req.model_dump())
    result = await asyncio.wait_for(fut, timeout=cap_s + RTT_MARGIN_S)
    return DecideResponse(**result)
