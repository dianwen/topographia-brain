"""FastAPI decision service.

Concurrency: AlphaBeta search is pure-Python CPU work that holds the GIL, so a single
process computes one move at a time. We use an async front door + a ProcessPoolExecutor
(workers ~= vCPUs) so the server computes N moves in parallel, one per worker. The event
loop only awaits, staying responsive and naturally queueing when all workers are busy.
See docs/bot.md §6.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from catanatron.models.enums import Action, ActionType, ActionPrompt, SETTLEMENT

from .bot import TIERS, decide
from .translate import build_game, action_to_intent, special_intent
from . import negotiate
from . import opening
from . import robber

# How long best_proposal may spend evaluating candidate offers (added to the move time).
_PROPOSE_BUDGET_S = 0.4

# Bound the wait a touch above the tier cap (transport backstop; see docs/bot.md §7).
RTT_MARGIN_S = 1.0
_DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _decide_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure CPU work, runs in a pool process: DTO -> chosen action intent.

    Dispatch order: steal (engine can't model) -> discard -> respond to an offer ->
    confirm my own answered offer -> normal move (which may instead propose a trade).
    Trades/discards are scored with the tuned value function (see negotiate.py).
    """
    gs = payload["state"]
    seat = payload["seat"]
    difficulty = payload.get("difficulty", "medium")
    seed = payload.get("seed", 0)
    allow_propose = payload.get("allow_propose", True)

    # Standalone steal step — Catanatron bundles it into MOVE_ROBBER, so handle it raw.
    special = special_intent(gs, seat)
    if special is not None:
        return {"intent": special, "info": {"special": "steal"}}

    game = build_game(gs, seat)
    seat_color = game.state.colors[seat]

    # Expert opening (Medium/Hard): a dedicated pip/diversity/port/pairing heuristic beats the
    # generic value-fn search for setup placement (see opening.py). Easy keeps the search.
    if game.state.is_initial_build_phase and difficulty in ("medium", "hard"):
        if game.state.current_prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
            act = Action(seat_color, ActionType.BUILD_SETTLEMENT, opening.best_initial_settlement(game, seat_color))
        else:
            snode = game.state.buildings_by_color[seat_color][SETTLEMENT][-1]
            act = Action(seat_color, ActionType.BUILD_ROAD, opening.best_initial_road(game, seat_color, snode))
        return {"intent": action_to_intent(act, game.state), "info": {"opening": difficulty}}

    pending = gs.get("pending_robber")
    proposals = gs.get("pending_trade_proposals") or []
    n = len(gs["players"])

    # Discard on a 7: keep the highest-value hand.
    if pending and pending.get("step") == "discard":
        owed = pending.get("remaining_discards", {})
        if str(seat) in owed or seat in owed:
            return {"intent": {"kind": "discard", "resources": negotiate.decide_discard(game, seat_color)},
                    "info": {"discard": True}}

    # Move the robber (Medium/Hard): a denial-aware heuristic beats the value-fn search, which
    # only scores the bot's own position and is ~blind to blocking opponents (see robber.py).
    # Easy keeps the search. The victim is chosen separately in the steal step (special_intent).
    if pending and pending.get("step") == "move" and difficulty in ("medium", "hard"):
        act = robber.best_robber_action(game, seat_color)
        if act is not None:
            return {"intent": action_to_intent(act, game.state), "info": {"robber": difficulty}}

    # Respond to a proposal awaiting this seat.
    for p in proposals:
        if p["proposer_id"] != seat and seat not in p.get("accepted_by", []) and seat not in p.get("rejected_by", []):
            return {"intent": negotiate.decide_response(game, seat_color, p), "info": {"respond": True}}

    # Confirm/cancel my own proposal once everyone else has responded.
    for p in proposals:
        if p["proposer_id"] == seat:
            answered = set(p.get("accepted_by", [])) | set(p.get("rejected_by", []))
            if all(o in answered for o in range(n) if o != seat):
                return {"intent": negotiate.decide_confirm(game, seat_color, p), "info": {"confirm": True}}

    # Normal move.
    action, info = decide(game, seat_color, difficulty, seed)
    intent = action_to_intent(action, game.state)

    # Or propose a trade instead — only on MY turn, AFTER rolling (pre-roll the only legal
    # move is ROLL; proposing then is rejected as "must roll dice first"), on a clean turn.
    if (
        allow_propose
        and pending is None
        and not proposals
        and gs.get("dice_rolled")
        and seat == gs["current_turn_player_id"]
    ):
        prop = negotiate.best_proposal(game, seat_color, time.monotonic() + _PROPOSE_BUDGET_S)
        if prop is not None:
            return {"intent": prop, "info": {**info, "proposed": True}}
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


class _AccessLogErrorsOnly(logging.Filter):
    """Drop uvicorn access-log lines for successful responses; keep errors (status >= 400).

    /decide is hit on every bot move, so the default per-request access log is pure
    noise. We still want failed responses surfaced, so let >= 400 through. The status
    code is the 5th positional arg of uvicorn's access record (client, method, path,
    http_version, status_code).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 5:
            try:
                return int(args[4]) >= 400
            except (TypeError, ValueError):
                return True
        return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogErrorsOnly())

log = logging.getLogger("catan_brain")

app = FastAPI(title="catan-brain", lifespan=lifespan)


class DecideRequest(BaseModel):
    state: Dict[str, Any]
    seat: int
    difficulty: str = "medium"
    seed: int = 0
    # The caller's "propose only once per turn" guard (see games.ts `proposedThisTurn`).
    # MUST be declared here: Pydantic drops undeclared body fields, so without this the
    # flag never reaches `_decide_worker` and defaults to True — the bot then re-proposes
    # the same trade every tick after a rejection/cancellation (an infinite offer loop).
    allow_propose: bool = True


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
    deadline = cap_s + RTT_MARGIN_S
    try:
        result = await asyncio.wait_for(fut, timeout=deadline)
    except asyncio.TimeoutError:
        # The bot didn't decide within its think cap; the caller has already (or will)
        # fall back to its local heuristic. Surface it so a persistently slow/wedged
        # worker is visible rather than silently degrading every move to the fallback.
        log.error(
            "decide timed out after %.1fs (seat %s, difficulty %s); no move returned",
            deadline,
            req.seat,
            req.difficulty,
        )
        raise
    return DecideResponse(**result)
