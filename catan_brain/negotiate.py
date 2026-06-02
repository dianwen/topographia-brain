"""Strong trade & discard decisions, scored with Catanatron's tuned value function.

Catanatron's engine can't *apply* a domestic trade or a chosen discard, but we don't need
it to: we mutate the reconstructed State's resource hands and score the result with the
tuned value function (the same one that drives play). See docs/bot.md.

  - discard (on a 7): pick the discard subset that maximizes the KEPT hand's value.
  - respond to an offer: accept iff it improves me and doesn't help a leading proposer more.
  - make an offer: only when a 1-card trade unlocks a build this turn AND an opponent would
    rationally accept; pick the swap that maximizes my best post-trade move.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.state_functions import player_key

from .geometry import RES_T2C
from .bot import value_fn, _expectiminimax, _Deadline

TOPO_RES = ["lumber", "brick", "wool", "grain", "ore"]
# Value-fn units. public_vps weight ~3e14, production ~1e8.
# A 1:1 swap with no immediate build only shifts the hand-synergy term (weight 1e2), so the
# value comparisons need just a small threshold — "the hand genuinely improves". The SAME gate
# governs all three trade decisions (respond / propose / confirm) so they stay consistent: a
# swap worth proposing is worth completing once accepted, and one worth accepting is worth its
# threshold. (They previously diverged — propose at RESP_EPS but confirm at a build-scale 1e6 —
# so the bot proposed hand-synergy swaps it then refused to complete, re-proposing forever.)
RESP_EPS = 1.0

# Build costs in Topographia resource names.
COSTS = {
    "road": {"lumber": 1, "brick": 1},
    "settlement": {"lumber": 1, "brick": 1, "wool": 1, "grain": 1},
    "city": {"ore": 3, "grain": 2},
}


def _hand(state, color) -> Dict[str, int]:
    key = player_key(state, color)
    return {t: state.player_state[f"{key}_{RES_T2C[t]}_IN_HAND"] for t in TOPO_RES}


def _with_hand_delta(game: Game, color, gain: Dict[str, int], lose: Dict[str, int]) -> Game:
    """A copy of `game` with `color`'s hand adjusted by +gain -lose (Topographia res names)."""
    g = game.copy()
    key = player_key(g.state, color)
    for t in TOPO_RES:
        d = gain.get(t, 0) - lose.get(t, 0)
        if d:
            g.state.player_state[f"{key}_{RES_T2C[t]}_IN_HAND"] += d
    return g


def value_after_swap(game: Game, color, gain: Dict[str, int], lose: Dict[str, int]) -> float:
    """Tuned value for `color` after a resource swap (no lookahead). For out-of-turn calls."""
    return value_fn(_with_hand_delta(game, color, gain, lose), color)


def value_after_swap_then_build(
    game: Game, color, gain: Dict[str, int], lose: Dict[str, int], deadline: float
) -> float:
    """Best value `color` can reach in ONE ply after the swap — captures build-enabling.
    Assumes it is `color`'s turn (the offer context). Falls back to a flat eval otherwise."""
    g = _with_hand_delta(game, color, gain, lose)
    try:
        g.state.playable_actions = generate_playable_actions(g.state)
        return _expectiminimax(g, color, 1, -math.inf, math.inf, deadline)
    except (_Deadline, Exception):
        return value_fn(g, color)


# ── leadership ──
def _leader_is_threat(game: Game, color) -> bool:
    """True if `color` leads on VP or is within one of winning (don't feed them)."""
    vps = [
        game.state.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"] for i in range(len(game.state.colors))
    ]
    idx = game.state.color_to_index[color]
    return vps[idx] == max(vps) or vps[idx] >= game.vps_to_win - 1


def _would_accept(game: Game, responder, proposer, offer: Dict[str, int], request: Dict[str, int]) -> bool:
    """Would `responder` rationally accept (receive `offer`, give `request`)?"""
    hand = _hand(game.state, responder)
    if any(hand[t] < request.get(t, 0) for t in TOPO_RES):
        return False  # can't afford what's asked of them
    d_resp = value_after_swap(game, responder, offer, request) - value_fn(game, responder)
    if d_resp <= RESP_EPS:
        return False
    if _leader_is_threat(game, proposer):
        d_prop = value_after_swap(game, proposer, request, offer) - value_fn(game, proposer)
        if d_prop > d_resp:
            return False  # helps the leading proposer more than us
    return True


# ── 1. respond to an offer ──
def decide_response(game: Game, color, proposal: Dict) -> Dict:
    """Return an accept_trade / reject_trade intent for `proposal` (awaiting this seat)."""
    offer = {t: int(proposal.get("offer", {}).get(t, 0)) for t in TOPO_RES}
    request = {t: int(proposal.get("request", {}).get(t, 0)) for t in TOPO_RES}
    proposer = game.state.colors[proposal["proposer_id"]]
    pid = proposal["id"]
    if _would_accept(game, color, proposer, offer, request):
        return {"kind": "accept_trade", "proposal_id": pid}
    return {"kind": "reject_trade", "proposal_id": pid}


# ── confirm/cancel my own fully-answered proposal ──
def decide_confirm(game: Game, color, proposal: Dict) -> Dict:
    """As proposer: complete with the least-threatening accepter, or cancel."""
    offer = {t: int(proposal.get("offer", {}).get(t, 0)) for t in TOPO_RES}
    request = {t: int(proposal.get("request", {}).get(t, 0)) for t in TOPO_RES}
    accepters: List[int] = list(proposal.get("accepted_by", []))
    pid = proposal["id"]
    # It's my turn, so judge by what I can BUILD after the trade (same lookahead that made me
    # propose it). My gain is independent of WHICH accepter I pick (terms are fixed).
    deadline = time.monotonic() + 0.3
    d_me = value_after_swap_then_build(game, color, request, offer, deadline) - value_after_swap_then_build(
        game, color, {}, {}, deadline
    )
    if not accepters or d_me <= RESP_EPS:
        return {"kind": "cancel_trade", "proposal_id": pid}
    # Give the cards to the least-threatening accepter (lowest VP).
    vp = lambda i: game.state.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"]
    acceptor = min(accepters, key=vp)
    return {"kind": "complete_trade", "proposal_id": pid, "acceptor_id": acceptor}


# ── 2. make an offer ──
def _can_build(game: Game, color, build: str) -> bool:
    board = game.state.board
    key = player_key(game.state, color)
    if build == "settlement":
        return (
            game.state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] > 0
            and len(board.buildable_node_ids(color)) > 0
        )
    if build == "city":
        return (
            game.state.player_state[f"{key}_CITIES_AVAILABLE"] > 0
            and len(game.state.buildings_by_color[color]["SETTLEMENT"]) > 0
        )
    if build == "road":
        return game.state.player_state[f"{key}_ROADS_AVAILABLE"] > 0 and len(board.buildable_edges(color)) > 0
    return False


def best_proposal(game: Game, color, deadline: float) -> Optional[Dict]:
    """Best 1:1 (give surplus, get a needed card) to propose, or None.

    Fires when I hold spare cards AND lack cards for a build I can actually pursue, the swap
    improves my best move (value-fn lookahead — biggest when it unlocks a build), and an
    opponent would rationally accept. Returns the most-improving such offer."""
    hand = _hand(game.state, color)
    pursuable = [b for b in ("city", "settlement", "road") if _can_build(game, color, b)]
    if not pursuable:
        return None
    # Needs: any resource I'm short of for a pursuable build. (If I can already afford one,
    # there's nothing to trade FOR.)
    needs = {t for b in pursuable for t in TOPO_RES if hand[t] < COSTS[b].get(t, 0)}
    if not needs:
        return None
    # Candidates: give a card I'm NOT short of, get a card I need. We don't pre-filter by a
    # rigid "surplus" rule — the value-fn lookahead below is the real gate: it penalizes
    # giving away a card I actually need (my best move's value drops) and rewards getting one
    # that unlocks a build, so only genuinely-improving swaps survive.
    givable = [t for t in TOPO_RES if hand[t] >= 1 and t not in needs]
    if not givable:
        return None

    opponents = [c for c in game.state.colors if c != color]
    v_now = value_after_swap_then_build(game, color, {}, {}, deadline)
    # Pre-rank cheaply (no lookahead), then verify the top few with the lookahead.
    cands = sorted(
        ((g, n) for g in givable for n in needs),
        key=lambda gn: value_after_swap(game, color, {gn[1]: 1}, {gn[0]: 1}),
        reverse=True,
    )
    best = None
    best_v = v_now + RESP_EPS  # require a genuine improvement (build-unlocking ones dominate)
    for give, need in cands[:6]:
        if time.monotonic() >= deadline:
            break
        offer, request = {give: 1}, {need: 1}
        v_trade = value_after_swap_then_build(game, color, {need: 1}, {give: 1}, deadline)
        if v_trade <= best_v:
            continue
        if any(_would_accept(game, opp, color, offer, request) for opp in opponents):
            best_v = v_trade
            best = {"kind": "propose_trade", "offer": offer, "request": request}
    return best


# ── 3. discard on a 7 ──
def _discard_candidates(hold: Dict[str, int], k: int, cap: int = 150) -> List[Dict[str, int]]:
    """Distinct discard multisets of size k (each ≤ holdings). Greedy (most-held) first; capped."""
    out: List[Dict[str, int]] = []
    # greedy baseline
    greedy = {t: 0 for t in TOPO_RES}
    need = k
    for t in sorted(TOPO_RES, key=lambda t: -hold[t]):
        take = min(need, hold[t])
        greedy[t] = take
        need -= take
    out.append(greedy)

    def rec(i: int, remaining: int, cur: Dict[str, int]):
        if len(out) >= cap:
            return
        if i == len(TOPO_RES):
            if remaining == 0 and cur != greedy:
                out.append(dict(cur))
            return
        t = TOPO_RES[i]
        for take in range(0, min(hold[t], remaining) + 1):
            cur[t] = take
            rec(i + 1, remaining - take, cur)
            if len(out) >= cap:
                return
        cur[t] = 0

    rec(0, k, {t: 0 for t in TOPO_RES})
    return out


def decide_discard(game: Game, color) -> Dict[str, int]:
    """Discard floor(hand/2): the subset that maximizes the value of the kept hand."""
    hold = _hand(game.state, color)
    total = sum(hold.values())
    k = total // 2
    if k <= 0:
        return {t: 0 for t in TOPO_RES}
    deadline = time.monotonic() + 0.30
    best, best_v = None, -math.inf
    for disc in _discard_candidates(hold, k):
        if best is not None and time.monotonic() >= deadline:
            break
        v = value_after_swap(game, color, {}, disc)
        if v > best_v:
            best_v, best = v, disc
    return best
