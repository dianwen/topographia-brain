"""AlphaBeta (expectiminimax) decision over Catanatron's engine.

We implement the search in-repo because Catanatron's `AlphaBetaPlayer` ships only on
a refactored GitHub master (not the stable PyPI package); see docs/bot.md. The search
runs on Catanatron 3.2.1's rules engine (`Game.copy/execute`, `generate_playable_actions`)
with our own value function over `player_state` features.

Difficulty is (target_depth, epsilon, time_cap):
  easy   = depth 1, eps 0.30, cap 0.10s
  medium = depth 2, eps 0.10, cap 0.40s
  hard   = depth 3, eps 0.00, cap 1.50s
Only Hard never blunders -> distinctly strongest (see docs/bot.md).
"""
from __future__ import annotations

import math
import random
import time
import os
from typing import Dict, Tuple

from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, SETTLEMENT, CITY
from catanatron.models.player import Color
from catanatron.state_functions import player_key

from .vendor.value import get_value_fn, DEFAULT_WEIGHTS, CONTENDER_WEIGHTS

TIERS: Dict[str, Tuple[int, float, float]] = {
    "easy": (1, 0.30, 0.10),
    "medium": (2, 0.10, 0.40),
    "hard": (3, 0.00, 1.50),
}

# 2d6 sum -> probability (for the forced-roll chance node).
_DICE_PROBAS = {}
for _i in range(1, 7):
    for _j in range(1, 7):
        _DICE_PROBAS[_i + _j] = _DICE_PROBAS.get(_i + _j, 0) + 1 / 36
# a representative (d1, d2) for each sum (engine only uses the sum for yields)
_DICE_REP = {s: (min(s - 1, 6), s - min(s - 1, 6)) for s in _DICE_PROBAS}


class _Deadline(Exception):
    pass


# Action-ordering priority (search best-looking moves first -> more pruning).
_ORDER = {
    ActionType.BUILD_CITY: 0,
    ActionType.BUILD_SETTLEMENT: 1,
    ActionType.BUY_DEVELOPMENT_CARD: 2,
    ActionType.PLAY_KNIGHT_CARD: 3,
    ActionType.BUILD_ROAD: 4,
    ActionType.MARITIME_TRADE: 5,
    ActionType.PLAY_YEAR_OF_PLENTY: 6,
    ActionType.PLAY_MONOPOLY: 6,
    ActionType.PLAY_ROAD_BUILDING: 6,
    ActionType.MOVE_ROBBER: 3,
    ActionType.DISCARD: 3,
    ActionType.END_TURN: 9,
}


def _order(actions):
    return sorted(actions, key=lambda a: _ORDER.get(a.action_type, 7))


# Catanatron's REAL tuned value function (vendored, GPL — see vendor/). "C" selects the
# optimizer-tuned CONTENDER_WEIGHTS; otherwise the hand-set DEFAULT_WEIGHTS (matches
# catanatron's default AlphaBetaPlayer). Override with BRAIN_VALUE_FN=C.
_USE_CONTENDER = os.environ.get("BRAIN_VALUE_FN") == "C"
_TUNED_FN = get_value_fn("contender_fn" if _USE_CONTENDER else "base_fn", None)
# The per-VP weight the tuned fn applies to PUBLIC victory points (dominates the score).
_PUBLIC_VPS_W = (CONTENDER_WEIGHTS if _USE_CONTENDER else DEFAULT_WEIGHTS)["public_vps"]

# ── Contested-bonus-VP ramp ──────────────────────────────────────────────────
# Longest Road and Largest Army each grant +2 PUBLIC VP, which the tuned value fn counts at
# the full (dominant) per-VP weight — so the search will chase a bonus (+2 VP) over building a
# settlement (+1 VP) even early, when a settlement's compounding production and un-stealable VP
# are worth more. But a bonus that *clinches the win* is worth every bit of its VP. So we scale
# the bonus's VP value by a ramp: a low floor early, rising to full as the holder nears the
# winning VP count. Implemented as a correction to the tuned fn (vendor/value.py stays verbatim):
# we subtract back the un-earned fraction of the bonus's VP value.
#   BRAIN_BONUS_VP_FLOOR — early-game multiplier on bonus VP (1.0 disables the discount entirely)
#   BRAIN_BONUS_VP_RAMP  — VP-from-win window over which the multiplier ramps floor → 1.0
_BONUS_VP_FLOOR = float(os.environ.get("BRAIN_BONUS_VP_FLOOR", "0.15"))
_BONUS_VP_RAMP = float(os.environ.get("BRAIN_BONUS_VP_RAMP", "4"))


def _bonus_vp_multiplier(actual_vp: float, vps_to_win: int) -> float:
    """0..1 weight for contested bonus VP, by how close the holder is to winning.

    `actual_vp` already includes the bonus being valued (we score post-move states), so
    `vps_to_win - actual_vp` is the VP still needed AFTER this bonus. At <=0 the bonus reaches
    the win -> full weight; beyond the ramp window -> the early floor; linear in between.
    """
    if _BONUS_VP_RAMP <= 0:
        return 1.0
    remaining = vps_to_win - actual_vp
    if remaining <= 0:
        return 1.0
    if remaining >= _BONUS_VP_RAMP:
        return _BONUS_VP_FLOOR
    return _BONUS_VP_FLOOR + (1.0 - _BONUS_VP_FLOOR) * (_BONUS_VP_RAMP - remaining) / _BONUS_VP_RAMP


def _simple_value(state, color: Color) -> float:
    """Fallback heuristic, used only if the tuned value fn raises on a reconstructed state."""
    key = player_key(state, color)
    vp = state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
    income = 0.0
    node_prod = state.board.map.node_production
    for node in state.buildings_by_color[color][SETTLEMENT]:
        income += sum(node_prod.get(node, {}).values())
    for node in state.buildings_by_color[color][CITY]:
        income += 2 * sum(node_prod.get(node, {}).values())
    lr = state.player_state[f"{key}_LONGEST_ROAD_LENGTH"]
    return 1000.0 * vp + 12.0 * income + 5.0 * lr


def value_fn(game: Game, color: Color) -> float:
    """Value of `game` for `color` via Catanatron's tuned value function, with contested
    bonus VP (Longest Road / Largest Army) discounted early and ramped to full near the win."""
    try:
        base = _TUNED_FN(game, color)
    except Exception:
        return _simple_value(game.state, color)
    if _BONUS_VP_FLOOR >= 1.0:
        return base
    ps = game.state.player_state
    key = player_key(game.state, color)
    bonus_vp = (2 if ps.get(f"{key}_HAS_ROAD") else 0) + (2 if ps.get(f"{key}_HAS_ARMY") else 0)
    if not bonus_vp:
        return base
    actual_vp = ps.get(f"{key}_ACTUAL_VICTORY_POINTS", ps[f"{key}_VICTORY_POINTS"])
    mult = _bonus_vp_multiplier(actual_vp, game.vps_to_win)
    # base counted the bonus's VP at full weight; subtract back the discounted-away fraction.
    return base - bonus_vp * (1.0 - mult) * _PUBLIC_VPS_W


def _expectiminimax(game: Game, p0: Color, depth: int, alpha: float, beta: float, deadline: float) -> float:
    if time.monotonic() >= deadline:
        raise _Deadline
    if game.winning_color() is not None or depth == 0:
        return value_fn(game, p0)

    state = game.state
    actions = state.playable_actions
    cur = state.current_color()

    # Forced-roll chance node -> expected value over dice sums (no pruning across chance).
    if len(actions) == 1 and actions[0].action_type == ActionType.ROLL:
        ev = 0.0
        for total, proba in _DICE_PROBAS.items():
            g2 = game.copy()
            g2.execute(Action(cur, ActionType.ROLL, _DICE_REP[total]), validate_action=False)
            ev += proba * _expectiminimax(g2, p0, depth - 1, alpha, beta, deadline)
        return ev

    maximizing = cur == p0
    if maximizing:
        best = -math.inf
        for a in _order(actions):
            g2 = game.copy()
            g2.execute(a, validate_action=False)
            best = max(best, _expectiminimax(g2, p0, depth - 1, alpha, beta, deadline))
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best if best != -math.inf else value_fn(game, p0)
    else:
        worst = math.inf
        for a in _order(actions):
            g2 = game.copy()
            g2.execute(a, validate_action=False)
            worst = min(worst, _expectiminimax(g2, p0, depth - 1, alpha, beta, deadline))
            beta = min(beta, worst)
            if beta <= alpha:
                break
        return worst if worst != math.inf else value_fn(game, p0)


def decide(game: Game, seat_color: Color, difficulty: str, seed: int) -> Tuple[Action, dict]:
    """Choose an action for `seat_color`. Returns (action, debug_info)."""
    target_depth, epsilon, cap_s = TIERS[difficulty]
    rng = random.Random(seed)
    actions = game.state.playable_actions
    info = {"n_actions": len(actions), "completed_depth": 0, "blunder": False}

    if len(actions) == 0:
        raise ValueError("no playable actions for reconstructed state")
    if len(actions) == 1:
        return actions[0], info

    # Epsilon blunder: skip the search entirely and play a random legal move. NOT during
    # initial placement — setup is only two settlements/roads, so a single random opening
    # cripples the whole game (looks "very weak"). Difficulty still differs in setup via
    # search depth, and epsilon weakens main play as intended.
    if epsilon and not game.state.is_initial_build_phase and rng.random() < epsilon:
        info["blunder"] = True
        return rng.choice(actions), info

    deadline = time.monotonic() + cap_s
    best_action = actions[0]
    ordered = _order(actions)
    for d in range(1, target_depth + 1):
        if time.monotonic() >= deadline:
            break
        try:
            best_val = -math.inf
            cur_best = ordered[0]
            alpha = -math.inf
            for a in ordered:
                g2 = game.copy()
                g2.execute(a, validate_action=False)
                v = _expectiminimax(g2, seat_color, d - 1, alpha, math.inf, deadline)
                if v > best_val:
                    best_val = v
                    cur_best = a
                alpha = max(alpha, best_val)
            best_action = cur_best
            info["completed_depth"] = d
            # search best-first next iteration (move the winner to the front)
            ordered = [cur_best] + [a for a in ordered if a is not cur_best]
        except _Deadline:
            break

    return best_action, info
