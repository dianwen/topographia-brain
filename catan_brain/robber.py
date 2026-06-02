"""Expert robber-placement heuristic (Medium/Hard), used for the move-robber step instead
of the generic tree search.

WHY THIS EXISTS: the search maximizes `value_fn(game, seat_color)`, which scores only the
bot's OWN position (its production, VP, roads). Blocking an opponent's tile doesn't change
the bot's own features, so at the search's shallow depth every candidate hex that isn't one
of the bot's own tiles looks ~identical -> placement was near-arbitrary and repetitive (the
enumeration-order tie-break kept re-picking the same hex / picking on a trailing player).

This scores each LEGAL target hex by the opponent production it denies: pip-weighted (high-
frequency numbers hurt more), cities counted double (they produce two), the current VP leader
weighted up (pressure who's winning), and the bot's OWN adjacent production penalized (never
shoot yourself). A small bonus favors a hex where a card-holding opponent is adjacent, so the
follow-up steal step (which picks the victim — see translate.special_intent) actually nets a
card. Pure board arithmetic, no tree search.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from catanatron.models.enums import RESOURCES, ActionType, SETTLEMENT, CITY
from catanatron.models.player import Color
from catanatron.state_functions import player_key

# Dice "pips" (number of dots) = relative frequency of each number. 7/desert => 0.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Weights (tunable).
W_LEADER = 1.5   # multiplier on production denied to the current VP leader (vs other rivals)
W_SELF = 2.0     # penalty multiplier for blocking the bot's OWN adjacent production
W_STEAL = 0.5    # flat bonus when ≥1 adjacent opponent holds cards (a steal will net something)


def _node_owners(state) -> Dict[int, Tuple[Color, bool]]:
    """node_id -> (owner color, is_city)."""
    owners: Dict[int, Tuple[Color, bool]] = {}
    for color in state.colors:
        for n in state.buildings_by_color[color][SETTLEMENT]:
            owners[n] = (color, False)
        for n in state.buildings_by_color[color][CITY]:
            owners[n] = (color, True)
    return owners


def _hand_sizes(state) -> Dict[Color, int]:
    sizes: Dict[Color, int] = {}
    for color in state.colors:
        key = player_key(state, color)
        sizes[color] = sum(state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES)
    return sizes


def _leader(state) -> Optional[Color]:
    """Color with the most ACTUAL victory points (ties -> first; weighting is a soft nudge)."""
    best, best_vp = None, -1
    for color in state.colors:
        vp = state.player_state.get(f"{player_key(state, color)}_ACTUAL_VICTORY_POINTS", 0)
        if vp > best_vp:
            best_vp, best = vp, color
    return best


def _coord_value(state, coord, seat_color, leader, owners, hands) -> float:
    tile = state.board.map.land_tiles.get(coord)
    if tile is None:
        return float("-inf")
    pip = PIPS.get(tile.number, 0)
    score = 0.0
    can_steal = False
    for node_id in tile.nodes.values():
        owner = owners.get(node_id)
        if owner is None:
            continue
        color, is_city = owner
        weight = pip * (2 if is_city else 1)
        if color == seat_color:
            score -= W_SELF * weight
        else:
            score += (W_LEADER if color == leader else 1.0) * weight
            if hands.get(color, 0) > 0:
                can_steal = True
    if can_steal:
        score += W_STEAL
    return score


def best_robber_action(game, seat_color: Color):
    """Best MOVE_ROBBER action for `seat_color`, or None if there are none to choose from."""
    state = game.state
    moves = [a for a in state.playable_actions if a.action_type == ActionType.MOVE_ROBBER]
    if not moves:
        return None
    owners = _node_owners(state)
    hands = _hand_sizes(state)
    leader = _leader(state)
    return max(moves, key=lambda a: _coord_value(state, a.value[0], seat_color, leader, owners, hands))
