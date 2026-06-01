"""Expert initial-placement heuristics (Medium/Hard), used during the setup phase instead
of the generic tree search.

A strong Catan opening is not just the highest pip count — it weighs, across BOTH of a
player's opening settlements: expected income (pips), resource diversity, the key
wood+brick (expansion) and wheat+ore (cities/dev) pairs, dice-number spread (variance), and
ports. The generic value-function search optimizes a single settlement's production and
doesn't plan the pair, so openings came out merely "good"; this makes them expert.

Pure pip/board arithmetic — no tree search — so it's fast (sub-millisecond over 54 nodes).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from catanatron.models.board import STATIC_GRAPH
from catanatron.models.enums import WOOD, BRICK, WHEAT, ORE, SETTLEMENT, CITY

# Dice "pips" (number of dots) = relative frequency of each number. 7/desert => 0.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Weights (tunable). Pips dominate (income); the rest shapes a balanced, expert opening.
W_PIP = 1.0           # per pip of expected income at this node
W_NODE_DIVERSITY = 1.5  # per distinct resource AT this node (3-resource spots are gold)
W_NEW_RESOURCE = 2.5    # per resource this node adds that my other settlement lacks
W_WOOD_BRICK = 3.0      # the combined pair covers BOTH wood and brick (early expansion)
W_WHEAT_ORE = 2.5       # the combined pair covers BOTH wheat and ore (cities / dev cards)
W_NUM_COLLISION = -1.0  # per dice number duplicated across my settlements (variance risk)
W_PORT_USED = 2.0       # a 2:1 port for a resource this node produces
W_PORT_OTHER = 0.5      # a 2:1 port for a resource it doesn't (or a future one)
W_PORT_GENERIC = 1.0    # a 3:1 port

_NO_PORT = object()


def _node_res_pips(game, node: int) -> Tuple[Dict[str, int], List[int]]:
    """{resource: total pips} and the list of dice numbers for a node's adjacent land tiles."""
    rp: Dict[str, int] = defaultdict(int)
    numbers: List[int] = []
    for tile in game.state.board.map.adjacent_tiles.get(node, []):
        if tile.resource is not None:
            rp[tile.resource] += PIPS.get(tile.number, 0)
            numbers.append(tile.number)
    return rp, numbers


def _port_of(game, node: int):
    """The port resource at `node` (None => 3:1 generic), or _NO_PORT if there's no port."""
    for resource, nodes in game.state.board.map.port_nodes.items():
        if node in nodes:
            return resource
    return _NO_PORT


def _owned(game, color) -> Tuple[Dict[str, int], set]:
    """Resources (with pips) and dice numbers already covered by `color`'s settlements/cities."""
    res: Dict[str, int] = defaultdict(int)
    numbers: set = set()
    for n in (
        game.state.buildings_by_color[color][SETTLEMENT]
        + game.state.buildings_by_color[color][CITY]
    ):
        rp, nums = _node_res_pips(game, n)
        for r, p in rp.items():
            res[r] += p
        numbers.update(nums)
    return res, numbers


def score_settlement(game, color, node: int) -> float:
    """Expert value of placing `color`'s next opening settlement at `node`."""
    rp, numbers = _node_res_pips(game, node)
    owned_res, owned_nums = _owned(game, color)

    score = W_PIP * sum(rp.values()) + W_NODE_DIVERSITY * len(rp)
    score += W_NEW_RESOURCE * sum(1 for r in rp if r not in owned_res)

    combined = set(owned_res) | set(rp)
    if WOOD in combined and BRICK in combined:
        score += W_WOOD_BRICK
    if WHEAT in combined and ORE in combined:
        score += W_WHEAT_ORE

    score += W_NUM_COLLISION * sum(1 for n in numbers if n in owned_nums)

    port = _port_of(game, node)
    if port is not _NO_PORT:
        if port is None:
            score += W_PORT_GENERIC
        else:
            score += W_PORT_USED if rp.get(port, 0) > 0 else W_PORT_OTHER
    return score


def best_initial_settlement(game, color) -> int:
    """Best opening settlement node for `color` (complements its existing settlement, if any)."""
    candidates = game.state.board.buildable_node_ids(color, initial_build_phase=True)
    return max(candidates, key=lambda n: score_settlement(game, color, n))


def best_initial_road(game, color, settlement_node: int) -> Tuple[int, int]:
    """Point the opening road toward the best open expansion node two steps out."""
    buildable = game.state.board.board_buildable_ids
    best_edge = None
    best_target = float("-inf")
    for nb in STATIC_GRAPH.neighbors(settlement_node):
        # Value of this direction = the best still-open settlement node reachable by extending
        # one more road from nb (i.e. two edges from our settlement).
        target = 0.0
        for m in STATIC_GRAPH.neighbors(nb):
            if m != settlement_node and m in buildable:
                target = max(target, score_settlement(game, color, m))
        if target > best_target:
            best_target = target
            best_edge = (settlement_node, nb)
    return best_edge
