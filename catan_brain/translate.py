"""Reconstruct a Catanatron State from a Topographia GameState, and map a chosen
Catanatron Action back to a compact Topographia "action intent".

The brain runs server-side and receives the FULL Topographia GameState (every
player's hidden dev cards included), so reconstruction is exact for resources,
dev cards, bank and board. Known divergences are documented in docs/bot.md and
flagged with `# DIVERGENCE:` comments.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from catanatron.models.player import Color, SimplePlayer
from catanatron.models.enums import (
    Action,
    ActionType,
    ActionPrompt,
    SETTLEMENT,
    CITY,
    ROAD,
    DEVELOPMENT_CARDS,
    RESOURCES,
)
from catanatron.models.board import Board, STATIC_GRAPH, longest_acyclic_path
from catanatron.models.actions import generate_playable_actions
from catanatron.state import State, PLAYER_INITIAL_STATE
from catanatron.state_functions import player_key
from catanatron.game import Game

from .geometry import (
    RES_T2C,
    RES_C2T,
    FREQDECK_ORDER,
    DEV_T2C,
    NODE_TO_KEY,
    KEY_TO_NODE,
    EDGE_KEY_TO_NODES,
    COORD_TO_HEX,
    HEX_TO_COORD,
    hex_to_cube,
    build_map,
)

SEAT_COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def _res_freqdeck(resmap: Dict[str, int]) -> List[int]:
    """Topographia ResourceMap -> catanatron [WOOD,BRICK,SHEEP,WHEAT,ORE]."""
    out = [0, 0, 0, 0, 0]
    for t_res, count in resmap.items():
        out[FREQDECK_ORDER.index(RES_T2C[t_res])] = count
    return out


def build_game(gs: Dict[str, Any], seat: int) -> Game:
    """Build a Catanatron Game whose state mirrors `gs`, ready for `seat` to decide."""
    players = gs["players"]
    n = len(players)
    colors = tuple(SEAT_COLORS[i] for i in range(n))

    # ── Map: actual resources/numbers + ports from the (random) Topographia board ──
    hex_tiles: Dict = {}
    robber_cube = None
    for key, hx in gs["board"]["hexes"].items():
        q, r = hx["q"], hx["r"]
        if hx["resource_type"] == "desert":
            hex_tiles[(q, r)] = (None, None)
        else:
            hex_tiles[(q, r)] = (RES_T2C[hx["resource_type"]], hx["token_value"])
        if hx["has_robber"]:
            robber_cube = hex_to_cube(q, r)

    port_nodes: Dict[Optional[str], List[int]] = defaultdict(list)
    for harbor in gs["board"].get("harbors", []):
        res = harbor["harbor_type"]
        catan_res = None if res == "generic" else RES_T2C[res]
        for inter_id in harbor["intersection_ids"]:
            node = KEY_TO_NODE.get(_inter_id_to_key(inter_id))
            if node is not None:
                port_nodes[catan_res].append(node)

    cmap = build_map(hex_tiles, port_nodes)

    # ── State scaffold (initialize=False so nothing is randomized) ──
    state = State([], None, initialize=False)
    state.players = [SimplePlayer(c) for c in colors]
    state.colors = colors
    state.color_to_index = {c: i for i, c in enumerate(colors)}
    state.discard_limit = gs["config"].get("discard_threshold", 7)
    state.board = Board(cmap)
    if robber_cube is not None:
        state.board.robber_coordinate = robber_cube
    state.buildings_by_color = {c: defaultdict(list) for c in colors}

    # ── player_state ──
    ps: Dict[str, Any] = {}
    for i in range(n):
        for k, v in PLAYER_INITIAL_STATE.items():
            ps[f"P{i}_{k}"] = v
    state.player_state = ps

    # ── Board buildings + roads (set directly, then rebuild caches) ──
    for key, inter in gs["board"]["intersections"].items():
        if inter["building"] is None or inter["player_id"] is None:
            continue
        node = KEY_TO_NODE[key]
        color = colors[inter["player_id"]]
        btype = CITY if inter["building"] == "city" else SETTLEMENT
        state.board.buildings[node] = (color, btype)
        state.buildings_by_color[color][btype].append(node)

    for key, edge in gs["board"]["edges"].items():
        if edge["player_id"] is None:
            continue
        a, b = EDGE_KEY_TO_NODES[key]
        color = colors[edge["player_id"]]
        state.board.roads[(a, b)] = color
        state.board.roads[(b, a)] = color
        state.buildings_by_color[color][ROAD].append((a, b))

    # During setup, initial_road_possibilities reads SETTLEMENT[-1] as the just-placed
    # settlement, so make sure the acting seat's last settlement is at the end of the list.
    if gs["phase"]["tag"] == "setup" and gs.get("setup_last_settlement_id") is not None:
        last_node = KEY_TO_NODE.get(_inter_id_to_key(gs["setup_last_settlement_id"]))
        setts = state.buildings_by_color[colors[seat]][SETTLEMENT]
        if last_node in setts:
            setts.remove(last_node)
            setts.append(last_node)

    _rebuild_board_caches(state)

    # ── Per-player counts: resources, dev cards, VPs, longest-road / army flags ──
    lr_owner = gs.get("longest_road_player_id")
    la_owner = gs.get("largest_army_player_id")
    for i, p in enumerate(players):
        key = f"P{i}"
        fd = _res_freqdeck(p["resources"])
        for ri, res in enumerate(FREQDECK_ORDER):
            ps[f"{key}_{res}_IN_HAND"] = fd[ri]
        # dev cards (full state -> every player's hidden cards are known to the server)
        hidden = p.get("dev_cards_hidden", {})
        for t_dev, c_dev in DEV_T2C.items():
            ps[f"{key}_{c_dev}_IN_HAND"] = hidden.get(t_dev, 0)
        # DIVERGENCE: Topographia only tracks knights_played per seat; other played
        # dev cards aren't itemised. Largest army only needs PLAYED_KNIGHT, so that's exact.
        ps[f"{key}_PLAYED_KNIGHT"] = p.get("knights_played", 0)

        color = colors[i]
        n_sett = len(state.buildings_by_color[color][SETTLEMENT])
        n_city = len(state.buildings_by_color[color][CITY])
        ps[f"{key}_SETTLEMENTS_AVAILABLE"] = p.get("settlements_left", 5 - n_sett)
        ps[f"{key}_CITIES_AVAILABLE"] = p.get("cities_left", 4 - n_city)
        ps[f"{key}_ROADS_AVAILABLE"] = p.get("roads_left", 15)

        has_road = lr_owner == i
        has_army = la_owner == i
        ps[f"{key}_HAS_ROAD"] = has_road
        ps[f"{key}_HAS_ARMY"] = has_army
        ps[f"{key}_LONGEST_ROAD_LENGTH"] = state.board.road_lengths.get(color, 0)

        public_vp = n_sett + 2 * n_city + (2 if has_road else 0) + (2 if has_army else 0)
        actual_vp = public_vp + hidden.get("victory_point", 0)
        ps[f"{key}_VICTORY_POINTS"] = public_vp
        ps[f"{key}_ACTUAL_VICTORY_POINTS"] = actual_vp

    # board longest-road owner (for value fn / consistency)
    if lr_owner is not None:
        state.board.road_color = colors[lr_owner]
        state.board.road_length = state.board.road_lengths.get(colors[lr_owner], 0)

    # ── Turn / phase flags ──
    cur_turn = gs["current_turn_player_id"]
    state.current_turn_index = cur_turn
    ps[f"P{cur_turn}_HAS_ROLLED"] = bool(gs.get("dice_rolled"))
    ps[f"P{cur_turn}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = (
        players[cur_turn].get("dev_cards_played_this_turn") is not None
    )

    phase = gs["phase"]
    pending = gs.get("pending_robber")
    state.is_initial_build_phase = phase["tag"] == "setup"
    state.is_road_building = gs.get("road_building_roads_remaining", 0) > 0
    state.free_roads_available = gs.get("road_building_roads_remaining", 0)
    state.is_discarding = bool(pending and pending.get("step") == "discard")
    state.is_moving_knight = bool(pending and pending.get("step") == "move")

    state.current_player_index = seat
    if state.is_initial_build_phase:
        state.current_prompt = (
            ActionPrompt.BUILD_INITIAL_ROAD
            if gs.get("setup_last_settlement_id") is not None
            else ActionPrompt.BUILD_INITIAL_SETTLEMENT
        )
    elif state.is_discarding:
        state.current_prompt = ActionPrompt.DISCARD
    elif state.is_moving_knight:
        state.current_prompt = ActionPrompt.MOVE_ROBBER
    else:
        state.current_prompt = ActionPrompt.PLAY_TURN

    # ── Dev deck (exact remaining order from Topographia) ──
    state.development_listdeck = [DEV_T2C[c] for c in gs.get("dev_deck", [])]

    # ── Bank ──
    state.resource_freqdeck = _res_freqdeck(gs["bank"])

    state.actions = []
    state.num_turns = gs.get("setup_placement_index", 0)
    state.playable_actions = generate_playable_actions(state)

    game = Game([], initialize=False)
    game.seed = 0
    game.id = gs.get("_game_id", "brain")
    game.vps_to_win = gs["config"].get("victory_points_to_win", 10)
    game.state = state
    return game


def _rebuild_board_caches(state: State) -> None:
    """Recompute connected_components, board_buildable_ids, and road_lengths from the
    directly-set buildings/roads (mirrors what build_settlement/build_road maintain)."""
    board = state.board
    # board_buildable_ids = land_nodes minus occupied nodes and their neighbours.
    buildable = set(board.map.land_nodes)
    for node in board.buildings:
        buildable.discard(node)
        for nb in STATIC_GRAPH.neighbors(node):
            buildable.discard(nb)
    board.board_buildable_ids = buildable

    # connected_components[color]: BFS over each colour's roads, not crossing enemy nodes.
    board.connected_components = defaultdict(list)
    board.road_lengths = defaultdict(int)
    for color in state.colors:
        color_edges = [e for e, c in board.roads.items() if c == color]
        adj: Dict[int, set] = defaultdict(set)
        nodes = set()
        for a, b in color_edges:
            adj[a].add(b)
            nodes.add(a)
            nodes.add(b)
        # include own buildings as (possibly singleton) components
        own_buildings = {n for n, (c, _t) in board.buildings.items() if c == color}
        nodes |= own_buildings

        seen = set()
        comps = []
        for start in nodes:
            if start in seen:
                continue
            comp = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.add(n)
                # don't expand through an enemy-owned node
                bcol = board.buildings.get(n)
                if bcol is not None and bcol[0] != color and n != start:
                    continue
                for m in adj[n]:
                    if m not in seen:
                        stack.append(m)
            comps.append(comp)
        board.connected_components[color] = comps
        # longest road length over this colour's components
        best = 0
        for comp in comps:
            if len(comp) >= 2:
                try:
                    best = max(best, len(longest_acyclic_path(board, comp, color)))
                except (ValueError, IndexError):
                    pass
        board.road_lengths[color] = best


def _inter_id_to_key(inter_id: Dict[str, Any]) -> str:
    """Topographia IntersectionId object {hexes:[{q,r}...]} -> string key."""
    hexes = sorted((h["q"], h["r"]) for h in inter_id["hexes"])
    return "|".join(f"{q},{r}" for q, r in hexes)


def _hex_from_cube(cube) -> Dict[str, int]:
    return {"q": cube[0], "r": cube[2]}


def action_to_intent(action: Action, state: State) -> Dict[str, Any]:
    """Map a chosen Catanatron Action to a compact Topographia action intent.
    The TS RemoteBrainPolicy expands board keys into structural ids using its board."""
    at = action.action_type
    v = action.value
    if at == ActionType.ROLL:
        return {"kind": "roll"}
    if at == ActionType.END_TURN:
        return {"kind": "end_turn"}
    if at == ActionType.BUILD_ROAD:
        edge_key = _edge_to_key(v)
        kind = "initial_road" if state.is_initial_build_phase else "build_road"
        return {"kind": kind, "edge_key": edge_key}
    if at == ActionType.BUILD_SETTLEMENT:
        kind = "initial_settlement" if state.is_initial_build_phase else "build_settlement"
        return {"kind": kind, "node_key": NODE_TO_KEY[v]}
    if at == ActionType.BUILD_CITY:
        return {"kind": "build_city", "node_key": NODE_TO_KEY[v]}
    if at == ActionType.BUY_DEVELOPMENT_CARD:
        return {"kind": "buy_dev"}
    if at == ActionType.PLAY_KNIGHT_CARD:
        return {"kind": "play_knight"}
    if at == ActionType.PLAY_ROAD_BUILDING:
        return {"kind": "play_road_building"}
    if at == ActionType.PLAY_YEAR_OF_PLENTY:
        # v is (Resource, Resource) or list
        res = [RES_C2T[r] for r in v if r is not None]
        return {"kind": "play_year_of_plenty", "resources": res}
    if at == ActionType.PLAY_MONOPOLY:
        return {"kind": "play_monopoly", "resource": RES_C2T[v]}
    if at == ActionType.MOVE_ROBBER:
        coordinate, robbed_color, _ = v
        steal_from = state.color_to_index[robbed_color] if robbed_color is not None else None
        return {"kind": "move_robber", "hex": _hex_from_cube(coordinate), "steal_from": steal_from}
    if at == ActionType.DISCARD:
        # Catanatron yields DISCARD with value=None (random at apply). Topographia needs a
        # concrete ResourceMap summing to floor(total/2); compute a greedy discard (most-held
        # first) from the acting player's hand. (In the integration discards stay on the local
        # heuristic; this is a safety path.)
        resmap = {t: 0 for t in RES_T2C}
        if v:
            for r in v:
                resmap[RES_C2T[r]] += 1
        else:
            key = player_key(state, action.color)
            hand = {t: state.player_state[f"{key}_{c}_IN_HAND"] for t, c in RES_T2C.items()}
            total = sum(hand.values())
            to_discard = total // 2
            for t in sorted(hand, key=lambda x: -hand[x]):
                take = min(to_discard, hand[t])
                resmap[t] = take
                to_discard -= take
                if to_discard <= 0:
                    break
        return {"kind": "discard", "resources": resmap}
    if at == ActionType.MARITIME_TRADE:
        # v is 5-tuple: (give, give, give?, give?, receive); trailing Nones for ports
        offering = [r for r in v[:-1] if r is not None]
        receive = v[-1]
        return {
            "kind": "bank_trade",
            "offer": RES_C2T[offering[0]],
            "offer_count": len(offering),
            "request": RES_C2T[receive],
        }
    raise ValueError(f"unmapped action type {at}")


def _edge_to_key(edge) -> str:
    a, b = edge
    ka, kb = NODE_TO_KEY[a], NODE_TO_KEY[b]
    return f"{ka}--{kb}" if ka <= kb else f"{kb}--{ka}"


def special_intent(gs: Dict[str, Any], seat: int) -> Optional[Dict[str, Any]]:
    """Handle decisions Catanatron's engine can't model directly. Returns an intent or None.

    DIVERGENCE: Topographia splits the robber into move_robber then steal; Catanatron bundles
    them into one MOVE_ROBBER. So the standalone 'steal' step has no Catanatron prompt — we
    pick the eligible victim holding the most cards (a reasonable heuristic).
    """
    pending = gs.get("pending_robber")
    if pending and pending.get("step") == "steal":
        eligible = pending.get("eligible_player_ids", [])
        if not eligible:
            return {"kind": "steal", "target_player_id": None}
        players = gs["players"]
        best = max(eligible, key=lambda pid: sum(players[pid]["resources"].values()))
        return {"kind": "steal", "target_player_id": best}
    return None
