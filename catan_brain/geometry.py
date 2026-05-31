"""Static Topographia <-> Catanatron board geometry.

Both engines use the same standard 19-hex board with the SAME axial convention:
Topographia axial (q, r) -> cube (x=q, z=r); Catanatron cube_to_axial = (x, z).
So tile positions coincide as the identity map, and the induced 54-node / 72-edge
bijection is exact (proven in scripts/verify_bijection.py).

Everything here is computed ONCE at import from Catanatron's BASE_MAP_TEMPLATE and
Topographia's corner geometry, then reused for every request.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    CatanMap,
    LandTile,
    NodeRef,
    Water,
    initialize_tiles,
)
from catanatron.models.coordinate_system import UNIT_VECTORS, Direction, add

# ── Resource / dev-card name maps (Topographia <-> Catanatron) ──
RES_T2C = {"lumber": "WOOD", "brick": "BRICK", "wool": "SHEEP", "grain": "WHEAT", "ore": "ORE"}
RES_C2T = {v: k for k, v in RES_T2C.items()}
FREQDECK_ORDER = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]  # catanatron bank index order

DEV_T2C = {
    "knight": "KNIGHT",
    "victory_point": "VICTORY_POINT",
    "road_building": "ROAD_BUILDING",
    "year_of_plenty": "YEAR_OF_PLENTY",
    "monopoly": "MONOPOLY",
}
DEV_C2T = {v: k for k, v in DEV_T2C.items()}

# Corner -> the two neighbouring tile directions that share it (from map.get_nodes_and_edges).
_NODEREF_NEIGHBORS = {
    NodeRef.NORTH: (Direction.NORTHWEST, Direction.NORTHEAST),
    NodeRef.NORTHEAST: (Direction.NORTHEAST, Direction.EAST),
    NodeRef.SOUTHEAST: (Direction.EAST, Direction.SOUTHEAST),
    NodeRef.SOUTH: (Direction.SOUTHEAST, Direction.SOUTHWEST),
    NodeRef.SOUTHWEST: (Direction.SOUTHWEST, Direction.WEST),
    NodeRef.NORTHWEST: (Direction.WEST, Direction.NORTHWEST),
}


def cube_to_hex(cube: Tuple[int, int, int]) -> Tuple[int, int]:
    """Catanatron cube (x, y, z) -> Topographia axial (q, r)."""
    return (cube[0], cube[2])


def hex_to_cube(q: int, r: int) -> Tuple[int, int, int]:
    """Topographia axial (q, r) -> Catanatron cube (x, y, z)."""
    return (q, -q - r, r)


def _inter_key(hexes) -> str:
    s = sorted(hexes)
    return "|".join(f"{q},{r}" for q, r in s)


def _edge_key(k1: str, k2: str) -> str:
    return f"{k1}--{k2}" if k1 <= k2 else f"{k2}--{k1}"


# ── Build the canonical geometry once ──
_BASE = CatanMap.from_template(BASE_MAP_TEMPLATE)

#: node_id -> Topographia intersection key, and reverse
NODE_TO_KEY: Dict[int, str] = {}
for _coord, _tile in _BASE.land_tiles.items():
    for _noderef, _node_id in _tile.nodes.items():
        _d1, _d2 = _NODEREF_NEIGHBORS[_noderef]
        _cubes = [_coord, add(_coord, UNIT_VECTORS[_d1]), add(_coord, UNIT_VECTORS[_d2])]
        NODE_TO_KEY[_node_id] = _inter_key([cube_to_hex(c) for c in _cubes])
KEY_TO_NODE: Dict[str, int] = {v: k for k, v in NODE_TO_KEY.items()}

#: land tile coordinate <-> Topographia hex (q, r)
COORD_TO_HEX: Dict[Tuple[int, int, int], Tuple[int, int]] = {
    c: cube_to_hex(c) for c in _BASE.land_tiles
}
HEX_TO_COORD: Dict[Tuple[int, int], Tuple[int, int, int]] = {v: k for k, v in COORD_TO_HEX.items()}

#: edge key -> (node_a, node_b) catanatron edge (sorted)
EDGE_KEY_TO_NODES: Dict[str, Tuple[int, int]] = {}
from catanatron.models.board import get_edges  # noqa: E402

for _a, _b in get_edges(frozenset(_BASE.land_nodes)):
    EDGE_KEY_TO_NODES[_edge_key(NODE_TO_KEY[_a], NODE_TO_KEY[_b])] = tuple(sorted((_a, _b)))

assert len(NODE_TO_KEY) == 54 and len(EDGE_KEY_TO_NODES) == 72, "bijection build failed"

# Capture the canonical per-coordinate node/edge geometry so we can rebuild maps with
# arbitrary (random-board) resources/numbers without re-running placement randomness.
_GEOMETRY = {coord: (tile.nodes, tile.edges) for coord, tile in _BASE.tiles.items()}
_LAND_COORDS = set(_BASE.land_tiles.keys())


def build_map(
    hex_tiles: Dict[Tuple[int, int], Tuple[Optional[str], Optional[int]]],
    port_nodes: Dict[Optional[str], List[int]],
) -> CatanMap:
    """Build a CatanMap with Topographia's actual resources/numbers/ports.

    hex_tiles: (q, r) -> (catan_resource | None for desert, number | None)
    port_nodes: catan_resource | None (3:1) -> list of node ids granting that port
    """
    tiles = {}
    tid = 0
    for coord, (nodes, edges) in _GEOMETRY.items():
        nodes = copy.copy(nodes)
        edges = copy.copy(edges)
        if coord in _LAND_COORDS:
            resource, number = hex_tiles[COORD_TO_HEX[coord]]
            tiles[coord] = LandTile(tid, resource, number, nodes, edges)
            tid += 1
        else:
            tiles[coord] = Water(nodes, edges)
    cmap = CatanMap.from_tiles(tiles)
    # Override the port cache directly from Topographia's harbors (we don't model Port tiles).
    cmap.port_nodes = {res: set(ids) for res, ids in port_nodes.items()}
    return cmap
