"""De-risk the crux: prove Topographia<->Catanatron board geometry is a clean bijection.

Topographia axial (q,r) -> cube (x=q, z=r). Catanatron cube_to_axial = (x, z).
If the 19 tile positions coincide and the induced 54 intersection keys + 72 edge
keys match Topographia's standard board exactly, the translation layer is feasible.
"""
from catanatron.models.map import BASE_MAP_TEMPLATE, CatanMap, NodeRef
from catanatron.models.board import get_edges
from catanatron.models.coordinate_system import UNIT_VECTORS, Direction, add

# ---- Topographia standard board (from src/board/standard.ts + geometry.ts) ----
STANDARD_HEXES = [
    (-2, 0), (-2, 1), (-2, 2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
    (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, -2), (1, -1), (1, 0),
    (1, 1), (2, -2), (2, -1), (2, 0),
]
# HEX_CORNER_DIRECTIONS from geometry.ts: 6 corners, each = 3 relative hex offsets.
CORNER_DIRS = [
    [(0, 0), (-1, 0), (0, -1)],
    [(0, 0), (0, -1), (1, -1)],
    [(0, 0), (1, -1), (1, 0)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (0, 1), (-1, 1)],
    [(0, 0), (-1, 1), (-1, 0)],
]


def topo_inter_key(hexes):
    # sortHexCoords: by q then r; key = "q,r|q,r|q,r"
    s = sorted(hexes)
    return "|".join(f"{q},{r}" for q, r in s)


def topo_edge_key(k1, k2):
    return f"{k1}--{k2}" if k1 <= k2 else f"{k2}--{k1}"


# Build Topographia intersections + edges.
topo_inters = set()
topo_edges = set()
for (q, r) in STANDARD_HEXES:
    corner_keys = []
    for offs in CORNER_DIRS:
        hexes = [(q + dq, r + dr) for dq, dr in offs]
        k = topo_inter_key(hexes)
        topo_inters.add(k)
        corner_keys.append(k)
    for i in range(6):
        topo_edges.add(topo_edge_key(corner_keys[i], corner_keys[(i + 1) % 6]))

# ---- Catanatron base map ----
cmap = CatanMap.from_template(BASE_MAP_TEMPLATE)

# NodeRef -> the two neighbor tile directions that share that corner (derived from
# get_nodes_and_edges in map.py).
NODEREF_NEIGHBORS = {
    NodeRef.NORTH: (Direction.NORTHWEST, Direction.NORTHEAST),
    NodeRef.NORTHEAST: (Direction.NORTHEAST, Direction.EAST),
    NodeRef.SOUTHEAST: (Direction.EAST, Direction.SOUTHEAST),
    NodeRef.SOUTH: (Direction.SOUTHEAST, Direction.SOUTHWEST),
    NodeRef.SOUTHWEST: (Direction.SOUTHWEST, Direction.WEST),
    NodeRef.NORTHWEST: (Direction.WEST, Direction.NORTHWEST),
}


def cube_to_topo_axial(cube):
    # cube (x,y,z) -> Topographia axial (q=x, r=z)
    return (cube[0], cube[2])


node_to_key = {}
conflicts = []
for coord, tile in cmap.land_tiles.items():
    for noderef, node_id in tile.nodes.items():
        d1, d2 = NODEREF_NEIGHBORS[noderef]
        cubes = [coord, add(coord, UNIT_VECTORS[d1]), add(coord, UNIT_VECTORS[d2])]
        hexes = [cube_to_topo_axial(c) for c in cubes]
        key = topo_inter_key(hexes)
        if node_id in node_to_key and node_to_key[node_id] != key:
            conflicts.append((node_id, node_to_key[node_id], key))
        node_to_key[node_id] = key

catan_inter_keys = set(node_to_key.values())

# Catanatron edges (land subgraph) -> Topographia edge keys
catan_edges = set()
for (a, b) in get_edges(frozenset(cmap.land_nodes)):
    if a in node_to_key and b in node_to_key:
        catan_edges.add(topo_edge_key(node_to_key[a], node_to_key[b]))

# ---- Report ----
print("Topographia: %d hexes, %d intersections, %d edges" % (
    len(STANDARD_HEXES), len(topo_inters), len(topo_edges)))
print("Catanatron : %d land tiles, %d nodes, %d node->key, %d edges" % (
    len(cmap.land_tiles), len(cmap.land_nodes), len(node_to_key), len(catan_edges)))
print("node->key conflicts (same node, 2 keys):", len(conflicts))
print("intersections bijection (sets equal):", catan_inter_keys == topo_inters,
      "| distinct catan keys:", len(catan_inter_keys))
print("edges bijection (sets equal):", catan_edges == topo_edges)

missing_i = topo_inters - catan_inter_keys
extra_i = catan_inter_keys - topo_inters
missing_e = topo_edges - catan_edges
extra_e = catan_edges - topo_edges
if missing_i or extra_i:
    print("  inter missing(topo-only):", list(missing_i)[:5], " extra(catan-only):", list(extra_i)[:5])
if missing_e or extra_e:
    print("  edge missing:", list(missing_e)[:5], " extra:", list(extra_e)[:5])

ok = (not conflicts and catan_inter_keys == topo_inters and catan_edges == topo_edges
      and len(node_to_key) == 54 and len(catan_edges) == 72)
print("\nBIJECTION VERIFIED:" , ok)
