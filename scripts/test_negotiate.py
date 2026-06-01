"""Isolation tests for the trade/discard deciders (value-function-driven)."""
import json
import sys

sys.path.insert(0, "/Users/dianwen/workspace/topographia/brain")
from catan_brain.translate import build_game  # noqa: E402
from catan_brain import negotiate  # noqa: E402

gs = json.load(open("/Users/dianwen/workspace/topographia/web/src/fixtures/standard-game.json"))
gs["phase"] = {"tag": "main_play"}
gs["dice_rolled"] = True
gs["current_turn_player_id"] = 0
gs["pending_robber"] = None
# The fixture is a FINISHED game (a player has 10 VP) -> winning_color() would make the
# search treat every node as terminal. Raise the target so it reads as a live mid-game.
gs["config"]["victory_points_to_win"] = 15
for p in gs["players"]:
    p.setdefault("dev_cards_hidden", {"knight": 0, "victory_point": 0, "road_building": 0,
                                      "year_of_plenty": 0, "monopoly": 0})
    p.setdefault("dev_cards_played_this_turn", None)


def fresh(resources0):
    g = json.loads(json.dumps(gs))
    g["players"][0]["resources"] = resources0
    return g


# 1. DISCARD: big hand -> discard floor(n/2), keep the most valuable cards.
g = fresh({"lumber": 4, "brick": 1, "wool": 1, "grain": 3, "ore": 5})  # 14 cards -> discard 7
game = build_game(g, 0)
disc = negotiate.decide_discard(game, game.state.colors[0])
total = sum(g["players"][0]["resources"].values())
print("DISCARD:", disc, "sum", sum(disc.values()), "expected", total // 2,
      "| ok:", sum(disc.values()) == total // 2 and all(disc[r] <= g["players"][0]["resources"][r] for r in disc))

# 2. RESPOND: proposer (P1, leader) offers wool for my ore. I hold ore.
g = fresh({"lumber": 1, "brick": 1, "wool": 0, "grain": 1, "ore": 2})
game = build_game(g, 0)
proposal = {"id": "t1", "proposer_id": 1, "offer": {"wool": 1}, "request": {"ore": 1},
            "accepted_by": [], "rejected_by": []}
resp = negotiate.decide_response(game, game.state.colors[0], proposal)
print("RESPOND:", resp)

# 3. OFFER: mutually beneficial. P0 one grain short of a CITY (owns settlements, spare lumber);
#    P1 one lumber short of a settlement (spare grain) -> P0 offers lumber for grain.
g = fresh({"lumber": 2, "brick": 0, "wool": 0, "grain": 1, "ore": 3})
g["players"][1]["resources"] = {"lumber": 0, "brick": 1, "wool": 1, "grain": 3, "ore": 0}
game = build_game(g, 0)
import time as _t  # noqa: E402
prop = negotiate.best_proposal(game, game.state.colors[0], _t.monotonic() + 0.5)
print("OFFER(mutual):", prop, "| ok:", prop is not None and prop["kind"] == "propose_trade")

# 4. OFFER when not one-short -> None (no junk proposals).
g = fresh({"lumber": 0, "brick": 0, "wool": 0, "grain": 0, "ore": 1})
game = build_game(g, 0)
prop = negotiate.best_proposal(game, game.state.colors[0], _t.monotonic() + 0.5)
print("OFFER(not-short):", prop)

print("OK")
