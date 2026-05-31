"""Isolation smoke test: reconstruct a Topographia GameState into Catanatron and decide.

Uses the real standard-board fixture (geometry/harbors/buildings/roads) to exercise the
full translation + search without a running server. Not a reachable position, but
structurally valid -> catches reconstruction/decision bugs.
"""
import json
import sys
import time

sys.path.insert(0, "/Users/dianwen/workspace/topographia/brain")

from catan_brain.translate import build_game, action_to_intent  # noqa: E402
from catan_brain.bot import decide  # noqa: E402

gs = json.load(open("/Users/dianwen/workspace/topographia/web/src/fixtures/standard-game.json"))

# Make it a live mid-game decision for player 0.
gs["phase"] = {"tag": "main_play"}
gs["dice_rolled"] = True
gs["current_turn_player_id"] = 0
gs["pending_robber"] = None
for p in gs["players"]:
    p.setdefault("dev_cards_hidden", {"knight": 1, "victory_point": 0, "road_building": 0,
                                      "year_of_plenty": 0, "monopoly": 0})
    p.setdefault("dev_cards_played_this_turn", None)

game = build_game(gs, 0)
st = game.state
print("prompt:", st.current_prompt, "| n playable:", len(st.playable_actions))
print("playable types:", sorted({a.action_type.value for a in st.playable_actions}))
print("P0 VP/actual:", st.player_state["P0_ACTUAL_VICTORY_POINTS"],
      "| road_lengths:", dict(st.board.road_lengths))
print("buildings on board:", len(st.board.buildings), "| roads:", len(st.board.roads) // 2)

for diff in ["easy", "medium", "hard"]:
    t = time.monotonic()
    action, info = decide(game, st.colors[0], diff, seed=42)
    dt = (time.monotonic() - t) * 1000
    intent = action_to_intent(action, st)
    print(f"  {diff:6s}: {action.action_type.value:22s} depth={info['completed_depth']} "
          f"blunder={info['blunder']} {dt:6.1f}ms -> intent={intent}")

print("\n--- rich-hand scenario (P0 has plenty of every resource) ---")
gs2 = json.loads(json.dumps(gs))
gs2["players"][0]["resources"] = {"lumber": 5, "brick": 5, "wool": 5, "grain": 5, "ore": 5}
game2 = build_game(gs2, 0)
st2 = game2.state
print("n playable:", len(st2.playable_actions),
      "| types:", sorted({a.action_type.value for a in st2.playable_actions}))
for diff in ["easy", "medium", "hard"]:
    t = time.monotonic()
    action, info = decide(game2, st2.colors[0], diff, seed=7)
    dt = (time.monotonic() - t) * 1000
    intent = action_to_intent(action, st2)
    print(f"  {diff:6s}: {action.action_type.value:22s} depth={info['completed_depth']} "
          f"n_actions={info['n_actions']} blunder={info['blunder']} {dt:7.1f}ms -> {intent}")

print("OK")
