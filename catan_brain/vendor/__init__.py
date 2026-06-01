"""Vendored from Catanatron (GPL-3.0), pinned commit in CATANATRON_COMMIT.txt.

Why vendored: Catanatron 3.2.1 (the stable PyPI release) ships the rules engine but
NOT the tuned value function / features (those live only on a refactored master whose
engine is incompatible). These two files use only stable engine APIs, so we vendor them
here (GPL side) and feed them our reconstructed Game. See ../bot.py and docs/bot.md.
"""
