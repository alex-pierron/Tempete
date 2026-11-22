"""Research code of the unlearning task (phase 1 and 2); not part of the kit.

Team A: methods that imitate the GOLD model (gold_soft sweeps, surrogate scores, candidates).
Team B: miniature bench that computes the TRUE official score on CIFAR (mini_bench, mini_score,
mini_core, mini_methods) and the shared figure toolbox (viz). No gold of the real task exists:
only the miniature bench and the leaderboard give a true score (see
`docs/hackathon/unlearning.md`, section 4).
"""
