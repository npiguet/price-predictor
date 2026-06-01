"""Draft agent package (generation 1): imitation policy + critic.

A new top-level package that mirrors ``sealed``'s hexagonal layout and reuses
its scorer, picker, greedy builder, embedding layout, card locator, checkpoint
plumbing, and Forge supervisor/worker pattern. ``draft`` imports from ``sealed``
and ``price_predictor``; never the reverse (one-way dependency rule, FR-002).
"""
