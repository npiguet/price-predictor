"""Ability effect model package: what each card ability does in play.

A third top-level package alongside ``price_predictor`` and ``sealed``, laid out
in the same hexagonal shape and mirroring ``src/draft/`` file-for-file. It ships
two artifacts: a per-ability embedding cache (one fixed-width vector ``e`` per
unique ability line, computed offline) and a state-conditional effect head that
predicts an ability's effect on a given game state.

``effects`` imports from ``sealed`` and ``price_predictor``; never the reverse
(one-way dependency rule, FR-002). The declared import surface is asserted by
``tests/unit/effects/test_import_boundaries.py``.
"""
