"""The knowledge-probe suite: what a checkpoint knows about the game, and where.

A package so its modules import one another by name. The entry scripts
(``run.py``, ``compare.py``, ``ladder_check.py`` and the grids) are still run
by path, and each puts ``scripts/`` on ``sys.path`` before importing the suite.

Nothing here does work at import time.
"""
