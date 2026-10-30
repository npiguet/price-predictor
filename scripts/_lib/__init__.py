"""Helpers shared by the analysis and probe scripts under ``scripts/``.

A script reaches this package by putting the ``scripts`` directory on its path::

    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # N: depth below scripts/
    from _lib import stats

Production code never imports from here; ``src`` stays the only library.
"""
