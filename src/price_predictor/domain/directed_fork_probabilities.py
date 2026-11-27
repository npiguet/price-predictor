"""How much a rule needs a directed fork: its probability from the records
each of its two sides (``fires``, ``does_not_fire``) already has.

One formula for every tool that steers directed forks: ``effects
collect-variants``, which counts the forks of its own run's records, and
``sealed match-outcomes``, which starts from the records per side
``scripts/rules_catalogue.py directed-rules`` writes and adds the forks its
workers keep. It lives here because ``sealed`` never imports ``effects``.
"""

from __future__ import annotations

from collections.abc import Sequence

#: A rule still short on a side forks at least this often: a rule one record
#: from its count is not starved of the chance to finish.
MIN_NEEDED_PROBABILITY = 0.05


def directed_fork_probability(side_counts: Sequence[int], count: int) -> float:
    """The probability a directed fork of a rule is taken at a boundary its
    precondition holds at, given the records of each of its sides and the
    ``count`` each side needs.

    0 once every side has reached ``count``; 1 while a side has none; between,
    the share of ``count`` the shorter side still misses, but never below
    :data:`MIN_NEEDED_PROBABILITY`.
    """
    if count < 1:
        raise ValueError(f"the count a side needs must be at least 1, got {count}")
    shorter = min(side_counts)
    if shorter >= count:
        return 0.0
    if shorter <= 0:
        return 1.0
    return max(MIN_NEEDED_PROBABILITY, (count - shorter) / count)
