"""The value types the corpus build's survey, placement, budgets and write
pass share, each defined once here.

Torch-free: ``build-corpus``'s survey and write workers import it.
"""

from __future__ import annotations

from collections.abc import Mapping

from effects.domain.rule_catalogue import Polarity

__all__ = ["CopyMap", "RecordHash", "RuleCell"]

#: A record's selection hash: ``corpus_curation.record_hash`` of its id under
#: ``--seed`` (a protection copy's, ``corpus_survey.protection_hash``). Every
#: selection choice is a function of it, and the write pass keys a record's
#: added copies by it.
RecordHash = int

#: ``(rule, polarity)``: one rule tag of a record, one cell of a game's
#: inventory.
RuleCell = tuple[str, Polarity]

#: ``game -> record hash -> drawn rule per added copy``: the rule budgets'
#: draws (``RuleBudgets.copies``), a record's protection copies under its
#: protection hash. A worker receives its own shard's games only.
CopyMap = Mapping[str, Mapping[RecordHash, tuple[str, ...]]]
