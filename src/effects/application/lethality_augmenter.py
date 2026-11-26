"""The lethality counterfactual copies of a training step (FR-063e)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from effects.domain.grid_edits import TextOverride
from effects.domain.lethality_counterfactuals import (
    BoardReading,
    CopyMaterial,
    Counterfactual,
    counterfactual,
    qualify,
)
from effects.domain.records import EffectRecord


@dataclass(frozen=True)
class LethalityAugmenter:
    """``--lethality-copies`` copies of each step record that qualifies
    (FR-063e), drawn before the build's own draws so a run without copies
    draws exactly as before."""

    #: ``--lethality-copies``: how many copies each qualifying record gets.
    copies: int
    #: ``entity -> keywords`` the board reading names a card's keywords by.
    keywords_of: Callable
    sidecars: object

    def copies_of(
        self, records: Sequence[EffectRecord], targets: Sequence, batcher,
    ) -> list[Counterfactual]:
        """The copies of the step's ``records``, whose ``targets`` they were
        derived to. Only a record's first natural row is copied: its other
        natural rows and its added rows are not (D15). Each copy's text edit is
        staged on ``batcher``."""
        made: list[Counterfactual] = []
        first_naturals = first_natural_rows(records)
        for index, (record, record_targets) in enumerate(zip(records, targets)):
            if record.ability and index in first_naturals:
                made.extend(self._copies_of_record(record, record_targets, batcher))
        return made

    def _copies_of_record(self, record: EffectRecord, targets, batcher) -> list[Counterfactual]:
        """``record``'s copies, none where it does not qualify."""
        key = record.ability[0]
        line = batcher.line_of(key)
        qualified = qualify(record, line, targets, BoardReading(self.keywords_of, self.sidecars))
        if qualified is None:
            return []
        made = []
        for number in range(self.copies):
            material = CopyMaterial(record, qualified, targets, line, batcher.text_of(key))
            copy = counterfactual(material, batcher.rng, number)
            self._stage_edit(batcher, copy, key)
            made.append(copy)
        return made

    @staticmethod
    def _stage_edit(batcher, copy: Counterfactual, key) -> None:
        """Hand the copy's edited text to ``batcher`` under its own key."""
        edit = copy.edit
        if edit.key is not None:
            batcher.override_text(TextOverride(
                edit.key, edit.text, key, script=edit.script, role_spans=edit.role_spans,
            ))


def first_natural_rows(records: Sequence[EffectRecord]) -> frozenset[int]:
    """The indices of each record's first natural row in a step: a natural row
    has no drawn rule, and a record's later natural copies repeat its id (D15)."""
    seen: set[str] = set()
    firsts = set()
    for index, record in enumerate(records):
        if record.curation is not None and record.curation.is_added:
            continue
        if record.record_id not in seen:
            seen.add(record.record_id)
            firsts.add(index)
    return frozenset(firsts)
