"""What a record is read as, derived once.

The survey reads each kept record several ways (the unsupervised counts, the
rules' carried outputs, the cell's outcome signature), and each reads
:func:`~effects.domain.effect_targets.derive_targets` of the same record. A
:class:`RecordReading` holds the record and derives its targets on the first
ask, so a record whose reads need none derives none and one whose reads need
them derives them once. The record's turn or step condition is held the same
way (:meth:`condition`), since its join counts toward the sidecar cache's join
gaps each time it is read.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.records import EffectRecord


class RecordReading:
    """One record, and what is derived from it, each derived on first ask."""

    __slots__ = ("_condition", "_targets", "record")

    def __init__(self, record: EffectRecord) -> None:
        self.record = record
        self._targets: RecordTargets | None = None
        self._condition: tuple[Any] | None = None

    @classmethod
    def or_new(cls, record: EffectRecord, reading: RecordReading | None) -> RecordReading:
        """``reading`` where the caller holds one of ``record``, else a new one."""
        return cls(record) if reading is None else reading

    @property
    def targets(self) -> RecordTargets:
        """:func:`~effects.domain.effect_targets.derive_targets` of the record."""
        if self._targets is None:
            self._targets = derive_targets(self.record)
        return self._targets

    def condition(self, derive: Callable[[], Any]) -> Any:
        """The record's turn or step condition: ``derive()`` on the first ask,
        its answer, None included, on every later one."""
        if self._condition is None:
            self._condition = (derive(),)
        return self._condition[0]
