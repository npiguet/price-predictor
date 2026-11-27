"""What a record is read as, derived once.

The survey reads each kept record several ways (the unsupervised counts, the
rules' carried outputs, the cell's outcome signature), and each reads
:func:`~effects.domain.effect_targets.derive_targets` of the same record. A
:class:`RecordReading` holds the record and derives its targets on the first
ask, so a record whose reads need none derives none and one whose reads need
them derives them once.
"""

from __future__ import annotations

from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.records import EffectRecord


class RecordReading:
    """One record, and what is derived from it, each derived on first ask."""

    __slots__ = ("_targets", "record")

    def __init__(self, record: EffectRecord) -> None:
        self.record = record
        self._targets: RecordTargets | None = None

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
