"""The closed keyword list curation records (``output_list_survey.keywords_named``)."""

from __future__ import annotations

from effects.domain.event_schema import Event, EventType
from effects.domain.output_list_survey import keywords_named
from effects.domain.records import RecordKind, ResolutionPayload
from tests.unit.effects.record_builders import make_record


def test_a_hexproof_from_a_quality_names_hexproof():
    """Granting hexproof from black is a hexproof change in the outputs, so
    the list that sizes the keyword columns names ``hexproof`` for it."""
    record = make_record(RecordKind.RESOLUTION, payload=ResolutionPayload(events=(
        Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
              params={"keywords": ["Hexproof:Black"]}),)))
    assert keywords_named(record, frozenset()) == {"hexproof"}
