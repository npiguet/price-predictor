"""``sealed match-outcomes`` records its caps under the members the collectors use."""

from __future__ import annotations

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.collector_connector import caps_document
from sealed.domain.collection_caps import caps_manifest


def test_the_properties_of_a_cap_set_read_back_as_the_collectors_document():
    caps = CollectionCaps(
        mana_cap=3, playability_rate=0.5, probe_keywords="deathtouch", random_seat_share=0.25,
        random_seat_probability=0.5, count_dropped_events=True,
        directed_fork_rules=("704.5f", "702.19e"), directed_rule_probabilities={"704.5f": 0.75},
        max_records_per_game=100,
    )
    assert caps_manifest(caps.as_system_properties()) == caps_document(caps)


def test_the_defaults_read_back_as_the_collectors_document():
    caps = CollectionCaps()
    assert caps_manifest(caps.as_system_properties()) == caps_document(caps)
