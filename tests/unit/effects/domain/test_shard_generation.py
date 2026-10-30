"""A record's generation is the presence of ``random_seat``, not its value (FR-033)."""

from __future__ import annotations

from effects.domain.shard_generation import ShardGeneration


def test_a_record_without_random_seat_is_gen1():
    assert ShardGeneration.of({"kind": "resolution"}) is ShardGeneration.GEN_1


def test_a_record_carrying_random_seat_is_gen2_whatever_its_value():
    """An on-policy gen-2 record writes ``false``; it is still gen-2."""
    assert ShardGeneration.of({"random_seat": False}) is ShardGeneration.GEN_2
    assert ShardGeneration.of({"random_seat": True}) is ShardGeneration.GEN_2
