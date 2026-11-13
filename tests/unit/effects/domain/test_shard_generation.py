"""A record's generation is the presence of ``random_seat``, not its value (FR-033)."""

from __future__ import annotations

import pytest

from effects.domain.shard_generation import (
    ShardGeneration,
    UnacceptedGenerationError,
    UnknownGenerationError,
)


def test_a_record_without_random_seat_is_gen1():
    assert ShardGeneration.of({"kind": "resolution"}) is ShardGeneration.GEN_1


def test_a_record_carrying_random_seat_is_gen2_whatever_its_value():
    """An on-policy gen-2 record writes ``false``; it is still gen-2."""
    assert ShardGeneration.of({"random_seat": False}) is ShardGeneration.GEN_2
    assert ShardGeneration.of({"random_seat": True}) is ShardGeneration.GEN_2


@pytest.mark.parametrize("value", [4, 2, "3", True, 3.0, None])
def test_a_generation_member_naming_no_known_generation_is_refused(value):
    """A record's generation decides how several fields read; a value naming
    none of them is no record any reader may guess at, gen-1 least of all."""
    with pytest.raises(UnknownGenerationError, match=repr(value)):
        ShardGeneration.of({"generation": value, "random_seat": False})


def test_an_unknown_generation_is_refused_as_an_unaccepted_one():
    """So every command's exit-status mapping for a refused generation holds it."""
    assert issubclass(UnknownGenerationError, UnacceptedGenerationError)


def test_generation_3_is_gen3_whatever_else_the_record_carries():
    assert ShardGeneration.of({"generation": 3, "random_seat": True}) is ShardGeneration.GEN_3
