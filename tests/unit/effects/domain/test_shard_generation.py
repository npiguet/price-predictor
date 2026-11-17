"""Every reader reads gen-3 records only; an older one is refused, named (FR-033)."""

from __future__ import annotations

import pytest

from effects.domain.shard_generation import (
    OlderGenerationError,
    ShardGeneration,
    UnacceptedGenerationError,
    UnknownGenerationError,
    older_generation_of,
    require_gen3,
)


def test_a_record_without_random_seat_is_named_gen1():
    assert older_generation_of({"kind": "resolution"}) is ShardGeneration.GEN_1


def test_a_record_carrying_random_seat_is_named_gen2_whatever_its_value():
    """An on-policy gen-2 record writes ``false``; it is still gen-2."""
    assert older_generation_of({"random_seat": False}) is ShardGeneration.GEN_2
    assert older_generation_of({"random_seat": True}) is ShardGeneration.GEN_2


@pytest.mark.parametrize(("raw", "named"), [
    ({"kind": "resolution", "random_seat": False}, "gen-2"),
    ({"kind": "resolution"}, "gen-1"),
])
def test_a_record_without_a_generation_member_is_refused_naming_its_generation(raw, named):
    with pytest.raises(OlderGenerationError, match=f"the record is {named} "):
        require_gen3(raw)


@pytest.mark.parametrize("value", [4, 2, "3", True, 3.0, None])
def test_a_generation_member_naming_no_known_generation_is_refused(value):
    """A record's generation decides how several fields read; a value naming
    none of them is no record any reader may guess at."""
    with pytest.raises(UnknownGenerationError, match=repr(value)):
        require_gen3({"generation": value, "random_seat": False})


def test_both_refusals_are_unaccepted_generations():
    """So every command's exit-status mapping for a refused generation holds them."""
    assert issubclass(OlderGenerationError, UnacceptedGenerationError)
    assert issubclass(UnknownGenerationError, UnacceptedGenerationError)


def test_generation_3_passes_whatever_else_the_record_carries():
    require_gen3({"generation": 3, "random_seat": True})
