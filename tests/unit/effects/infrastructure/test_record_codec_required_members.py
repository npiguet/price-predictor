"""A gen-3 record carries every member :data:`gen3_members.ALWAYS` names on
every instance of its site, so the decoder reads each strictly: a missing one
is a decode error naming its site and member, not a default. Each case is a
real gen-3 record less one member."""

from __future__ import annotations

import copy
import gzip
import json
import re
from functools import cache
from pathlib import Path

import pytest

from effects.domain.gen3_members import ALWAYS, MissingMemberError, Site, site_blocks
from effects.domain.shard_generation import OlderGenerationError
from effects.infrastructure.record_codec import SupersededPayloadError, record_from_dict
from effects.infrastructure.shard_io import RecordDecodeError, read_records
from tests.unit.effects.builders import gen3_fixture_records


@cache
def _fixture() -> tuple[dict, ...]:
    return tuple(gen3_fixture_records())


def _first_carrying(site: Site, member: str, start: int = 0) -> int:
    """The index of the fixture's first record from ``start`` on holding an
    instance of ``site`` that carries ``member``."""
    rows = _fixture()
    return next(index for index in range(start, len(rows))
                if any(found is site and member in block
                       for found, block in site_blocks(rows[index])))


def _dropped(index: int, site: Site, member: str) -> dict:
    """A copy of fixture record ``index`` with ``member`` dropped from its
    first instance of ``site`` that carries it."""
    row = copy.deepcopy(_fixture()[index])
    block = next(block for found, block in site_blocks(row)
                 if found is site and member in block)
    del block[member]
    return row


#: The members whose absence an earlier check refuses first, by its own error.
_REFUSED_EARLIER: dict[tuple[Site, str], type[Exception]] = {
    (Site.RECORD, "generation"): OlderGenerationError,
    (Site.COST_HALF, "events"): SupersededPayloadError,
    (Site.EFFECT_HALF, "outcome"): SupersededPayloadError,
}

_OWED = [pytest.param(site, name, id=f"{site.value}.{name}")
         for site, names in ALWAYS.items() for name in names
         if (site, name) not in _REFUSED_EARLIER]


@pytest.mark.parametrize(("site", "member"), _OWED)
def test_a_record_without_an_always_member_does_not_decode(site, member):
    raw = _dropped(_first_carrying(site, member), site, member)

    with pytest.raises(MissingMemberError, match=re.escape(f"no {site.value}.{member}")):
        record_from_dict(raw)


@pytest.mark.parametrize(("site", "member"), list(_REFUSED_EARLIER))
def test_a_member_an_earlier_check_owes_keeps_its_own_error(site, member):
    raw = _dropped(_first_carrying(site, member), site, member)

    with pytest.raises(_REFUSED_EARLIER[site, member]):
        record_from_dict(raw)


def _shard_with(directory: Path, index: int, row: dict) -> Path:
    """The fixture as one gzip shard, record ``index`` replaced by ``row``."""
    rows = list(_fixture())
    rows[index] = row
    shard = directory / "run.0-0.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8", newline="\n") as out:
        out.writelines(json.dumps(each) + "\n" for each in rows)
    return shard


@pytest.mark.parametrize(("site", "member"), [
    (Site.RECORD, "fork"), (Site.COMBAT, "blocks"), (Site.CONTRIBUTION, "abilities"),
])
def test_a_shard_line_without_a_member_is_refused_at_its_shard_and_line(
        tmp_path, site, member):
    index = _first_carrying(site, member, start=len(_fixture()) // 2)
    shard = _shard_with(tmp_path, index, _dropped(index, site, member))

    with pytest.raises(RecordDecodeError) as caught:
        list(read_records(tmp_path))

    message = str(caught.value)
    assert message.startswith(f"{shard}, line {index + 1}: "), message
    assert f"no {site.value}.{member}" in message, message
    assert (caught.value.path, caught.value.line) == (shard, index + 1)


def test_validate_corpus_names_the_member_twice_and_cuts_the_shard_there(tmp_path):
    """The presence check sees the raw line before it is decoded, so it
    names the member; the decode then ends the shard's read at that line."""
    from tests.unit.effects.application.validate_corpus_ports import validate_directory

    index = _first_carrying(Site.COMBAT, "blocks", start=len(_fixture()) // 2)
    shard = _shard_with(tmp_path, index, _dropped(index, Site.COMBAT, "blocks"))

    findings = {finding.name: finding for finding in validate_directory(tmp_path).findings}

    presence = next(finding for name, finding in findings.items()
                    if name.startswith("gen-3 fields"))
    decodes = findings["every shard line decodes"]
    assert not presence.ok and "payload<combat>.blocks 1" in presence.detail[0]
    assert not decodes.ok
    assert decodes.detail[0].startswith(f"{shard}, line {index + 1}: ")
    assert "no payload<combat>.blocks" in decodes.detail[0]
