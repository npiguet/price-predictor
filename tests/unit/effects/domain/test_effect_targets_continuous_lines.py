"""What a continuous record says was lost and gained on each characteristic line
(finding 10.3): a lone ``=`` in ``keywords``, the whole-class type tokens, a line
set to what the entity already held, and a contribution's ``abilities``.

The records are real: ``continuous-lines.jsonl`` holds six lines of the review
pilots, each as the collector wrote it, and the gen-3 fixture holds the rest. A
continuous record's board is the board without its static, so an entity's own
keywords, types and colours there are what it held before the static applied.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import cache
from pathlib import Path

from effects.domain.effect_head_layout import FIELDS_BY_NAME, HeadField
from effects.domain.effect_targets import derive_targets
from effects.domain.input_vocabularies import CARD_TYPES, SUPERTYPES
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import EffectRecord, RecordKind
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.state_snapshot import COLORS
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import read_shard
from tests.unit.effects.builders import GEN3_RECORDS

#: Cut from the review pilots (``output/effects/gen3/scratch/2026-11-2*-review-pilot*``):
#: pilot 2's Witness Protection on Wrecking Crew, Sinister Strength on Maggot
#: Carrier, Minimus Containment on Livaan and Mothdust Changeling's own
#: changeling; pilot 5's Evil Presence on a Plains; pilot 4's Lignify on Lys
#: Alana Scarblade. No byte of a line was changed.
_LINES = Path(__file__).parents[3] / "fixtures" / "effects" / "continuous-lines.jsonl"
_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"

_WITNESS_PROTECTION = "fe8ab2a4-fefe-40a0-ab81-e88ff48be5da.4-mwql5i2gx9ky.216"
_SINISTER_STRENGTH = "fe8ab2a4-fefe-40a0-ab81-e88ff48be5da.4-mwqktx3u57ch.5201"
_MINIMUS_CONTAINMENT = "fe8ab2a4-fefe-40a0-ab81-e88ff48be5da.4-mwqkxs6x775c.9001"
_EVIL_PRESENCE = "cd2b8419-91d0-4b47-975c-5a3d77b88315.5-mwr4y3j591ak.1623"
_MOTHDUST_CHANGELING = "fe8ab2a4-fefe-40a0-ab81-e88ff48be5da.5-mwql1n2n6dqg.1075"
_LIGNIFY = f"{_RUN}.2-mwqvwtly1zvq.9965"

_VOCABULARIES = {
    HeadField.KEYWORDS_GAINED: KEYWORDS_CHANGED, HeadField.KEYWORDS_LOST: KEYWORDS_CHANGED,
    HeadField.TYPES_GAINED: CARD_TYPES, HeadField.TYPES_LOST: CARD_TYPES,
    HeadField.SUBTYPES_GAINED: SUBTYPES_CHANGED, HeadField.SUBTYPES_LOST: SUBTYPES_CHANGED,
    HeadField.SUPERTYPES_GAINED: SUPERTYPES, HeadField.SUPERTYPES_LOST: SUPERTYPES,
    HeadField.COLORS_GAINED: COLORS, HeadField.COLORS_LOST: COLORS,
}


@cache
def _pilot_lines() -> dict[str, EffectRecord]:
    with _LINES.open(encoding="utf-8") as lines:
        records = [record_from_dict(json.loads(line)) for line in lines if line.strip()]
    return {record.record_id: record for record in records}


@cache
def _fixture() -> dict[str, EffectRecord]:
    return {record.record_id: record for record in read_shard(GEN3_RECORDS)}


def _target(record: EffectRecord, entity: str, name: str):
    """The value ``name`` is scored on for ``entity``: written or silent; None where absent."""
    return derive_targets(record)[entity].supervised(FIELDS_BY_NAME[name])


def _members(record: EffectRecord, entity: str, name: str) -> set[str] | None:
    """The members a multi-binary target sets, or None where it is absent."""
    value = _target(record, entity, name)
    if value is None:
        return None
    return {member for member, bit in zip(_VOCABULARIES[name], value, strict=True) if bit}


def _as_goldhound(record: EffectRecord) -> EffectRecord:
    """Wrecking Crew's record with its creature made Goldhound (Artifact Creature —
    Treasure Dog, first strike, menace). No pilot holds Witness Protection on
    Goldhound; this one holds it on a creature with two keyword lines, edited
    in its types, subtypes and keywords alone."""
    crew = next(entity for entity in record.state.entities if entity.id == "E109")
    assert [line.keyword for line in crew.lines] == ["reach", "trample"]
    goldhound = replace(
        crew, name="Goldhound", types=("artifact", "creature"), subtypes=("treasure", "dog"),
        granted_temporary=replace(crew.granted_temporary, keywords=("first strike", "menace")),
        lines=tuple(replace(line, fragment=fragment)
                    for line, fragment in zip(crew.lines, ("First Strike", "Menace"), strict=True)),
    )
    entities = tuple(
        goldhound if entity.id == "E109" else entity for entity in record.state.entities)
    return replace(record, state=replace(record.state, entities=entities))


class TestWitnessProtection:
    """"Enchanted creature loses all abilities and is a green and white Citizen
    creature with base power and toughness 1/1": keywords ``=``, types
    ``creature citizen -all-card-types -all-creature-types``."""

    def test_goldhound_loses_both_keywords_its_artifact_type_and_both_subtypes(self):
        record = _as_goldhound(_pilot_lines()[_WITNESS_PROTECTION])
        assert _members(record, "E109", HeadField.KEYWORDS_LOST) == {"first_strike", "menace"}
        assert _members(record, "E109", HeadField.TYPES_LOST) == {"artifact"}
        assert _members(record, "E109", HeadField.SUBTYPES_LOST) == {"treasure", "dog"}

    def test_goldhound_gains_the_citizen_and_not_the_creature_it_already_was(self):
        record = _as_goldhound(_pilot_lines()[_WITNESS_PROTECTION])
        assert _members(record, "E109", HeadField.TYPES_GAINED) == set()
        assert _members(record, "E109", HeadField.SUBTYPES_GAINED) == {"citizen"}
        assert _members(record, "E109", HeadField.KEYWORDS_GAINED) == set()

    def test_wrecking_crew_loses_reach_trample_and_its_creature_types(self):
        record = _pilot_lines()[_WITNESS_PROTECTION]
        assert _members(record, "E109", HeadField.KEYWORDS_LOST) == {"reach", "trample"}
        assert _members(record, "E109", HeadField.TYPES_LOST) == set()
        assert _members(record, "E109", HeadField.SUBTYPES_LOST) == {"human", "warrior"}
        assert _members(record, "E109", HeadField.COLORS_LOST) == {"R"}
        assert _members(record, "E109", HeadField.COLORS_GAINED) == {"W", "G"}

    def test_its_lost_abilities_are_abilities_lost(self):
        record = _pilot_lines()[_WITNESS_PROTECTION]
        assert _target(record, "E109", HeadField.ABILITIES_LOST) == 1
        assert _target(record, "E109", HeadField.ABILITIES_GAINED) == 0


class TestALineSetToWhatItWas:
    def test_a_black_creature_made_black_gains_no_colour(self):
        """Sinister Strength's ``= B`` on Maggot Carrier, the shape of
        Chainer's colour contribution on a black creature: nothing changes."""
        record = _pilot_lines()[_SINISTER_STRENGTH]
        assert _members(record, "E5", HeadField.COLORS_GAINED) == set()
        assert _members(record, "E5", HeadField.COLORS_LOST) == set()


class TestAWholeClassRemoved:
    def test_a_creature_made_a_treasure_loses_its_creature_subtypes_with_its_type(self):
        """Minimus Containment: ``-all-card-types -all-artifact-types artifact
        treasure`` on a legendary Dragon Shaman, which stays legendary."""
        record = _pilot_lines()[_MINIMUS_CONTAINMENT]
        assert _members(record, "E57", HeadField.TYPES_LOST) == {"creature"}
        assert _members(record, "E57", HeadField.TYPES_GAINED) == {"artifact"}
        assert _members(record, "E57", HeadField.SUBTYPES_LOST) == {"dragon", "shaman"}
        assert _members(record, "E57", HeadField.SUBTYPES_GAINED) == {"treasure"}
        assert _members(record, "E57", HeadField.SUPERTYPES_LOST) == set()

    def test_a_contribution_s_granted_and_taken_abilities_are_counted(self):
        record = _pilot_lines()[_MINIMUS_CONTAINMENT]
        assert _target(record, "E57", HeadField.ABILITIES_GAINED) == 1
        assert _target(record, "E57", HeadField.ABILITIES_LOST) == 1

    def test_a_plains_made_a_swamp_loses_plains(self):
        """Evil Presence: ``-all-land-types swamp``; a basic land type is a land type."""
        record = _pilot_lines()[_EVIL_PRESENCE]
        assert _members(record, "E16", HeadField.SUBTYPES_LOST) == {"plains"}
        assert _members(record, "E16", HeadField.SUBTYPES_GAINED) == {"swamp"}
        assert _members(record, "E16", HeadField.TYPES_LOST) == set()

    def test_lignify_takes_a_keyword_no_input_flag_keeps(self):
        """Lignify: keywords ``=``, types ``treefolk -all-creature-types`` on an
        Elf Assassin with fear. Fear has an output column and no input flag:
        what is lost is every keyword with a column, not every kept one."""
        record = _pilot_lines()[_LIGNIFY]
        assert "fear" not in KEPT_KEYWORD_FLAGS
        assert _members(record, "E38", HeadField.KEYWORDS_LOST) == {"fear"}
        assert _members(record, "E38", HeadField.SUBTYPES_LOST) == {"elf", "assassin"}
        assert _members(record, "E38", HeadField.SUBTYPES_GAINED) == {"treefolk"}


class TestEveryCreatureType:
    def test_a_changeling_s_gained_subtypes_are_unknown(self):
        """``all-creature-types`` gains every creature type, which no column states."""
        record = _pilot_lines()[_MOTHDUST_CHANGELING]
        assert _target(record, "E6", HeadField.SUBTYPES_GAINED) is None
        assert _members(record, "E6", HeadField.SUBTYPES_LOST) == set()


class TestTheFixture:
    def test_a_reconfigured_spider_loses_its_creature_subtype_with_creature(self):
        """Webspinner Cuff attached: ``-creature`` on an Equipment Spider."""
        record = _fixture()[f"{_RUN}.0-mwqvze8h7jrz.7394"]
        assert _members(record, "E37", HeadField.TYPES_LOST) == {"creature"}
        assert _members(record, "E37", HeadField.SUBTYPES_LOST) == {"spider"}

    def test_deep_freeze_on_a_creature_without_keywords_loses_none_and_its_abilities(self):
        """Deep Freeze: keywords ``= defender``, two ``-`` ability tokens."""
        record = _fixture()[f"{_RUN}.1-mwqvsypgt7o0.88"]
        assert _members(record, "E1", HeadField.KEYWORDS_LOST) == set()
        assert _members(record, "E1", HeadField.KEYWORDS_GAINED) == {"defender"}
        assert _target(record, "E1", HeadField.ABILITIES_LOST) == 1
        assert _target(record, "E1", HeadField.ABILITIES_GAINED) == 0

    def test_a_role_s_granted_static_is_one_ability_gained(self):
        record = _fixture()[f"{_RUN}.4-mwqw395jccfe.7923"]
        assert _target(record, "E137", HeadField.ABILITIES_GAINED) == 1
        assert _target(record, "E137", HeadField.ABILITIES_LOST) == 0

    def test_a_contribution_with_no_ability_tokens_changed_no_ability(self):
        record = _fixture()[f"{_RUN}.0-mwqvqdyrb9c0.118"]
        assert _target(record, "E22", HeadField.ABILITIES_GAINED) == 0
        assert _target(record, "E22", HeadField.ABILITIES_LOST) == 0

    def test_a_resolution_s_abilities_stay_unsupervised_where_unwritten(self):
        """An ``Animate`` of its own source names no subject, so a resolution's
        silence on abilities is no "none changed"."""
        halves = [record for record in _fixture().values()
                  if record.kind is RecordKind.RESOLUTION
                  and sampling_class(record) is SamplingClass.RESOLUTION_EFFECT]
        unwritten = [
            entry.supervised(FIELDS_BY_NAME[name])
            for record in halves for entry in derive_targets(record).values()
            for name in (HeadField.ABILITIES_GAINED, HeadField.ABILITIES_LOST)
            if name not in entry.fields
        ]
        assert unwritten and set(unwritten) == {None}
