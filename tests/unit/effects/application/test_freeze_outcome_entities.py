"""A rule test of a per-entity rule counts only rows naming an outcome entity
(finding 7.8, contracts/selection-set.md, Freeze step 4).

The control score reads a rule's per-entity outputs at the tag's outcome
entities, so a row naming none is read nowhere: for a rule whose effective
outputs are all per-entity, the freeze counts only rows naming one toward n
and the standard error, and leaves out, naming it, a rule with a polarity
holding none. A rule with a verdict, global or created-objects output keeps
every row. The records are the real gen-3 fixture's, their games edited into
the two halves, one game a record (a standard error counts games, finding 12.7).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.application.freeze_selection_set import (
    FreezePorts,
    ProbeItem,
    RuleTestRules,
    SelectionSetRefused,
    freeze_selection_set,
    left_out_lines,
)
from effects.domain.rule_catalogue import MapVerdict, Polarity, RuleMapEntry
from effects.domain.rule_detectors import RuleTag
from effects.domain.rule_output_carriage import carried_outputs, read_at_entities_only
from effects.domain.score_scales import auc_standard_error
from effects.domain.selection_set import NO_OUTCOME_ENTITY, error_of, item_key
from tests.unit.effects.application.test_freeze_selection_set import (
    GD,
    LENIENT,
    _Grid,
    _Losses,
    _manifest,
    _reader,
)

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
RULE = "121.2"
#: Carried at the actor player by every record the fixture below keeps.
OUTPUTS = ("gate",)
HALVES = {GD: {"selection": [f"s{i}" for i in range(16)],
               "reporting": [f"r{i}" for i in range(16)]}}


@pytest.fixture(scope="module")
def fixture_records():
    """32 rule-test-eligible fixture records, 16 in each half, one game each."""
    from tests.unit.effects.builders import curated_fixture_records

    eligible = [r for r in curated_fixture_records()
                if not (r.fork or r.interventional or r.synthetic or r.aimed_at is not None
                        or r.directed is not None)
                and carried_outputs(r, [r.actor_player], OUTPUTS)]
    assert len(eligible) >= 32
    return [replace(record, game_id=f"s{i}" if i < 16 else f"r{i - 16}")
            for i, record in enumerate(eligible[:32])]


def _tags(records, *, fires=10, near_with_entity=0):
    """Per half: ``fires`` rows naming the actor as outcome entity, the rest
    ``does_not_fire``, the first ``near_with_entity`` of them naming it too."""
    tags = {}
    for half in ("s", "r"):
        rows = [r for r in records if r.game_id[0] == half]
        for i, record in enumerate(rows):
            polarity = FIRES if i < fires else NOT
            named = polarity is FIRES or i - fires < near_with_entity
            tags[record.record_id] = [RuleTag(
                RULE, polarity, observed=polarity is FIRES, condition=(record.actor_player,),
                outcome=(record.actor_player,) if named else (), scenario=f"{RULE}-1")]
    return tags


class _Probes:
    """One binary probe type, positive on the records of the even-numbered games."""

    def items(self, records):
        for record in records:
            yield ProbeItem("dies", "binary", record.record_id, "E1",
                            float(int(record.game_id[1:]) % 2))


def _freeze(records, tags, *, entity_read={RULE: OUTPUTS}):
    rules = RuleTestRules(tag=lambda record: tags.get(record.record_id, []),
                          declared=frozenset({RULE}), trained=frozenset({RULE}),
                          entity_read=entity_read)
    return freeze_selection_set(
        _manifest(HALVES), {GD: _reader(records)},
        ports=FreezePorts(rules=rules, probes=_Probes(), grids=[_Grid()], deviance=_Losses()),
        thresholds=LENIENT)


def test_a_per_entity_rule_whose_near_misses_name_no_entity_is_left_out_and_named(
        fixture_records):
    with pytest.raises(SelectionSetRefused) as refused:
        _freeze(fixture_records, _tags(fixture_records))
    selection = refused.value.selection
    for half in ("selection", "reporting"):
        assert RULE not in selection.rule_tests[GD][half]
        left = selection.left_out[GD][half]["rule_tests"]
        assert [(item.item, item.reason) for item in left] == [
            (RULE, f"{NO_OUTCOME_ENTITY}: 10 of 10 fires, 0 of 6 does_not_fire name one")]
        assert selection.standard_errors[item_key(GD, half, "rule_tests", RULE)] is None
    assert any(f"{RULE} ({NO_OUTCOME_ENTITY}" in line for line in left_out_lines(selection))


def test_with_entities_on_both_polarities_the_rule_is_kept(fixture_records):
    frozen = _freeze(fixture_records, _tags(fixture_records, near_with_entity=6))
    for half in ("selection", "reporting"):
        sides = frozen.rule_tests[GD][half][RULE]
        assert (len(sides["fires"]), len(sides["does_not_fire"])) == (10, 6)
        assert all(row.entities for rows in sides.values() for row in rows)


def test_only_rows_naming_an_entity_count_toward_n_and_the_standard_error(fixture_records):
    frozen = _freeze(fixture_records, _tags(fixture_records, near_with_entity=3))
    for half in ("selection", "reporting"):
        sides = frozen.rule_tests[GD][half][RULE]
        assert (len(sides["fires"]), len(sides["does_not_fire"])) == (10, 3)
        error = frozen.standard_errors[item_key(GD, half, "rule_tests", RULE)]
        assert error_of(error) == pytest.approx(auc_standard_error(10, 3))


def test_a_rule_read_at_the_record_keeps_its_rows_without_an_entity(fixture_records):
    frozen = _freeze(fixture_records, _tags(fixture_records), entity_read={})
    for half in ("selection", "reporting"):
        sides = frozen.rule_tests[GD][half][RULE]
        assert (len(sides["fires"]), len(sides["does_not_fire"])) == (10, 6)


def test_a_rule_is_read_at_entities_only_when_every_effective_output_is_per_entity():
    assert read_at_entities_only(("gate", "zone_outcome"))
    assert not read_at_entities_only(("gate", "can_play"))
    assert not read_at_entities_only(("created_objects",))
    assert not read_at_entities_only(())


def test_the_rules_read_at_entities_only_come_from_the_manifest_s_effective_outputs():
    from effects.infrastructure.rules_catalogue_io import load_scenarios

    manifest = replace(
        _manifest(HALVES),
        rule_map={rule: RuleMapEntry(rule, MapVerdict.TRAINED, ("gate", "can_play"))
                  for rule in ("121.2", "601.2", "704.5a")},
        effective_outputs={"121.2": ("gate", "zone_outcome"), "601.2": ("gate", "can_play"),
                           "704.5a": ()})
    rules = RuleTestRules.of(load_scenarios(), manifest)
    assert rules.entity_read == {"121.2": ("gate", "zone_outcome")}
