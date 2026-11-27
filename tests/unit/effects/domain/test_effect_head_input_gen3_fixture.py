"""The real gen-3 records lay out at one width per slot kind, and run
through the model, relations included."""

from __future__ import annotations

import pytest

from effects.domain.card_features import card_features, card_segments, counter_rows
from effects.domain.input_layout import SlotKind
from tests.unit.effects.domain.effect_head_input_support import (
    E_DIM,
    _build,
)


def _entities_and_lines(records):
    """Every entity of ``records``' boards, and every slot line of those."""
    from effects.domain.slot_lines import slot_lines

    entities = [entity for record in records for entity in record.state.entities]
    return entities, [line for entity in entities for line in slot_lines(entity)]


class TestTheGen3Fixture:
    """The real gen-3 records lay out at one width per slot kind."""

    @pytest.fixture(scope="class")
    def records(self):
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records

        return [record_from_dict(raw) for raw in gen3_fixture_records()]

    def test_every_slot_kind_keeps_one_width(self, records):
        from effects.domain.slot_widths import layout_widths

        widths = layout_widths()
        seen: dict = {}
        for record in records:
            for slot in _build(record).slots:
                seen.setdefault(slot.kind, set()).add(len(slot.features))
        assert seen == {kind: {widths[kind]} for kind in seen}

    def test_the_fixture_reaches_the_new_inputs(self, records):
        from effects.domain.slot_lines import keyword_line_fragment

        entities, lines = _entities_and_lines(records)
        assert any(not line.active for line in lines)
        # Cut before ``grant`` existed: its granted keyword lines carry none.
        assert any(line.granted and keyword_line_fragment(line.key) for line in lines)
        assert any(counter_rows(entity) for entity in entities)
        assert any(entity.chosen and entity.chosen.type for entity in entities)


class TestTheGen3FixtureInTheModel:
    """The fixture's widths are the model's, relations included."""

    def test_the_fixture_runs_through_the_model(self, gen3_records):
        import torch

        from effects.domain.effect_model import EffectModel, EffectModelConfig
        from effects.domain.slot_lines import ABILITY_FEATURE_WIDTH
        from effects.domain.slot_widths import layout_widths
        from effects.domain.surface_collation import collate_surfaces

        records = list(gen3_records.values())[:16]
        widths = layout_widths()
        entity, record = records[0].state.entities[0], records[0]
        assert widths[SlotKind.CARD] == len(card_features(entity, record)) == max(
            part.stop for part in card_segments(entity, record).values())
        assert widths[SlotKind.ABILITY] == ABILITY_FEATURE_WIDTH
        surfaces = [_build(record) for record in records]
        inputs = collate_surfaces(surfaces, e_dim=E_DIM, widths=widths)
        assert inputs.relations.shape[0] == sum(len(s.relations) for s in surfaces) > 0
        torch.manual_seed(0)
        model = EffectModel(EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
            ability_features=widths[SlotKind.ABILITY], e_dim=E_DIM,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        ))
        hidden = model(inputs)
        assert hidden.shape[:2] == inputs.slot_kinds.shape
        assert torch.isfinite(hidden).all()
