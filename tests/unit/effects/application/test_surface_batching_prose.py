"""The prose surface (FR-069) and the relations under the masks."""

from __future__ import annotations

import torch

from effects.domain.baselines import BaselineMasks
from effects.domain.input_layout import EffectHeadInput, Slot, SlotKind
from effects.domain.provenance import SidecarLine
from effects.domain.records import (
    EffectRecord,
)
from effects.domain.state_snapshot import (
    CardLine,
    EntityState,
)
from effects.infrastructure.sidecar_io import SidecarCache
from tests.unit.effects.application.surface_batching_support import (
    _ACT,
    _Sidecars,
)
from tests.unit.effects.builders import (
    lines_of,
    make_batcher,
    make_record,
    make_snapshot,
)

# ── the prose surface (FR-069) ───────────────────────────────────────────


def test_a_prose_text_is_encoded_with_its_lines_role_spans():
    from effects.domain.ability_encoder import role_index
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.provenance import RoleSpan

    prose = "{T}: draw a card."
    line = SidecarLine(
        line_index=0, line_kind="activated", provenance=(_ACT,),
        script_text="AB$ Draw | Cost$ T",
        role_spans=(RoleSpan(0, 4, "cost"), RoleSpan(5, len(prose), "effect")), prose_offset=0,
    )
    batcher = make_batcher(
        tokenizer=AbilityTokenizer({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[MASK]": 3}),
        sidecars=_Sidecars(), masks=BaselineMasks(), surface="prose", e_dim=4,
        widths={}, device=torch.device("cpu"),
    )
    (encoded,) = batcher.stage({prose: line}).prepared
    assert encoded.role_ids[1] == role_index("cost")
    assert encoded.role_ids[-1] == role_index("effect")
    # The script surface's spans would index the wrong text; a script fallback
    # on the prose surface reads none.
    (fallback,) = batcher.stage({line.script_text: line}).prepared
    assert set(fallback.role_ids) == {0}


class TestRelationsUnderTheMasks:
    """T102: the board's links are state, so the no-state baseline reads none."""

    def _inputs(self):
        from effects.domain.input_layout import SlotKind
        from effects.domain.surface_collation import collate_surfaces

        slots = tuple(Slot(kind=SlotKind.CARD, position=i, features=(1.0,)) for i in range(3))
        return collate_surfaces([EffectHeadInput(slots=slots, relations=((0, 1, 2),))],
                                e_dim=4, widths={SlotKind.CARD: 1, SlotKind.GLOBAL: 40,
                                                 SlotKind.PLAYER: 1})

    def test_no_state_drops_every_relation(self):
        from effects.domain.baselines import Baseline, baseline_masks
        from tests.unit.effects.builders import make_batcher

        batcher = make_batcher(masks=baseline_masks(Baseline.NO_STATE))
        assert batcher._masked(self._inputs()).relations.shape == (0, 4)

    def test_the_full_model_keeps_them(self):
        from effects.domain.baselines import Baseline, baseline_masks
        from tests.unit.effects.builders import make_batcher

        batcher = make_batcher(masks=baseline_masks(Baseline.FULL))
        assert batcher._masked(self._inputs()).relations.tolist() == [[0, 0, 1, 2]]


def _flying_batcher(**settings):
    from effects.domain.ability_tokenizer import AbilityTokenizer
    from effects.domain.keyword_definitions import KeywordDefinition

    words = ["[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "flying", "cant", "be"]
    tokenizer = AbilityTokenizer(
        {word: index for index, word in enumerate(words)},
        {"Flying": KeywordDefinition(
            keyword="Flying",
            reminder_template="cant be blocked except by creatures with flying or reach",
        )},
    )
    return make_batcher(
        tokenizer=tokenizer, sidecars=_Sidecars(), masks=settings.pop("masks", BaselineMasks()),
        surface="prose", e_dim=4, device=torch.device("cpu"), **settings,
    )


def _resolution(entities) -> EffectRecord:
    return make_record(
        record_id="run.0.1", run_id="run", game_id="run.0.1", ability=(_ACT,),
        state=make_snapshot(entities=entities),
    )


def test_a_keyword_line_with_no_script_reads_its_reminder_text():
    """Root spec § 9.2: a keyword an effect granted is a line read through its
    reminder text; a keyword the definitions do not know reads as written."""
    from effects.domain.slot_lines import keyword_line_key

    batcher = _flying_batcher(widths={})
    flying = keyword_line_key("Flying")
    assert batcher.text_of(flying) == "cant be blocked except by creatures with flying or reach"
    assert batcher.line_of(flying) is None
    assert batcher.options_for(flying) == ()
    assert batcher.keyword_of_key(flying) == "flying"
    assert batcher.text_of(keyword_line_key("Mystery:3")) == "Mystery:3"


def test_a_batch_encodes_the_reminder_text_of_a_granted_keyword_line():

    entity = EntityState(
        id="E1", name="hornclaw", zone="battlefield", controller="P0",
        lines=(CardLine(keys=(), kind="keyword", granted=True, fragment="Flying"),),
    )
    record = _resolution(entities=(entity,))
    texts = _flying_batcher(widths={}).batch_texts([record])
    assert "cant be blocked except by creatures with flying or reach" in texts


def test_no_state_zeroes_the_line_flags_and_the_card_ids():
    from effects.domain.slot_widths import layout_widths

    entity = EntityState(
        id="E1", name="soldier", zone="battlefield", controller="P0",
        subtypes=("human",), counters={"OIL": 2}, lines=lines_of(printed=(_ACT,)),
    )
    record = _resolution(entities=(entity,))
    widths = layout_widths()
    full = _flying_batcher(widths=widths).prepare([record]).inputs
    blind = _flying_batcher(
        widths=widths, masks=BaselineMasks(zero_state=True)).prepare([record]).inputs
    assert full.slot_features[SlotKind.ABILITY].abs().sum() > 0
    assert full.subtype_ids.abs().sum() > 0 and full.counter_counts.abs().sum() > 0
    assert blind.slot_features[SlotKind.ABILITY].abs().sum() == 0
    for name in ("subtype_ids", "chosen_type_ids", "real_face_subtype_ids",
                 "counter_ids", "counter_counts"):
        assert getattr(blind, name).abs().sum() == 0, name


def test_a_floating_effects_card_reads_its_source_lines_text():
    """Root spec § 9.3 (T114): a timestamped floating effect's ``[CARD]``
    is followed by its source line, encoded with the batch like any other."""
    from effects.domain.surface_builder import command_zone_entities
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import (
        GEN3_SIDECAR_FOLDERS,
        gen3_fixture_records,
    )

    record = next(
        record_from_dict(raw) for raw in gen3_fixture_records()
        if raw["record_id"].endswith(".0-mwqvqdyrb9c0.37"))
    floating, = command_zone_entities(record)
    sidecars = SidecarCache({folder.name: folder for folder in GEN3_SIDECAR_FOLDERS})
    batcher = make_batcher(
        tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    source = batcher.text_of(floating.lines[0].keys[0])
    texts = batcher.batch_texts([record])
    assert source is not None and source in texts
    rows = {text: row for row, text in enumerate(sorted(texts))}
    surface = batcher.surface_builder(rows).build(record)
    at = next(index for index, slot in enumerate(surface.slots)
              if slot.kind is SlotKind.CARD and slot.entity_id == floating.id)
    assert surface.slots[at + 1].e == rows[source]
