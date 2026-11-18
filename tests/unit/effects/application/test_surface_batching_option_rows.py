"""Decision records and option rows (FR-060a, FR-063a)."""

from __future__ import annotations

from pathlib import Path

import torch

from effects.application.surface_batching import SurfaceBatcher
from effects.domain.baselines import BaselineMasks
from effects.domain.effect_head_input import ACT_SLOT, SlotKind
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    EffectRecord,
    PlayabilityDecisionPayload,
)
from effects.domain.state_snapshot import (
    CardLine,
    LineKind,
)
from effects.infrastructure.shard_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache
from tests.unit.effects.builders import (
    make_batcher,
    make_entity,
    make_record,
    make_snapshot,
    slots_of_kind,
)

# ── decision records and option rows (FR-060a, FR-063a) ─────────────────

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


def _fixture_batcher() -> tuple[SurfaceBatcher, list[EffectRecord]]:
    """A batcher over the gen-3 fixture's sidecars. Its tokenizer knows only
    the specials but holds the keyword definitions a keyless keyword line is
    read through."""
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions
    from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, GEN3_SIDECAR_FOLDERS

    records = list(read_shard(_FIXTURES / "gen3-records.jsonl.gz"))
    cards, tokens = GEN3_SIDECAR_FOLDERS
    sidecars = SidecarCache({"cardsfolder": cards, "tokenscripts": tokens})
    tokenizer = AbilityTokenizer(
        {token: index for index, token in enumerate(SPECIAL_TOKENS)},
        load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS), surface="script",
    )
    batcher = make_batcher(
        tokenizer=tokenizer, sidecars=sidecars, masks=BaselineMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    return batcher, records


def test_a_decision_records_act_slot_reads_its_candidates_e():
    batcher, records = _fixture_batcher()
    decisions = [
        r for r in records
        if isinstance(r.payload, PlayabilityDecisionPayload)
        and batcher.text_of(r.payload.candidates[0].ability[0]) is not None
    ]
    assert decisions, "the fixture holds a decision with a convertible candidate"
    record = decisions[0]
    texts = batcher.batch_texts([record])
    rows = {text: row for row, text in enumerate(sorted(texts))}
    surface = batcher.surface_builder(rows).build(record)
    act = surface.slots[ACT_SLOT]
    candidate_text = batcher.text_of(record.payload.candidates[0].ability[0])
    assert act.e == rows[candidate_text]


def test_every_option_row_on_a_real_surface_follows_its_root():
    """FR-063a on the gen-3 fixture: a mode row comes right after its root's
    row or another mode row of the same card, never first."""
    batcher, records = _fixture_batcher()
    texts = batcher.batch_texts(records)
    rows = {text: row for row, text in enumerate(sorted(texts))}
    for record in records:
        slots = batcher.surface_builder(rows).build(record).slots
        for before, slot in zip(slots, slots[1:]):
            if slot.option:
                assert before.kind is SlotKind.ABILITY
                assert before.entity_id == slot.entity_id


def test_a_real_charm_gets_its_three_mode_rows_after_its_root():
    """FR-063a on Fire Magic as the converter writes it."""
    gen3 = _FIXTURES / "gen3-sidecars"
    sidecars = SidecarCache({"cardsfolder": gen3 / "cardsfolder"})
    root = ProvenanceKey("cardsfolder/f/fire_magic.txt", 0, "spell", 0)
    batcher = make_batcher(
        tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
        surface="script", e_dim=4, widths={}, device=torch.device("cpu"),
    )
    modes = batcher.options_for(root)
    assert [mode.option for mode in modes] == [0, 1, 2]
    fire_magic = make_entity(
        "E1", "Fire Magic", zone="stack",
        lines=(CardLine(keys=(root,), kind=LineKind.SPELL, granted=False),))
    record = make_record(ability=(root,), state=make_snapshot(entities=(fire_magic,)))
    texts = batcher.batch_texts([record])
    assert len(texts) == 4  # the root and each mode's own chain
    rows = {text: row for row, text in enumerate(sorted(texts))}
    abilities = slots_of_kind(batcher.surface_builder(rows).build(record), SlotKind.ABILITY)
    assert [slot.option for slot in abilities] == [False, True, True, True]
    assert [slot.e for slot in abilities[1:]] == [
        rows[batcher.text_of(mode)] for mode in modes
    ]
