"""A card's lines as ``[ABILITY]`` slots: which lines get one, context
dropout, a key with no line, a charm's mode rows and a keyword of several
lines."""

from __future__ import annotations

import random

import pytest

from effects.domain.input_layout import SlotKind
from effects.domain.provenance import ProvenanceKey
from effects.domain.state_snapshot import Refs
from effects.domain.surface_builder import SurfaceBuilder
from tests.unit.effects.builders import (
    lines_of,
    slots_of_kind,
)
from tests.unit.effects.domain.effect_head_input_support import (
    _ANTHEM,
    _BOLT,
    _VECTORS,
    _WARD,
    _ZERO,
    E_DIM,
    _build,
    _entity,
    _flags,
    _Lines,
    _record,
    _snapshot,
)


class TestAbilityTokens:
    def test_printed_and_attachment_granted_lines_become_tokens(self):
        entity = _entity("E1", lines=lines_of((_ANTHEM,), (_WARD,)))
        surface = _build(_record(state=_snapshot(entities=(entity,))))
        vectors = [s.e_vector for s in slots_of_kind(surface, SlotKind.ABILITY)]
        assert vectors == [_VECTORS[_ANTHEM], _VECTORS[_WARD]]

    def test_an_unmet_static_is_one_more_token_flagged_inactive(self):
        """The lines in force read as tokens, printed or granted; an unmet
        static is one more token, flagged inactive (T074)."""
        from effects.domain.state_snapshot import CardLine

        lines = (
            CardLine(keys=(_ANTHEM,), kind="static", granted=False, active=True),
            CardLine(keys=(_WARD,), kind="static", granted=True, active=True),
            CardLine(keys=(_BOLT,), kind="static", granted=False, active=False),
        )
        unmet = _build(_record(state=_snapshot(entities=(_entity("E1", lines=lines),))))
        met = _build(_record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of((_ANTHEM,), (_WARD,))),))))
        assert unmet.slots[:-1] == met.slots
        inactive = unmet.slots[-1]
        assert inactive.e_vector == _VECTORS[_BOLT] and _flags(inactive)["active"] == 0.0

    def test_context_dropout_can_drop_a_context_ability(self):
        record = _record()
        kept = _build(record, context_dropout=0.0)
        dropped = _build(record, context_dropout=1.0, rng=random.Random(1))
        assert len(slots_of_kind(dropped, SlotKind.ABILITY)) < len(
            slots_of_kind(kept, SlotKind.ABILITY)
        )

    def test_the_acting_entitys_abilities_are_never_dropped(self):
        """Dropping the ability under study would make the record unanswerable."""
        state = _snapshot(refs=Refs(source="E1"))
        surface = _build(
            _record(state=state), context_dropout=1.0, rng=random.Random(1),
        )
        acting = [s for s in slots_of_kind(surface, SlotKind.ABILITY) if s.entity_id == "E1"]
        assert len(acting) == 1

    def test_dropout_is_reproducible_under_a_seeded_rng(self):
        record = _record()
        first = _build(record, context_dropout=0.5, rng=random.Random(3))
        second = _build(record, context_dropout=0.5, rng=random.Random(3))
        assert len(first.slots) == len(second.slots)


class TestKeysWithNoLine:
    """A key the sidecar maps to no line is not an ability (FR-073).

    Forge attaches an implicit cast-this-permanent object to every permanent,
    and the converter renders no line for it. Without the predicate the
    surface carried a zero-vector token for it on almost every entity.
    """

    _PHANTOM = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0)

    def test_a_key_with_no_line_gets_no_ability_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of(printed=(self._PHANTOM, _ANTHEM))),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        abilities = slots_of_kind(surface, SlotKind.ABILITY)
        assert [slot.e_vector for slot in abilities] == [_VECTORS[_ANTHEM]]

    def test_positions_stay_contiguous_after_a_skip(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of(printed=(self._PHANTOM, _ANTHEM, _WARD))),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        positions = [
            slot.position for slot in surface.slots
            if slot.kind in (SlotKind.CARD, SlotKind.ABILITY)
        ]
        assert positions == [0, 1, 2]

    def test_a_masked_line_keeps_its_zero_token(self):
        """The state-only control zeroes ``e`` but keeps the geometry."""
        record = _record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of(printed=(_ANTHEM,))),)))
        surface = _build(record, e_for=lambda key: None, has_line=lambda key: True)
        assert [slot.e_vector for slot in slots_of_kind(surface, SlotKind.ABILITY)] == [_ZERO]

    def test_without_the_predicate_every_key_keeps_its_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of(printed=(self._PHANTOM,))),
        )))
        abilities = slots_of_kind(_build(record), SlotKind.ABILITY)
        assert [slot.e_vector for slot in abilities] == [_ZERO]


class TestOptionRows:
    """FR-063a: a charm's mode rows follow its root row inside the card's block."""

    _CHARM = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0)

    def _modes(self):
        return tuple(
            ProvenanceKey(self._CHARM.script_file, 0, "spell", 0, option)
            for option in range(3)
        )

    def _surface(self, options_for):
        vectors = {self._CHARM: (0.5,) * E_DIM}
        vectors |= {mode: (float(mode.option),) * E_DIM for mode in self._modes()}
        record = _record(state=_snapshot(entities=(
            _entity("E1", lines=lines_of(printed=(self._CHARM, _ANTHEM))),
        )))
        return _build(
            record, e_for=lambda key: vectors.get(key, _VECTORS.get(key)),
            options_for=options_for,
        )

    def test_a_charm_gets_its_root_then_its_modes_in_choices_order(self):
        modes = self._modes()
        surface = self._surface(
            lambda key: modes if key == self._CHARM else (),
        )
        abilities = slots_of_kind(surface, SlotKind.ABILITY)
        assert [slot.option for slot in abilities] == [False, True, True, True, False]
        assert [slot.e_vector[0] for slot in abilities[1:4]] == [0.0, 1.0, 2.0]
        assert [slot.position for slot in abilities] == [1, 2, 3, 4, 5]

    def test_no_options_callback_adds_no_rows(self):
        abilities = slots_of_kind(self._surface(None), SlotKind.ABILITY)
        assert [slot.option for slot in abilities] == [False, False]

    def test_the_batcher_reads_the_modes_off_the_sidecar(self):
        import torch

        from effects.domain.baselines import BaselineMasks
        from effects.domain.provenance import ProvenanceSidecar, SidecarLine
        from tests.unit.effects.builders import make_batcher

        modes = self._modes()
        sidecar = ProvenanceSidecar(
            card="cryptic command", script_file=self._CHARM.script_file,
            lines=(
                SidecarLine(3, "spell", (self._CHARM,), script_text="Choices$ SV1,SV2,SV3",
                    prose_offset=0),
                *(SidecarLine(4 + m.option, "option", (m,), script_text=f"SV1: DB$ M{m.option}",
                    prose_offset=0)
                  for m in modes),
                SidecarLine(7, "option", (), script_text="", prose_offset=0),
            ),
        )

        class _Sidecars:
            def get(self, script_file):
                return sidecar

            def line_for(self, key):
                return sidecar.line_for(key)

            def prose_for(self, key):
                return None

        batcher = make_batcher(
            tokenizer=None, sidecars=_Sidecars(), masks=BaselineMasks(),
            surface="script", e_dim=E_DIM, widths={}, device=torch.device("cpu"),
        )
        assert batcher.options_for(self._CHARM) == modes
        assert batcher.options_for(modes[0]) == ()


class TestAKeywordOfSeveralLines:
    """A board line naming a keyword's bare key, where the keyword prints
    several lines (a saga's chapters), is one slot per line, in line order:
    the real gen-3 record of Braids's Frightful Return on the battlefield, against
    the current converter's sidecar."""

    _SAGA = "cardsfolder/b/braidss_frightful_return.txt"

    @pytest.fixture(scope="class")
    def record(self):
        import itertools

        from effects.infrastructure.shard_io import read_shard
        from tests.unit.effects.builders import GEN3_RECORDS

        return next(itertools.islice(read_shard(GEN3_RECORDS), 276, None))

    @pytest.fixture(scope="class")
    def sidecars(self):
        from effects.infrastructure.sidecar_io import SidecarCache
        from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS

        return SidecarCache({folder.name: folder for folder in GEN3_SIDECAR_FOLDERS})

    def _chapter(self, index: int) -> ProvenanceKey:
        from effects.domain.provenance import BuiltTrait

        return ProvenanceKey(self._SAGA, 0, "keyword", 0, built=BuiltTrait("trigger", index))

    def test_the_surface_gives_each_chapter_line_its_slot(self, record, sidecars):
        saga = next(e for e in record.state.entities if e.name == "Braids's Frightful Return")
        vectors = {self._chapter(n): (float(n + 1),) * E_DIM for n in range(3)}
        surface = SurfaceBuilder(
            _Lines(e_for=vectors.get,
                   board_keys=lambda key: sidecars.get(key.script_file).board_keys(key)),
            e_dim=E_DIM, rng=random.Random(0),
        ).build(record)
        abilities = [slot for slot in slots_of_kind(surface, SlotKind.ABILITY)
                     if slot.entity_id == saga.id]
        assert [slot.e_vector[0] for slot in abilities] == [1.0, 2.0, 3.0, 0.0]

    def test_the_batcher_stages_every_chapter_s_text(self, record, sidecars):
        import torch

        from effects.domain.baselines import BaselineMasks
        from tests.unit.effects.builders import make_batcher

        batcher = make_batcher(
            tokenizer=None, sidecars=sidecars, masks=BaselineMasks(),
            surface="script", e_dim=E_DIM, widths={}, device=torch.device("cpu"),
        )
        chapters = [line.script_text for line in sidecars.get(self._SAGA).lines
                    if line.line_kind == "chapter"]
        staged = batcher.batch_texts([record])
        assert len(chapters) == 3
        assert all(any(text.startswith(chapter.split(" |")[0]) for text in staged)
                   for chapter in chapters)
