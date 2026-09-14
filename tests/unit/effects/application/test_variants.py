"""Synthetic script variants (T138).

Four properties: a variant record carries ``synthetic`` and ``variant_of``, no
variant is generated from a held-out card, variants contribute no pairing loss,
and no variant is ever converted to prose.

The held-out rule is the one that would be silently wrong. Perturbing a
parameter changes the text, so a variant of a held-out card carries a text that
is *not* held out and would reach training — teaching that card's mechanics
under an edit the split cannot see. Skipping generation is the only point where
the rule can be enforced, because by collection time the text no longer matches
anything in the holdout.
"""

from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

import pytest

from effects.application.collect_variants import (
    DEFAULT_VARIANT_VOLUME,
    CollectVariantsConfig,
    GeneratedVariant,
    _deck_text,
    generate_variants,
    perturb_script,
    variant_budget,
)
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.script_variants import (
    MAX_NUMERIC_SHIFT,
    NUMERIC_PARAM_KEYS,
    SELECTOR_WHITELIST,
    PerturbationKind,
    double,
    numeric_params,
    perturb,
    selector_params,
    shift,
    variant_name,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)

_BOLT_SCRIPT = (
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target."
)


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "r.0.1", "run_id": "r", "timestamp": "t", "game_id": "r.0.1",
        "kind": RecordKind.RESOLUTION, "moment": Moment.RESOLUTION,
        "actor_player": "P0", "state": _SNAPSHOT,
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestVariantRecordFlags:
    def test_a_variant_record_is_synthetic_and_names_its_source(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        assert record.synthetic
        assert record.variant_of == "Lightning Bolt"

    def test_variant_of_requires_the_synthetic_flag(self):
        with pytest.raises(ValueError, match="variant_of"):
            _record(variant_of="Lightning Bolt")

    def test_the_synthetic_flag_never_reaches_the_model(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        fields = record.model_input_fields()
        assert "synthetic" not in fields
        assert "variant_of" not in fields

    def test_a_variant_is_otherwise_an_ordinary_record(self):
        record = _record(synthetic=True, variant_of="Lightning Bolt")
        assert record.kind is RecordKind.RESOLUTION
        assert set(record.model_input_fields()) == {
            "kind", "moment", "subkind", "actor_player", "ability", "state",
            "payload",
        }


class TestNumericPerturbation:
    def test_a_shift_moves_the_value_by_at_most_three(self):
        rng = random.Random(0)
        for _ in range(50):
            assert abs(shift(10, rng) - 10) <= MAX_NUMERIC_SHIFT

    def test_a_shift_is_floored_at_zero(self):
        """A negative count is not a smaller effect, it is a script Forge
        rejects — and a rejected script never reaches a game."""
        rng = random.Random(0)
        for _ in range(50):
            assert shift(1, rng) >= 0

    def test_doubling_keeps_the_card_recognisably_the_same(self):
        assert double(3) == 6
        assert double(0) == 0

    def test_the_numeric_keys_are_a_whitelist(self):
        """A shifted CounterType would produce a script that fails to load
        rather than a card that plays differently."""
        assert "NumDmg" in NUMERIC_PARAM_KEYS
        assert "CounterType" not in NUMERIC_PARAM_KEYS
        assert "ValidTgts" not in NUMERIC_PARAM_KEYS

    def test_numeric_params_reads_only_whitelisted_numbers(self):
        found = numeric_params(_BOLT_SCRIPT)
        assert found == {"NumDmg": 3}

    def test_a_perturbed_script_differs_in_exactly_one_parameter(self):
        rng = random.Random(1)
        result = perturb(_BOLT_SCRIPT, rng)
        assert result is not None
        perturbed, perturbation = result
        assert perturbed != _BOLT_SCRIPT
        assert perturbation.key == "NumDmg"
        # Every other parameter survives untouched.
        assert "ValidTgts$ Any" in perturbed
        assert "Cost$ R" in perturbed

    def test_a_line_with_nothing_to_perturb_is_skipped(self):
        assert perturb("Name:Lightning Bolt", random.Random(0)) is None
        assert perturb("Types:Instant", random.Random(0)) is None


class TestSelectorPerturbation:
    def test_a_whitelisted_selector_is_swappable(self):
        script = "Mode$ Continuous | Affected$ Creature | AddPower$ 1"
        assert selector_params(script) == {"Affected": "Creature"}

    def test_a_swap_produces_a_selector_forge_recognises(self):
        script = "Mode$ Continuous | Affected$ Creature | AddPower$ 1"
        rng = random.Random(7)
        for _ in range(20):
            result = perturb(script, rng)
            if result is None:
                continue
            perturbed, perturbation = result
            if perturbation.kind == PerturbationKind.SELECTOR:
                assert perturbation.replacement in SELECTOR_WHITELIST
                assert perturbation.replacement != perturbation.original
                assert perturbation.replacement in perturbed

    def test_the_whitelist_holds_only_composable_restrictions(self):
        for selector in SELECTOR_WHITELIST:
            assert " " not in selector
            assert selector[0].isupper()


class TestVariantGeneration:
    def _write_source(self, folder, stem, name, script):
        (folder / f"{stem}.txt").write_text(
            f"Name:{name}\nManaCost:R\nTypes:Instant\n{script}\n"
            f"Oracle:{name} deals 3 damage.\n",
            encoding="utf-8",
        )

    def test_a_variant_is_written_with_its_own_name(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=5,
        )
        assert variants
        assert variants[0].source_card == "Lightning Bolt"
        assert "Lightning Bolt" in variants[0].name
        assert variants[0].name != "Lightning Bolt"
        assert f"Name:{variants[0].name}" in variants[0].path.read_text(
            encoding="utf-8"
        )

    def test_no_variant_is_generated_from_a_held_out_card(self, tmp_path):
        """A perturbed text is not itself held out, so the variant would
        reach training and teach the held-out card's mechanics anyway."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants",
            held_out=frozenset({"Lightning Bolt"}), limit=5,
        )
        assert variants == []

    def test_a_variant_is_never_converted_to_prose(self, tmp_path):
        """There is no oracle text for a card nobody printed."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        output = tmp_path / "variants"
        generate_variants(source, output, held_out=frozenset(), limit=5)
        # The variant tree holds Forge source scripts. The sidecar beside them
        # is written by the Java parser, not here.
        for path in output.glob("*.txt"):
            assert path.read_text(encoding="utf-8").startswith("Name:")

    def test_a_variant_file_is_named_for_the_key_that_resolves_it(
        self, tmp_path,
    ):
        """The runtime keys a variant as ``variant-scripts/{sanitized name}.txt``
        and the reader looks for a sidecar at exactly that path, so a filename
        derived from anything else makes every variant record unjoinable."""
        from price_predictor.infrastructure.card_filenames import (
            sanitize_card_name,
        )

        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=1,
        )
        assert variants
        assert variants[0].path.name == (
            f"{sanitize_card_name(variants[0].name)}.txt"
        )

    def test_a_variant_name_survives_sanitizing_without_punctuation(self):
        """A parenthesised suffix reads as a set code on a Forge deck-list line
        and survives both filename sanitizers verbatim."""
        from price_predictor.infrastructure.card_filenames import (
            sanitize_card_name,
        )

        stem = sanitize_card_name(variant_name("Lightning Bolt", 0))
        assert stem == "lightning_bolt_variant_0"
        assert "(" not in stem and ")" not in stem

    def test_the_limit_is_respected(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(10):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=3,
        )
        assert len(variants) == 3

    def test_generation_is_reproducible_under_a_seed(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(5):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        first = generate_variants(
            source, tmp_path / "a", held_out=frozenset(), limit=3, seed=5,
        )
        second = generate_variants(
            source, tmp_path / "b", held_out=frozenset(), limit=3, seed=5,
        )
        assert [v.name for v in first] == [v.name for v in second]

    def test_a_card_with_no_perturbable_line_is_skipped(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        (source / "bear.txt").write_text(
            "Name:Grizzly Bears\nManaCost:1 G\nTypes:Creature Bear\nPT:2/2\n",
            encoding="utf-8",
        )
        assert generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=5,
        ) == []

    def test_perturb_script_finds_the_first_perturbable_line(self):
        lines = ["Name:Bolt", "Types:Instant", _BOLT_SCRIPT]
        result = perturb_script(lines, random.Random(3))
        assert result is not None
        perturbed, _description = result
        assert perturbed[0] == "Name:Bolt"
        assert perturbed[2] != _BOLT_SCRIPT


class TestVolumeCap:
    def test_the_budget_is_a_fraction_of_the_real_corpus(self):
        """Expressed against the corpus so it scales with it, rather than
        needing a new absolute every time the corpus grows."""
        assert variant_budget(1000, 0.2) == 200
        assert variant_budget(0, 0.2) == 0

    def test_the_default_volume_is_the_contract(self):
        assert DEFAULT_VARIANT_VOLUME == 0.2
        assert CollectVariantsConfig().variant_volume == 0.2

    def test_a_larger_volume_allows_more(self):
        assert variant_budget(1000, 0.5) > variant_budget(1000, 0.2)


class TestPairingLoss:
    def test_a_variant_has_only_the_script_surface(self):
        """So it contributes no pairing term: there is no prose to pair with."""
        from effects.domain.ability_encoder import (
            SURFACE_PROSE,
            SURFACE_SCRIPT,
            encoding_text,
        )

        class _Line:
            script_text = "NumDmg$ 7"

        line = _Line()
        assert encoding_text(line, None, SURFACE_SCRIPT) == "NumDmg$ 7"
        # No prose exists, so the prose surface falls back to the script rather
        # than pairing two different readings of one line.
        assert encoding_text(line, None, SURFACE_PROSE) == "NumDmg$ 7"

    def test_a_real_card_has_both_surfaces_and_can_pair(self):
        from effects.domain.ability_encoder import (
            SURFACE_PROSE,
            SURFACE_SCRIPT,
            encoding_text,
        )

        class _Line:
            script_text = "NumDmg$ 3"

        line = _Line()
        prose = "deal 3 damage to any target"
        assert encoding_text(line, prose, SURFACE_SCRIPT) == "NumDmg$ 3"
        assert encoding_text(line, prose, SURFACE_PROSE) == prose


def test_variant_name_derives_from_its_source():
    assert variant_name("Lightning Bolt", 0) == "Lightning Bolt Variant 0"
    assert variant_name("Lightning Bolt", 1) != variant_name("Lightning Bolt", 0)


class TestRunClosesTheSupervisor:
    """final-fix-3.md item 5: same finding, same fix shape as
    ``collect_coverage.run`` (see ``TestRunClosesTheSupervisor`` in
    ``test_coverage.py`` for the full rationale) -- ``run()`` built a
    ``CollectorSupervisor`` and played a round through it, but never called
    ``.stop()``, so the worker log handles opened for F4's latched
    effect-record failure reporters stayed open until the interpreter exited
    on its own.
    """

    def _config(self, tmp_path, **overrides):
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        variant_scripts = tmp_path / "variant-scripts"
        return CollectVariantsConfig(
            effect_records=tmp_path / "records",
            forge_cards_path=forge_cards,
            variant_scripts=variant_scripts,
            **overrides,
        )

    def _variant(self, tmp_path) -> GeneratedVariant:
        """A ``GeneratedVariant`` whose ``.path`` is a real file on disk.

        ``generate_variants`` is stubbed out below rather than exercised for
        real, so nothing else would write this file -- and ``run()`` now
        reads it (``_deck_text``) to build the round's decks.
        """
        path = tmp_path / "variant-scripts" / "lightning_bolt_variant_0.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Name:Lightning Bolt Variant 0\nManaCost:R\nTypes:Instant\n",
            encoding="utf-8",
        )
        return GeneratedVariant(
            name="Lightning Bolt Variant 0", source_card="Lightning Bolt",
            path=path, perturbation="numeric",
        )

    def _patch_prerequisites(self, monkeypatch, variant):
        """Everything ``run()`` needs before it ever reaches the supervisor,
        stood in for rather than exercised for real: the corpus size, the
        variant generator (which otherwise reads real Forge card scripts) and
        the sidecar writer (which otherwise needs a real JVM).

        The sidecar writer is patched on ``collect_variants`` itself, not on
        ``effects.infrastructure.variant_sidecar_connector`` where the class
        lives -- ``run()`` now resolves the name at module scope (so
        ``TestTheVariantRoundPlaysTheVariants`` can patch it there too), and
        patching the source module no longer reaches a name ``run()`` never
        looks up again.
        """
        from unittest.mock import MagicMock

        import effects.application.collect_variants as collect_variants
        import effects.infrastructure.record_io as record_io

        # Growing, not constant: `run` now compares the corpus before and
        # after the round and refuses to report success when it did not move.
        readings = iter([1000, 1100])
        monkeypatch.setattr(
            record_io, "count_records", lambda path: next(readings),
        )
        monkeypatch.setattr(
            collect_variants, "generate_variants",
            lambda *a, **k: [variant],
        )
        sidecar = MagicMock()
        sidecar.run.return_value = 0
        monkeypatch.setattr(
            collect_variants, "VariantSidecarConnector",
            MagicMock(return_value=sidecar),
        )
        return collect_variants

    def test_stop_runs_after_an_ordinary_round(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import effects.infrastructure.collector_connector as collector_connector

        variant = self._variant(tmp_path)
        collect_variants = self._patch_prerequisites(monkeypatch, variant)
        supervisor = MagicMock()
        supervisor.interrupted = False
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        code = collect_variants.run(self._config(tmp_path))

        assert code == 0
        supervisor.play_round.assert_called_once()
        supervisor.stop.assert_called_once()

    def test_stop_runs_even_if_the_round_raises(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import effects.infrastructure.collector_connector as collector_connector

        variant = self._variant(tmp_path)
        collect_variants = self._patch_prerequisites(monkeypatch, variant)
        supervisor = MagicMock()
        supervisor.play_round.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )

        with pytest.raises(RuntimeError, match="boom"):
            collect_variants.run(self._config(tmp_path))

        supervisor.stop.assert_called_once()


class TestTheHoldoutNameBoundary:
    """Forge source scripts carry printed case; the holdout carries converted.

    `generate_variants` reads `Name:` straight out of Forge's own script, so it
    sees `Soul Echo`, while the held-out list built from the converted tree says
    `soul echo`. Unfolded, the skip never fires and every held-out card gets a
    variant — putting its mechanics into training under a text the split cannot
    recognise, which is exactly what the skip exists to prevent.
    """

    def test_a_printed_case_script_is_skipped_by_a_converted_case_holdout(
        self, tmp_path,
    ) -> None:
        source = tmp_path / "cards"
        source.mkdir()
        (source / "bolt.txt").write_text(
            "Name:Lightning Bolt\nManaCost:R\nTypes:Instant\n"
            "A:SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any\n",
            encoding="utf-8",
        )

        variants = generate_variants(
            source, tmp_path / "variants",
            held_out=frozenset({"lightning bolt"}), limit=5,
        )

        assert variants == []


class TestTheVariantRoundPlaysTheVariants:
    def test_the_scripts_are_staged_and_decked(self, tmp_path):
        from effects.application import collect_variants
        from effects.infrastructure import collector_connector

        source = tmp_path / "cards"
        source.mkdir()
        (source / "bolt.txt").write_text(
            "Name:Bolt\nManaCost:R\nTypes:Instant\n"
            "A:SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any\n",
            encoding="utf-8",
        )
        records = tmp_path / "records"
        records.mkdir()
        (records / "seed.jsonl").write_text("{}\n" * 100, encoding="utf-8")

        supervisor = MagicMock()
        with patch.object(collector_connector, "CollectorSupervisor",
                          return_value=supervisor) as ctor, \
             patch.object(collect_variants, "VariantSidecarConnector") as sidecar:
            sidecar.return_value.run.return_value = 0
            collect_variants.run(
                collect_variants.CollectVariantsConfig(
                    effect_records=records,
                    forge_cards_path=source,
                    variant_scripts=tmp_path / "variants",
                    decks_per_round=3,
                )
            )

        assert ctor.call_args.kwargs["variant_scripts"] == tmp_path / "variants"
        decks_file = supervisor.play_round.call_args.args[0]
        assert decks_file.read_text(encoding="utf-8").strip(), "no variant decks"
        assert supervisor.play_round.call_args.kwargs["matches"] == 3


class TestDeckTextManaCostIsParseable:
    """Fix round 1: `_deck_text` rendered the perturbed *script's* raw
    ``ManaCost:`` line verbatim -- Forge's own space-separated shards, e.g.
    ``"2 R R"``. `compute_basic_lands` reads its input through
    `convert_mana_cost`, which extracts mana symbols with a ``{...}`` regex
    and expects the *converted corpus's* brace-delimited shape (``"{2}{R}
    {R}"``). Fed the raw shards, that regex matches nothing,
    `convert_mana_cost` returns ``""``, and `ManaCost.parse` reads every
    variant as costless -- silently, every time, not just when the line is
    missing. The deck still gets built and still looks plausible (a file
    full of 40-card decks), which is exactly the failure class this whole
    plan exists to remove: a collector that appears to work while collecting
    nothing useful.
    """

    def test_a_mono_red_variant_gets_a_red_manabase(self, tmp_path):
        """23 copies of a mono-red "2 R R" variant (the only candidate in a
        one-card pool, so every nonland slot is it) should fill every basic
        with Mountains. Against the bug, `compute_basic_lands` sees zero
        pips for every color, falls back to its colorless case, and splits
        the 17 basics evenly across all five (3/4/3/4/3, confirmed against
        the unfixed code before this test was written)."""
        path = tmp_path / "bolt_variant.txt"
        path.write_text(
            "Name:Bolt Variant 0\nManaCost:2 R R\nTypes:Instant\n"
            "A:SP$ DealDamage | NumDmg$ 7 | ValidTgts$ Any\n",
            encoding="utf-8",
        )
        variant = GeneratedVariant(
            name="Bolt Variant 0", source_card="Bolt", path=path,
            perturbation="numeric",
        )

        from effects.application.collect_coverage import build_coverage_decks

        texts = {variant.name: _deck_text(variant)}
        decks = build_coverage_decks(
            {variant.name: 1.0}, texts, 1, rng=random.Random(0),
        )

        deck = decks[0]
        assert len(deck) == 40
        assert deck.count("Mountain") == 17, (
            f"expected all 17 basics to be Mountain, got {deck}"
        )
        for other in ("Plains", "Island", "Swamp", "Forest"):
            assert deck.count(other) == 0, f"unexpected {other} in {deck}"

    def test_a_land_variants_no_cost_still_parses_as_costless(self, tmp_path):
        """Forge's own literal spelling for a land's ``ManaCost:`` line is
        the two words ``no cost``, not a color shard. Brace-wrapping splits
        it into two "shards" (``{no}{cost}``) -- this pins that
        ``convert_mana_cost`` rejoins them back to the exact literal
        ``ManaCost.parse`` special-cases, rather than the fix turning a land
        into a card with bogus, unparseable mana symbols."""
        from effects.application.collect_coverage import _NonlandText
        from sealed.domain.manabase import compute_basic_lands

        path = tmp_path / "plains_variant.txt"
        path.write_text(
            "Name:Plains Variant 0\nManaCost:no cost\nTypes:Basic Land\n",
            encoding="utf-8",
        )
        variant = GeneratedVariant(
            name="Plains Variant 0", source_card="Plains", path=path,
            perturbation="selector",
        )

        # Zero pips either way -- the point is this must land in the same
        # all-zero-pips (colorless-fallback) case a missing ManaCost: line
        # does, not raise and not inject a spurious color.
        lands = compute_basic_lands([_NonlandText(_deck_text(variant))] * 23)
        assert sum(lands.values()) == 17
        assert set(lands) == {"Plains", "Island", "Swamp", "Mountain", "Forest"}


class TestTheRunReportsWhatItCollected:
    """Final review, IMPORTANT 5: ``run`` returned 0 with no evidence that a
    single variant record had been written.

    It counted the corpus once, before play, purely to size the budget, and
    never counted it again -- so a round in which every deck materialized as
    17 basics (the whole-run case if the staged custom cards do not land in
    Forge's card database under the exact names the decks file spells) exited
    0 having collected nothing, and nothing said so.
    """

    def _fixture(self, tmp_path, monkeypatch, counts):
        """A run whose only unknown is what ``count_records`` reports.

        ``counts`` is consumed one value per call: the first is the
        pre-flight corpus size the budget is taken from, the second is the
        size after the round.
        """
        from unittest.mock import MagicMock

        import effects.application.collect_variants as collect_variants
        import effects.infrastructure.collector_connector as collector_connector
        import effects.infrastructure.record_io as record_io

        path = tmp_path / "variant-scripts" / "bolt_variant_0.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "Name:Bolt Variant 0\nManaCost:R\nTypes:Instant\n", encoding="utf-8",
        )
        variant = GeneratedVariant(
            name="Bolt Variant 0", source_card="Bolt", path=path,
            perturbation="numeric",
        )
        readings = iter(counts)
        monkeypatch.setattr(
            record_io, "count_records", lambda directory: next(readings),
        )
        monkeypatch.setattr(
            collect_variants, "generate_variants", lambda *a, **k: [variant],
        )
        sidecar = MagicMock()
        sidecar.run.return_value = 0
        monkeypatch.setattr(
            collect_variants, "VariantSidecarConnector",
            MagicMock(return_value=sidecar),
        )
        supervisor = MagicMock(spec=collector_connector.CollectorSupervisor)
        supervisor.interrupted = False
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        config = CollectVariantsConfig(
            effect_records=tmp_path / "records",
            forge_cards_path=forge_cards,
            variant_scripts=tmp_path / "variant-scripts",
        )
        return collect_variants, config, supervisor

    def test_a_round_that_collected_nothing_fails_loudly(
        self, tmp_path, monkeypatch, caplog,
    ):
        import logging

        collect_variants, config, supervisor = self._fixture(
            tmp_path, monkeypatch, [1000, 1000],
        )

        with caplog.at_level(logging.ERROR):
            code = collect_variants.run(config)

        supervisor.play_round.assert_called_once()
        assert code != 0, (
            "a variant run that collected no record at all reported success"
        )
        assert any(
            "no new effect records" in record.getMessage().lower()
            for record in caplog.records
        ), "nothing in the log said the round collected nothing"

    def test_a_round_that_collected_records_says_how_many(
        self, tmp_path, monkeypatch, caplog,
    ):
        import logging

        collect_variants, config, _ = self._fixture(
            tmp_path, monkeypatch, [1000, 1042],
        )

        with caplog.at_level(logging.INFO):
            code = collect_variants.run(config)

        assert code == 0
        assert any(
            "42" in record.getMessage() for record in caplog.records
        ), "the run never reported how many records it added"

    def test_an_interrupted_round_does_not_report_success(
        self, tmp_path, monkeypatch,
    ):
        """Final review, IMPORTANT 3, on the variant side: ``ForgeWorkerPool``
        swallows SIGINT, so a round the operator stopped returns exactly like
        one that finished."""
        collect_variants, config, supervisor = self._fixture(
            tmp_path, monkeypatch, [1000, 1042],
        )
        supervisor.interrupted = True

        assert collect_variants.run(config) == 130
