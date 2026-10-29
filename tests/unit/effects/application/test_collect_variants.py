"""Synthetic script variants (T138).

Four properties: a variant record carries ``synthetic`` and ``variant_of``, no
variant is generated from a held-out card, variants contribute no pairing loss
(``domain/test_ability_encoder.py``), and no variant is ever converted to prose.

The held-out rule is the one that would be silently wrong. Perturbing a
parameter changes the text, so a variant of a held-out card carries a text that
is *not* held out and would reach training — teaching that card's mechanics
under an edit the split cannot see. Skipping generation is the only point where
the rule can be enforced, because by collection time the text no longer matches
anything in the holdout.
"""

from __future__ import annotations

import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from effects.application.collect_variants import (
    RANDOM_SEED,
    CollectVariantsConfig,
    GeneratedVariant,
    _deck_candidate,
    generate_variants,
    loadable_variants,
    perturb_script,
    variant_budget,
    volume_ceiling,
)
from effects.domain.records import EffectRecord
from effects.domain.script_variants import (
    is_multi_face,
    variant_name,
)
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from tests.unit.effects.builders import make_record
from tests.unit.effects.domain.test_script_variants import _BOLT_SCRIPT


def _record(**overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return make_record(**defaults)


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

    def test_a_larger_volume_allows_more(self):
        assert variant_budget(1000, 0.5) > variant_budget(1000, 0.2)

    def test_the_volume_is_measured_against_the_corpus_not_the_destination(self):
        """Writing variants to their own directory must not zero the budget.

        `--variant-volume` scales the cap with the corpus that already exists.
        When variants are written into that same tree the destination *is* the
        corpus, but a run that keeps them in `records/variants/` starts with an
        empty destination — and measuring there caps the run at zero records
        against a corpus of millions.
        """
        config = CollectVariantsConfig(
            effect_records=Path("output/effects/records/variants"),
            corpus_records=Path("output/effects/records"),
        )
        assert config.volume_source() == Path("output/effects/records")

    def test_the_volume_source_defaults_to_the_destination(self):
        """Unset, the corpus and the destination are the same directory."""
        config = CollectVariantsConfig(
            effect_records=Path("output/effects/records"),
        )
        assert config.volume_source() == Path("output/effects/records")

    def test_the_record_count_stops_once_the_cap_cannot_bind(self):
        """Counting the corpus exactly is minutes of gzip for an inert number.

        The budget caps how many variant *scripts* are generated, and there are
        only ever as many candidates as source scripts. Past
        `source_scripts / volume` records the cap cannot bind, so the count
        stops there rather than gunzipping the rest of the corpus.
        """
        assert volume_ceiling(33_700, 0.2) == 168_501
        assert volume_ceiling(33_700, 0.0) == 0

    def test_a_ceiling_stops_the_count_early(self, tmp_path):
        from effects.infrastructure.record_io import count_records, write_shard

        for index in range(4):
            write_shard(
                tmp_path / f"run.0-{index}.jsonl.gz",
                [_record(record_id=f"r.{index}.{n}") for n in range(10)],
            )
        assert count_records(tmp_path) == 40
        assert count_records(tmp_path, ceiling=15) == 15

    def test_the_seed_is_settable_so_a_second_run_draws_differently(self):
        """A repeat run is the only way to top up, and it needs a new draw.

        `collect-coverage` re-weights each round by how far a card is from its
        target, so its rounds make progress whatever the seed. A variant round
        weights every variant equally and plays once, so with the seed pinned a
        second invocation regenerates the same perturbations, builds the same
        decks, and collects nothing.
        """
        from effects.infrastructure.cli import build_parser

        assert CollectVariantsConfig().seed == RANDOM_SEED
        assert CollectVariantsConfig(seed=7).seed == 7
        assert build_parser().parse_args(["collect-variants"]).seed == RANDOM_SEED
        assert build_parser().parse_args(
            ["collect-variants", "--seed", "7"],
        ).seed == 7

    def test_two_seeds_perturb_the_same_script_differently(self, tmp_path):
        source = tmp_path / "cards"
        (source / "b").mkdir(parents=True)
        (source / "b" / "bolt.txt").write_text(
            "Name:Bolt\nManaCost:R\nTypes:Instant\n"
            "A:SP$ DealDamage | Cost$ R | NumDmg$ 3 | ValidTgts$ Any\n"
            "Oracle:Bolt deals 3 damage to any target.\n",
            encoding="utf-8",
        )
        drawn = {
            seed: [
                v.perturbation for v in generate_variants(
                    source, tmp_path / f"out{seed}",
                    held_out=frozenset(), limit=50, seed=seed,
                )
            ]
            for seed in range(1, 21)
        }
        assert len({tuple(v) for v in drawn.values()}) > 1


class TestRunClosesTheSupervisor:
    """The same property as ``collect_coverage.run``'s (see
    ``TestRunClosesTheSupervisor`` in ``test_collect_coverage.py`` for why): a run
    that builds a ``CollectorSupervisor`` and plays a round through it must
    call ``.stop()``, or the worker log handles opened for the latched
    effect-record failure reporters stay open until the interpreter exits on
    its own.
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
        reads it (``_deck_candidate``) to build the round's decks.
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
        ``effects.infrastructure.one_shot_workers`` where the function
        lives -- ``run()`` now resolves the name at module scope (so
        ``TestTheVariantRoundPlaysTheVariants`` can patch it there too), and
        patching the source module no longer reaches a name ``run()`` never
        looks up again.
        """
        from unittest.mock import MagicMock

        import effects.application.collect_variants as collect_variants
        import effects.infrastructure.record_io as record_io

        # The volume source's size alone: the growth check counts the shards
        # the round adds, which a test's supervisor writes itself.
        monkeypatch.setattr(record_io, "count_records", lambda path, **_: 1000)
        monkeypatch.setattr(
            collect_variants, "generate_variants",
            lambda *a, **k: [variant],
        )
        # Stubbed like the sidecar writer beside it: the real one reads the
        # list VariantSidecarMain writes from a staged Forge database, which
        # needs a JVM. Its own behaviour is covered by
        # TestOnlyLoadableVariantsAreDecked.
        monkeypatch.setattr(
            collect_variants, "loadable_variants", lambda generated, _root: generated,
        )
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=0),
        )
        return collect_variants

    def test_stop_runs_after_an_ordinary_round(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        import effects.infrastructure.collector_connector as collector_connector

        variant = self._variant(tmp_path)
        collect_variants = self._patch_prerequisites(monkeypatch, variant)
        supervisor = MagicMock()
        supervisor.interrupted = False
        supervisor.play_round.side_effect = _writes_a_shard(tmp_path / "records", 100)
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
             patch.object(collect_variants, "write_variant_sidecars",
                          return_value=0), \
             patch.object(collect_variants, "loadable_variants",
                          side_effect=lambda generated, _root: generated):
            # The loadable list is written by the sidecar writer, which is
            # mocked here, so the file it would leave behind does not exist.
            # TestOnlyLoadableVariantsAreDecked covers the filter itself.
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


class TestVariantManaCostIsParseable:
    """A variant's cost once reached the manabase as the perturbed *script's* raw
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

        from effects.application.collector_common import build_coverage_decks

        decks = build_coverage_decks(
            {variant.name: 1.0}, {variant.name: _deck_candidate(variant)}, 1,
            rng=random.Random(0),
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
        lands = compute_basic_lands([_deck_candidate(variant)] * 23)
        assert sum(lands.values()) == 17
        assert set(lands) == {"Plains", "Island", "Swamp", "Mountain", "Forest"}


class TestTheVariantCandidateMatchesAConvertedCard:
    def test_a_variant_hands_the_manabase_what_a_converted_card_would(
        self, tmp_path,
    ):
        """The same cost reaches ``compute_basic_lands`` from either source, so
        a variant gets the basics a converted card of that cost gets."""
        from effects.domain.coverage import DeckCandidate

        path = tmp_path / "bolt_variant.txt"
        path.write_text(
            "Name:Bolt Variant 0\nManaCost:2 R R\nTypes:Instant\n",
            encoding="utf-8",
        )
        variant = GeneratedVariant(
            name="Bolt Variant 0", source_card="Bolt", path=path,
            perturbation="numeric",
        )

        assert _deck_candidate(variant) == DeckCandidate.from_converted_text(
            "Bolt Variant 0", "name: bolt variant 0\nmana cost: {2}{R}{R}\n",
        )

    def test_a_script_without_a_cost_names_none(self, tmp_path):
        path = tmp_path / "x.txt"
        path.write_text("Name:X Variant 0\nTypes:Instant\n", encoding="utf-8")
        variant = GeneratedVariant(
            name="X Variant 0", source_card="X", path=path, perturbation="numeric",
        )

        assert _deck_candidate(variant).mana_cost is None


class TestTheRunReportsWhatItCollected:
    """``run`` used to return 0 with no evidence that a
    single variant record had been written.

    It counted the corpus once, before play, purely to size the budget, and
    never counted it again -- so a round in which every deck materialized as
    17 basics (the whole-run case if the staged custom cards do not land in
    Forge's card database under the exact names the decks file spells) exited
    0 having collected nothing, and nothing said so.
    """

    def _fixture(self, tmp_path, monkeypatch, source_records, added):
        """A run whose only unknowns are the corpus sizes.

        ``source_records`` is what ``count_records`` reports for the *volume
        source* the budget is taken from, which can name a different directory
        from the destination -- that is what ``--corpus-records`` is for -- so
        the growth check must never subtract it. ``added`` is how many records
        the round writes, in a shard of its own beside one already in the
        destination, which must not count as growth either.
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
        monkeypatch.setattr(
            record_io, "count_records", lambda directory, **_: source_records,
        )
        monkeypatch.setattr(
            collect_variants, "generate_variants", lambda *a, **k: [variant],
        )
        # As above: the real filter needs a Forge database staged with the
        # variants. TestOnlyLoadableVariantsAreDecked covers it directly.
        monkeypatch.setattr(
            collect_variants, "loadable_variants", lambda generated, _root: generated,
        )
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=0),
        )
        records = tmp_path / "records"
        _writes_a_shard(records, 7, name="earlier.0-a.jsonl")()
        supervisor = MagicMock(spec=collector_connector.CollectorSupervisor)
        supervisor.interrupted = False
        if added:
            supervisor.play_round.side_effect = _writes_a_shard(records, added)
        monkeypatch.setattr(
            collector_connector, "CollectorSupervisor",
            MagicMock(return_value=supervisor),
        )
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        config = CollectVariantsConfig(
            effect_records=records,
            forge_cards_path=forge_cards,
            variant_scripts=tmp_path / "variant-scripts",
        )
        return collect_variants, config, supervisor

    def test_a_round_that_collected_nothing_fails_loudly(
        self, tmp_path, monkeypatch, caplog,
    ):
        import logging

        collect_variants, config, supervisor = self._fixture(
            tmp_path, monkeypatch, 1000, added=0,
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
            tmp_path, monkeypatch, 1000, added=42,
        )

        with caplog.at_level(logging.INFO):
            code = collect_variants.run(config)

        assert code == 0
        assert any(
            "Collected 42 new effect records" in record.getMessage()
            for record in caplog.records
        ), "the run never reported how many records it added"

    def test_a_corpus_counted_elsewhere_never_enters_the_growth_figure(
        self, tmp_path, monkeypatch, caplog,
    ):
        """The reported figure was "Interrupted after -39332737 new records".

        ``--corpus-records`` sizes the budget against a corpus the variants are
        not written into. Subtracting that count from a count of the
        destination reports a run against a 39M-record corpus writing to an
        empty directory as having collected minus thirty-nine million.
        """
        import logging

        collect_variants, config, _ = self._fixture(
            tmp_path, monkeypatch, 39_000_000, added=42,
        )

        with caplog.at_level(logging.INFO):
            code = collect_variants.run(config)

        assert code == 0
        assert any("42" in r.getMessage() for r in caplog.records)
        assert not any("-38" in r.getMessage() for r in caplog.records), (
            "the growth figure was taken against the wrong directory"
        )

    def test_an_interrupted_round_does_not_report_success(
        self, tmp_path, monkeypatch,
    ):
        """Ctrl-C on the variant side: ``ForgeWorkerPool``
        swallows SIGINT, so a round the operator stopped returns exactly like
        one that finished."""
        collect_variants, config, supervisor = self._fixture(
            tmp_path, monkeypatch, 1000, added=42,
        )
        supervisor.interrupted = True

        assert collect_variants.run(config) == 130


def _writes_a_shard(directory: Path, records: int, *, name: str = "round.0-b.jsonl"):
    """A stand-in for a round's workers: one new shard of ``records`` lines."""

    def write(*_args, **_kwargs) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("{}\n" * records, encoding="utf-8")

    return write


class TestOnlyLoadableVariantsAreDecked:
    """A perturbation can produce a script Forge declines to load.

    The script is still written and still gets a sidecar, so nothing before
    the card database can tell it apart. A decks-only round refuses a deck
    naming a card Forge does not know rather than playing basics, so an
    unfiltered deck build is not a rounding error: at roughly one variant in
    nine unloadable, almost every deck of 23 nonlands holds at least one. A
    real 4,000-deck round lost 350 of its 354 workers to exactly that.
    """

    def _generated(self, tmp_path, names):
        return [
            GeneratedVariant(name=name, source_card="x",
                             path=tmp_path / "x.txt", perturbation="numeric")
            for name in names
        ]

    def test_a_variant_forge_rejected_is_not_decked(self, tmp_path):
        (tmp_path / "loadable.txt").write_text(
            "Bolt Variant 0" + chr(10) + "Shock Variant 2" + chr(10),
            encoding="utf-8",
        )
        kept = loadable_variants(
            self._generated(
                tmp_path,
                ["Bolt Variant 0", "Broken Variant 1", "Shock Variant 2"],
            ),
            tmp_path,
        )
        assert [v.name for v in kept] == ["Bolt Variant 0", "Shock Variant 2"]

    def test_a_missing_list_is_refused_rather_than_ignored(self, tmp_path):
        """No list is no evidence, and decking anyway is the defect itself."""
        with pytest.raises(FileNotFoundError, match="loadable"):
            loadable_variants(self._generated(tmp_path, ["Bolt Variant 0"]),
                              tmp_path)


class TestMultiFaceCardsAreLeftAlone:
    """Renaming one face of a two-faced card breaks the whole card database.

    A variant takes its own name by rewriting the script's first `Name:`, which
    on a two-faced card renames the front and leaves the back answering to the
    original. Forge does not skip the resulting card: `StaticData` throws while
    constructing the database, so every Forge startup that sees the script
    dies. One run staged such a script and every subsequent JVM -- variant run
    or not -- failed with `mainPart is null` until the directory was cleared.
    """

    def test_an_alternate_section_marks_a_multi_face_script(self):
        assert is_multi_face(["Name:Front", "Types:Creature", "ALTERNATE"])
        assert not is_multi_face(["Name:Bolt", "Types:Instant"])

    def test_no_variant_is_generated_from_a_multi_face_card(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        (source / "dfc.txt").write_text(
            "Name:Front Face" + chr(10) + "ManaCost:R" + chr(10)
            + "Types:Instant" + chr(10)
            + "A:SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any" + chr(10)
            + "ALTERNATE" + chr(10) + "Name:Back Face" + chr(10)
            + "Types:Instant" + chr(10),
            encoding="utf-8",
        )
        assert generate_variants(
            source, tmp_path / "variants", held_out=frozenset(), limit=5,
        ) == []


class TestNoVariantPrintsAHeldOutText:
    """A perturbation can land on a held-out text, or under template on its template.

    Its source card is not held out, so the name test passes it; nothing else
    would keep a held-out text's mechanics out of training (G14).
    """

    _HELD = "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 2"

    def _variant(self, tmp_path, name: str, text: str) -> GeneratedVariant:
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.infrastructure.sidecar_io import sidecar_path_for
        from tests.unit.effects.builders import write_sidecar

        path = tmp_path / f"{name}.txt"
        path.write_text(f"Name:{name}\n", encoding="utf-8")
        key = ProvenanceKey(f"variant-scripts/{name}.txt", 0, "spell", 0)
        write_sidecar(
            ProvenanceSidecar(
                card=name, script_file=f"variant-scripts/{name}.txt",
                lines=(SidecarLine(
                    line_index=0, line_kind="spell", provenance=(key,), script_text=text,
                ),),
            ),
            sidecar_path_for(path),
        )
        return GeneratedVariant(name=name, source_card="x", path=path, perturbation="numeric")

    def _holdout(self, unit: HoldoutUnit):
        from effects.domain.split import text_keyed_holdout

        return text_keyed_holdout(
            {"Held": "cardsfolder/h/held.txt"}, {"Held": [self._HELD]},
            HoldoutRule(1000, 8, unit),
        )

    def test_a_variant_printing_a_held_out_template_is_dropped_with_its_files(self, tmp_path):
        from effects.application.collect_variants import drop_held_out_variants
        from effects.infrastructure.sidecar_io import sidecar_path_for

        held = self._variant(
            tmp_path, "act_of_treason_variant_1",
            "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        )
        clean = self._variant(tmp_path, "shock_variant_2", "SP$ Draw | NumCards$ 2")

        kept = drop_held_out_variants([held, clean], self._holdout(HoldoutUnit.TEMPLATE))

        assert kept == [clean]
        assert not held.path.exists()
        assert not sidecar_path_for(held.path).exists()

    def test_under_the_text_unit_only_the_text_itself_is_held_out(self, tmp_path):
        from effects.application.collect_variants import drop_held_out_variants

        near = self._variant(
            tmp_path, "near_variant_1", "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        )
        same = self._variant(tmp_path, "same_variant_2", self._HELD)

        assert drop_held_out_variants([near, same], self._holdout(HoldoutUnit.TEXT)) == [near]

    def test_the_rule_the_collector_is_given_decides_what_is_dropped(self, tmp_path):
        """A6: under ``build-corpus --holdout-permille 1000`` this text is held
        out, under the default it is not; a collector applying the default
        would deck it into training games' boards."""
        from effects.application.collect_variants import _keep_unheld
        from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
        from effects.domain.split import HOLDOUT_PERMILLE
        from effects.domain.text_holdout import masked_template, text_is_held_out
        from effects.infrastructure.sidecar_io import sidecar_path_for
        from tests.unit.effects.builders import write_sidecar

        assert not text_is_held_out(masked_template(self._HELD), permille=HOLDOUT_PERMILLE)
        card = tmp_path / "cardsfolder" / "h" / "held.txt"
        card.parent.mkdir(parents=True)
        card.write_text("name: Held\ntypes: instant\n", encoding="utf-8")
        write_sidecar(
            ProvenanceSidecar(
                card="Held", script_file="cardsfolder/h/held.txt",
                lines=(SidecarLine(
                    line_index=0, line_kind="spell", script_text=self._HELD,
                    provenance=(ProvenanceKey("cardsfolder/h/held.txt", 0, "spell", 0),),
                ),),
            ),
            sidecar_path_for(card),
        )
        variants = tmp_path / "variant-scripts"
        variants.mkdir()

        def config(**rule) -> CollectVariantsConfig:
            return CollectVariantsConfig(
                cards_folders=(tmp_path / "cardsfolder",), holdout=HoldoutRule(**rule),
            )

        printed = self._variant(
            variants, "held_variant_1", "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        )
        clean = self._variant(variants, "draw_variant_2", "SP$ Draw | NumCards$ 2")
        assert _keep_unheld(config(), [printed, clean]) == [printed, clean]
        assert _keep_unheld(config(permille=1000), [printed, clean]) == [clean]
