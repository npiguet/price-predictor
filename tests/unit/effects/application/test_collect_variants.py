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
    PROTECTION_QUALITY_WHITELIST,
    CollectVariantsConfig,
    GeneratedVariant,
    generate_variants,
    loadable_variants,
    perturb_script,
    swap_protection_quality,
)
from effects.domain.records import EffectRecord
from effects.domain.rule_outcomes import ProtectionQuality, QualityKind
from effects.domain.script_variants import (
    is_multi_face,
    script_mana_cost,
    variant_name,
)
from effects.domain.split import HoldoutRule
from effects.domain.text_holdout import HoldoutUnit
from tests.unit.effects.builders import make_record, make_snapshot
from tests.unit.effects.domain.test_script_variants import _BOLT_SCRIPT


@pytest.fixture(autouse=True)
def _no_real_unstaging(monkeypatch):
    """A run ends by emptying Forge's real custom-cards directory; never here."""
    from effects.application import collect_variants

    monkeypatch.setattr(collect_variants, "unstage_variants", MagicMock(return_value=0))


def _record(**overrides) -> EffectRecord:
    """A gen-2 record unless ``generation=3`` is passed: the round shards
    here end no game with a marker, which only a gen-3 game owes."""
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1",
                      "generation": None}
    defaults.update(overrides)
    if defaults["generation"] is None:
        defaults.setdefault("state", make_snapshot(links=None))
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
            source, tmp_path / "variants", held_out=frozenset(),
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
            held_out=frozenset({"Lightning Bolt"}),
        )
        assert variants == []

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
            source, tmp_path / "variants", held_out=frozenset(),
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

        stem = sanitize_card_name(variant_name("Lightning Bolt", "0a1b2c3d"))
        assert stem == "lightning_bolt_variant_0a1b2c3d"
        assert "(" not in stem and ")" not in stem

    def test_a_variant_is_named_by_its_perturbed_script_s_digest(self, tmp_path):
        """FR-058b: the name says which script it is, so two runs that perturb
        one card alike name it alike and two that perturb it differently
        never share a name."""
        import re

        from effects.domain.script_variants import variant_digest

        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        (variant,) = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(),
        )
        written = variant.path.read_text(encoding="utf-8").splitlines()
        perturbed = [
            "Name:Lightning Bolt" if line.startswith("Name:") else line for line in written
        ]
        assert re.fullmatch(r"Lightning Bolt Variant [0-9a-f]{8}", variant.name)
        assert variant.name == variant_name("Lightning Bolt", variant_digest(perturbed))

    def test_a_second_run_never_rewrites_a_variant_of_the_first(self, tmp_path):
        """Python R3: a top-up run with another seed used to write a different
        perturbation under an earlier variant's name, so the earlier records
        joined the later text."""
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(8):
            self._write_source(source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT)
        output = tmp_path / "variants"
        first = generate_variants(source, output, held_out=frozenset(), seed=1)
        before = {v.path: v.path.read_bytes() for v in first}

        second = generate_variants(source, output, held_out=frozenset(), seed=2)

        assert {path: path.read_bytes() for path in before} == before
        assert {v.name for v in second} != {v.name for v in first}, (
            "two seeds perturbed every card alike; the test shows nothing"
        )

    def test_an_existing_variant_file_is_reused_as_it_is(self, tmp_path):
        """A file already under the name is the same script (the name is its
        digest), so it is decked again and never written over, whatever it
        holds."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        output = tmp_path / "variants"
        (fresh,) = generate_variants(source, output, held_out=frozenset())
        fresh.path.write_text("kept as it was\n", encoding="utf-8")

        (again,) = generate_variants(source, output, held_out=frozenset())

        assert again.name == fresh.name and fresh.fresh and not again.fresh
        assert again.path.read_text(encoding="utf-8") == "kept as it was\n"

    def test_every_eligible_script_gets_one_variant(self, tmp_path):
        """FR-058a: the untargeted round varies every eligible script once."""
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(10):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        variants = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(),
        )
        assert sorted(v.source_card for v in variants) == sorted(
            f"Bolt {index}" for index in range(10))

    def test_a_variant_carries_the_cost_its_script_declares(self, tmp_path):
        """Read once, as the script is written, rather than off disk per deck."""
        source = tmp_path / "cards"
        source.mkdir()
        self._write_source(source, "bolt", "Lightning Bolt", _BOLT_SCRIPT)
        (variant,) = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(),
        )
        assert variant.mana_cost == "{R}"

    def test_generation_is_reproducible_under_a_seed(self, tmp_path):
        source = tmp_path / "cards"
        source.mkdir()
        for index in range(5):
            self._write_source(
                source, f"bolt{index}", f"Bolt {index}", _BOLT_SCRIPT,
            )
        first = generate_variants(
            source, tmp_path / "a", held_out=frozenset(), seed=5,
        )
        second = generate_variants(
            source, tmp_path / "b", held_out=frozenset(), seed=5,
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
            source, tmp_path / "variants", held_out=frozenset(),
        ) == []

    def test_perturb_script_finds_the_first_perturbable_line(self):
        lines = ["Name:Bolt", "Types:Instant", _BOLT_SCRIPT]
        result = perturb_script(lines, random.Random(3))
        assert result is not None
        perturbed, _description = result
        assert perturbed[0] == "Name:Bolt"
        assert perturbed[2] != _BOLT_SCRIPT


class TestProtectionQualitySwap:
    """FR-058: a protection keyword's quality swaps to a rare one from a
    checked-in whitelist, spelled as protection copies spell the quality."""

    _KNIGHT = [
        "Name:White Knight", "ManaCost:W W", "Types:Creature Human Knight",
        "PT:2/2", "K:First Strike", "K:Protection from black",
    ]

    def test_every_whitelisted_quality_is_rare_and_spelled_as_a_copy_spells_it(self):
        for fragment in PROTECTION_QUALITY_WHITELIST:
            quality = ProtectionQuality.parse(fragment)
            assert quality is not None, fragment
            assert quality.kind is not QualityKind.COLOUR, fragment
            assert quality.fragment == fragment
        kinds = {ProtectionQuality.parse(f).kind for f in PROTECTION_QUALITY_WHITELIST}
        assert kinds == {QualityKind.CARD_TYPE, QualityKind.CREATURE_TYPE,
                         QualityKind.CONDITION, QualityKind.EVERYTHING}

    @pytest.mark.parametrize("line", [
        "K:Protection from red",
        "K:Protection:Artifact",
        "K:Protection:Goblin",
        "K:Protection:Instant:instants",
        "K:Protection:Card.MultiColor:multicolored",
        "K:Protection from everything",
    ])
    def test_a_protection_line_swaps_to_another_whitelisted_quality(self, line):
        for seed in range(20):
            swapped = swap_protection_quality(line, random.Random(seed))
            assert swapped is not None
            perturbed, description = swapped
            replacement = perturbed.removeprefix("K:")
            assert replacement in PROTECTION_QUALITY_WHITELIST
            assert ProtectionQuality.parse(replacement) != ProtectionQuality.parse(line[2:])
            assert description == f"selector-swap: {line} -> {replacement}"

    @pytest.mark.parametrize("line", [
        "K:Flying",
        "K:Protection:Player.Opponent:each of your opponents",
        "K:Protection:Creature.nonSpirit:non-Spirit creatures",
        "A:SP$ Pump | KW$ Protection from red | ValidTgts$ Creature",
    ])
    def test_any_other_line_is_left_to_the_other_perturbations(self, line):
        assert swap_protection_quality(line, random.Random(0)) is None

    def test_a_card_with_protection_has_its_quality_swapped(self):
        perturbed, description = perturb_script(self._KNIGHT, random.Random(1))
        assert perturbed[:-1] == self._KNIGHT[:-1]
        assert perturbed[-1] in {f"K:{f}" for f in PROTECTION_QUALITY_WHITELIST}
        assert description.startswith("selector-swap: K:Protection from black -> ")

    def test_a_variant_of_a_protection_card_is_written_with_the_swapped_quality(
        self, tmp_path,
    ):
        source = tmp_path / "cards"
        source.mkdir()
        (source / "white_knight.txt").write_text("\n".join(self._KNIGHT) + "\n", encoding="utf-8")
        [variant] = generate_variants(
            source, tmp_path / "variants", held_out=frozenset(),
        )
        written = variant.path.read_text(encoding="utf-8").splitlines()
        assert written[0] == f"Name:{variant.name}"
        assert "K:Protection from black" not in written
        assert written[-1].removeprefix("K:") in PROTECTION_QUALITY_WHITELIST


class TestTheSeed:
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
                    held_out=frozenset(), seed=seed,
                )
            ]
            for seed in range(1, 21)
        }
        assert len({tuple(v) for v in drawn.values()}) > 1


def _written(name: str, source: str, path: Path, *, perturbation: str = "numeric"):
    """A variant of the script at ``path``, its cost read as generation reads it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return GeneratedVariant(
        name=name, source_card=source, path=path, perturbation=perturbation,
        mana_cost=script_mana_cost(lines),
    )


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
            held_out=frozenset({"lightning bolt"}),
        )

        assert variants == []


class TestTheVariantRoundPlaysTheVariants:
    def test_the_scripts_are_staged_and_decked(self, tmp_path, fake_supervisor):
        from effects.application import collect_variants

        fake_supervisor.run_id = "run-123"
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

        with patch.object(collect_variants, "write_variant_sidecars",
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
                    decks_per_round=3, workers=1,
                    # No converted tree: the held-out text check is skipped,
                    # rather than reading the machine's real output tree.
                    cards_folders=(tmp_path / "no-converted-cards",),
                )
            )

        constructed = fake_supervisor.constructor.call_args.kwargs
        assert constructed["variant_scripts"] == tmp_path / "variants"
        assert fake_supervisor.play_round.call_args.args[0], "no variant decks"
        assert fake_supervisor.play_round.call_args.kwargs["matches"] == 3
        # The run's own decks file (its path is the supervisor's: every
        # worker reloads it at its start, so two runs never share one).
        fake_supervisor.decks_path.assert_called_with("variant")
        assert fake_supervisor.play_round.call_args.args[1] is (
            fake_supervisor.decks_path.return_value
        )


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
        variant = _written("Bolt Variant 0", "Bolt", path)

        from effects.application.collector_common import build_coverage_decks

        decks = build_coverage_decks(
            {variant.name: 1.0}, {variant.name: variant.deck_candidate()}, 1,
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
        variant = _written("Plains Variant 0", "Plains", path, perturbation="selector")

        # Zero pips either way -- the point is this must land in the same
        # all-zero-pips (colorless-fallback) case a missing ManaCost: line
        # does, not raise and not inject a spurious color.
        lands = compute_basic_lands([variant.deck_candidate()] * 23)
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
        variant = _written("Bolt Variant 0", "Bolt", path)

        assert variant.deck_candidate() == DeckCandidate.from_converted_text(
            "Bolt Variant 0", "name: bolt variant 0\nmana cost: {2}{R}{R}\n",
        )

_RUN = "run-1"


def _variant_record(script_file: str, game: str, n: int, **overrides) -> EffectRecord:
    """A resolution acting through ``script_file``'s first spell line."""
    from effects.domain.provenance import ProvenanceKey

    return _record(
        record_id=f"{game}.{n}", game_id=game, run_id=_RUN,
        ability=(ProvenanceKey(script_file, 0, "spell", 0),), **overrides,
    )


def _round_writes(directory: Path, *rounds: list[EffectRecord], run: str = _RUN):
    """A stand-in for the workers: the n-th round writes ``rounds[n]`` as a
    shard of this run's, and a round past the last writes nothing."""
    from effects.infrastructure.shard_io import write_shard

    played = iter(rounds)

    def write(*_args, **_kwargs) -> None:
        records = next(played, None)
        if records:
            directory.mkdir(parents=True, exist_ok=True)
            index = len(list(directory.glob(f"{run}.*")))
            write_shard(directory / f"{run}.0-{index}.jsonl.gz", records)

    return write


def test_a_damaged_variant_shard_is_counted_up_to_the_damage_once(tmp_path):
    """One damaged shard must not stop a collection run every round."""
    import gzip

    from effects.application.collect_variants import VariantGameCounter
    from effects.domain.variant_rounds import VariantGames
    from effects.infrastructure.record_codec import format_record_line

    bolt = "variant-scripts/bolt_variant_0a1b2c3d.txt"
    members = [gzip.compress((format_record_line(_variant_record(bolt, game, 1)) + "\n")
                             .encode("utf-8")) for game in ("g1", "g2", "g3")]
    members[1] = b"XX" + members[1][2:]
    shard = tmp_path / f"{_RUN}.0-a.jsonl.gz"
    shard.write_bytes(b"".join(members))
    counter = VariantGameCounter(tmp_path, _RUN, workers=1)
    units = {"Bolt": VariantGames(name="Bolt", script_file=bolt)}
    counter.update(units)
    assert units["Bolt"].games == {"g1"}
    assert len(counter.damaged) == 1 and str(shard) in counter.damaged[0]
    counter.update(units)
    assert len(counter.damaged) == 1


class TestTheUntargetedRound:
    """FR-058a: one variant of every eligible script, played in rounds
    weighted by the games each lacks, until each has ``--variant-games``
    games or has retired; its games counted from this run's own shards."""

    BOLT = "variant-scripts/bolt_variant_0a1b2c3d.txt"
    SHOCK = "variant-scripts/shock_variant_99aa00ff.txt"

    def _fixture(self, tmp_path, monkeypatch, supervisor, *rounds, **config):
        import effects.application.collect_variants as collect_variants

        variants = []
        for name, card in (("Bolt Variant 0a1b2c3d", "Bolt"), ("Shock Variant 99aa00ff", "Shock")):
            path = tmp_path / "variant-scripts" / f"{name.lower().replace(' ', '_')}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Name:{name}\nManaCost:R\nTypes:Instant\n", encoding="utf-8")
            variants.append(GeneratedVariant(
                name=name, source_card=card, path=path, perturbation="numeric",
                mana_cost="{R}",
            ))
        monkeypatch.setattr(collect_variants, "generate_variants", lambda *a, **k: variants)
        # The real filter needs a Forge database staged with the variants;
        # TestOnlyLoadableVariantsAreDecked covers it directly.
        monkeypatch.setattr(
            collect_variants, "loadable_variants", lambda generated, _root: generated,
        )
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=0),
        )
        records = tmp_path / "records"
        records.mkdir()
        supervisor.run_id = _RUN
        supervisor.play_round.side_effect = _round_writes(records, *rounds)
        forge_cards = tmp_path / "forge-cards"
        forge_cards.mkdir()
        config.setdefault("cards_folders", (tmp_path / "no-converted-cards",))
        return collect_variants, CollectVariantsConfig(
            effect_records=records, forge_cards_path=forge_cards,
            variant_scripts=tmp_path / "variant-scripts", decks_per_round=4,
            workers=1, **config,
        )

    def _games(self, script_file, *games):
        return [_variant_record(script_file, game, n) for n, game in enumerate(games)]

    def test_rounds_end_once_every_variant_has_its_games(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        import logging

        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor,
            self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1"),
            self._games(self.SHOCK, "g3"),
        )
        with caplog.at_level(logging.INFO):
            assert collect_variants.run(config) == 0

        assert fake_supervisor.play_round.call_count == 2
        assert "Collected 4 new effect records; 2 of 2 variants played" in caplog.text

    def test_each_round_weights_a_variant_by_the_games_it_lacks(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """After the first round only Shock lacks a game, so it alone is decked."""
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor,
            self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1"),
            self._games(self.SHOCK, "g3"),
        )
        collect_variants.run(config)

        second = fake_supervisor.play_round.call_args_list[1].args[0]
        nonlands = {card for deck in second for card in deck} - {
            "Plains", "Island", "Swamp", "Mountain", "Forest"}
        assert nonlands == {"Shock Variant 99aa00ff"}

    def test_a_game_counts_once_however_many_records_it_holds(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor,
            self._games(self.BOLT, "g1", "g1", "g1") + self._games(self.SHOCK, "g1", "g2"),
            self._games(self.BOLT, "g2"),
        )
        assert collect_variants.run(config) == 0
        assert fake_supervisor.play_round.call_count == 2

    def test_a_decision_candidate_naming_the_variant_counts_its_game(self):
        from effects.domain.provenance import ProvenanceKey
        from effects.domain.records import (
            Candidate,
            PlayabilityDecisionPayload,
            PlayabilitySubkind,
            RecordKind,
        )
        from effects.domain.variant_rounds import VariantShardGames

        key = ProvenanceKey(self.BOLT, 0, "spell", 0)
        found = VariantShardGames()
        found.add(make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION, game_id="g7",
            payload=PlayabilityDecisionPayload(candidates=(Candidate(
                ability=(key,), can_play=True, affordable=True, has_legal_target=True,
            ),)),
        ))
        assert found.games == {self.BOLT: {"g7"}}

    def test_a_cut_game_gives_its_variant_no_game(self, tmp_path):
        """WP7 / P1 #1: a gen-3 game without its game-end marker is one
        build-corpus drops; it must not finish a variant's games."""
        from effects.application.collect_variants import count_variant_shard
        from effects.infrastructure.shard_io import write_shard
        from tests.unit.effects.builders import with_game_end_markers

        ended = with_game_end_markers([_variant_record(self.BOLT, "g1", 0, generation=3)])
        cut = [_variant_record(self.BOLT, "g2", 0, generation=3),
               _variant_record(self.SHOCK, "g2", 1, generation=3)]
        shard = tmp_path / f"{_RUN}.0-a.jsonl.gz"
        write_shard(shard, ended + cut)

        found = count_variant_shard(shard)

        assert found.games == {self.BOLT: {"g1"}}
        assert found.cut_games == 1
        assert found.records == 4, "every record is still read"

    def test_the_round_reports_its_cut_games(self, tmp_path, monkeypatch, fake_supervisor):
        from tests.unit.effects.builders import with_game_end_markers

        records = with_game_end_markers([
            _variant_record(self.BOLT, "g1", 0, generation=3),
            _variant_record(self.BOLT, "g2", 0, generation=3),
            _variant_record(self.SHOCK, "g3", 0, generation=3),
            _variant_record(self.SHOCK, "g4", 0, generation=3),
        ])
        cut = _variant_record(self.SHOCK, "g5", 0, generation=3)
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, [*records, cut],
        )

        assert collect_variants.run(config) == 0
        assert [call.args for call in fake_supervisor.count_cut_games.call_args_list] == [
            (0,), (1,),
        ]

    def test_another_run_s_shard_never_counts(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """python, two runs on one directory: only this run's shards say what
        its variants played."""
        from effects.infrastructure.shard_io import write_shard

        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, no_progress_rounds=1,
        )
        write_shard(
            config.effect_records / "other-run.0-a.jsonl.gz",
            self._games(self.BOLT, "x1", "x2") + self._games(self.SHOCK, "x1", "x2"),
        )
        assert collect_variants.run(config) == 1

    def test_a_continued_run_s_games_count_and_its_finished_variants_are_not_replayed(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """V4 #6: an interrupted run restarted with the same seed replayed
        every variant it had finished, doubling their games. ``--continues``
        counts the earlier run's shards first."""
        from effects.infrastructure.shard_io import write_shard

        rounds = [self._games(self.SHOCK, "g1", "g2")]
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, *rounds, continues=("earlier",),
        )
        write_shard(
            config.effect_records / "earlier.0-a.jsonl.gz", self._games(self.BOLT, "x1", "x2"),
        )

        assert collect_variants.run(config) == 0

        decked = {name for call in fake_supervisor.play_round.call_args_list
                  for deck in call.args[0] for name in deck}
        assert "Bolt Variant 0a1b2c3d" not in decked
        assert "Shock Variant 99aa00ff" in decked

    def test_a_run_that_could_deck_nothing_fails_and_says_so(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """No unfinished variant yields a deck: no round is played, the
        variants still lack their games, and the run must not read as done."""
        collect_variants, config = self._fixture(tmp_path, monkeypatch, fake_supervisor)
        monkeypatch.setattr(collect_variants, "build_coverage_decks", lambda *a, **k: [])

        assert collect_variants.run(config) == 1
        fake_supervisor.play_round.assert_not_called()
        assert "already has its" not in caplog.text
        assert "2 variant(s) still short" in caplog.text

    def test_a_continued_run_with_nothing_left_plays_no_round_and_succeeds(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        from effects.infrastructure.shard_io import write_shard

        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, continues=("earlier",),
        )
        write_shard(
            config.effect_records / "earlier.0-a.jsonl.gz",
            self._games(self.BOLT, "x1", "x2") + self._games(self.SHOCK, "x1", "x2"),
        )

        assert collect_variants.run(config) == 0
        fake_supervisor.play_round.assert_not_called()

    def test_a_variant_shown_without_a_new_game_retires(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """Bolt has its two games; Shock sits on the board in every round's
        games and never acts, so it retires after --no-progress-rounds."""
        import logging

        from effects.domain.provenance import ProvenanceKey
        from tests.unit.effects.builders import make_entity, make_snapshot

        shock_on_board = make_snapshot(entities=(make_entity(
            "E9", "Shock Variant 99aa00ff", printed=(ProvenanceKey(self.SHOCK, 0, "spell", 0),),
        ),))
        shown = [_variant_record(self.BOLT, "g9", 0, state=shock_on_board)]
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor,
            self._games(self.BOLT, "g1", "g2"), shown, shown, no_progress_rounds=2,
        )
        with caplog.at_level(logging.INFO):
            assert collect_variants.run(config) == 0

        assert fake_supervisor.play_round.call_count == 3
        assert "1 retired short of 2 games" in caplog.text

    def test_a_round_that_collected_nothing_fails_loudly(
        self, tmp_path, monkeypatch, caplog, fake_supervisor,
    ):
        """A round whose decks all materialized as basics writes no variant
        record; the variants retire unseen and the run exits 1."""
        import logging

        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, no_progress_rounds=1,
        )
        with caplog.at_level(logging.ERROR):
            code = collect_variants.run(config)

        assert code == 1
        assert "no new effect records" in caplog.text.lower()

    def test_each_generated_variant_is_logged_with_its_source_and_perturbation(
        self, tmp_path, monkeypatch, caplog, fake_supervisor,
    ):
        """So a variant's records can be traced back to what was changed."""
        import logging

        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, no_progress_rounds=1,
        )
        with caplog.at_level(logging.INFO):
            collect_variants.run(config)

        assert "Bolt Variant 0a1b2c3d, from Bolt: numeric" in caplog.text

    def test_a_step_with_nothing_to_go_on_ends_the_run_with_its_code(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """The sidecar pass's own exit code, not a blanket 1."""
        collect_variants, config = self._fixture(tmp_path, monkeypatch, fake_supervisor)
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=3),
        )

        assert collect_variants.run(config) == 3
        fake_supervisor.play_round.assert_not_called()

    def test_the_staging_is_claimed_before_anything_is_staged(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """Another command's first JVM leaves a staging alone while its owner
        runs: the owner is recorded before the sidecar pass stages."""
        collect_variants, config = self._fixture(tmp_path, monkeypatch, fake_supervisor)
        order: list[str] = []
        monkeypatch.setattr(collect_variants, "claim_staging", lambda: order.append("claim"))
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars",
            lambda _path: order.append("sidecar pass") or 0,
        )

        collect_variants.run(config)

        assert order[:2] == ["claim", "sidecar pass"]

    def test_a_sidecar_pass_past_its_timeout_ends_the_run_with_1(
        self, tmp_path, monkeypatch, fake_supervisor, caplog,
    ):
        """P1 #12: the one-shot JVM is killed at its timeout; the run says so."""
        from price_predictor.infrastructure.forge_jvm import ForgeWorkerTimeout

        collect_variants, config = self._fixture(tmp_path, monkeypatch, fake_supervisor)
        monkeypatch.setattr(collect_variants, "write_variant_sidecars", MagicMock(
            side_effect=ForgeWorkerTimeout("VariantSidecarMain ran past its 3600 s timeout"),
        ))

        assert collect_variants.run(config) == 1
        fake_supervisor.play_round.assert_not_called()
        assert "ran past its 3600 s timeout" in caplog.text

    @pytest.mark.parametrize("sidecar_code, plays", [(0, True), (0, False), (3, False)])
    def test_the_round_s_staged_variants_are_taken_out_however_it_ends(
        self, tmp_path, monkeypatch, fake_supervisor, sidecar_code, plays,
    ):
        """FR-058b, python R8: the sidecar pass and every worker stage the
        tree into Forge's custom-cards directory, which every later Forge
        startup on the machine reads."""
        rounds = [self._games(self.BOLT, "g1", "g2") + self._games(self.SHOCK, "g1", "g2")]
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, *(rounds if plays else []),
            no_progress_rounds=1,
        )
        monkeypatch.setattr(
            collect_variants, "write_variant_sidecars", MagicMock(return_value=sidecar_code),
        )
        unstage = MagicMock(return_value=1)
        monkeypatch.setattr(collect_variants, "unstage_variants", unstage)

        collect_variants.run(config)

        unstage.assert_called_once_with()

    def test_an_interrupted_round_does_not_report_success(
        self, tmp_path, monkeypatch, fake_supervisor,
    ):
        """Ctrl-C on the variant side: ``ForgeWorkerPool`` swallows SIGINT, so
        a round the operator stopped returns exactly like one that finished."""
        collect_variants, config = self._fixture(
            tmp_path, monkeypatch, fake_supervisor, self._games(self.BOLT, "g1"),
        )
        fake_supervisor.interrupted = True

        assert collect_variants.run(config) == 130
        assert fake_supervisor.play_round.call_count == 1


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
        (tmp_path / "loadable.lst").write_text(
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

    def test_the_list_is_no_card_script(self):
        """Python R1: Forge loads every ``.txt`` staged from the variant tree as
        a card, and a list of names with no ``Name:`` line stops every Forge
        startup on the machine."""
        from effects.application.collect_variants import LOADABLE_FILE

        # The property, not the spelling: the sibling below pins the spelling
        # to the Java writer, which would follow any rename to a ``.txt``.
        assert not LOADABLE_FILE.lower().endswith(".txt")

    def test_the_list_is_the_one_variant_sidecar_main_writes(self):
        import re

        from effects.application.collect_variants import LOADABLE_FILE

        java = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main"
                / "java" / "com" / "pricepredictor" / "connector" / "VariantSidecarMain.java")
        declared = re.search(r'LOADABLE_FILE\s*=\s*"([^"]+)"', java.read_text(encoding="utf-8"))
        assert declared is not None and declared.group(1) == LOADABLE_FILE

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
            source, tmp_path / "variants", held_out=frozenset(),
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

    def test_a_reused_variant_printing_a_held_out_text_stays_on_disk(self, tmp_path):
        """FR-058b: only a variant no deck has held yet is deleted. A reused
        one may have records from an earlier run, which join its file by key."""
        import dataclasses

        from effects.application.collect_variants import drop_held_out_variants
        from effects.infrastructure.sidecar_io import sidecar_path_for

        held = dataclasses.replace(self._variant(
            tmp_path, "act_of_treason_variant_1",
            "SP$ DealDamage | ValidTgts$ Artifact | NumDmg$ 5",
        ), fresh=False)

        assert drop_held_out_variants([held], self._holdout(HoldoutUnit.TEMPLATE)) == []
        assert held.path.exists() and sidecar_path_for(held.path).exists()

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
