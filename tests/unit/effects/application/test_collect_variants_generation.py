"""Generating synthetic script variants (T138): what is written, what is skipped
(held-out, multi-face, nothing to perturb), the name and seed it takes, and the
protection-quality swap.

The held-out rule is the one that would be silently wrong. Perturbing a
parameter changes the text, so a variant of a held-out card carries a text that
is *not* held out and would reach training. Skipping generation is the only
point where the rule can be enforced, because by collection time the text no
longer matches anything in the holdout.
"""

from __future__ import annotations

import random

import pytest

from effects.application.collect_variants import (
    PROTECTION_QUALITY_WHITELIST,
    generate_variants,
    perturb_script,
    swap_protection_quality,
)
from effects.domain.rule_outcomes import ProtectionQuality, QualityKind
from effects.domain.script_variants import is_multi_face, variant_name
from tests.unit.effects.domain.test_script_variants import _BOLT_SCRIPT


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
            held_out=frozenset({"lightning bolt"}),
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
