"""A variant reaches the manabase as a converted card of its cost would."""

from __future__ import annotations

import random
from pathlib import Path

from effects.application.collect_variants import GeneratedVariant
from effects.domain.script_variants import script_mana_cost


def _written(name: str, source: str, path: Path, *, perturbation: str = "numeric"):
    """A variant of the script at ``path``, its cost read as generation reads it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return GeneratedVariant(
        name=name, source_card=source, path=path, perturbation=perturbation,
        mana_cost=script_mana_cost(lines),
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
