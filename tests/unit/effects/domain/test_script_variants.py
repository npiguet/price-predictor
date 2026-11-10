"""Perturbing one script parameter (T138): a whitelisted number shifts or
doubles, floored at zero, and a selector swaps from a checked-in whitelist."""

from __future__ import annotations

import random

from effects.domain.script_variants import (
    MAX_NUMERIC_SHIFT,
    SELECTOR_WHITELIST,
    PerturbationKind,
    numeric_params,
    perturb,
    script_mana_cost,
    selector_params,
    shift,
    variant_digest,
    variant_name,
)

_BOLT_SCRIPT = (
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target."
)


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


def test_variant_name_derives_from_its_source_and_its_script_s_digest():
    """FR-058b: ``<card> Variant <h>``, so one perturbed script has one name."""
    assert variant_name("Lightning Bolt", "0a1b2c3d") == "Lightning Bolt Variant 0a1b2c3d"


def test_the_digest_is_the_first_eight_hex_digits_of_the_script_s_sha256():
    import hashlib

    lines = ["Name:Bolt", "A:SP$ DealDamage | NumDmg$ 4"]
    expected = hashlib.sha256(b"Name:Bolt\nA:SP$ DealDamage | NumDmg$ 4\n").hexdigest()
    assert variant_digest(lines) == expected[:8]
    assert variant_digest(["Name:Bolt", "A:SP$ DealDamage | NumDmg$ 5"]) != variant_digest(lines)


class TestScriptManaCost:
    def test_forge_shards_are_brace_delimited_as_the_converted_corpus_spells_them(self):
        assert script_mana_cost(["Name:X", "ManaCost:2 R R"]) == "{2}{R}{R}"

    def test_a_lands_no_cost_keeps_the_literal_manacost_parse_special_cases(self):
        assert script_mana_cost(["ManaCost:no cost"]) == "{no}{cost}"

    def test_a_script_declaring_none_names_none(self):
        assert script_mana_cost(["Name:X", "Types:Instant"]) is None


class TestSignedValues:
    """python review, perturbation oddities: a ``+N`` value (``NumAtt$ +2``,
    most pumps) never parsed as a number, and a ``-N`` one was floored to 0 or
    1, turning a debuff into nothing or a buff."""

    _PUMP = "A:SP$ Pump | ValidTgts$ Creature | NumAtt$ +2 | NumDef$ +2"
    _DEBUFF = "A:SP$ Pump | ValidTgts$ Creature | NumAtt$ -2 | NumDef$ -2"

    def test_a_plus_value_is_a_number(self):
        assert numeric_params(self._PUMP) == {"NumAtt": 2, "NumDef": 2}

    def test_a_pump_stays_a_pump_written_with_its_sign(self):
        rng = random.Random(0)
        numeric = 0
        for _ in range(100):
            result = perturb(self._PUMP, rng)
            assert result is not None
            perturbed, perturbation = result
            if perturbation.kind == PerturbationKind.SELECTOR:
                continue
            numeric += 1
            assert perturbation.original == "+2"
            assert perturbation.replacement.startswith("+")
            assert int(perturbation.replacement) >= 1
            assert f"{perturbation.key}$ {perturbation.replacement}" in perturbed
        assert numeric > 0

    def test_a_debuff_stays_a_debuff(self):
        rng = random.Random(0)
        numeric = 0
        for _ in range(100):
            result = perturb(self._DEBUFF, rng)
            assert result is not None
            perturbed, perturbation = result
            if perturbation.kind == PerturbationKind.SELECTOR:
                continue
            numeric += 1
            assert perturbation.original == "-2"
            assert int(perturbation.replacement) <= -1
            assert f"{perturbation.key}$ {perturbation.replacement}" in perturbed
        assert numeric > 0


class TestDeckBuildingHints:
    """python review: a ``DeckHints:Type$ Creature`` swap changes nothing in a
    game, only what the AI's deck builder prefers."""

    def test_a_deck_building_hint_is_not_perturbed(self):
        rng = random.Random(0)
        for line in (
            "DeckHints:Type$Creature",
            "DeckHints:Type$ Creature",
            "DeckNeeds:Type$Artifact",
            "DeckHas:Ability$Token & Type$Creature",
            "SVar:AIPreference:SacCost$Creature",
        ):
            assert perturb(line, rng) is None, line
