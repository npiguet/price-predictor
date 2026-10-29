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


def test_variant_name_derives_from_its_source():
    assert variant_name("Lightning Bolt", 0) == "Lightning Bolt Variant 0"
    assert variant_name("Lightning Bolt", 1) != variant_name("Lightning Bolt", 0)


class TestScriptManaCost:
    def test_forge_shards_are_brace_delimited_as_the_converted_corpus_spells_them(self):
        assert script_mana_cost(["Name:X", "ManaCost:2 R R"]) == "{2}{R}{R}"

    def test_a_lands_no_cost_keeps_the_literal_manacost_parse_special_cases(self):
        assert script_mana_cost(["ManaCost:no cost"]) == "{no}{cost}"

    def test_a_script_declaring_none_names_none(self):
        assert script_mana_cost(["Name:X", "Types:Instant"]) is None
