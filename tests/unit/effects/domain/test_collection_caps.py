"""The caps, and the two places that must agree about them.

Every cap is a per-worker-process quantity the Python supervisor cannot
observe, so it travels to the JVM as a system property. That makes the property
names a contract between three parties that cannot all import each other:
``sealed`` owns ``match-outcomes``, ``effects`` owns the coverage and variant
collectors, and neither may import the other's CLI. This file is where the two
sides are pinned together, because it is in the effects suite and may see both.
"""

from __future__ import annotations

import argparse

from effects.domain.collection_caps import (
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_MANA_CAP,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBES_PER_GAME,
    CollectionCaps,
)
from effects.infrastructure.cli import CAP_DEFAULTS
from effects.infrastructure.cli import announce_probe_state as effects_announce
from sealed.infrastructure.cli import (
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_MANA_CAP,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBE_KEYWORDS,
    EFFECT_PROBES_PER_GAME,
    _effect_collection_caps,
)
from sealed.infrastructure.cli import announce_probe_state as sealed_announce


class TestBothSidesAnnounceProbesTheSameWay:
    """The probe startup line is restated on both sides, so it is pinned here.

    ``--probes-per-game`` reads like the switch and is the budget;
    ``--probe-keywords`` is the switch and defaults to naming nothing. Two
    commands that said different things about that would be worse than one that
    said nothing, because an operator would learn to trust the wrong one.
    """

    def test_both_call_the_empty_case_disabled_and_name_the_flag(self):
        for line in (sealed_announce("", 2), effects_announce("", 2)):
            assert "DISABLED" in line
            assert "--probe-keywords" in line

    def test_both_produce_the_same_line_for_the_same_inputs(self):
        for keywords, budget in (("", 2), ("trample", 2), ("wither,infect", 5)):
            assert sealed_announce(keywords, budget) == effects_announce(
                keywords, budget
            )

    def test_the_two_defaults_both_leave_probes_off(self):
        assert EFFECT_PROBE_KEYWORDS == CAP_DEFAULTS["probe_keywords"] == ""
        assert "DISABLED" in effects_announce(
            CAP_DEFAULTS["probe_keywords"], DEFAULT_PROBES_PER_GAME,
        )


class TestTheTwoSidesAgree:
    """A default that drifts between the two commands makes two corpora that
    mean different things and nothing says which is which."""

    def test_the_defaults_match(self):
        assert EFFECT_MANA_CAP == DEFAULT_MANA_CAP
        assert EFFECT_PLAYABILITY_RATE == DEFAULT_PLAYABILITY_RATE
        assert EFFECT_INTERVENTIONS_PER_GAME == DEFAULT_INTERVENTIONS_PER_GAME
        assert EFFECT_PROBES_PER_GAME == DEFAULT_PROBES_PER_GAME
        assert EFFECT_PROBE_KEYWORDS == CollectionCaps().probe_keywords

    def test_the_property_names_match(self):
        """Both sides must spell the -D properties the worker reads identically."""
        sealed_side = _effect_collection_caps(argparse.Namespace())
        effects_side = CollectionCaps().as_system_properties()
        assert set(sealed_side) == set(effects_side)

    def test_the_default_values_match_property_for_property(self):
        assert _effect_collection_caps(argparse.Namespace()) == (
            CollectionCaps().as_system_properties()
        )


class TestPropertyNames:
    def test_every_property_is_namespaced_for_the_collectors(self):
        """A worker started without instrumentation ignores the lot."""
        for name in CollectionCaps().as_system_properties():
            assert name.startswith("effect.")

    def test_underscores_become_dots(self):
        properties = CollectionCaps().as_system_properties()
        assert "effect.mana.cap" in properties
        assert "effect.interventions.per.game" in properties
        assert "effect.probe.keywords" in properties

    def test_values_are_strings(self):
        """They become -D arguments, so a non-string would stringify later and
        somewhere less obvious."""
        for value in CollectionCaps().as_system_properties().values():
            assert isinstance(value, str)


class TestFromArgs:
    def test_it_reads_the_flags_the_parsers_install(self):
        args = argparse.Namespace(
            mana_cap=3, playability_rate=0.5, interventions_per_game=1,
            probes_per_game=0, probe_keywords="first_strike,trample",
        )
        caps = CollectionCaps.from_args(args)
        assert caps.mana_cap == 3
        assert caps.playability_rate == 0.5
        assert caps.probe_keywords == "first_strike,trample"

    def test_a_missing_flag_falls_back_to_the_default(self):
        """The evaluator and the trainer carry no cap flags, and both build a
        config from their own namespace."""
        caps = CollectionCaps.from_args(argparse.Namespace())
        assert caps == CollectionCaps()


class TestTheManaCap:
    def test_it_defaults_to_one_per_game(self):
        """A Mountain taps a dozen times a game for the same R, and the repeats
        observe a board that barely moved."""
        assert CollectionCaps().mana_cap == 1

    def test_it_is_overridable_for_more_mana_coverage(self):
        assert CollectionCaps(mana_cap=5).mana_cap == 5
