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
import re
from pathlib import Path

import pytest

from effects.domain.collection_caps import (
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_LEGALITY_RATE,
    DEFAULT_MANA_CAP,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_PROBABILITY,
    DEFAULT_RANDOM_SEAT_SHARE,
    CollectionCaps,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from effects.infrastructure.cli import CAP_DEFAULTS
from effects.infrastructure.cli import announce_probe_state as effects_announce
from sealed.infrastructure.cli import (
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_LEGALITY_RATE,
    EFFECT_MANA_CAP,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBE_KEYWORDS,
    EFFECT_PROBES_PER_GAME,
    EFFECT_RANDOM_SEAT_PROBABILITY,
    EFFECT_RANDOM_SEAT_SHARE,
    _effect_collection_caps,
)
from sealed.infrastructure.cli import announce_probe_state as sealed_announce


class TestBothSidesAnnounceProbesTheSameWay:
    """The probe startup line is restated on both sides, so it is pinned here.

    ``--probes-per-game`` reads like the switch and is the budget;
    ``--probe-keywords`` is the switch, and an emptied one names nothing. Two
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

    def test_the_two_defaults_both_turn_probes_on(self):
        assert EFFECT_PROBE_KEYWORDS == CAP_DEFAULTS["probe_keywords"]
        line = effects_announce(
            CAP_DEFAULTS["probe_keywords"], DEFAULT_PROBES_PER_GAME,
        )
        assert "DISABLED" not in line
        assert "first_strike" in line and "infect" in line


class TestTheProbeKeywordDefault:
    """Probes are on by default, over every keyword gate 2 checks."""

    def test_it_names_all_eight_damage_step_keywords(self):
        assert DEFAULT_PROBE_KEYWORDS == (
            "first_strike,double_strike,deathtouch,lifelink,trample,"
            "indestructible,wither,infect"
        )

    def test_it_is_gate_twos_own_table(self):
        """Derived from the table, so a keyword gate 2 checks is never one
        collection skips."""
        assert DEFAULT_PROBE_KEYWORDS.split(",") == list(KEYWORDS_BY_NAME)

    def test_it_reaches_the_worker(self):
        assert CollectionCaps().as_system_properties()[
            "effect.probe.keywords"
        ] == DEFAULT_PROBE_KEYWORDS

    def test_an_empty_list_still_turns_probes_off(self):
        caps = CollectionCaps(probe_keywords="")
        assert caps.as_system_properties()["effect.probe.keywords"] == ""


class TestTheTwoSidesAgree:
    """A default that drifts between the two commands makes two corpora that
    mean different things and nothing says which is which."""

    def test_the_defaults_match(self):
        assert EFFECT_MANA_CAP == DEFAULT_MANA_CAP
        assert EFFECT_PLAYABILITY_RATE == DEFAULT_PLAYABILITY_RATE
        assert EFFECT_INTERVENTIONS_PER_GAME == DEFAULT_INTERVENTIONS_PER_GAME
        assert EFFECT_PROBES_PER_GAME == DEFAULT_PROBES_PER_GAME
        assert EFFECT_PROBE_KEYWORDS == CollectionCaps().probe_keywords
        assert EFFECT_LEGALITY_RATE == DEFAULT_LEGALITY_RATE
        assert EFFECT_RANDOM_SEAT_SHARE == DEFAULT_RANDOM_SEAT_SHARE
        assert EFFECT_RANDOM_SEAT_PROBABILITY == DEFAULT_RANDOM_SEAT_PROBABILITY

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
        assert "effect.legality.rate" in properties

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


#: The Java side of the contract. Read rather than mirrored in a constant,
#: because a mirrored list is a third copy to drift.
_JAVA_CAPS = (
    Path(__file__).resolve().parents[4]
    / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
    / "connector" / "effects" / "PatchedCollectors.java"
)


class TestTheJvmReadsExactlyWhatThePythonSideWrites:
    """The caps the worker reads and the caps a supervisor can set are one set.

    They were not. ``effect.legality.rate`` was read by
    ``CollectionCaps.fromSystemProperties`` and settable by no command, so the
    legality sampler ran at a rate a javadoc called ``--legality-rate``, a flag
    that did not exist.
    A property one side reads and the other never writes is invisible from
    both.
    """

    def _java_property_names(self) -> set[str]:
        """Every ``effect.*`` property literal the connector's caps record names.

        Read out of the source rather than restated, because a restated list is
        a third copy and would drift the way the first two did. The whole file
        is scanned: ``CollectionCaps.fromSystemProperties`` is the only place
        that spells these literals, so a new one appearing anywhere else in it
        is still a property the Java side reads.
        """
        source = _JAVA_CAPS.read_text(encoding="utf-8")
        return set(re.findall(r'"(effect\.[a-z.]+)"', source))

    @pytest.mark.skipif(
        not _JAVA_CAPS.exists(), reason="the connector source is not checked out"
    )
    def test_neither_side_names_a_property_the_other_does_not(self):
        assert self._java_property_names() == set(
            CollectionCaps().as_system_properties()
        )


class TestNoSnapshotDepthKnob:
    """Every record collects all four tiers, so no command can choose a depth."""

    def test_the_caps_carry_no_tier_field(self):
        assert "snapshot_tiers" not in {
            field for field in CollectionCaps.__dataclass_fields__
        }

    def test_no_property_names_a_depth(self):
        assert "effect.snapshot.tiers" not in CollectionCaps().as_system_properties()
        assert "effect.snapshot.tiers" not in _effect_collection_caps(
            argparse.Namespace()
        )


class TestTheLegalityRate:
    def test_what_if_queries_are_off_by_default(self):
        """Real declarations are written whatever the rate; the rate keeps
        only what-ifs, which are oversampled by how often the AI re-asks."""
        assert CollectionCaps().legality_rate == DEFAULT_LEGALITY_RATE == 0.0

    def test_it_reaches_the_property_the_worker_samples_from(self):
        properties = CollectionCaps(legality_rate=0.25).as_system_properties()
        assert properties["effect.legality.rate"] == "0.25"


class TestTheRandomSeat:
    """FR-021: both knobs travel to the worker like every other cap."""

    def test_it_is_off_by_default(self):
        caps = CollectionCaps()
        assert caps.random_seat_share == 0.0
        assert caps.random_seat_probability == 0.0

    def test_both_reach_their_properties(self):
        properties = CollectionCaps(
            random_seat_share=0.125, random_seat_probability=0.25,
        ).as_system_properties()
        assert properties["effect.random.seat.share"] == "0.125"
        assert properties["effect.random.seat.probability"] == "0.25"

    def test_the_sealed_supervisor_writes_the_same_properties(self):
        args = argparse.Namespace(
            random_seat_share=0.125, random_seat_probability=0.25,
        )
        sealed_side = _effect_collection_caps(args)
        assert sealed_side["effect.random.seat.share"] == "0.125"
        assert sealed_side["effect.random.seat.probability"] == "0.25"

    def test_an_unset_probability_travels_as_the_default(self):
        """argparse leaves it None; the share is 0 whenever it is."""
        args = argparse.Namespace(random_seat_probability=None)
        assert _effect_collection_caps(args)["effect.random.seat.probability"] == (
            str(DEFAULT_RANDOM_SEAT_PROBABILITY)
        )
