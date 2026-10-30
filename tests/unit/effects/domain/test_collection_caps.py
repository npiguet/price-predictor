"""The caps, and the two places that must agree about them.

Every cap is a per-worker-process quantity the Python supervisor cannot
observe, so it travels to the JVM as a system property. ``sealed`` writes the
flags every collecting supervisor shares and their defaults, ``effects`` owns
the caps record the collectors hand the worker, and ``sealed`` may not import
``effects``; this file is in the effects suite, may see both, and pins them
together and against the property literals the Java side reads.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from effects.domain.collection_caps import (
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_MANA_CAP,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_PROBABILITY,
    DEFAULT_RANDOM_SEAT_SHARE,
    CollectionCaps,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from sealed.infrastructure.cli import _effect_collection_caps
from sealed.infrastructure.effect_collection_flags import (
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_MANA_CAP,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBE_KEYWORDS,
    EFFECT_PROBES_PER_GAME,
    EFFECT_RANDOM_SEAT_PROBABILITY,
    EFFECT_RANDOM_SEAT_SHARE,
    announce_probe_state,
)


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

    def test_the_default_announces_probes_on(self):
        line = announce_probe_state(DEFAULT_PROBE_KEYWORDS, DEFAULT_PROBES_PER_GAME)
        assert "DISABLED" not in line
        assert "first_strike" in line and "infect" in line


class TestTheTwoSidesAgree:
    """A default that drifts between the two commands makes two corpora that
    mean different things and nothing says which is which."""

    def test_the_defaults_match(self):
        assert EFFECT_MANA_CAP == DEFAULT_MANA_CAP
        assert EFFECT_PLAYABILITY_RATE == DEFAULT_PLAYABILITY_RATE
        assert EFFECT_INTERVENTIONS_PER_GAME == DEFAULT_INTERVENTIONS_PER_GAME
        assert EFFECT_PROBES_PER_GAME == DEFAULT_PROBES_PER_GAME
        assert EFFECT_PROBE_KEYWORDS == CollectionCaps().probe_keywords
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


#: The Java side of the contract. Read rather than mirrored in a constant,
#: because a mirrored list is a third copy to drift.
_JAVA_CAPS = (
    Path(__file__).resolve().parents[4]
    / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
    / "connector" / "effects" / "CollectionCaps.java"
)


class TestTheJvmReadsExactlyWhatThePythonSideWrites:
    """The caps the worker reads and the caps a supervisor can set are one set.

    A property one side reads and the other never writes is invisible from
    both: the worker runs at its default, or the flag does nothing, and no
    record says so.
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

    def test_no_property_names_a_depth(self):
        assert "effect.snapshot.tiers" not in CollectionCaps().as_system_properties()
        assert "effect.snapshot.tiers" not in _effect_collection_caps(
            argparse.Namespace()
        )


class TestTheRandomSeat:
    """FR-021: both knobs travel to the worker like every other cap."""

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
