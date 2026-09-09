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
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_SNAPSHOT_TIERS,
    CollectionCaps,
    parse_snapshot_tiers,
)
from effects.infrastructure.cli import CAP_DEFAULTS
from effects.infrastructure.cli import announce_probe_state as effects_announce
from sealed.infrastructure.cli import (
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_LEGALITY_RATE,
    EFFECT_MANA_CAP,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBE_KEYWORDS,
    EFFECT_PROBES_PER_GAME,
    EFFECT_SNAPSHOT_TIERS,
    _effect_collection_caps,
)
from sealed.infrastructure.cli import _snapshot_tiers as sealed_snapshot_tiers
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
        assert EFFECT_LEGALITY_RATE == DEFAULT_LEGALITY_RATE
        assert EFFECT_SNAPSHOT_TIERS == DEFAULT_SNAPSHOT_TIERS

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
        assert "effect.snapshot.tiers" in properties
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

    They were not. ``effect.snapshot.tiers`` and ``effect.legality.rate`` were
    read by ``CollectionCaps.fromSystemProperties``, documented in
    ``MatchWorkerMain``, and settable by no command — so the tier-4 snapshot
    stage three needs took a code edit to request, and the legality sampler ran
    at a rate a javadoc called ``--legality-rate``, a flag that did not exist.
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


class TestTheSnapshotTierVector:
    """Stage three's hand-and-graveyard tier, reachable at last."""

    def test_it_defaults_to_the_three_stage_one_tiers(self):
        assert CollectionCaps().snapshot_tiers == "1,2,3"

    def test_tier_four_is_expressible(self):
        assert CollectionCaps(snapshot_tiers="1,2,3,4").snapshot_tiers == "1,2,3,4"

    def test_it_travels_as_the_comma_separated_string_the_jvm_parses(self):
        caps = CollectionCaps(snapshot_tiers="1,2,3,4")
        assert caps.as_system_properties()["effect.snapshot.tiers"] == "1,2,3,4"

    @pytest.mark.parametrize("text", ["1,2", "1,2,3", "1,2,3,4", " 1 , 2 "])
    def test_a_prefix_of_the_four_tiers_is_accepted(self, text):
        assert parse_snapshot_tiers(text)

    @pytest.mark.parametrize("text", ["1,3", "2,3", "1", "", "1,2,4", "3,2,1"])
    def test_anything_but_a_prefix_is_refused(self, text):
        """Tiers are cumulative, and the battlefield goes into every snapshot
        regardless — a vector omitting 2 would tell a reader the board was
        uncollected while the board sits in `entities`."""
        with pytest.raises(ValueError):
            parse_snapshot_tiers(text)

    def test_a_non_numeric_vector_is_refused_rather_than_silently_defaulted(self):
        """The JVM's own reader falls back on this, which is the one behaviour
        an operator cannot see."""
        with pytest.raises(ValueError):
            parse_snapshot_tiers("1,two,3")

    def test_the_caps_object_refuses_it_at_construction(self):
        with pytest.raises(ValueError):
            CollectionCaps(snapshot_tiers="1,3")

    @pytest.mark.parametrize("text", ["1,3", "1", "1,two,3"])
    def test_both_commands_refuse_the_same_vectors(self, text):
        """The rule is restated on the sealed side, which never imports this
        one, so the two spellings are pinned together here."""
        with pytest.raises(argparse.ArgumentTypeError):
            sealed_snapshot_tiers(text)
        with pytest.raises(ValueError):
            parse_snapshot_tiers(text)

    @pytest.mark.parametrize("text", ["1,2", "1,2,3", "1,2,3,4"])
    def test_both_commands_accept_the_same_vectors(self, text):
        assert sealed_snapshot_tiers(text) == text
        assert parse_snapshot_tiers(text)


class TestTheLegalityRate:
    def test_it_defaults_to_the_playability_rate(self):
        """Same number, different channel: the two subkinds arrive at very
        different volumes from the same priority pass, so the rates are
        separate knobs that happen to start equal."""
        assert CollectionCaps().legality_rate == DEFAULT_LEGALITY_RATE

    def test_it_reaches_the_property_the_worker_samples_from(self):
        properties = CollectionCaps(legality_rate=0.25).as_system_properties()
        assert properties["effect.legality.rate"] == "0.25"
