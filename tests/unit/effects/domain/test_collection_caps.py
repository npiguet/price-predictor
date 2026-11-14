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
from dataclasses import fields
from pathlib import Path

import pytest

from effects.domain.collection_caps import (
    DEFAULT_DIRECTED_FORK_PROBABILITY,
    DEFAULT_DIRECTED_FORKS_PER_GAME,
    DEFAULT_INTERVENTIONS_PER_GAME,
    DEFAULT_MANA_CAP,
    DEFAULT_MAX_RECORDS_PER_GAME,
    DEFAULT_PLAIN_CANDIDATE_RATE,
    DEFAULT_PLAYABILITY_RATE,
    DEFAULT_PROBE_KEYWORDS,
    DEFAULT_PROBES_PER_GAME,
    DEFAULT_RANDOM_SEAT_PROBABILITY,
    DEFAULT_RANDOM_SEAT_SHARE,
    DEFAULT_ZONE_CANDIDATE_RATE,
    CollectionCaps,
)
from effects.domain.damage_step_keywords import KEYWORDS_BY_NAME
from sealed.infrastructure.cli import _effect_collection_caps
from sealed.infrastructure.effect_collection_flags import (
    DROPPED_EVENTS_SUFFIX,
    EFFECT_COUNT_DROPPED_EVENTS,
    EFFECT_DIRECTED_FORK_PROBABILITY,
    EFFECT_DIRECTED_FORK_RULES,
    EFFECT_DIRECTED_FORKS_PER_GAME,
    EFFECT_INTERVENTIONS_PER_GAME,
    EFFECT_MANA_CAP,
    EFFECT_MAX_RECORDS_PER_GAME,
    EFFECT_PLAIN_CANDIDATE_RATE,
    EFFECT_PLAYABILITY_RATE,
    EFFECT_PROBE_KEYWORDS,
    EFFECT_PROBES_PER_GAME,
    EFFECT_RANDOM_SEAT_PROBABILITY,
    EFFECT_RANDOM_SEAT_SHARE,
    EFFECT_ZONE_CANDIDATE_RATE,
    add_cap_flags,
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


def test_the_contract_defaults_not_pinned_beside_their_own_rule():
    """The caps whose literal default no class below pins: one mana record per
    key, two interventions and two probes a game, no random seat, no dropped
    event counts (features 023 and 025, ``contracts/cli.md``). The collectors'
    parsers are held to ``CollectionCaps``' defaults by
    ``test_cli_contract_collectors.py``."""
    caps = CollectionCaps()
    assert (caps.mana_cap, caps.interventions_per_game, caps.probes_per_game) == (1, 2, 2)
    assert (caps.random_seat_share, caps.random_seat_probability) == (0.0, 0.0)
    assert caps.count_dropped_events is False


class TestTheTwoSidesAgree:
    """A default that drifts between the two commands makes two corpora that
    mean different things and nothing says which is which."""

    def test_the_defaults_match(self):
        assert EFFECT_MANA_CAP == DEFAULT_MANA_CAP
        assert EFFECT_PLAYABILITY_RATE == DEFAULT_PLAYABILITY_RATE
        assert EFFECT_ZONE_CANDIDATE_RATE == DEFAULT_ZONE_CANDIDATE_RATE
        assert EFFECT_PLAIN_CANDIDATE_RATE == DEFAULT_PLAIN_CANDIDATE_RATE
        assert EFFECT_INTERVENTIONS_PER_GAME == DEFAULT_INTERVENTIONS_PER_GAME
        assert EFFECT_PROBES_PER_GAME == DEFAULT_PROBES_PER_GAME
        assert EFFECT_PROBE_KEYWORDS == CollectionCaps().probe_keywords
        assert EFFECT_RANDOM_SEAT_SHARE == DEFAULT_RANDOM_SEAT_SHARE
        assert EFFECT_RANDOM_SEAT_PROBABILITY == DEFAULT_RANDOM_SEAT_PROBABILITY
        assert EFFECT_COUNT_DROPPED_EVENTS == CollectionCaps().count_dropped_events
        assert EFFECT_DIRECTED_FORK_RULES == CollectionCaps().directed_fork_rules
        assert EFFECT_DIRECTED_FORKS_PER_GAME == DEFAULT_DIRECTED_FORKS_PER_GAME
        assert EFFECT_DIRECTED_FORK_PROBABILITY == DEFAULT_DIRECTED_FORK_PROBABILITY
        assert EFFECT_MAX_RECORDS_PER_GAME == DEFAULT_MAX_RECORDS_PER_GAME

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
        """Every property, ``effect.aimed.at`` included, which only an aimed
        run sends."""
        assert self._java_property_names() == set(
            CollectionCaps(aimed_at=()).as_system_properties()
        )


class TestTheAimedAtRules:
    """FR-009a: ``effect.aimed.at`` travels by its presence; the worker reads
    an unset property as a run aimed at nothing and a blank one as a run aimed
    at no rule in particular."""

    def test_a_run_aimed_at_nothing_sends_no_property(self):
        assert "effect.aimed.at" not in CollectionCaps().as_system_properties()

    def test_a_run_aimed_at_no_rule_sends_it_blank(self):
        assert CollectionCaps(aimed_at=()).as_system_properties()["effect.aimed.at"] == ""

    def test_the_rules_travel_comma_separated(self):
        properties = CollectionCaps(aimed_at=("704.5f", "702.19b")).as_system_properties()
        assert properties["effect.aimed.at"] == "704.5f,702.19b"


class TestTheDroppedEventCount:
    """FR-041: ``effect.count.dropped.events`` is spelled ``true``/``false`` on
    both sides, the spelling the worker's flag reader takes, and the file the
    workers append to is the one the supervisors sum."""

    def test_it_is_off_on_both_sides(self):
        assert CollectionCaps().as_system_properties()[
            "effect.count.dropped.events"
        ] == "false"
        assert _effect_collection_caps(argparse.Namespace())[
            "effect.count.dropped.events"
        ] == "false"

    def test_both_sides_spell_it_on_alike(self):
        assert CollectionCaps(count_dropped_events=True).as_system_properties()[
            "effect.count.dropped.events"
        ] == "true"
        assert _effect_collection_caps(argparse.Namespace(count_dropped_events=True))[
            "effect.count.dropped.events"
        ] == "true"

    @pytest.mark.skipif(
        not _JAVA_CAPS.exists(), reason="the connector source is not checked out"
    )
    def test_the_counts_file_suffix_is_the_jvms(self):
        source = (_JAVA_CAPS.parent / "DroppedEvents.java").read_text(encoding="utf-8")
        assert re.search(
            r'static final String SUFFIX = "([^"]+)";', source
        ).group(1) == DROPPED_EVENTS_SUFFIX


class TestDecisionSampling:
    """FR-033b: stage 1 at 1 in 100, a zone candidate written at 1 in 10,000
    and a plain one at 1 in 1,000, on both sides and in the JVM's defaults."""

    def test_the_defaults(self):
        assert DEFAULT_PLAYABILITY_RATE == 0.01
        assert DEFAULT_ZONE_CANDIDATE_RATE == 0.0001
        assert DEFAULT_PLAIN_CANDIDATE_RATE == 0.001

    @pytest.mark.skipif(
        not _JAVA_CAPS.exists(), reason="the connector source is not checked out"
    )
    def test_the_jvm_defaults_are_the_same(self):
        """A worker started without the properties samples as a supervisor
        that sends the defaults would."""
        source = _JAVA_CAPS.read_text(encoding="utf-8")
        defaults = re.search(
            r"return new CollectionCaps\(\s*\d+, ([0-9.]+), ([0-9.]+), ([0-9.]+),", source
        )
        assert tuple(float(v) for v in defaults.groups()) == (
            DEFAULT_PLAYABILITY_RATE, DEFAULT_ZONE_CANDIDATE_RATE,
            DEFAULT_PLAIN_CANDIDATE_RATE,
        )

    def test_the_rates_reach_their_properties(self):
        properties = CollectionCaps(
            playability_rate=0.5, zone_candidate_rate=0.125, plain_candidate_rate=0.25,
        ).as_system_properties()
        assert properties["effect.playability.rate"] == "0.5"
        assert properties["effect.zone.candidate.rate"] == "0.125"
        assert properties["effect.plain.candidate.rate"] == "0.25"


class TestThePerGameRecordCap:
    """FR-041a: ``--max-records-per-game`` defaults to 5000 on both sides
    and in the JVM, and travels as ``effect.max.records.per.game``."""

    def test_the_default_is_5000(self):
        assert DEFAULT_MAX_RECORDS_PER_GAME == 5000
        assert CollectionCaps().as_system_properties()[
            "effect.max.records.per.game"
        ] == "5000"
        assert _effect_collection_caps(argparse.Namespace())[
            "effect.max.records.per.game"
        ] == "5000"

    def test_the_flag_reaches_the_property_on_both_sides(self):
        parser = argparse.ArgumentParser()
        add_cap_flags(parser)
        args = parser.parse_args(["--max-records-per-game", "250"])
        assert args.max_records_per_game == 250
        assert _effect_collection_caps(args)["effect.max.records.per.game"] == "250"
        assert CollectionCaps(max_records_per_game=250).as_system_properties()[
            "effect.max.records.per.game"
        ] == "250"

    def test_a_cap_below_one_is_refused(self):
        parser = argparse.ArgumentParser()
        add_cap_flags(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--max-records-per-game", "0"])

    @pytest.mark.skipif(
        not _JAVA_CAPS.exists(), reason="the connector source is not checked out"
    )
    def test_the_jvm_default_is_the_same(self):
        source = _JAVA_CAPS.read_text(encoding="utf-8")
        assert re.search(
            r"DEFAULT_MAX_RECORDS_PER_GAME = (\d+);", source
        ).group(1) == str(DEFAULT_MAX_RECORDS_PER_GAME)


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


class TestTheDirectedForks:
    """FR-012: no rule, two a game and a quarter of matching boundaries by
    default, spelled alike on both sides and in the JVM's defaults."""

    def test_the_defaults(self):
        assert CollectionCaps().directed_fork_rules == ()
        assert DEFAULT_DIRECTED_FORKS_PER_GAME == 2
        assert DEFAULT_DIRECTED_FORK_PROBABILITY == 0.25

    def test_the_property_spellings(self):
        properties = CollectionCaps(
            directed_fork_rules=("704.5f", "702.19b"), directed_forks_per_game=3,
            directed_fork_probability=1.0,
        ).as_system_properties()
        assert properties["effect.directed.fork.rules"] == "704.5f,702.19b"
        assert properties["effect.directed.forks.per.game"] == "3"
        assert properties["effect.directed.fork.probability"] == "1.0"

    def test_no_rule_travels_blank_and_takes_no_fork(self):
        assert CollectionCaps().as_system_properties()["effect.directed.fork.rules"] == ""

    def test_the_sealed_supervisor_writes_the_same_properties(self):
        args = argparse.Namespace(
            directed_fork_rules=("704.5f", "702.19b"), directed_forks_per_game=3,
            directed_fork_probability=1.0,
        )
        assert {
            key: value for key, value in _effect_collection_caps(args).items()
            if key.startswith("effect.directed.")
        } == {
            "effect.directed.fork.rules": "704.5f,702.19b",
            "effect.directed.forks.per.game": "3",
            "effect.directed.fork.probability": "1.0",
        }

    @pytest.mark.skipif(
        not _JAVA_CAPS.exists(), reason="the connector source is not checked out"
    )
    def test_the_jvm_defaults_are_the_same(self):
        """A worker started without the properties takes none, at the same
        budget and probability a supervisor would send."""
        source = _JAVA_CAPS.read_text(encoding="utf-8")
        defaults = re.search(
            r"Optional\.empty\(\), false,\s*List\.of\(\), (\d+), ([0-9.]+),"
            r" DEFAULT_MAX_RECORDS_PER_GAME\);", source
        )
        assert defaults is not None
        assert int(defaults.group(1)) == DEFAULT_DIRECTED_FORKS_PER_GAME
        assert float(defaults.group(2)) == DEFAULT_DIRECTED_FORK_PROBABILITY


class TestTheJvmDefaultsAreThePythonDefaultsArgumentByArgument:
    """Every positional argument of the Java ``CollectionCaps.defaults()``,
    against the Python field of the same place: the classes above read some
    of them, which is how a divergence in the others passed."""

    @staticmethod
    def _java_defaults() -> dict[str, str]:
        source = _JAVA_CAPS.read_text(encoding="utf-8")
        components = re.search(r"public record CollectionCaps\((.*?)\) \{", source, re.S).group(1)
        names = [part.split()[-1] for part in re.sub(r"<[^<>]*(<[^<>]*>)?[^<>]*>", "", components)
                 .split(",")]
        body = re.search(
            r"static CollectionCaps defaults\(\) \{\s*return new CollectionCaps\((.*?)\);",
            source, re.S).group(1)
        values = [value.strip() for value in body.split(",")]
        assert len(names) == len(values) == len(fields(CollectionCaps))
        return dict(zip(names, values, strict=True))

    @staticmethod
    def _python(name: str):
        return getattr(CollectionCaps(), name)

    @staticmethod
    def _camel(name: str) -> str:
        head, *rest = name.split("_")
        return head + "".join(word.title() for word in rest)

    def test_the_components_are_the_python_fields_in_order(self):
        assert list(self._java_defaults()) == [self._camel(f.name) for f in fields(CollectionCaps)]

    @pytest.mark.parametrize("name", [f.name for f in fields(CollectionCaps)
                                      if f.name not in ("probe_keywords", "aimed_at")])
    def test_each_default_is_the_python_one(self, name):
        java = self._java_defaults()[self._camel(name)]
        python = self._python(name)
        if java == "DEFAULT_MAX_RECORDS_PER_GAME":
            java = re.search(r"DEFAULT_MAX_RECORDS_PER_GAME = (\d+);",
                             _JAVA_CAPS.read_text(encoding="utf-8")).group(1)
        if isinstance(python, bool):
            assert java == str(python).lower()
        elif isinstance(python, tuple):
            assert java == "List.of()" and python == ()
        else:
            assert type(python)(java) == python

    def test_an_unaimed_worker_reads_no_rule_as_aimed(self):
        """Python's ``None`` and Java's ``Optional.empty()``: no ``aimed_at``."""
        assert self._java_defaults()["aimedAt"] == "Optional.empty()"
        assert CollectionCaps().aimed_at is None

    def test_the_probe_keywords_default_is_the_python_one(self):
        """A worker started without ``effect.probe.keywords`` probes what a
        default run sends: the same keywords in the same order, which the
        JVM's probe rotation reads."""
        assert self._java_defaults()["probeKeywords"] == "DEFAULT_PROBE_KEYWORDS"
        listed = re.search(r"DEFAULT_PROBE_KEYWORDS = List\.of\((.*?)\);",
                           _JAVA_CAPS.read_text(encoding="utf-8"), re.S).group(1)
        java = [word.strip().strip('"') for word in listed.split(",")]
        assert java == CollectionCaps().probe_keywords.split(",")
        assert len(java) == 8
