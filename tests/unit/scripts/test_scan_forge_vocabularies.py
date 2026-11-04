"""``scan_forge_vocabularies.py`` and the module it writes (T112, root spec § 9.4, § 9.6)."""

from __future__ import annotations

import pytest
import scan_forge_vocabularies as sfv

from effects.domain.forge_vocabularies import KEYWORD_COUNTERS, NAMED_COUNTERS, SUBTYPES

_TYPE_LISTS = """\
[BasicTypes]
Plains
Island:Islands
[LandTypes]
Cave:Caves
[CreatureTypes]
Time Lord:Time Lords
Human:Humans
[PlanarTypes]
Bolas's Meditation Realm
"""

_COUNTER_ENUM = """\
public enum CounterEnumType implements CounterType {

    M1M1("-1/-1", "-1/-1", 255, 110, 106, CounterAiCategory.Negative),
    P1P1("+1/+1", "+1/+1", 96, 226, 23, CounterAiCategory.Positive),

    LOYALTY("LOYAL", 198, 198, 198, CounterAiCategory.Positive),

    ACQUIREDTASTE("acquired taste", "ATASTE", 139, 19, 69, CounterAiCategory.Positive),

    // Player Counters

    ENERGY("ENRGY", CounterAiCategory.Positive),

    ;

    CounterEnumType(final String counterOnCardDisplayName, CounterAiCategory aiCategory) {
        this(counterOnCardDisplayName, 255, 255, 255, aiCategory);
    }
}
"""

_KEYWORD_COUNTERS = """\
    static ImmutableList<String> keywordCounter = ImmutableList.of(
            "Flying", "First Strike", "Double Strike");
"""


def test_every_section_lists_subtypes_in_file_order():
    assert sfv.subtypes(_TYPE_LISTS) == (
        "plains", "island", "cave", "time_lord", "human", "bolas's_meditation_realm")


def test_a_counter_is_spelled_as_a_record_spells_it():
    assert sfv.named_counters(_COUNTER_ENUM) == (
        "M1M1", "P1P1", "LOYALTY", "ACQUIRED TASTE", "ENERGY")


def test_keyword_counters_are_in_the_package_s_spelling():
    assert sfv.keyword_counters(_KEYWORD_COUNTERS) == (
        "flying", "first_strike", "double_strike")


def test_the_written_module_reads_back(tmp_path):
    lists = {"SUBTYPES": ("human",), "NAMED_COUNTERS": ("P1P1",),
             "KEYWORD_COUNTERS": ("flying",)}
    target = tmp_path / "vocabularies.py"
    target.write_text(sfv.render(lists), encoding="utf-8")
    namespace: dict = {}
    exec(target.read_text(encoding="utf-8"), namespace)
    assert {name: namespace[name] for name in lists} == lists


def test_the_checked_in_lists_hold_what_the_input_relies_on():
    assert {"plains", "human", "aura", "equipment", "saga"} <= set(SUBTYPES)
    assert {"P1P1", "M1M1", "LOYALTY", "OIL"} <= set(NAMED_COUNTERS)
    assert len(KEYWORD_COUNTERS) == 15 and "first_strike" in KEYWORD_COUNTERS


_REAL = pytest.mark.skipif(not (sfv.FORGE / sfv.TYPE_LISTS).is_file(),
                           reason="no ../forge checkout")


@_REAL
def test_the_checked_in_module_is_the_scan_of_forge():
    """Rerun ``python scripts/scan_forge_vocabularies.py`` after a Forge merge."""
    assert sfv.OUTPUT.read_text(encoding="utf-8") == sfv.render(sfv.scan(sfv.FORGE))
