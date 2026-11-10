"""The variant rounds' rules (root spec § 8.15; FR-058a, FR-058c, FR-058d):
short pairs, the shortfall split, retirement, and which source lines a
sidecar line's chained text is made of."""

from __future__ import annotations

import pytest

from effects.domain.provenance import ProvenanceKey
from effects.domain.rule_catalogue import MapVerdict, Polarity, RuleMapEntry
from effects.domain.variant_rounds import (
    ShortPair,
    VariantGames,
    acting_lines,
    counted_carriers,
    gained,
    missing_games,
    rules_to_retire,
    short_pairs,
    variant_shares,
)

_COUNTS = {
    "generation": "gen-3",
    "rules": {
        "702.2b": {"fires": {"forge_ai": 150, "aimed": 30},
                   "does_not_fire": {"random_seat": 20, "forge_ai": 10}},
        "704.5f": {"fires": {"forge_ai": 99}, "does_not_fire": {"forge_ai": 400}},
        "613.1": {"fires": {"forge_ai": 1}},
    },
}
_HEADS = {
    "702.2b": RuleMapEntry("702.2b", MapVerdict.TRAINED),
    "704.5f": RuleMapEntry("704.5f", MapVerdict.TRAINED),
    "702.19b": RuleMapEntry("702.19b", MapVerdict.TRAINED),
    "613.1": RuleMapEntry("613.1", MapVerdict.RECORDED),
}
_DETECTED = frozenset({"702.2b", "704.5f", "702.19b", "613.1"})


def test_a_pair_short_of_the_floor_over_all_sources_is_short():
    pairs = short_pairs(_COUNTS, _HEADS, _DETECTED, floor=100)
    assert ShortPair("702.2b", Polarity.DOES_NOT_FIRE, 30, 70) in pairs
    assert ShortPair("704.5f", Polarity.FIRES, 99, 1) in pairs
    assert not any(p.rule == "702.2b" and p.polarity is Polarity.FIRES for p in pairs)


def test_a_rule_no_record_tags_is_short_in_both_polarities():
    pairs = short_pairs(_COUNTS, _HEADS, _DETECTED, floor=100)
    assert {(p.rule, p.polarity, p.shortfall) for p in pairs if p.rule == "702.19b"} == {
        ("702.19b", Polarity.FIRES, 100), ("702.19b", Polarity.DOES_NOT_FIRE, 100)}


def test_only_trained_rules_with_a_declared_detector_are_steered():
    pairs = short_pairs(_COUNTS, _HEADS, _DETECTED - {"704.5f"}, floor=100)
    assert {p.rule for p in pairs} == {"702.2b", "702.19b"}


def test_a_retired_rule_has_no_short_pair():
    pairs = short_pairs(_COUNTS, _HEADS, _DETECTED, floor=100, retired={"702.19b"})
    assert "702.19b" not in {p.rule for p in pairs}


def test_the_variants_split_by_shortfall_at_least_one_each():
    pairs = [ShortPair("a", Polarity.FIRES, 0, 300), ShortPair("b", Polarity.FIRES, 99, 1),
             ShortPair("c", Polarity.DOES_NOT_FIRE, 0, 699)]
    shares = variant_shares(pairs, 100)
    assert shares == {("a", Polarity.FIRES): 30, ("b", Polarity.FIRES): 1,
                      ("c", Polarity.DOES_NOT_FIRE): 69}


def test_a_rule_retires_when_no_short_polarity_gained_enough():
    pairs = [ShortPair("a", Polarity.FIRES, 0, 100), ShortPair("a", Polarity.DOES_NOT_FIRE, 0, 100),
             ShortPair("b", Polarity.FIRES, 10, 90)]
    after = {"rules": {"a": {"does_not_fire": {"aimed": 3}}, "b": {"fires": {"forge_ai": 10}}}}
    gains = gained({"rules": {"b": {"fires": {"forge_ai": 10}}}}, after, pairs)
    assert gains == {"a": 3, "b": 0}
    assert rules_to_retire(gains, min_gain=1) == {"b"}
    assert rules_to_retire(gains, min_gain=4) == {"a", "b"}


def test_a_variant_s_deck_weight_is_the_games_it_lacks():
    units = {
        "x": VariantGames("x", "variant-scripts/x.txt", games={"g1"}),
        "y": VariantGames("y", "variant-scripts/y.txt", games={"g1", "g2"}),
        "z": VariantGames("z", "variant-scripts/z.txt", retired=True),
    }
    assert missing_games(units, 2) == {"x": 1.0}


_SERVANT = [
    "Name:Servant of the Stinger",
    "ManaCost:1 B",
    "Types:Creature Human Warlock",
    "PT:1/3",
    "K:Deathtouch",
    "T:Mode$ DamageDone | ValidSource$ Card.Self | Execute$ TrigSearch | CombatDamage$ True "
    "| ValidTarget$ Player | TriggerZones$ Battlefield | TriggerDescription$ Whenever it hits.",
    "SVar:TrigSearch:AB$ ChangeZone | Cost$ Sac<1/CARDNAME> | Origin$ Library "
    "| Destination$ Hand | ChangeType$ Card | Mandatory$ True",
    "DeckHas:Ability$Sacrifice",
]


@pytest.mark.parametrize("script_text, lines", [
    ("Deathtouch", [4]),
    ("CombatDamage$ True | Execute$ SV1 | Mode$ DamageDone | TriggerDescription$ Whenever "
     "it hits. | TriggerZones$ Battlefield | ValidSource$ Card.Self | ValidTarget$ Player "
     "[SEG] SV1: AB$ ChangeZone | ChangeType$ Card | Cost$ Sac<1/CARDNAME> | Destination$ "
     "Hand | Mandatory$ True | Origin$ Library", [5, 6]),
    ("SP$ DealDamage | NumDmg$ 3", []),
])
def test_a_sidecar_line_s_chain_is_found_among_the_source_lines(script_text, lines):
    """The sidecar sorts a segment's parameters and relabels its chain
    (``SV1``); the source line is found by its parameters all the same."""
    assert acting_lines(_SERVANT, script_text) == lines


def test_the_counts_carriers_read_back_as_keys_on_their_script():
    counts = {"carriers": {"702.2b": [{
        "script_file": "cardsfolder/s/servant_of_the_stinger.txt", "records": 4,
        "lines": [{"key": {"face": 0, "trait_kind": "keyword", "index_within_kind": 0},
                   "records": 4}],
    }]}}
    (carrier,) = counted_carriers(counts, "702.2b")
    assert carrier.keys == (ProvenanceKey(
        "cardsfolder/s/servant_of_the_stinger.txt", 0, "keyword", 0),)
    assert counted_carriers(counts, "704.5f") == []
