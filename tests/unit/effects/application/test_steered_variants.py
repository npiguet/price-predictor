"""``collect-variants --round steered`` on a fixture catalogue and fixture
counts (T188; FR-058c, FR-058d)."""

from __future__ import annotations

import json
import random
from unittest.mock import MagicMock

import pytest

from effects.application import steered_variants
from effects.application.collect_variants import CollectVariantsConfig, decks_of
from effects.application.steered_variants import (
    SteeredGeneration,
    combined_counts,
    rule_carriers,
    script_index,
    steer,
)
from effects.domain.coverage import DeckCandidate
from effects.domain.provenance import ProvenanceKey
from effects.domain.rule_catalogue import Polarity
from effects.domain.variant_rounds import ShortPair, VariantGames, detected_rules
from effects.infrastructure.rules_catalogue_io import load_heads, load_scenarios

_SCRIPTS = {
    "s/servant_of_the_stinger.txt": [
        "Name:Servant of the Stinger", "ManaCost:1 B", "Types:Creature Human Warlock",
        "PT:1/3", "K:Deathtouch",
    ],
    "g/grizzly_bears.txt": ["Name:Grizzly Bears", "ManaCost:1 G", "Types:Creature Bear",
                            "PT:2/2"],
    "v/venom_slinger.txt": [
        "Name:Venom Slinger", "ManaCost:2 B", "Types:Creature Snake", "PT:1/1", "K:Deathtouch",
        "A:AB$ DealDamage | Cost$ T | ValidTgts$ Creature | NumDmg$ 1 | SpellDescription$ ping.",
    ],
    "h/held_card.txt": ["Name:Held Card", "ManaCost:B", "Types:Creature Rat", "PT:1/1",
                        "K:Deathtouch"],
    "t/two_faces.txt": ["Name:Front Face", "ManaCost:B", "Types:Creature Rat", "PT:1/1",
                        "K:Deathtouch", "ALTERNATE", "Name:Back Face", "Types:Creature Rat"],
}
_SLINGER_PING = ProvenanceKey("cardsfolder/v/venom_slinger.txt", 0, "spell", 1)
_SLINGER_TOUCH = ProvenanceKey("cardsfolder/v/venom_slinger.txt", 0, "keyword", 0)
_SCRIPT_TEXTS = {
    _SLINGER_PING: "AB$ DealDamage | Cost$ T | NumDmg$ 1 | SpellDescription$ ping. "
                   "| ValidTgts$ Creature",
    _SLINGER_TOUCH: "Deathtouch",
}


def _scenario(sid, rule, cards, variants):
    return {"id": sid, "rules": [rule], "record_kind": "combat", "cards": cards,
            "detector": {"family": "keyword_combat", "record_kind": "combat", "outcome": "o",
                         "near_miss": "n", "params": {"keyword": "deathtouch"}},
            "variants": [{"label": "base", "polarity": "fires", "delta": "", "expected": ""},
                         *variants]}


@pytest.fixture
def world(tmp_path):
    forge = tmp_path / "cardsfolder"
    for relative, lines in _SCRIPTS.items():
        (forge / relative).parent.mkdir(parents=True, exist_ok=True)
        (forge / relative).write_text("\n".join(lines) + "\n", encoding="utf-8")
    rules = tmp_path / "rules"
    (rules / "scenarios").mkdir(parents=True)
    (rules / "heads").mkdir()
    scenario = _scenario("702.2b-1", "702.2b",
                         ["Servant of the Stinger", "Grizzly Bears", "Held Card", "Front Face"], [
        {"label": "no_touch", "polarity": "does_not_fire", "delta": "d", "expected": "",
         "operation": "keyword", "script_edit": {
             "card": "Servant of the Stinger", "remove": ["K:Deathtouch"], "add": []}},
        {"label": "bear_touch", "polarity": "fires", "delta": "d", "expected": "",
         "operation": "keyword", "script_edit": {
             "card": "Grizzly Bears", "remove": [], "add": ["K:Deathtouch"]}},
        {"label": "held", "polarity": "does_not_fire", "delta": "d", "expected": "",
         "operation": "keyword", "script_edit": {
             "card": "Held Card", "remove": ["K:Deathtouch"], "add": []}},
    ])
    (rules / "scenarios" / "702.2.json").write_text(
        json.dumps({"unit": "702.2", "scenarios": [scenario]}), encoding="utf-8")
    (rules / "heads" / "702.2.jsonl").write_text(
        json.dumps({"id": "702.2b", "output": "trained", "outputs": ["zone_outcome"]}) + "\n",
        encoding="utf-8")
    counts = {"generation": "gen-3", "rules": {
        "702.2b": {"fires": {"forge_ai": 150}, "does_not_fire": {"forge_ai": 10}}},
        "carriers": {"702.2b": [
            {"script_file": "cardsfolder/v/venom_slinger.txt", "records": 9, "lines": [
                {"key": _SLINGER_PING.sidecar_dict(), "records": 9},
                {"key": _SLINGER_TOUCH.sidecar_dict(), "records": 9}]},
            {"script_file": "tokenscripts/b_1_1_rat_deathtouch.txt", "records": 3, "lines": []},
        ]}}
    catalogue = load_scenarios(rules)
    generation = SteeredGeneration(
        catalogue=catalogue, index=script_index(sorted(forge.rglob("*.txt"))),
        forge_cards_path=forge, held_out=frozenset({"held card"}),
        script_text_of=_SCRIPT_TEXTS.get, output_path=tmp_path / "variant-scripts",
        rng=random.Random(3),
    )
    return generation, counts, rules


_SHORT = ShortPair("702.2b", Polarity.DOES_NOT_FIRE, 10, 90)


def test_a_rule_s_carriers_are_its_scenario_cards_and_its_counted_cards(world):
    generation, counts, _ = world
    carriers = rule_carriers(
        "702.2b", generation.catalogue, counts, index=generation.index,
        forge_cards_path=generation.forge_cards_path, held_out=generation.held_out,
    )
    assert set(carriers) == {"Servant of the Stinger", "Grizzly Bears", "Venom Slinger"}
    assert carriers["Venom Slinger"].keys == {_SLINGER_PING, _SLINGER_TOUCH}


def test_a_carrier_takes_its_pair_s_script_edits_then_near_misses(world):
    generation, counts, _ = world
    made = generation.variants([_SHORT], counts, total=5)

    assert [v.perturbation for v in made][0] == "script_edit 702.2b-1/no_touch"
    servant = made[0]
    assert "K:Deathtouch" not in servant.path.read_text(encoding="utf-8").splitlines()
    assert servant.name.startswith("Servant of the Stinger Variant ")
    assert {v.rule for v in made} == {"702.2b"}
    near = made[1:]
    assert near and all(v.source_card == "Venom Slinger" for v in near)
    assert all(v.perturbation.startswith("near miss: ") for v in near)
    for variant in near:
        lines = variant.path.read_text(encoding="utf-8").splitlines()
        assert lines[-1] != _SCRIPTS["v/venom_slinger.txt"][-1]
        assert lines[1:-1] == _SCRIPTS["v/venom_slinger.txt"][1:-1]


def test_a_pair_s_edits_are_those_of_its_own_polarity(world):
    generation, counts, _ = world
    fires = ShortPair("702.2b", Polarity.FIRES, 0, 100)
    made = generation.pair_variants(fires, rule_carriers(
        "702.2b", generation.catalogue, counts, index=generation.index,
        forge_cards_path=generation.forge_cards_path, held_out=generation.held_out), 1)
    assert [v.perturbation for v in made] == ["script_edit 702.2b-1/bear_touch"]


def test_no_variant_comes_from_a_held_out_or_multi_faced_card(world):
    generation, counts, _ = world
    made = generation.variants([_SHORT], counts, total=50)
    assert not {v.source_card for v in made} & {"Held Card", "Front Face"}


def test_a_carrier_and_edit_yield_one_variant_in_the_whole_round(world):
    generation, counts, _ = world
    first = generation.variants([_SHORT], counts, total=5)
    second = generation.variants([_SHORT], counts, total=5)
    assert not {v.name for v in first} & {v.name for v in second}
    assert "script_edit 702.2b-1/no_touch" not in {v.perturbation for v in second}


def test_a_pair_takes_no_more_than_its_share(world):
    generation, counts, _ = world
    assert len(generation.variants([_SHORT], counts, total=2)) == 2


def test_a_steered_deck_holds_one_rule_s_variants_and_names_it():
    units = {
        "a1": VariantGames("a1", "variant-scripts/a1.txt", rule="702.2b"),
        "a2": VariantGames("a2", "variant-scripts/a2.txt", rule="702.2b"),
        "b1": VariantGames("b1", "variant-scripts/b1.txt", rule="704.5f"),
    }
    candidates = {name: DeckCandidate(name, "{B}") for name in units}
    decks, rules = decks_of({"a1": 1.0, "a2": 2.0, "b1": 1.0}, candidates, 30,
                            random.Random(1), units, by_rule=True)
    assert len(decks) == len(rules) == 30
    assert set(rules) == {"702.2b", "704.5f"}
    for deck, rule in zip(decks, rules, strict=True):
        assert {units[c].rule for c in deck if c in units} == {rule}


def test_the_corpus_counts_and_the_run_s_counts_add_up():
    total = combined_counts(
        {"rules": {"a": {"fires": {"forge_ai": 2}}}},
        {"rules": {"a": {"fires": {"forge_ai": 1, "aimed": 4}}, "b": {"fires": {"aimed": 1}}}},
    )
    assert total == {"rules": {"a": {"fires": {"forge_ai": 3, "aimed": 4}},
                               "b": {"fires": {"aimed": 1}}}}


class _Ruled:
    """The run's rule counts, one document per recount: the n-th ``update``
    makes the n-th document current."""

    def __init__(self, *documents):
        self._documents = [{"rules": {}}, *documents]
        self._at = 0

    def update(self):
        self._at = min(self._at + 1, len(self._documents) - 1)

    def document(self):
        return self._documents[self._at]


def _steer(world, monkeypatch, ruled, **config):
    generation, counts, rules = world
    played = []

    def play(units, candidates, supervisor, games, config, *, by_rule, rng):
        played.append((dict(units), by_rule))
        return False

    monkeypatch.setattr(steered_variants, "play_variant_rounds", play)
    monkeypatch.setattr(steered_variants, "prepare_variants", lambda config, variants: variants)
    config = CollectVariantsConfig(steer_variants=3, rules_dir=rules, **config)
    catalogue = load_scenarios(rules)
    units: dict = {}
    result = steer(config, generation, counts, load_heads(rules, catalogue.rules()),
                   detected_rules(catalogue), units, MagicMock(), MagicMock(), ruled)
    return result, played, units


def test_a_rule_whose_short_polarity_gained_nothing_retires(world, monkeypatch):
    played_any, played, _ = _steer(world, monkeypatch, _Ruled())
    assert played_any and len(played) == 1
    units, by_rule = played[0]
    assert by_rule and {unit.rule for unit in units.values()} == {"702.2b"}


def test_a_rule_that_gained_plays_again_with_new_variants_until_the_last_iteration(
    world, monkeypatch,
):
    gaining = [{"rules": {"702.2b": {"does_not_fire": {"aimed": n}}}} for n in (5, 10, 15)]
    _, played, units = _steer(world, monkeypatch, _Ruled(*gaining), steer_iterations=3)
    assert len(played) == 3
    names = [set(round_units) for round_units, _ in played]
    assert not (names[0] & names[1]) and not (names[1] & names[2])
    assert set(units) == names[0] | names[1] | names[2]


def test_the_round_ends_when_no_short_pair_is_left(world, monkeypatch):
    filled = {"rules": {"702.2b": {"does_not_fire": {"aimed": 90}}}}
    _, played, _ = _steer(world, monkeypatch, _Ruled(filled), steer_iterations=5)
    assert len(played) == 1


def test_a_corpus_with_no_short_pair_plays_nothing(world, monkeypatch):
    generation, counts, _ = world
    counts["rules"]["702.2b"]["does_not_fire"]["forge_ai"] = 500
    played_any, played, _ = _steer(world, monkeypatch, _Ruled())
    assert not played_any and played == []


def test_the_script_index_finds_every_face_s_name(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("Name:Front\nALTERNATE\nName:Back\n", encoding="utf-8")
    assert script_index([path]) == {"Front": path, "Back": path}


def test_collect_steered_unstages_however_it_ends(world, monkeypatch, tmp_path, fake_supervisor):
    """FR-058b: the round's staged variants leave Forge's custom-cards directory."""
    from effects.application import collect_variants

    _, counts, rules = world
    counts_path = tmp_path / "counts.json"
    counts_path.write_text(json.dumps(counts), encoding="utf-8")
    unstage = MagicMock(return_value=0)
    monkeypatch.setattr(steered_variants, "unstage_variants", unstage)
    monkeypatch.setattr(steered_variants, "steer", MagicMock(side_effect=RuntimeError("boom")))
    forge = tmp_path / "cardsfolder"
    config = CollectVariantsConfig(
        round=collect_variants.VariantRound.STEERED, rule_counts=counts_path, rules_dir=rules,
        forge_cards_path=forge, effect_records=tmp_path / "records",
        variant_scripts=tmp_path / "variant-scripts", cards_folders=(tmp_path / "none",),
    )
    with pytest.raises(RuntimeError):
        collect_variants.run(config)
    unstage.assert_called_once_with()


def test_the_steered_round_is_what_round_steered_runs(monkeypatch, tmp_path):
    from effects.application import collect_variants

    collect = MagicMock(return_value=0)
    monkeypatch.setattr(steered_variants, "collect_steered", collect)
    (tmp_path / "cards").mkdir()
    config = CollectVariantsConfig(round=collect_variants.VariantRound.STEERED,
                                   forge_cards_path=tmp_path / "cards")
    assert collect_variants.run(config) == 0
    collect.assert_called_once()

