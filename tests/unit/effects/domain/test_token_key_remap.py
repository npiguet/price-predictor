"""Resolving an old token key to its script (FR-151)."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.token_key_remap import (
    TokenKeyRemapper,
    TokenScriptFacts,
    load_token_script_facts,
    parse_token_script,
)

GOBLIN = (
    "Name:Goblin Token\nManaCost:no cost\nColors:red\nTypes:Creature Goblin\nPT:1/1\nOracle:\n"
)
GOBLIN_HASTE = GOBLIN.replace("PT:1/1\n", "PT:1/1\nK:Haste\n")
FOOD = (
    "Name:Food Token\nManaCost:no cost\nTypes:Artifact Food\n"
    "A:AB$ GainLife | Cost$ 2 T Sac<1/CARDNAME/this token> | LifeAmount$ 3\nOracle:\n"
)
SPIRIT_X = (
    "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\nPT:*/*\nOracle:\n"
)
SPIRIT_11 = (
    "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\n"
    "PT:1/1\nK:Flying\nOracle:\n"
)


def _key(trait_kind: str) -> dict:
    """A printed key of the given trait kind, on an arbitrary carrying script."""
    return {
        "script_file": "cardsfolder/s/some_token.txt",
        "face": 0,
        "trait_kind": trait_kind,
        "index_within_kind": 0,
    }


def test_parse_reads_name_colors_types_pt_and_trait_counts():
    facts = parse_token_script("r_1_1_goblin_haste", GOBLIN_HASTE)
    assert facts == TokenScriptFacts(
        stem="r_1_1_goblin_haste", name="goblin token", colors=frozenset({"R"}),
        core_types=frozenset({"creature"}), subtypes=frozenset({"goblin"}), pt=("1", "1"),
        trait_counts={"spell": 0, "trigger": 0, "static": 0, "replacement": 0, "keyword": 1},
    )
    food = parse_token_script("c_a_food_sac", FOOD)
    assert food.colors == frozenset() and food.pt is None
    assert food.core_types == {"artifact"} and food.subtypes == {"food"}
    assert food.trait_counts["spell"] == 1


def test_parse_returns_none_without_a_name():
    assert parse_token_script("x", "Types:Creature\n") is None


def test_load_groups_scripts_by_printed_name(tmp_path: Path):
    (tmp_path / "r_1_1_goblin.txt").write_text(GOBLIN, encoding="utf-8")
    (tmp_path / "r_1_1_goblin_haste.txt").write_text(GOBLIN_HASTE, encoding="utf-8")
    (tmp_path / "c_a_food_sac.txt").write_text(FOOD, encoding="utf-8")
    by_name = load_token_script_facts(tmp_path)
    assert sorted(f.stem for f in by_name["goblin token"]) == [
        "r_1_1_goblin", "r_1_1_goblin_haste",
    ]
    assert [f.stem for f in by_name["food token"]] == ["c_a_food_sac"]


def _write_scripts(tmp_path: Path) -> None:
    scripts = (
        ("r_1_1_goblin", GOBLIN),
        ("r_1_1_goblin_haste", GOBLIN_HASTE),
        ("c_a_food_sac", FOOD),
        ("w_x_x_spirit", SPIRIT_X),
        ("w_1_1_spirit_flying", SPIRIT_11),
    )
    for stem, text in scripts:
        (tmp_path / f"{stem}.txt").write_text(text, encoding="utf-8")


@pytest.fixture
def remapper(tmp_path: Path) -> TokenKeyRemapper:
    _write_scripts(tmp_path)
    return TokenKeyRemapper(
        load_token_script_facts(tmp_path),
        converted_card_files=frozenset({"cardsfolder/f/food_chain.txt"}),
        token_sidecars=frozenset({
            "r_1_1_goblin", "r_1_1_goblin_haste", "c_a_food_sac", "w_x_x_spirit",
        }),
    )


@pytest.fixture
def remapper_with_flying_spirit(tmp_path: Path) -> TokenKeyRemapper:
    _write_scripts(tmp_path)
    return TokenKeyRemapper(
        load_token_script_facts(tmp_path),
        converted_card_files=frozenset({"cardsfolder/f/food_chain.txt"}),
        token_sidecars=frozenset({
            "r_1_1_goblin", "r_1_1_goblin_haste", "c_a_food_sac",
            "w_x_x_spirit", "w_1_1_spirit_flying",
        }),
    )


def test_an_old_token_key_is_recognised_by_its_spaced_stem(remapper):
    assert remapper.is_old_token_key("cardsfolder/f/food_token.txt") == "food token"
    assert remapper.is_old_token_key("cardsfolder/f/food_chain.txt") is None      # a real card
    assert remapper.is_old_token_key("cardsfolder/m/marit_lage.txt") is None      # no such name
    assert remapper.is_old_token_key("tokenscripts/c_a_food_sac.txt") is None     # already right


def test_a_unique_name_resolves_without_context(remapper):
    assert remapper.resolve("food token") == "c_a_food_sac"


def test_an_ambiguous_name_needs_the_entity(remapper):
    assert remapper.resolve("goblin token") is None


def test_colors_types_and_pt_narrow_the_candidates(remapper_with_flying_spirit):
    spirit = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["spirit"], "pt": {"base": [1, 1]},
    }
    # 1/1 rules out w_x_x_spirit? No: a * side matches anything, so both survive on P/T
    # alone… and the keyword count settles it: the entity shows one keyword key, plus the
    # ever-present spell key for the permanent's own cast (see resolve()'s docstring).
    keys = [_key("spell"), _key("keyword")]
    resolved = remapper_with_flying_spirit.resolve("spirit token", entity=spirit, printed_keys=keys)
    assert resolved == "w_1_1_spirit_flying"


def test_trait_counts_split_same_stat_scripts(remapper):
    goblin = {
        "colors": ["R"], "types": ["creature"], "subtypes": ["goblin"], "pt": {"base": [1, 1]},
    }
    plain = remapper.resolve("goblin token", entity=goblin, printed_keys=[_key("spell")])
    hasty = remapper.resolve(
        "goblin token", entity=goblin, printed_keys=[_key("spell"), _key("keyword")],
    )
    assert (plain, hasty) == ("r_1_1_goblin", "r_1_1_goblin_haste")


def test_a_stem_without_a_converted_sidecar_is_never_chosen(remapper):
    spirit = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["spirit"], "pt": {"base": [1, 1]},
    }
    keys = [_key("spell"), _key("keyword")]
    # w_1_1_spirit_flying has no sidecar in this fixture, so the one candidate left is unusable.
    assert remapper.resolve("spirit token", entity=spirit, printed_keys=keys) is None


def test_a_mismatching_entity_resolves_nothing(remapper):
    elf = {"colors": ["G"], "types": ["creature"], "subtypes": ["elf"], "pt": {"base": [1, 1]}}
    assert remapper.resolve("goblin token", entity=elf, printed_keys=[_key("spell")]) is None
