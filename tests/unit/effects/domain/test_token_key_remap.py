"""Resolving an old token key to its script (FR-151)."""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.domain.token_key_remap import (
    RemapCounts,
    TokenKeyRemapper,
    TokenScriptFacts,
    load_token_script_facts,
    parse_token_script,
    remap_record_dict,
)

GOBLIN = (
    "Name:Goblin Token\nManaCost:no cost\nColors:red\nTypes:Creature Goblin\nPT:1/1\nOracle:\n"
)
GOBLIN_HASTE = GOBLIN.replace("PT:1/1\n", "PT:1/1\nK:Haste\n")
WHITE_GOBLIN = GOBLIN.replace("Colors:red\n", "Colors:white\n")
FOOD = (
    "Name:Food Token\nManaCost:no cost\nTypes:Artifact Food\n"
    "A:AB$ GainLife | Cost$ 2 T Sac<1/CARDNAME/this token> | LifeAmount$ 3\nOracle:\n"
)
SOLDIER_11 = (
    "Name:Soldier Token\nManaCost:no cost\nColors:white\nTypes:Creature Soldier\n"
    "PT:1/1\nOracle:\n"
)
SOLDIER_22 = SOLDIER_11.replace("PT:1/1\n", "PT:2/2\n")
SPIRIT_X = (
    "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\nPT:*/*\nOracle:\n"
)
SPIRIT_11 = (
    "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\n"
    "PT:1/1\nK:Flying\nOracle:\n"
)
ANGEL = (
    "Name:Angel Token\nManaCost:no cost\nColors:white\nTypes:Creature Angel\nPT:4/4\nOracle:\n"
)
ANGEL_LEGENDARY = ANGEL.replace("Types:Creature Angel\n", "Types:Legendary Creature Angel\n")


def _key(trait_kind: str = "spell") -> dict:
    """A printed key of the given trait kind, on an arbitrary carrying script."""
    return {
        "script_file": "cardsfolder/s/some_token.txt",
        "face": 0,
        "trait_kind": trait_kind,
        "index_within_kind": 0,
    }


def test_parse_reads_name_colors_types_and_pt():
    facts = parse_token_script("r_1_1_goblin_haste", GOBLIN_HASTE)
    assert facts == TokenScriptFacts(
        stem="r_1_1_goblin_haste", name="goblin token", colors=frozenset({"R"}),
        core_types=frozenset({"creature"}), supertypes=frozenset(),
        subtypes=frozenset({"goblin"}), pt=("1", "1"),
    )
    food = parse_token_script("c_a_food_sac", FOOD)
    assert food.colors == frozenset() and food.pt is None
    assert food.core_types == {"artifact"} and food.subtypes == {"food"}


def test_parse_buckets_a_supertype_apart_from_the_core_types():
    facts = parse_token_script("w_4_4_angel_legendary", ANGEL_LEGENDARY)
    assert facts.supertypes == {"legendary"}
    assert facts.core_types == {"creature"} and facts.subtypes == {"angel"}


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


#: The stems every fixture below writes. "Goblin Token" is carried by two
#: scripts of different colours, so a red goblin entity settles at step 2;
#: the haste variant is written only by the fixture that tests refusal.
SCRIPTS = (
    ("r_1_1_goblin", GOBLIN),
    ("w_1_1_goblin", WHITE_GOBLIN),
    ("c_a_food_sac", FOOD),
    ("w_1_1_soldier", SOLDIER_11),
    ("w_2_2_soldier", SOLDIER_22),
    ("w_x_x_spirit", SPIRIT_X),
    ("w_1_1_spirit_flying", SPIRIT_11),
    ("w_4_4_angel", ANGEL),
    ("w_4_4_angel_legendary", ANGEL_LEGENDARY),
)


def _write_scripts(tmp_path: Path, *, haste: bool = False) -> None:
    for stem, text in SCRIPTS:
        (tmp_path / f"{stem}.txt").write_text(text, encoding="utf-8")
    if haste:
        (tmp_path / "r_1_1_goblin_haste.txt").write_text(GOBLIN_HASTE, encoding="utf-8")


@pytest.fixture
def remapper(tmp_path: Path) -> TokenKeyRemapper:
    _write_scripts(tmp_path)
    return TokenKeyRemapper(
        load_token_script_facts(tmp_path),
        converted_card_files=frozenset({"cardsfolder/f/food_chain.txt"}),
        # Every stem but w_2_2_soldier, which is what
        # test_a_stem_without_a_converted_sidecar_is_never_chosen turns on.
        token_sidecars=frozenset(
            stem for stem, _ in SCRIPTS if stem != "w_2_2_soldier"
        ),
    )


@pytest.fixture
def remapper_with_hasty_goblin(tmp_path: Path) -> TokenKeyRemapper:
    _write_scripts(tmp_path, haste=True)
    return TokenKeyRemapper(
        load_token_script_facts(tmp_path),
        converted_card_files=frozenset({"cardsfolder/f/food_chain.txt"}),
        token_sidecars=frozenset(
            [stem for stem, _ in SCRIPTS] + ["r_1_1_goblin_haste"],
        ),
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


def test_colors_types_and_pt_narrow_the_candidates(remapper):
    """Two white Soldier scripts of different sizes: the entity's P/T picks one."""
    soldier = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["soldier"], "pt": {"base": [1, 1]},
    }
    resolved = remapper.resolve("soldier token", entity=soldier, printed_keys=[_key()])
    assert resolved == "w_1_1_soldier"


def test_a_supertype_the_entity_lacks_rules_a_script_out(remapper):
    """Legendary is a supertype, not a subtype: the entity's own field decides."""
    plain = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["angel"], "pt": {"base": [4, 4]},
    }
    legendary = dict(plain, supertypes=["legendary"])
    assert remapper.resolve("angel token", entity=plain, printed_keys=[_key()]) == "w_4_4_angel"
    assert remapper.resolve(
        "angel token", entity=legendary, printed_keys=[_key()],
    ) == "w_4_4_angel_legendary"


def test_same_stat_scripts_differing_by_a_keyword_refuse(remapper_with_hasty_goblin):
    """A snapshot carries no keyword key at all, so K:Haste is invisible here."""
    goblin = {
        "colors": ["R"], "types": ["creature"], "subtypes": ["goblin"], "pt": {"base": [1, 1]},
    }
    resolved = remapper_with_hasty_goblin.resolve(
        "goblin token", entity=goblin, printed_keys=[_key()],
    )
    assert resolved is None


def test_a_star_pt_script_matches_any_size_and_keeps_the_name_ambiguous(remapper):
    """A PT of */* matches whatever the entity shows, so the Spirit pair never splits."""
    spirit = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["spirit"], "pt": {"base": [1, 1]},
    }
    assert remapper.resolve("spirit token", entity=spirit, printed_keys=[_key()]) is None


def test_a_stem_without_a_converted_sidecar_is_never_chosen(remapper):
    soldier = {
        "colors": ["W"], "types": ["creature"], "subtypes": ["soldier"], "pt": {"base": [2, 2]},
    }
    # w_2_2_soldier is the only candidate left, and this fixture holds no
    # sidecar for it, so the one survivor is unusable.
    assert remapper.resolve("soldier token", entity=soldier, printed_keys=[_key()]) is None


def test_a_mismatching_entity_resolves_nothing(remapper):
    elf = {"colors": ["G"], "types": ["creature"], "subtypes": ["elf"], "pt": {"base": [1, 1]}}
    assert remapper.resolve("goblin token", entity=elf, printed_keys=[_key()]) is None


def _prov_key(script_file: str, kind: str = "spell", index: int = 0) -> dict:
    """A provenance-key dict naming ``script_file``, for the raw-JSON tests below."""
    return {"script_file": script_file, "face": 0, "trait_kind": kind, "index_within_kind": index}


def _record(entities, ability=None, extra=None) -> dict:
    return {
        "record_id": "r1", "kind": "resolution", "ability": ability or [],
        "state": {"entities": entities, "players": []}, "payload": extra or {},
    }


def _goblin_entity() -> dict:
    # A token entity's printed list holds exactly one key -- (spell, 0), the
    # permanent's own cast spell. The snapshot builder walks spell abilities
    # only, so a K:/T:/S:/R: line produces no printed key at all.
    script_file = "cardsfolder/g/goblin_token.txt"
    return {
        "id": "E1", "name": "Goblin Token", "colors": ["R"], "types": ["creature"],
        "subtypes": ["goblin"], "pt": {"base": [1, 1]},
        "printed": [_prov_key(script_file, "spell")],
        "granted_attached": [], "granted_temporary": {"abilities": []},
    }


def test_entity_keys_are_rewritten_with_the_entitys_context(remapper):
    data = _record([_goblin_entity()])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    printed = data["state"]["entities"][0]["printed"]
    assert printed[0]["trait_kind"] == "spell"
    assert printed[0]["script_file"] == "tokenscripts/r_1_1_goblin.txt"
    assert counts.remapped == 1 and not counts.ambiguous


def test_the_acting_ability_reuses_the_entitys_resolution(remapper):
    ability = [_prov_key("cardsfolder/g/goblin_token.txt", "keyword")]
    data = _record([_goblin_entity()], ability=ability)
    remap_record_dict(data, remapper, RemapCounts())
    assert data["ability"][0]["script_file"] == "tokenscripts/r_1_1_goblin.txt"


def _elf_entity() -> dict:
    # Carries the same old key as the goblin above but never matches any
    # goblin script's colours/subtypes, so it resolves to nothing (mirrors
    # test_a_mismatching_entity_resolves_nothing's fixture).
    script_file = "cardsfolder/g/goblin_token.txt"
    return {
        "id": "E2", "name": "Elf Token", "colors": ["G"], "types": ["creature"],
        "subtypes": ["elf"], "pt": {"base": [1, 1]},
        "printed": [_prov_key(script_file, "spell")],
        "granted_attached": [], "granted_temporary": {"abilities": []},
    }


def test_entities_that_disagree_each_resolve_on_their_own_context(remapper):
    """Two entities sharing one old key are never conflated (review round 1)."""
    script_file = "cardsfolder/g/goblin_token.txt"
    data = _record([_goblin_entity(), _elf_entity()], ability=[_prov_key(script_file, "spell")])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    goblin_printed = data["state"]["entities"][0]["printed"]
    elf_printed = data["state"]["entities"][1]["printed"]
    assert goblin_printed[0]["script_file"] == "tokenscripts/r_1_1_goblin.txt"
    # The elf never matches any goblin script's colours/subtypes, so its own
    # key -- despite naming the same old file -- stays untouched.
    assert elf_printed[0]["script_file"] == script_file
    # The entities disagreed (one resolved, one didn't), so the acting key
    # is not trusted to follow either of them and stays untouched too.
    assert data["ability"][0]["script_file"] == script_file
    assert counts.remapped == 1
    # The elf's leftover key (final pass) and the acting key (visit_other)
    # are each counted where they were visited.
    assert counts.ambiguous == {"goblin_token": 2}


def test_entities_that_agree_let_the_acting_key_follow_them(remapper):
    """Two entities resolving to the same stem let the acting key reuse it."""
    script_file = "cardsfolder/g/goblin_token.txt"
    entity_a, entity_b = _goblin_entity(), _goblin_entity()
    entity_b["id"] = "E2"
    data = _record([entity_a, entity_b], ability=[_prov_key(script_file, "spell")])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    for entity in data["state"]["entities"]:
        for key in entity["printed"]:
            assert key["script_file"] == "tokenscripts/r_1_1_goblin.txt"
    # By name alone "goblin token" is ambiguous (test_an_ambiguous_name_needs_the_entity);
    # the acting key resolves anyway because both entities agreed.
    assert data["ability"][0]["script_file"] == "tokenscripts/r_1_1_goblin.txt"
    assert counts.remapped == 3 and not counts.ambiguous


def test_a_key_with_no_carrying_entity_resolves_by_name_only(remapper):
    data = _record([], ability=[_prov_key("cardsfolder/f/food_token.txt")])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    assert data["ability"][0]["script_file"] == "tokenscripts/c_a_food_sac.txt"
    data = _record([], ability=[_prov_key("cardsfolder/g/goblin_token.txt")])
    remap_record_dict(data, remapper, counts)
    assert data["ability"][0]["script_file"] == "cardsfolder/g/goblin_token.txt"
    assert counts.ambiguous == {"goblin_token": 1}


def test_keys_inside_payloads_are_reached(remapper):
    payload = {
        "candidates": [
            {"ability": [_prov_key("cardsfolder/f/food_token.txt")], "responsible_static": []},
        ],
    }
    data = _record([], extra=payload)
    remap_record_dict(data, remapper, RemapCounts())
    assert data["payload"]["candidates"][0]["ability"][0]["script_file"] == (
        "tokenscripts/c_a_food_sac.txt"
    )


def test_a_real_card_key_is_untouched(remapper):
    data = _record([], ability=[_prov_key("cardsfolder/f/food_chain.txt")])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    assert data["ability"][0]["script_file"] == "cardsfolder/f/food_chain.txt"
    assert counts.remapped == 0 and not counts.ambiguous


def test_counts_merge():
    a, b = RemapCounts(), RemapCounts()
    a.remapped, b.remapped = 2, 3
    a.ambiguous["x"] += 1
    b.ambiguous["x"] += 2
    a.merge(b)
    assert a.remapped == 5 and a.ambiguous == {"x": 3}
