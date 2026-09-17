# Token Key Remap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the token abilities the existing raw corpus lost to the collector's old mis-key (`cardsfolder/f/food_token.txt` for a token that lives at `tokenscripts/c_a_food_sac.txt`) by remapping those keys in `build-corpus`, and make future collections write the right key and the right token id in the first place.

**Outcome (2026-09-17, after the whole-plan review):** The remap corrects a contentless key rather than recovering lost ability text, and the rebuild was cancelled. Every old token key in the corpus is `(spell, 0)` — the permanent's own cast-spell trait — which is a `dropped_keys` entry in 835 of 839 token sidecars, so it maps to no text before or after the rewrite; the token's ability lines were already keyed under `tokenscripts/<stem>.txt` by the old collector and resolve as they are. The trait-count step was also dropped: a token entity carries no `K:`/`T:`/`S:`/`R:` key at all, so the count preferred whichever script had no such lines and named the wrong script in about 7.8% of its resolutions. Same-stat scripts differing only by a keyword now stay ambiguous. What remains is worth keeping as an archival correction — a key naming a real file rather than one no tree holds — and FR-150's collector fix still matters for future collections.

**Architecture:** A pure domain module resolves an old token key to a token script by printed name, then by the entity's colours, types, supertypes and P/T, and refuses to guess when candidates remain. `build-corpus` applies it to every record's raw JSON in both passes before anything else reads the record, so the survey's keys, the rarity table, the caps and the written shards all see the corrected keys; the manifest records how many keys were remapped and which stems stayed ambiguous. On the collector side the snapshot's `token_script_id` becomes the script stem instead of the printed name, and the fat JAR is rebuilt so the next collection uses both fixes.

**Tech Stack:** Python 3.12 (`effects` package), pytest; Java 17 / Maven for `forge-connector`.

**Spec:** `docs/superpowers/specs/2026-09-16-corpus-and-trainer-rework.md` (finding 8, FR-150) and this plan's own contract below. Amends spec `specs/023-ability-effect-model/spec.md` with FR-151 (remap) and FR-078 (`token_script_id` is the script stem).

## Global Constraints

- Work on branch `corpus-rework` in the worktree `.worktrees/corpus-rework`; tests with `PYTHONPATH=src python -m pytest ...`; Maven with `-Dforge.dir=C:/Users/nicol/IdeaProjects/forge`.
- **A training run is reading `output/effects/corpus/` right now.** Nothing in this plan may write to that directory. The rebuilt corpus goes to `output/effects/corpus-remapped/` (`--output`), and the swap is a separate, later decision.
- The remap never invents a file: a key is rewritten only to `tokenscripts/<stem>.txt` where `<stem>` has a sidecar under `output/tokenscripts/`. Ambiguity leaves the key untouched and is counted.
- Two builds of one raw corpus at one seed still produce identical manifests.
- Every manifest field added has a `.get` default in `CorpusManifest.from_dict`.
- Before editing anything under `specs/`, load the repo's `feature-workflow` rules (`.claude/skills/feature-workflow/SKILL.md`).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## The contract (what the remap does)

An **old token key** is a provenance key whose `script_file` is not a file the converted card tree holds and whose filename stem, with underscores turned into spaces, equals the `Name:` of at least one Forge token script (`Food Token` → `food_token`). The old collector derived such keys from `card.getName()` for tokens rebuilt in forked games.

Resolution, in order, stops at the first step that leaves exactly one candidate:

1. All token scripts whose `Name:` equals the stem's spaced name.
2. Those whose `Colors:`, core `Types:`, subtypes and `PT:` match the entity carrying the key (an entity in `state.entities` whose `printed` list holds the key). A `*` in either P/T side matches anything; an entity with no `pt` matches a script with no `PT:`.
3. Those whose count of scripted lines per trait kind (`A:`→spell, `T:`→trigger, `S:`→static, `R:`→replacement, `K:`→keyword) equals the entity's count of printed keys per `trait_kind`.

A key that reaches no single candidate stays as it is. A key that resolves is rewritten to `tokenscripts/<stem>.txt` with `face`, `trait_kind` and `index_within_kind` unchanged (the forked copy's trait order equals the script's, proven by `CopiedTokenProvenanceTest`). Keys outside `state.entities` (the record's acting `ability`, playability candidates, `responsible_static`, `replaced_by`) reuse the resolution of the entity that carried the same old key in this record; without one they resolve by name alone (step 1) or stay.

---

## File map

| File | Responsibility |
|---|---|
| Create `src/effects/domain/token_key_remap.py` | `TokenScriptFacts`, `parse_token_script`, `TokenKeyRemapper`, `remap_record_dict`, `RemapCounts` |
| Modify `src/effects/domain/corpus_manifest.py` | `token_keys_remapped`, `token_keys_ambiguous`, `forge_tokenscripts` |
| Modify `src/effects/application/build_corpus.py` | build the remapper, read records through it in both passes, count, log, manifest |
| Modify `src/effects/infrastructure/cli.py` | `--forge-tokenscripts`, `--no-remap-token-keys` |
| Modify `forge-connector/.../SnapshotBuilder.java`, `ProvenanceKey.java` | `token_script_id` = script stem |
| Modify `specs/023-ability-effect-model/spec.md`, `quickstart.md`, `src/effects/CLAUDE.md` | FR-151, FR-078 wording, schema note |
| Tests | `tests/unit/effects/domain/test_token_key_remap.py`, additions to `test_build_corpus.py`, `test_corpus_manifest.py`; Java `SnapshotBuilderTest` or `CopiedTokenProvenanceTest` |

---

### Task 1: Token script facts and the resolver

**Files:**
- Create: `src/effects/domain/token_key_remap.py`
- Test: `tests/unit/effects/domain/test_token_key_remap.py`

**Interfaces:**
- `TokenScriptFacts(stem: str, name: str, colors: frozenset[str], core_types: frozenset[str], subtypes: frozenset[str], pt: tuple[str, str] | None, trait_counts: dict[str, int])` — frozen dataclass. `colors` are single letters `W U B R G`; `core_types` and `subtypes` lowercase; `pt` is the two sides of `PT:` as strings (`("1","1")`, `("*","*")`) or `None`; `trait_counts` keys are `spell trigger static replacement keyword`.
- `parse_token_script(stem: str, text: str) -> TokenScriptFacts | None` — `None` when there is no `Name:` line.
- `load_token_script_facts(directory: Path) -> dict[str, list[TokenScriptFacts]]` — printed name (lowercase) → facts, over every `*.txt` in the raw Forge tokenscripts directory.
- `class TokenKeyRemapper` with `__init__(self, facts_by_name, *, converted_card_files: frozenset[str], token_sidecars: frozenset[str])` where `converted_card_files` are the `cardsfolder/...` script files the converted card tree holds (values of `load_card_files`) and `token_sidecars` the stems with a sidecar under `output/tokenscripts/`. Methods: `is_old_token_key(script_file) -> str | None` (the spaced name when the key is a remap candidate, else `None`); `resolve(name, *, entity=None, printed_keys=()) -> str | None` (the stem, or `None` when ambiguous or unknown), where `entity` is the entity's raw dict (`colors`, `types`, `subtypes`, `pt`) and `printed_keys` the entity's raw key dicts.
- `COLOR_WORDS = {"white": "W", "blue": "U", "black": "B", "red": "R", "green": "G"}`; `TRAIT_LINES = {"A": "spell", "T": "trigger", "S": "static", "R": "replacement", "K": "keyword"}`.

- [ ] **Step 1: Write the failing tests**

```python
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

GOBLIN = "Name:Goblin Token\nManaCost:no cost\nColors:red\nTypes:Creature Goblin\nPT:1/1\nOracle:\n"
GOBLIN_HASTE = GOBLIN.replace("PT:1/1\n", "PT:1/1\nK:Haste\n")
FOOD = "Name:Food Token\nManaCost:no cost\nTypes:Artifact Food\nA:AB$ GainLife | Cost$ 2 T Sac<1/CARDNAME/this token> | LifeAmount$ 3\nOracle:\n"
SPIRIT_X = "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\nPT:*/*\nOracle:\n"
SPIRIT_11 = "Name:Spirit Token\nManaCost:no cost\nColors:white\nTypes:Creature Spirit\nPT:1/1\nK:Flying\nOracle:\n"


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
    assert sorted(f.stem for f in by_name["goblin token"]) == ["r_1_1_goblin", "r_1_1_goblin_haste"]
    assert [f.stem for f in by_name["food token"]] == ["c_a_food_sac"]


@pytest.fixture
def remapper(tmp_path: Path) -> TokenKeyRemapper:
    for stem, text in (("r_1_1_goblin", GOBLIN), ("r_1_1_goblin_haste", GOBLIN_HASTE),
                       ("c_a_food_sac", FOOD), ("w_x_x_spirit", SPIRIT_X), ("w_1_1_spirit_flying", SPIRIT_11)):
        (tmp_path / f"{stem}.txt").write_text(text, encoding="utf-8")
    return TokenKeyRemapper(
        load_token_script_facts(tmp_path),
        converted_card_files=frozenset({"cardsfolder/f/food_chain.txt"}),
        token_sidecars=frozenset({"r_1_1_goblin", "r_1_1_goblin_haste", "c_a_food_sac", "w_x_x_spirit"}),
    )


def test_an_old_token_key_is_recognised_by_its_spaced_stem(remapper):
    assert remapper.is_old_token_key("cardsfolder/f/food_token.txt") == "food token"
    assert remapper.is_old_token_key("cardsfolder/f/food_chain.txt") is None      # a real card
    assert remapper.is_old_token_key("cardsfolder/m/marit_lage.txt") is None      # no such token name
    assert remapper.is_old_token_key("tokenscripts/c_a_food_sac.txt") is None     # already right


def test_a_unique_name_resolves_without_context(remapper):
    assert remapper.resolve("food token") == "c_a_food_sac"


def test_an_ambiguous_name_needs_the_entity(remapper):
    assert remapper.resolve("goblin token") is None


def test_colors_types_and_pt_narrow_the_candidates(remapper):
    spirit = {"colors": ["W"], "types": ["creature"], "subtypes": ["spirit"], "pt": {"base": [1, 1]}}
    # 1/1 rules out w_x_x_spirit? No: a * side matches anything, so both survive on P/T alone…
    # …and the keyword count settles it: the entity shows one keyword key.
    keys = [{"script_file": "cardsfolder/s/spirit_token.txt", "face": 0, "trait_kind": "keyword", "index_within_kind": 0}]
    assert remapper.resolve("spirit token", entity=spirit, printed_keys=keys) == "w_1_1_spirit_flying"


def test_trait_counts_split_same_stat_scripts(remapper):
    goblin = {"colors": ["R"], "types": ["creature"], "subtypes": ["goblin"], "pt": {"base": [1, 1]}}
    plain = remapper.resolve("goblin token", entity=goblin, printed_keys=[])
    hasty = remapper.resolve("goblin token", entity=goblin, printed_keys=[
        {"script_file": "cardsfolder/g/goblin_token.txt", "face": 0, "trait_kind": "keyword", "index_within_kind": 0}])
    assert (plain, hasty) == ("r_1_1_goblin", "r_1_1_goblin_haste")


def test_a_stem_without_a_converted_sidecar_is_never_chosen(remapper):
    spirit = {"colors": ["W"], "types": ["creature"], "subtypes": ["spirit"], "pt": {"base": [1, 1]}}
    keys = [{"script_file": "cardsfolder/s/spirit_token.txt", "face": 0, "trait_kind": "keyword", "index_within_kind": 0}]
    # w_1_1_spirit_flying has no sidecar in this fixture, so the one candidate left is unusable.
    assert remapper.resolve("spirit token", entity=spirit, printed_keys=keys) is None


def test_a_mismatching_entity_resolves_nothing(remapper):
    elf = {"colors": ["G"], "types": ["creature"], "subtypes": ["elf"], "pt": {"base": [1, 1]}}
    assert remapper.resolve("goblin token", entity=elf, printed_keys=[]) is None
```

Note the fixture in `test_a_stem_without_a_converted_sidecar_is_never_chosen` deliberately omits `w_1_1_spirit_flying` from `token_sidecars`, which makes the earlier `test_colors_types_and_pt_narrow_the_candidates` fail as written. Give that earlier test its own remapper with the sidecar present (a second fixture `remapper_with_flying_spirit`) so both assertions hold.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects/domain/test_token_key_remap.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the module**

```python
"""Resolving the old collector's token keys to the token scripts they meant (FR-151).

Tokens rebuilt by Forge's ``GameCopier`` in forked games carried no paper
card, so the collector keyed their abilities to a path derived from the
printed name — ``cardsfolder/f/food_token.txt`` — which no tree holds. The
collector is fixed (FR-150), but every shard collected before the fix still
carries those keys. This module turns them back into ``tokenscripts/<stem>.txt``
where that can be done honestly: by name when one script carries the name,
else by the entity's colours, types and P/T, else by how many printed keys of
each trait kind the entity shows against how many lines the script declares.
Anything still ambiguous stays as it is and is counted, never guessed.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

COLOR_WORDS: dict[str, str] = {
    "white": "W", "blue": "U", "black": "B", "red": "R", "green": "G",
}
TRAIT_LINES: dict[str, str] = {
    "A": "spell", "T": "trigger", "S": "static", "R": "replacement", "K": "keyword",
}
TRAIT_KINDS: tuple[str, ...] = ("spell", "trigger", "static", "replacement", "keyword")
#: Core card types; anything else on a ``Types:`` line is a subtype.
CORE_TYPES: frozenset[str] = frozenset({
    "artifact", "creature", "enchantment", "instant", "land", "planeswalker",
    "sorcery", "battle", "kindred", "tribal", "legendary", "basic", "snow", "token",
})
_OLD_KEY = re.compile(r"^cardsfolder/[^/]+/([a-z0-9_'-]+)\.txt$")


@dataclass(frozen=True, slots=True)
class TokenScriptFacts:
    stem: str
    name: str
    colors: frozenset[str]
    core_types: frozenset[str]
    subtypes: frozenset[str]
    pt: tuple[str, str] | None
    trait_counts: dict[str, int]


def parse_token_script(stem: str, text: str) -> TokenScriptFacts | None:
    """The facts a Forge token script declares, or None without a ``Name:``."""
    name = None
    colors: set[str] = set()
    core: set[str] = set()
    subs: set[str] = set()
    pt: tuple[str, str] | None = None
    counts = {kind: 0 for kind in TRAIT_KINDS}
    for raw in text.splitlines():
        key, sep, value = raw.partition(":")
        if not sep:
            continue
        value = value.strip()
        if key == "Name":
            name = value.lower()
        elif key == "Colors":
            for word in re.split(r"[,\s]+", value.lower()):
                if word in COLOR_WORDS:
                    colors.add(COLOR_WORDS[word])
        elif key == "Types":
            for word in value.lower().split():
                (core if word in CORE_TYPES else subs).add(word)
        elif key == "PT":
            left, _, right = value.partition("/")
            pt = (left.strip(), right.strip())
        elif key in TRAIT_LINES:
            counts[TRAIT_LINES[key]] += 1
    if name is None:
        return None
    return TokenScriptFacts(
        stem=stem, name=name, colors=frozenset(colors), core_types=frozenset(core),
        subtypes=frozenset(subs), pt=pt, trait_counts=counts,
    )


def load_token_script_facts(directory: Path) -> dict[str, list[TokenScriptFacts]]:
    """Printed name (lowercase) -> the token scripts that carry it."""
    out: dict[str, list[TokenScriptFacts]] = {}
    for path in sorted(Path(directory).glob("*.txt")):
        facts = parse_token_script(path.stem, path.read_text(encoding="utf-8", errors="replace"))
        if facts is not None:
            out.setdefault(facts.name, []).append(facts)
    return out


def _pt_matches(script: tuple[str, str] | None, entity_pt) -> bool:
    if script is None:
        return entity_pt is None
    if entity_pt is None:
        return False
    base = entity_pt.get("base") if isinstance(entity_pt, Mapping) else None
    if not base or len(base) != 2:
        return True
    for side, want in zip(script, base):
        if side in ("*", "X") or not side.lstrip("+-").isdigit():
            continue
        if int(side) != int(want):
            return False
    return True


class TokenKeyRemapper:
    """Resolves old token keys; safe to pickle into a worker process."""

    def __init__(
        self,
        facts_by_name: Mapping[str, Sequence[TokenScriptFacts]],
        *,
        converted_card_files: frozenset[str],
        token_sidecars: frozenset[str],
    ) -> None:
        self.facts_by_name = {name: tuple(f) for name, f in facts_by_name.items()}
        self.converted_card_files = converted_card_files
        self.token_sidecars = token_sidecars

    def is_old_token_key(self, script_file: str) -> str | None:
        """The spaced token name when ``script_file`` is a remap candidate."""
        match = _OLD_KEY.match(script_file)
        if match is None or script_file in self.converted_card_files:
            return None
        name = match.group(1).replace("_", " ")
        return name if name in self.facts_by_name else None

    def resolve(self, name: str, *, entity=None, printed_keys: Iterable[Mapping] = ()) -> str | None:
        """The one stem ``name`` can mean here, or None."""
        candidates = [f for f in self.facts_by_name.get(name, ()) if f.stem in self.token_sidecars]
        if len(candidates) == 1:
            return candidates[0].stem
        if entity is None or not candidates:
            return None
        colors = frozenset(entity.get("colors") or ())
        types = {t.lower() for t in entity.get("types") or ()}
        subtypes = {t.lower() for t in entity.get("subtypes") or ()}
        pt = entity.get("pt")
        candidates = [
            f for f in candidates
            if f.colors == colors and f.core_types <= types | {"token"} and f.core_types >= types - {"token"}
            and f.subtypes == subtypes and _pt_matches(f.pt, pt)
        ]
        if len(candidates) == 1:
            return candidates[0].stem
        if not candidates:
            return None
        seen = Counter(k.get("trait_kind") for k in printed_keys)
        counts = {kind: seen.get(kind, 0) for kind in TRAIT_KINDS}
        candidates = [f for f in candidates if f.trait_counts == counts]
        return candidates[0].stem if len(candidates) == 1 else None
```

The `core_types` comparison tolerates the entity listing the pseudo-type `token`; if the corpus's entities never carry it, simplify to `f.core_types == types`. Check one real entity in `output/effects/corpus/training/shard-00001.jsonl.gz` during implementation and say which in the report.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects/domain/test_token_key_remap.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/token_key_remap.py tests/unit/effects/domain/test_token_key_remap.py
git commit -m "feat(effects): resolve the old collector's token keys to token scripts (FR-151)"
```

---

### Task 2: Remapping one record's raw JSON

**Files:**
- Modify: `src/effects/domain/token_key_remap.py` (append)
- Test: `tests/unit/effects/domain/test_token_key_remap.py` (append)

**Interfaces:**
- `RemapCounts` — mutable dataclass: `remapped: int = 0`, `ambiguous: Counter[str]` (old stem → keys left), `merge(other)`.
- `remap_record_dict(data: dict, remapper: TokenKeyRemapper, counts: RemapCounts) -> None` — rewrites in place every provenance-key dict (`{"script_file","face","trait_kind","index_within_kind"}`) whose `script_file` is an old token key, following the contract; entities first (each with its own context), then every other key list in the record with the per-record memo, then name-only.

- [ ] **Step 1: Write the failing tests**

```python
from effects.domain.token_key_remap import RemapCounts, remap_record_dict


def _key(script_file, kind="spell", index=0):
    return {"script_file": script_file, "face": 0, "trait_kind": kind, "index_within_kind": index}


def _record(entities, ability=None, extra=None):
    data = {"record_id": "r1", "kind": "resolution", "ability": ability or [],
            "state": {"entities": entities, "players": []}, "payload": extra or {}}
    return data


def test_entity_keys_are_rewritten_with_the_entitys_context(remapper):
    goblin = {"id": "E1", "name": "Goblin Token", "colors": ["R"], "types": ["creature"], "subtypes": ["goblin"],
              "pt": {"base": [1, 1]}, "printed": [_key("cardsfolder/g/goblin_token.txt", "keyword")],
              "granted_attached": [], "granted_temporary": {"abilities": []}}
    data = _record([goblin])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    assert data["state"]["entities"][0]["printed"][0]["script_file"] == "tokenscripts/r_1_1_goblin_haste.txt"
    assert data["state"]["entities"][0]["printed"][0]["trait_kind"] == "keyword"
    assert counts.remapped == 1 and not counts.ambiguous


def test_the_acting_ability_reuses_the_entitys_resolution(remapper):
    goblin = {"id": "E1", "name": "Goblin Token", "colors": ["R"], "types": ["creature"], "subtypes": ["goblin"],
              "pt": {"base": [1, 1]}, "printed": [_key("cardsfolder/g/goblin_token.txt", "keyword")],
              "granted_attached": [], "granted_temporary": {"abilities": []}}
    data = _record([goblin], ability=[_key("cardsfolder/g/goblin_token.txt", "keyword")])
    remap_record_dict(data, remapper, RemapCounts())
    assert data["ability"][0]["script_file"] == "tokenscripts/r_1_1_goblin_haste.txt"


def test_a_key_with_no_carrying_entity_resolves_by_name_only(remapper):
    data = _record([], ability=[_key("cardsfolder/f/food_token.txt")])
    counts = RemapCounts()
    remap_record_dict(data, remapper, counts)
    assert data["ability"][0]["script_file"] == "tokenscripts/c_a_food_sac.txt"
    data = _record([], ability=[_key("cardsfolder/g/goblin_token.txt")])
    remap_record_dict(data, remapper, counts)
    assert data["ability"][0]["script_file"] == "cardsfolder/g/goblin_token.txt"
    assert counts.ambiguous == {"goblin_token": 1}


def test_keys_inside_payloads_are_reached(remapper):
    payload = {"candidates": [{"ability": [_key("cardsfolder/f/food_token.txt")], "responsible_static": []}]}
    data = _record([], extra=payload)
    remap_record_dict(data, remapper, RemapCounts())
    assert data["payload"]["candidates"][0]["ability"][0]["script_file"] == "tokenscripts/c_a_food_sac.txt"


def test_a_real_card_key_is_untouched(remapper):
    data = _record([], ability=[_key("cardsfolder/f/food_chain.txt")])
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects/domain/test_token_key_remap.py -v -k "record or merge"`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

Append to `token_key_remap.py`:

```python
@dataclass
class RemapCounts:
    remapped: int = 0
    ambiguous: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ambiguous is None:
            self.ambiguous = Counter()

    def merge(self, other: "RemapCounts") -> None:
        self.remapped += other.remapped
        self.ambiguous.update(other.ambiguous)


_KEY_FIELDS = {"script_file", "face", "trait_kind", "index_within_kind"}


def _is_key(obj) -> bool:
    return isinstance(obj, Mapping) and _KEY_FIELDS <= obj.keys()


def _stem_of(script_file: str) -> str:
    return script_file.rsplit("/", 1)[-1][:-4]


def _rewrite(key: dict, stem: str) -> None:
    key["script_file"] = f"tokenscripts/{stem}.txt"


def _walk(obj, visit) -> None:
    """Call ``visit`` on every provenance-key dict below ``obj``."""
    if isinstance(obj, dict):
        if _is_key(obj):
            visit(obj)
            return
        for value in obj.values():
            _walk(value, visit)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, visit)


def remap_record_dict(data: dict, remapper: TokenKeyRemapper, counts: RemapCounts) -> None:
    """Rewrite every old token key in one record's JSON, in place."""
    memo: dict[str, str | None] = {}

    def resolve_for_entity(entity: dict, script_file: str) -> str | None:
        name = remapper.is_old_token_key(script_file)
        if name is None:
            return None
        if script_file not in memo:
            memo[script_file] = remapper.resolve(
                name, entity=entity, printed_keys=entity.get("printed") or (),
            )
        return memo[script_file]

    state = data.get("state") or {}
    for entity in state.get("entities") or ():
        def visit_entity_key(key, entity=entity):
            stem = resolve_for_entity(entity, key["script_file"])
            if stem is not None:
                _rewrite(key, stem)
                counts.remapped += 1
        for field in ("printed", "granted_attached"):
            _walk(entity.get(field) or [], visit_entity_key)
        _walk((entity.get("granted_temporary") or {}).get("abilities") or [], visit_entity_key)

    def visit_other(key):
        script_file = key["script_file"]
        name = remapper.is_old_token_key(script_file)
        if name is None:
            return
        if script_file in memo:
            stem = memo[script_file]
        else:
            stem = memo[script_file] = remapper.resolve(name)
        if stem is None:
            counts.ambiguous[_stem_of(script_file)] += 1
            return
        _rewrite(key, stem)
        counts.remapped += 1

    for field, value in data.items():
        if field == "state":
            continue
        _walk(value, visit_other)
    # Entity keys that stayed ambiguous are counted once per key.
    for entity in state.get("entities") or ():
        def count_left(key):
            if remapper.is_old_token_key(key["script_file"]) is not None:
                counts.ambiguous[_stem_of(key["script_file"])] += 1
        for field in ("printed", "granted_attached"):
            _walk(entity.get(field) or [], count_left)
        _walk((entity.get("granted_temporary") or {}).get("abilities") or [], count_left)
```

The memo means an old key that an entity resolved is reused for the acting ability even when that key appears with a different `trait_kind`; the memo is keyed on the `script_file` alone, which is the intended behaviour (one stem per old file per record).

- [ ] **Step 4: Run to verify they pass**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects/domain/test_token_key_remap.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/token_key_remap.py tests/unit/effects/domain/test_token_key_remap.py
git commit -m "feat(effects): remap every old token key in a record's JSON (FR-151)"
```

---

### Task 3: build-corpus applies the remap in both passes

**Files:**
- Modify: `src/effects/domain/corpus_manifest.py`
- Modify: `src/effects/application/build_corpus.py`
- Modify: `src/effects/infrastructure/cli.py`
- Test: `tests/unit/effects/domain/test_corpus_manifest.py`, `tests/unit/effects/application/test_build_corpus.py`, `test_build_corpus_survey.py`

**Interfaces:**
- `CorpusManifest` gains `token_keys_remapped: int = 0`, `token_keys_ambiguous: dict[str, int] = {}` (old stem → keys left, at most the 100 largest), `forge_tokenscripts: str = ""`.
- `BuildCorpusConfig` gains `forge_tokenscripts: Path | None = Path("../forge/forge-gui/res/tokenscripts")`, `remap_token_keys: bool = True`.
- `SurveyConfig` and `WriteConfig` gain `remapper: TokenKeyRemapper | None`.
- `ShardSurvey`, `Survey`, `WriteResult` gain `remap: RemapCounts`.
- New reader in `build_corpus.py`: `read_shard_remapped(path, remapper, counts) -> Iterator[EffectRecord]` — `iter_shard_lines` → `json.loads` → `remap_record_dict` when `remapper` is given → `record_from_dict`. Both `survey_shard` and `write_shard_pass` read through it.
- CLI: `--forge-tokenscripts PATH` (default above) and `--no-remap-token-keys`.
- The build refuses (`BuildCorpusError`) when `remap_token_keys` is on and `forge_tokenscripts` is not a directory, naming the path and the flag that turns the remap off.
- Manifest and log: "token keys remapped N; M left ambiguous over K stems, most: goblin_token (…), …". Two builds at one seed must agree on the counts (the remap is deterministic).

- [ ] **Step 1: Write the failing tests**

`test_corpus_manifest.py`: extend the round-trip and legacy tests with the three new fields (defaults `0`, `{}`, `""`).

`test_build_corpus_survey.py`:

```python
def test_the_survey_reads_records_through_the_remap(raw_shard_factory, tmp_path):
    """A survey key is the remapped key, so rarity and caps see the token script."""
    from effects.application.build_corpus import SurveyConfig, init_survey_worker, survey_shard
    from effects.domain.token_key_remap import TokenKeyRemapper, load_token_script_facts
    (tmp_path / "ts").mkdir()
    (tmp_path / "ts" / "c_a_food_sac.txt").write_text(
        "Name:Food Token\nTypes:Artifact Food\nA:AB$ GainLife | Cost$ 2 T\n", encoding="utf-8")
    remapper = TokenKeyRemapper(load_token_script_facts(tmp_path / "ts"),
                                converted_card_files=frozenset(), token_sidecars=frozenset({"c_a_food_sac"}))
    shard = raw_shard_factory([make_resolution("r", game="g1", ability=ProvenanceKey("cardsfolder/f/food_token.txt", 0, "spell", 1))])
    init_survey_worker(SurveyConfig(records_dir=str(shard.parent), held_out_names=frozenset(),
                                    held_out_script_files=frozenset(), text_cap=200, seed=1, max_events=64,
                                    remapper=remapper))
    out = survey_shard(shard.name)
    assert list(out.key_records) == ["tokenscripts/c_a_food_sac.txt|0|spell|1"]
    assert out.remap.remapped == 1
```

`test_build_corpus.py` (the `a_corpus` fixture writes a converted tree with sidecars; extend it with a `tokenscripts/c_a_food_sac` sidecar plus a raw Forge tokenscripts dir under `a_corpus.forge_tokenscripts`, and one training-game resolution record keyed `cardsfolder/f/food_token.txt`):

```python
def test_the_build_remaps_old_token_keys_and_records_the_counts(tmp_path, a_corpus):
    out = tmp_path / "out"
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
                                   forge_tokenscripts=a_corpus.forge_tokenscripts, workers=1,
                                   game_disjoint_target=1)) == 0
    store = CorpusStore(out)
    keys = {k.script_file for r in read_records(store.training_dir) for k in (r.ability or ())}
    assert "tokenscripts/c_a_food_sac.txt" in keys and "cardsfolder/f/food_token.txt" not in keys
    manifest = store.load()
    assert manifest.token_keys_remapped >= 1
    assert manifest.forge_tokenscripts == str(a_corpus.forge_tokenscripts)


def test_the_remap_can_be_turned_off(tmp_path, a_corpus):
    out = tmp_path / "out"
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
                                   forge_tokenscripts=a_corpus.forge_tokenscripts, remap_token_keys=False,
                                   workers=1, game_disjoint_target=1)) == 0
    keys = {k.script_file for r in read_records(CorpusStore(out).training_dir) for k in (r.ability or ())}
    assert "cardsfolder/f/food_token.txt" in keys
    assert CorpusStore(out).load().token_keys_remapped == 0


def test_a_missing_forge_tokenscripts_dir_is_refused(tmp_path, a_corpus):
    with pytest.raises(BuildCorpusError, match="forge-tokenscripts"):
        build(BuildCorpusConfig(records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=tmp_path / "out",
                                forge_tokenscripts=tmp_path / "nowhere", workers=1))


def test_the_cli_exposes_the_remap_flags():
    from effects.infrastructure.cli import build_parser
    args = build_parser().parse_args(["build-corpus", "--forge-tokenscripts", "x", "--no-remap-token-keys"])
    assert args.forge_tokenscripts == "x" and args.remap_token_keys is False
    assert build_parser().parse_args(["build-corpus"]).remap_token_keys is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects/application/test_build_corpus.py tests/unit/effects/application/test_build_corpus_survey.py tests/unit/effects/domain/test_corpus_manifest.py -q -k "remap or refused_forge or legacy or round_trip"`
Expected: FAIL (`TypeError` on `remapper=`, `AttributeError`s)

- [ ] **Step 3: Implement**

`corpus_manifest.py`: add the three fields with defaults and `.get` parsing.

`build_corpus.py`:

```python
from effects.domain.token_key_remap import RemapCounts, TokenKeyRemapper, remap_record_dict


def read_shard_remapped(path: Path, remapper: TokenKeyRemapper | None, counts: RemapCounts):
    """Every complete record of one shard, its old token keys remapped first.

    The remap works on the raw JSON before the record is built, so both passes
    see the corrected keys and nothing downstream has to know they were wrong.
    """
    import json

    from effects.infrastructure.record_io import iter_shard_lines, record_from_dict

    for line in iter_shard_lines(Path(path)):
        stripped = line.strip()
        if not stripped:
            continue
        data = json.loads(stripped)
        if remapper is not None:
            remap_record_dict(data, remapper, counts)
        yield record_from_dict(data)
```

`SurveyConfig` and `WriteConfig` gain `remapper: TokenKeyRemapper | None = None`; `ShardSurvey`, `Survey`, `WriteResult` gain `remap: RemapCounts = field(default_factory=RemapCounts)`; `survey_shard` and `write_shard_pass` iterate `read_shard_remapped(path, config.remapper, out.remap)`; `merge_surveys` and `absorb` call `.remap.merge(...)`.

`BuildCorpusConfig`: the two fields. In `build()`, after `card_files = load_card_files(cards_folder)`:

```python
    remapper = None
    if config.remap_token_keys:
        from effects.domain.token_key_remap import load_token_script_facts

        forge_tokens = Path(config.forge_tokenscripts) if config.forge_tokenscripts else None
        if forge_tokens is None or not forge_tokens.is_dir():
            raise BuildCorpusError(
                f"--forge-tokenscripts {forge_tokens} is not a directory. The old "
                "collector keyed forked tokens to cardsfolder paths; remapping them "
                "needs Forge's raw token scripts (their Colors/PT lines) to tell "
                "same-name scripts apart. Point the flag at "
                "<forge checkout>/forge-gui/res/tokenscripts/, or pass "
                "--no-remap-token-keys to build without the remap."
            )
        token_dir = folders.get("tokenscripts")
        token_sidecars = frozenset(
            p.stem for p in Path(token_dir).glob("*.txt")
        ) if token_dir is not None and Path(token_dir).is_dir() else frozenset()
        remapper = TokenKeyRemapper(
            load_token_script_facts(forge_tokens),
            converted_card_files=frozenset(card_files.values()),
            token_sidecars=token_sidecars,
        )
        logger.info(
            "Token key remap: %d token name(s) over %d script(s); %d converted token sidecar(s).",
            len(remapper.facts_by_name), sum(len(v) for v in remapper.facts_by_name.values()),
            len(token_sidecars),
        )
```

Pass `remapper=remapper` into `SurveyConfig(...)` and `WriteConfig(...)`. After the write pass, log and store:

```python
    remap = written.remap
    top = remap.ambiguous.most_common(5)
    logger.info(
        "token keys               %9d remapped to tokenscripts/; %d left ambiguous over %d stem(s)%s",
        remap.remapped, sum(remap.ambiguous.values()), len(remap.ambiguous),
        (" — most: " + ", ".join(f"{s} ({n})" for s, n in top)) if top else "",
    )
    if survey.remap.remapped != remap.remapped:
        logger.warning("The survey remapped %d key(s) and the write pass %d; the passes disagree.",
                       survey.remap.remapped, remap.remapped)
```

Manifest: `token_keys_remapped=remap.remapped`, `token_keys_ambiguous=dict(remap.ambiguous.most_common(100))`, `forge_tokenscripts=str(config.forge_tokenscripts) if remapper is not None else ""`.

`cli.py` `_build_corpus_parser`: `--forge-tokenscripts` (type str, default `../forge/forge-gui/res/tokenscripts`, help: raw Forge token scripts used to remap the old collector's token keys) and `--no-remap-token-keys` (`dest="remap_token_keys", action="store_false"`); wire both into `BuildCorpusConfig`.

Note `converted_card_files` must be the converted *cardsfolder* files with their tree prefix (`cardsfolder/a/….txt`), which is what `load_card_files` returns as values (check `train_effect_model.py:~630`, `out[name] = f"cardsfolder/{relative}"`).

- [ ] **Step 4: Run the build suites**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects -q`
Expected: all pass, one pre-existing umap warning. `test_two_builds_of_one_corpus_agree` must still pass.

- [ ] **Step 5: Commit**

```bash
git add src/effects/domain/corpus_manifest.py src/effects/application/build_corpus.py src/effects/infrastructure/cli.py tests/unit/effects
git commit -m "feat(effects): build-corpus remaps the old collector's token keys in both passes (FR-151)"
```

---

### Task 4: The collector writes the token script id, not the printed name

**Files:**
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceKey.java` (`tokenScriptStem` becomes package-private `tokenScriptStemOf`)
- Modify: `forge-connector/src/main/java/com/pricepredictor/connector/effects/SnapshotBuilder.java:509-510`
- Test: `forge-connector/src/test/java/com/pricepredictor/connector/effects/CopiedTokenProvenanceTest.java` (append) and `SnapshotBuilderTest.java` if it has a token case

**Interfaces:**
- `ProvenanceKey.tokenScriptStemOf(Card) -> String | null` (package-private static; same body as today's private `tokenScriptStem`, which becomes a one-line delegate or is renamed).
- `SnapshotBuilder.entityToJson` writes `"token_script_id": <stem>` for a token whose stem resolves, `<printed name>` for a token whose stem does not (the old behaviour, kept so the field is never null for a token), `null` for a non-token.

- [ ] **Step 1: Write the failing test**

In `CopiedTokenProvenanceTest`, add after the existing assertions (the fixture already renders the fork's snapshot as JSON):

```java
    /** FR-078 calls the field a token-script id; the printed name collapses distinct scripts. */
    @Test
    void theSnapshotNamesTheTokenScriptNotThePrintedName() {
        // reuse the fixture: build the board, fork it, render both snapshots
        ...
        assertTrue(mainline.contains("\"token_script_id\":\"c_a_food_sac\""), mainline);
        assertTrue(forked.contains("\"token_script_id\":\"c_a_food_sac\""), forked);
        assertFalse(forked.contains("\"token_script_id\":\"Food Token\""), forked);
    }
```

Factor the fixture's board-and-fork setup into a private helper if the existing test inlines it, so both tests share it.

- [ ] **Step 2: Run to verify it fails**

Run: `cd forge-connector && mvn -q test -Dforge.dir=C:/Users/nicol/IdeaProjects/forge -Dtest=CopiedTokenProvenanceTest`
Expected: the new test fails on the first `assertTrue` (the field holds `Food Token`).

- [ ] **Step 3: Implement**

`ProvenanceKey.java`: rename `private static String tokenScriptStem(Card host)` to `static String tokenScriptStemOf(Card host)` (package-private) and update its one caller in `scriptFileOf`; keep the javadoc.

`SnapshotBuilder.java:509-510`:

```java
                + ",\"token_script_id\":"
                + Json.string(tokenScriptIdOf(card))
```

with, in the same class:

```java
    /**
     * The token's script stem — what FR-078 means by a token-script id — or
     * the printed name when no script can be named (an engine-built token with
     * no image key), so a token is never reported as a non-token.
     */
    private static String tokenScriptIdOf(Card card) {
        if (!card.isToken()) {
            return null;
        }
        String stem = ProvenanceKey.tokenScriptStemOf(card);
        return stem != null ? stem : card.getName();
    }
```

- [ ] **Step 4: Run the connector tests that touch snapshots**

Run: `cd forge-connector && mvn -q test -Dforge.dir=C:/Users/nicol/IdeaProjects/forge -Dtest=CopiedTokenProvenanceTest,SnapshotBuilderTest,ProvenanceKeyTest`
Expected: all pass. If `SnapshotBuilderTest` pins `"token_script_id":"<printed name>"` for a `PaperToken`-backed token, update that expectation to the stem and say so in the report.

- [ ] **Step 5: Commit**

```bash
git add forge-connector/src/main/java/com/pricepredictor/connector/effects/ProvenanceKey.java forge-connector/src/main/java/com/pricepredictor/connector/effects/SnapshotBuilder.java forge-connector/src/test/java/com/pricepredictor/connector/effects/CopiedTokenProvenanceTest.java
git commit -m "fix(connector): token_script_id is the token's script stem, not its printed name (FR-078)"
```

---

### Task 5: Docs and the record-schema note

**Files:**
- Modify: `specs/023-ability-effect-model/spec.md` (FR-078 wording; new FR-151), `quickstart.md` (build-corpus flags and output line), `src/effects/CLAUDE.md` (record schema: `token_script_id` value space changed on 2026-09-17; the remap and its manifest fields)
- Modify: `docs/superpowers/specs/2026-09-16-corpus-and-trainer-rework.md` amendment table (FR-078, FR-151 rows)

- [ ] **Step 1: Apply the amendments** (load `.claude/skills/feature-workflow/SKILL.md` first)

FR-151 (new, after FR-150): `build-corpus` MUST remap a provenance key whose script file the converted card tree does not hold and whose filename stem names a token script to `tokenscripts/<stem>.txt`, resolving the stem by name, then by the carrying entity's colours, types and P/T, then by its printed-key counts per trait kind, and MUST leave a key untouched when more than one script remains; it MUST record `token_keys_remapped`, `token_keys_ambiguous` and `forge_tokenscripts` in the manifest; `--no-remap-token-keys` turns the step off.

FR-078: `token_script_id` MUST be the token's script stem (`c_a_food_sac`), falling back to the printed name only for a token whose script cannot be named; shards collected before 2026-09-17 carry printed names in that field.

Quickstart: the two new flags in the build-corpus flag table; one line under the manifest field list; a sentence in the build's expected-output section about the "token keys remapped" log line.

- [ ] **Step 2: Run the docs-related checks**

Run: `PYTHONPATH=src python -m pytest tests/unit/effects -q` (unchanged code; should stay green) and `grep -n "FR-151\|token_script_id" specs/023-ability-effect-model/spec.md | head`.

- [ ] **Step 3: Commit**

```bash
git add specs/023-ability-effect-model docs/superpowers/specs src/effects/CLAUDE.md
git commit -m "docs(effects): FR-151 token key remap; token_script_id is the script stem"
```

---

### Task 6: Rebuild the remapped corpus beside the live one, and the connector JAR

**Files:** none (operational). Run from the main checkout with the worktree's code; **never** touch `output/effects/corpus/` while the training run reads it.

- [ ] **Step 1: Pre-flight the remap on real shards (minutes, read-only)**

```bash
cd /c/Users/nicol/IdeaProjects/price-predictor && PYTHONPATH=.worktrees/corpus-rework/src python - <<'EOF'
import glob, json
from pathlib import Path
from effects.domain.token_key_remap import RemapCounts, TokenKeyRemapper, load_token_script_facts, remap_record_dict
from effects.application.train_effect_model import load_card_files
from effects.infrastructure.record_io import iter_shard_lines
facts = load_token_script_facts(Path("../forge/forge-gui/res/tokenscripts"))
cards = frozenset(load_card_files(Path("output/cardsfolder")).values())
sidecars = frozenset(p.stem for p in Path("output/tokenscripts").glob("*.txt"))
r = TokenKeyRemapper(facts, converted_card_files=cards, token_sidecars=sidecars)
c = RemapCounts()
for p in sorted(glob.glob("output/effects/records/full-strength/*.jsonl.gz"))[:8]:
    for line in iter_shard_lines(Path(p)):
        if line.strip(): remap_record_dict(json.loads(line), r, c)
print("remapped", c.remapped, "ambiguous", sum(c.ambiguous.values()), c.ambiguous.most_common(10))
EOF
```

Expected: the remapped count dominates the ambiguous count (the earlier measurement put unique-name keys at 41% of lookups and colour/type/P/T resolution at 353 of 419 same-name groups), and the most-ambiguous stems are the keyword-only variants (goblin, insect, elemental).

- [ ] **Step 2: Build the remapped corpus to a new directory (about 2h)**

```bash
cd /c/Users/nicol/IdeaProjects/price-predictor && PYTHONPATH=.worktrees/corpus-rework/src python -u -m effects build-corpus \
    --records-dir output/effects/records/ --output output/effects/corpus-remapped/ \
    --variant-scripts output/effects/variant-scripts/ --vocab-path models/effects/vocab-script.txt \
    --forge-tokenscripts ../forge/forge-gui/res/tokenscripts/ \
    > output/effects/reports/build-corpus-remapped-20260917.log 2>&1
```

Check the log's "token keys … remapped" line and the manifest's `token_keys_remapped` / `token_keys_ambiguous`; compare `per_stratum` and `delivered_mix` with the 02:44 build (they should be near-identical: the remap changes keys, not which records exist).

- [ ] **Step 3: Rebuild the connector JAR in the worktree**

```bash
cd /c/Users/nicol/IdeaProjects/price-predictor/.worktrees/corpus-rework/forge-connector && mvn -q package -DskipTests -Dforge.dir=C:/Users/nicol/IdeaProjects/forge
```

The Python workers load `forge-connector/target/forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar` relative to the checkout they run from, so the main checkout's collections pick up the fix only after this branch is merged and `mvn package` is run there. Say so in the hand-off.

- [ ] **Step 4: Record**

Note the pre-flight figures, the build's manifest figures and the JAR rebuild in the hand-off; the corpus swap (`corpus-remapped` → `corpus`) waits until the current training run has finished and been evaluated.

---

## Self-review

- **Contract coverage.** Recognition (Task 1 `is_old_token_key`), the three resolution steps (Task 1 `resolve`), the honest refusal (Task 1 sidecar gate and ambiguity), every key site (Task 2 walker), both build passes and the manifest (Task 3), the collector's future output (Task 4), docs (Task 5), the safe rebuild (Task 6).
- **Type consistency.** `TokenKeyRemapper(facts_by_name, *, converted_card_files, token_sidecars)`, `resolve(name, *, entity, printed_keys)`, `remap_record_dict(data, remapper, counts)`, `RemapCounts.merge`, `read_shard_remapped(path, remapper, counts)` are used with the same names in every task.
- **Live-run safety.** Task 6 writes only to `output/effects/corpus-remapped/`; the global constraint forbids the live directory.
- **Known gaps.** Keys whose stem spaces to a token name that is also a real card name are excluded by the converted-card check; a token whose printed name never matches any script `Name:` (e.g. an engine-built named token) is not a remap candidate and stays unresolved, as today.
