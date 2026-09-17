"""Resolving the old collector's token keys to the token scripts they meant (FR-151).

Tokens rebuilt by Forge's ``GameCopier`` in forked games carried no paper
card, so the collector keyed their abilities to a path derived from the
printed name -- ``cardsfolder/f/food_token.txt`` -- which no tree holds. The
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
#: Core card types; anything else on a ``Types:`` line is a subtype. This
#: deliberately excludes the supertypes Forge scripts can also put on a
#: ``Types:`` line (``Legendary``, ``Basic``, ``Snow``) and the pseudo-type
#: ``Token``: a real corpus entity's ``types`` field carries neither
#: (supertypes live in a separate field there, and entities carry no ``token``
#: pseudo-type at all), so bucketing those words as core would make
#: ``f.core_types == types`` fail for every such script. Bucketing them as
#: subtypes instead is not exact either -- a legendary token's ``subtypes``
#: would then carry ``legendary`` while the entity's does not -- but it is
#: the same, accepted kind of mismatch as any other, and such a script simply
#: fails the colours/types/P-T narrowing step and falls through to the
#: trait-count step or stays ambiguous.
CORE_TYPES: frozenset[str] = frozenset({
    "artifact", "creature", "enchantment", "instant", "land", "planeswalker",
    "sorcery", "battle", "kindred", "tribal",
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

    def _settle(self, candidates: Sequence[TokenScriptFacts]) -> str | None:
        """A stem when ``candidates`` holds exactly one script with a sidecar."""
        if len(candidates) != 1:
            return None
        stem = candidates[0].stem
        return stem if stem in self.token_sidecars else None

    def resolve(
        self, name: str, *, entity=None, printed_keys: Iterable[Mapping] = (),
    ) -> str | None:
        """The one stem ``name`` can mean here, or None when ambiguous or unknown.

        Narrows in three steps, stopping at the first that leaves exactly one
        candidate: by printed name, then by the carrying ``entity``'s colours,
        core types, subtypes and P/T, then by how many printed keys of each
        trait kind the entity shows (``printed_keys``) against how many lines
        each remaining script declares. The trait-kind counts are compared
        with the script's own ``spell`` count bumped by one: every token
        carries one ``spell`` key at index 0 for the permanent's own cast
        spell regardless of whether its script has an ``A:`` line, so an
        entity's raw ``printed_keys`` always shows one more ``spell`` than
        the script text does.
        """
        candidates = list(self.facts_by_name.get(name, ()))
        resolved = self._settle(candidates)
        if resolved is not None:
            return resolved
        if entity is None or not candidates:
            return None

        colors = frozenset(c.upper() for c in (entity.get("colors") or ()))
        types = {t.lower() for t in entity.get("types") or ()}
        subtypes = {t.lower() for t in entity.get("subtypes") or ()}
        pt = entity.get("pt")
        candidates = [
            f for f in candidates
            if f.colors == colors and f.core_types == types and f.subtypes == subtypes
            and _pt_matches(f.pt, pt)
        ]
        resolved = self._settle(candidates)
        if resolved is not None:
            return resolved
        if not candidates:
            return None

        seen = Counter(k.get("trait_kind") for k in printed_keys)
        counts = {kind: seen.get(kind, 0) for kind in TRAIT_KINDS}
        candidates = [
            f for f in candidates
            if {**f.trait_counts, "spell": f.trait_counts["spell"] + 1} == counts
        ]
        return self._settle(candidates)


@dataclass
class RemapCounts:
    """How many old token keys one pass over the corpus touched.

    ``ambiguous`` counts a key that stayed as it was, keyed by the old
    script's filename stem (``goblin_token``), so a build's log can name
    which token names most need a converted sidecar or richer context to
    settle.
    """

    remapped: int = 0
    ambiguous: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.ambiguous is None:
            self.ambiguous = Counter()

    def merge(self, other: "RemapCounts") -> None:
        """Fold another counts (e.g. another shard's) into this one."""
        self.remapped += other.remapped
        self.ambiguous.update(other.ambiguous)


_KEY_FIELDS = {"script_file", "face", "trait_kind", "index_within_kind"}


def _is_key(obj: object) -> bool:
    """Whether ``obj`` is a provenance-key dict rather than a plain container."""
    return isinstance(obj, Mapping) and _KEY_FIELDS <= obj.keys()


def _stem_of(script_file: str) -> str:
    return script_file.rsplit("/", 1)[-1][:-4]


def _rewrite(key: dict, stem: str) -> None:
    key["script_file"] = f"tokenscripts/{stem}.txt"


def _walk(obj, visit) -> None:
    """Call ``visit`` on every provenance-key dict below ``obj``.

    Stops descending into a dict once it is itself a key dict, so a key's
    own fields never get mistaken for nested containers holding more keys.
    """
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
    """Rewrite every old token key in one record's raw JSON, in place.

    Entities resolve first, each against its own colours, types, P/T and
    printed-key counts (the contract's steps 2-3). Every other key list in
    the record -- the acting ``ability``, playability candidates,
    ``responsible_static``, ``replaced_by``, and anything else under a
    non-``state`` field -- then reuses whichever stem an entity already
    found for the same old ``script_file`` in this record, falling back to
    resolving by name alone (step 1) when no entity carried it. A key that
    stays ambiguous is counted once, keyed by the old script's stem.
    """
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
    entities = state.get("entities") or ()

    for entity in entities:
        def visit_entity_key(key: dict, entity: dict = entity) -> None:
            stem = resolve_for_entity(entity, key["script_file"])
            if stem is not None:
                _rewrite(key, stem)
                counts.remapped += 1

        for field in ("printed", "granted_attached"):
            _walk(entity.get(field) or [], visit_entity_key)
        _walk((entity.get("granted_temporary") or {}).get("abilities") or [], visit_entity_key)

    def visit_other(key: dict) -> None:
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

    # Entity keys that stayed ambiguous are counted once per key, here
    # rather than inside visit_entity_key above: a rewritten key's
    # script_file no longer looks like an old key, so this second pass
    # over the same fields only ever finds the ones that never resolved.
    for entity in entities:
        def count_left(key: dict) -> None:
            if remapper.is_old_token_key(key["script_file"]) is not None:
                counts.ambiguous[_stem_of(key["script_file"])] += 1

        for field in ("printed", "granted_attached"):
            _walk(entity.get(field) or [], count_left)
        _walk((entity.get("granted_temporary") or {}).get("abilities") or [], count_left)
