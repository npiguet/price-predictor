"""Resolving the old collector's token keys to the token scripts they meant (FR-151).

Tokens rebuilt by Forge's ``GameCopier`` in forked games carried no paper
card, so the collector keyed their abilities to a path derived from the
printed name -- ``cardsfolder/f/food_token.txt`` -- which no tree holds. The
collector is fixed (FR-150), but every shard collected before the fix still
carries those keys. This module turns them back into ``tokenscripts/<stem>.txt``
where that can be done honestly: by name when one script carries the name,
else by the entity's colours, types, supertypes and P/T. Anything still
ambiguous stays as it is and is counted, never guessed.

What the corpus actually holds decides how far the narrowing can go. A token
entity's ``printed`` list carries one key only -- ``(spell, 0)``, the
permanent's own cast spell -- because the Java snapshot builder walks the
card's spell abilities alone: a script's ``K:``, ``T:``, ``S:`` and ``R:``
lines produce no printed key at all. So the snapshot cannot say whether a
token has haste, and two same-stat scripts that differ only by a keyword or
an ability are indistinguishable here. Such a name stays ambiguous rather
than being settled by a count the snapshot never carried.

That one key is also why this is an archival correction rather than a
recovery: ``(spell, 0)`` maps to no converted line before or after the
rewrite -- it is a ``dropped_keys`` entry in all but a handful of token
sidecars -- while the token's *ability* lines were already keyed correctly
under ``tokenscripts/<stem>.txt`` by the old collector.

Known limitation: step 2 compares the snapshot's **in-game** colours, types
and P/T, so a token an anthem or a colour-changing effect has altered can
match a different script of the same name than the one it was created from.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from effects.domain.provenance import KEY_DICT_FIELDS

COLOR_WORDS: dict[str, str] = {
    "white": "W", "blue": "U", "black": "B", "red": "R", "green": "G",
}
#: The supertypes a Forge ``Types:`` line can carry. A corpus entity keeps
#: these in a field of their own, so they are bucketed apart from the core
#: types rather than mixed into either side.
SUPERTYPE_WORDS: frozenset[str] = frozenset({"legendary", "basic", "snow"})
#: Core card types; anything else on a ``Types:`` line that is not a supertype
#: is a subtype. This deliberately excludes the pseudo-type ``Token``, which no
#: entity carries.
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
    supertypes: frozenset[str]
    subtypes: frozenset[str]
    pt: tuple[str, str] | None


def _script_values(text: str) -> dict[str, list[str]]:
    """Every ``Key:value`` line's stripped value, by key, in script order."""
    values: dict[str, list[str]] = {}
    for raw in text.splitlines():
        key, sep, value = raw.partition(":")
        if sep:
            values.setdefault(key, []).append(value.strip())
    return values


def _colors_of(lines: Iterable[str]) -> frozenset[str]:
    return frozenset(
        COLOR_WORDS[word]
        for value in lines
        for word in re.split(r"[,\s]+", value.lower())
        if word in COLOR_WORDS
    )


def _types_of(lines: Iterable[str]) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """``(core types, supertypes, subtypes)`` across every ``Types:`` line."""
    core: set[str] = set()
    supers: set[str] = set()
    subs: set[str] = set()
    for value in lines:
        for word in value.lower().split():
            if word in CORE_TYPES:
                core.add(word)
            elif word in SUPERTYPE_WORDS:
                supers.add(word)
            else:
                subs.add(word)
    return frozenset(core), frozenset(supers), frozenset(subs)


def _pt_of(value: str) -> tuple[str, str]:
    left, _, right = value.partition("/")
    return left.strip(), right.strip()


def parse_token_script(stem: str, text: str) -> TokenScriptFacts | None:
    """The facts a Forge token script declares, or None without a ``Name:``.

    A key written twice keeps its last ``Name:`` and ``PT:`` and the union of
    its ``Colors:`` and ``Types:`` lines.
    """
    values = _script_values(text)
    names = values.get("Name")
    if not names:
        return None
    core, supers, subs = _types_of(values.get("Types", ()))
    pts = values.get("PT")
    return TokenScriptFacts(
        stem=stem, name=names[-1].lower(), colors=_colors_of(values.get("Colors", ())),
        core_types=core, supertypes=supers, subtypes=subs,
        pt=_pt_of(pts[-1]) if pts else None,
    )


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
        # script_file -> is_old_token_key's answer. The remap asks about every
        # key of every record, ~56 a record over two passes of the corpus, and
        # the distinct script files are bounded by the card pool.
        self._old_names: dict[str, str | None] = {}

    def is_old_token_key(self, script_file: str) -> str | None:
        """The spaced token name when ``script_file`` is a remap candidate."""
        try:
            return self._old_names[script_file]
        except KeyError:
            name = self._old_names[script_file] = self._old_name(script_file)
            return name

    def _old_name(self, script_file: str) -> str | None:
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
        self, name: str, *, entity=None,
    ) -> str | None:
        """The one stem ``name`` can mean here, or None when ambiguous or unknown.

        Narrows in two steps, stopping at the first that leaves exactly one
        candidate: by printed name, then by the carrying ``entity``'s colours,
        core types, supertypes, subtypes and P/T. There is no third step:
        a snapshot shows a token's spell abilities only, so two same-stat
        scripts of one name that differ by a keyword (``r_1_1_goblin`` against
        ``r_1_1_goblin_haste``) or by an ability look identical from here and
        must refuse rather than pick one.
        """
        candidates = list(self.facts_by_name.get(name, ()))
        resolved = self._settle(candidates)
        if resolved is not None:
            return resolved
        if entity is None or not candidates:
            return None

        colors = frozenset(c.upper() for c in (entity.get("colors") or ()))
        types = {t.lower() for t in entity.get("types") or ()}
        supertypes = {t.lower() for t in entity.get("supertypes") or ()}
        subtypes = {t.lower() for t in entity.get("subtypes") or ()}
        pt = entity.get("pt")
        candidates = [
            f for f in candidates
            if f.colors == colors and f.core_types == types
            and f.supertypes == supertypes and f.subtypes == subtypes
            and _pt_matches(f.pt, pt)
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
    ambiguous: Counter = field(default_factory=Counter)

    def merge(self, other: RemapCounts) -> None:
        """Fold another counts (e.g. another shard's) into this one."""
        self.remapped += other.remapped
        self.ambiguous.update(other.ambiguous)

    def reset(self) -> None:
        """Back to zero, for a counter reused record after record."""
        self.remapped = 0
        self.ambiguous.clear()


def _stem_of(script_file: str) -> str:
    return script_file.rsplit("/", 1)[-1].removesuffix(".txt")


def _rewrite(key: dict, stem: str) -> None:
    key["script_file"] = f"tokenscripts/{stem}.txt"


def _key_dicts(obj) -> Iterator[dict]:
    """Every provenance-key dict below ``obj``, in document order.

    Stops descending into a dict once it is itself a key dict, so a key's
    own fields never get mistaken for nested containers holding more keys.
    """
    if isinstance(obj, dict):
        if KEY_DICT_FIELDS <= obj.keys():
            yield obj
            return
        for value in obj.values():
            yield from _key_dicts(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _key_dicts(item)


def _old_keys(obj, remapper: TokenKeyRemapper) -> Iterator[tuple[dict, str, str]]:
    """``(key, script_file, token name)`` for every old token key below ``obj``."""
    for key in _key_dicts(obj):
        script_file = key["script_file"]
        name = remapper.is_old_token_key(script_file)
        if name is not None:
            yield key, script_file, name


def _entity_key_sites(entity: dict) -> tuple:
    """An entity's key lists in its raw JSON, as ``EntityState`` names them."""
    return (
        entity.get("printed") or [],
        entity.get("granted_attached") or [],
        (entity.get("granted_temporary") or {}).get("abilities") or [],
    )


def remap_record_dict(data: dict, remapper: TokenKeyRemapper, counts: RemapCounts) -> None:
    """Rewrite every old token key in one record's raw JSON, in place.

    Each entity resolves *independently* against its own colours, types,
    supertypes and P/T (the contract's step 2) -- never against
    another entity's resolution -- so two entities sharing one old
    ``script_file`` (two token copies with different characteristics on one
    board) are never conflated. Every other key list in the record -- the
    acting ``ability``, playability candidates, ``responsible_static``,
    ``replaced_by``, and anything else under a non-``state`` field -- may
    then reuse an old file's resolution only when every entity that carried
    it agreed on the same stem; otherwise it falls back to resolving by
    name alone (step 1). A key that stays ambiguous is counted once, keyed
    by the old script's stem.
    """
    entities = (data.get("state") or {}).get("entities") or ()
    per_file_stems, entity_ambiguous = _remap_entity_keys(entities, remapper, counts)

    # Trust an old file's resolution for the record's other key sites only
    # when every entity that carried it agreed on one non-None stem.
    memo: dict[str, str | None] = {
        script_file: next(iter(stems))
        for script_file, stems in per_file_stems.items()
        if len(stems) == 1 and None not in stems
    }
    outside_state = [value for field_name, value in data.items() if field_name != "state"]
    for key, script_file, name in _old_keys(outside_state, remapper):
        if script_file in memo:
            stem = memo[script_file]
        else:
            stem = memo[script_file] = remapper.resolve(name)
        if stem is None:
            counts.ambiguous[_stem_of(script_file)] += 1
        else:
            _rewrite(key, stem)
            counts.remapped += 1

    counts.ambiguous.update(entity_ambiguous)


def _remap_entity_keys(
    entities, remapper: TokenKeyRemapper, counts: RemapCounts,
) -> tuple[dict[str, set[str | None]], list[str]]:
    """Rewrite each entity's old keys against that entity alone.

    Returns, per old script_file, every stem (or None) some entity carrying it
    resolved to -- the record-wide memo trusts a file only when every entity
    that carried it agreed -- and the stems of the entity keys that stayed
    ambiguous. Those are counted by the caller only after the other key
    sites, so the counter sees them in the order it always has: the manifest
    keeps the 100 most common, and ties keep insertion order.
    """
    per_file_stems: dict[str, set[str | None]] = {}
    ambiguous: list[str] = []
    for entity in entities:
        resolved: dict[str, str | None] = {}
        for site in _entity_key_sites(entity):
            for key, script_file, name in _old_keys(site, remapper):
                if script_file not in resolved:
                    resolved[script_file] = remapper.resolve(name, entity=entity)
                stem = resolved[script_file]
                per_file_stems.setdefault(script_file, set()).add(stem)
                if stem is None:
                    ambiguous.append(_stem_of(script_file))
                else:
                    _rewrite(key, stem)
                    counts.remapped += 1
    return per_file_stems, ambiguous
