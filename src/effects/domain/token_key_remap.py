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
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

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


def parse_token_script(stem: str, text: str) -> TokenScriptFacts | None:
    """The facts a Forge token script declares, or None without a ``Name:``."""
    name = None
    colors: set[str] = set()
    core: set[str] = set()
    supers: set[str] = set()
    subs: set[str] = set()
    pt: tuple[str, str] | None = None
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
                if word in CORE_TYPES:
                    core.add(word)
                elif word in SUPERTYPE_WORDS:
                    supers.add(word)
                else:
                    subs.add(word)
        elif key == "PT":
            left, _, right = value.partition("/")
            pt = (left.strip(), right.strip())
    if name is None:
        return None
    return TokenScriptFacts(
        stem=stem, name=name, colors=frozenset(colors), core_types=frozenset(core),
        supertypes=frozenset(supers), subtypes=frozenset(subs), pt=pt,
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

        Narrows in two steps, stopping at the first that leaves exactly one
        candidate: by printed name, then by the carrying ``entity``'s colours,
        core types, supertypes, subtypes and P/T. There is no third step:
        a snapshot shows a token's spell abilities only, so two same-stat
        scripts of one name that differ by a keyword (``r_1_1_goblin`` against
        ``r_1_1_goblin_haste``) or by an ability look identical from here and
        must refuse rather than pick one.

        ``printed_keys`` is accepted and unused, so a caller that has the
        entity's keys to hand needs no second call shape.
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
    state = data.get("state") or {}
    entities = state.get("entities") or ()

    # Per old script_file, every stem (or None) some entity carrying it
    # resolved to -- the record-wide memo below trusts a file only when
    # every entity that carried it agreed.
    per_file_stems: dict[str, set[str | None]] = {}

    for entity in entities:
        entity_cache: dict[str, str | None] = {}

        def resolve_for_entity(
            script_file: str, entity: dict = entity, entity_cache: dict = entity_cache,
        ) -> str | None:
            name = remapper.is_old_token_key(script_file)
            if name is None:
                return None
            if script_file not in entity_cache:
                entity_cache[script_file] = remapper.resolve(
                    name, entity=entity, printed_keys=entity.get("printed") or (),
                )
            return entity_cache[script_file]

        def visit_entity_key(key: dict, resolve_for_entity=resolve_for_entity) -> None:
            script_file = key["script_file"]
            stem = resolve_for_entity(script_file)
            if remapper.is_old_token_key(script_file) is not None:
                per_file_stems.setdefault(script_file, set()).add(stem)
            if stem is not None:
                _rewrite(key, stem)
                counts.remapped += 1

        for field in ("printed", "granted_attached"):
            _walk(entity.get(field) or [], visit_entity_key)
        _walk((entity.get("granted_temporary") or {}).get("abilities") or [], visit_entity_key)

    # Trust an old file's resolution for the record's other key sites only
    # when every entity that carried it agreed on one non-None stem.
    memo: dict[str, str | None] = {
        script_file: next(iter(stems))
        for script_file, stems in per_file_stems.items()
        if len(stems) == 1 and None not in stems
    }

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
