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
