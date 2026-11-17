"""The text an ability is encoded from, resolved once per key.

The one definition of the key-to-text join: ``build-corpus``, the trainer's
weighting and the batcher must agree on it exactly, or every lookup misses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from effects.domain.ability_encoder import encoding_text
from effects.domain.ability_tokenizer import Surface
from effects.domain.provenance import ProvenanceKey, SidecarLine, SidecarLookup, first_text
from effects.domain.records import EffectRecord, acting_keys
from effects.infrastructure.sidecar_io import (
    SidecarCache,
    converted_text_path,
    sidecar_paths,
)


def resolve_key(
    key: ProvenanceKey, sidecars: SidecarLookup, surface: Surface,
) -> tuple[str | None, SidecarLine | None]:
    """``(text, line)`` for one provenance key on ``surface``, read through
    the join (``SidecarCache.join``, which counts every key it reads as no
    line); ``(None, None)`` where the key names no line.

    A line with neither surface reads as the empty string, as the ability
    cache encodes it (``encoding_text``). A key the sidecar does not describe
    raises ``KeyError`` (the contract's fail-loudly case).
    """
    line = sidecars.line_for(key)
    if line is None:
        return None, None
    return encoding_text(line, sidecars.prose_for(key), surface), line


def text_for_key(key: ProvenanceKey, sidecars: SidecarLookup, surface: Surface) -> str | None:
    """One provenance key's text on ``surface``, or None where it has none."""
    return resolve_key(key, sidecars, surface)[0]


def texts_by_key(roots: Mapping[str, Path], surface: Surface) -> dict[ProvenanceKey, str]:
    """``key -> text`` for every key a line under ``roots`` (tree name ->
    folder) declares, through the join, spelled as a training batch looks it
    up; a line with neither surface reads as the empty string."""
    sidecars = SidecarCache(roots)
    return {key: text_for_key(key, sidecars, surface)
            for key in _declared_keys(roots, sidecars)}


def _declared_keys(roots: Mapping[str, Path], sidecars: SidecarCache):
    """Every key a line declares, tree by tree, in path and line order."""
    for tree, root in roots.items():
        root = Path(root)
        for path in sidecar_paths(root):
            relative = converted_text_path(path).relative_to(root).as_posix()
            for line in sidecars.get(f"{tree}/{relative}").lines:
                yield from line.provenance


def known_texts(roots: Mapping[str, Path], surface: Surface) -> list[str]:
    """Every distinct text a line under ``roots`` (tree name -> folder) encodes
    as on ``surface``, sorted, spelled as a training batch looks it up."""
    return sorted(set(texts_by_key(roots, surface).values()))


class KeyResolver:
    """:func:`resolve_key` and a charm's option keys, memoised for a run.

    A ``KeyError`` is deliberately not cached: remembered as "no text", the
    fail-loudly case would go quiet.
    """

    def __init__(self, sidecars: SidecarLookup, surface: Surface) -> None:
        self.sidecars = sidecars
        self.surface = surface
        self._resolved: dict[ProvenanceKey, tuple[str | None, SidecarLine | None]] = {}
        self._options: dict[ProvenanceKey, tuple[ProvenanceKey, ...]] = {}
        self._board: dict[ProvenanceKey, tuple[ProvenanceKey, ...]] = {}
        #: ``text -> keys resolved to it``: how a truncated text is named.
        self._keys_by_text: dict[str, list[ProvenanceKey]] = defaultdict(list)

    def resolve(self, key: ProvenanceKey) -> tuple[str | None, SidecarLine | None]:
        cached = self._resolved.get(key)
        if cached is None:
            cached = self._resolved[key] = resolve_key(key, self.sidecars, self.surface)
            if cached[0] is not None:
                self._keys_by_text[cached[0]].append(key)
        return cached

    def text_of(self, key: ProvenanceKey) -> str | None:
        return self.resolve(key)[0]

    def acting_text(self, record: EffectRecord) -> str | None:
        """The acting line's text, or None where no line acts (``combat``,
        ``attackers``, ``blockers``), which samples uniformly."""
        return first_text(acting_keys(record), self.text_of)

    def keys_of(self, text: str) -> list[ProvenanceKey]:
        """The keys resolved to ``text`` so far."""
        return list(self._keys_by_text.get(text, ()))

    def options_for(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The keys of the charm-mode lines that follow ``key``'s line (FR-063a);
        empty for a non-root line or a key no sidecar describes."""
        cached = self._options.get(key)
        if cached is not None:
            return cached
        found: tuple[ProvenanceKey, ...] = ()
        # A mode is never a root: its siblings follow it in the sidecar too,
        # and must not be read as its own options.
        if key.option is None:
            try:
                sidecar = self.sidecars.get(key.script_file)
                row = sidecar.row_for(key)
            except KeyError:
                row = None
            if row is not None:
                found = tuple(
                    next(k for k in sidecar.lines[r].provenance if k.option is not None)
                    for r in sidecar.option_rows_after(row)
                )
        self._options[key] = found
        return found

    def board_keys(self, key: ProvenanceKey) -> tuple[ProvenanceKey, ...]:
        """The keys a board line of ``key`` reads as slots
        (:meth:`ProvenanceSidecar.board_keys`): a saga's bare keyword key is
        its chapter lines; the key itself where no sidecar answers."""
        cached = self._board.get(key)
        if cached is None:
            try:
                cached = self.sidecars.get(key.script_file).board_keys(key)
            except KeyError:
                cached = (key,)
            self._board[key] = cached
        return cached
