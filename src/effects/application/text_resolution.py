"""How an ability key becomes the text a batch encodes it from.

One :class:`TextResolution` per batcher: the run's memoised key-to-text join,
a keyword line's reminder text, and the overrides a step's counterfactual
copies and a probe's edits put in front of both.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from effects.application.ability_text import KeyResolver
from effects.application.text_staging import TextStager
from effects.domain.grid_edits import TextOverride
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.slot_lines import keyword_line_fragment


class TextResolution:
    """``(text, line)`` of every ability key on one run's surface.

    A key reads, in order: its override, a keyword line's reminder text, the
    join. The overrides are apart from the resolver's memo, which a step's
    batcher borrows from the run, because a copy's text belongs to its step
    alone.
    """

    def __init__(self, resolver: KeyResolver, stager: TextStager) -> None:
        self.resolver = resolver
        self.stager = stager
        #: ``key -> (text, line)`` for :meth:`override` keys.
        self._overrides: dict[ProvenanceKey, tuple[str, SidecarLine | None]] = {}

    def fresh(self) -> TextResolution:
        """The same join and stager with no override."""
        return TextResolution(self.resolver, self.stager)

    @property
    def overrides(self) -> dict[ProvenanceKey, tuple[str, SidecarLine | None]]:
        return self._overrides

    def resolve(self, key) -> tuple[str | None, SidecarLine | None]:
        """``(text, line)`` for one key: its override, a keyword line's
        reminder text, else the join."""
        override = self._overrides.get(key)
        if override is not None:
            return override
        fragment = keyword_line_fragment(key)
        if fragment is not None:
            return self._keyword_line_text(fragment), None
        return self.resolver.resolve(key)

    def _keyword_line_text(self, fragment: str) -> str:
        """The text a keyword line with no script of its own is encoded from:
        its keyword's reminder text, its values filled in (root spec § 9.2);
        the fragment itself for a keyword the definitions do not expand."""
        keywords = self.stager.tokenizer.keywords
        line = keywords.line_of(fragment)
        if line is None or not keywords.expands(line.name):
            return fragment
        return keywords.line_text(line.name, line.details) or fragment

    def text_of(self, key) -> str | None:
        """The text one ability key is encoded from, on this run's surface:
        its single definition. A key the sidecar does not describe raises."""
        return self.resolve(key)[0]

    def line_of(self, key):
        return self.resolve(key)[1]

    def override(self, edit: TextOverride) -> None:
        """Make ``edit.key`` read ``edit.text`` for this resolution's life, with
        ``edit.like``'s line, its script and role spans replaced where the edit
        gives them (FR-063e)."""
        self._overrides[edit.key] = (edit.text, _edited_line(self.line_of(edit.like), edit))

    @contextmanager
    def reading_as(self, key, text: str) -> Iterator[None]:
        """Inside the block ``key`` reads ``text``, keeping its own line: how a
        probe has the encoder read one edited script."""
        previous = self._overrides.get(key)
        self.override(TextOverride(key, text, like=key))
        try:
            yield
        finally:
            if previous is None:
                del self._overrides[key]
            else:
                self._overrides[key] = previous

    def _is_synthetic(self, key) -> bool:
        """An overridden key or a keyword line: one the sidecar has no say in."""
        return key in self._overrides or keyword_line_fragment(key) is not None

    def options_for(self, key) -> tuple:
        """The keys of the charm-mode lines that follow ``key``'s line
        (FR-063a); none for an overridden key or a keyword line."""
        if self._is_synthetic(key):
            return ()
        return self.resolver.options_for(key)

    def board_keys(self, key) -> tuple:
        """The keys a board line of ``key`` reads as slots: a saga's bare
        keyword key is its chapter lines (``KeyResolver.board_keys``); an
        overridden key or a keyword line is itself."""
        if self._is_synthetic(key):
            return (key,)
        return self.resolver.board_keys(key)

    def room_of(self, key) -> str | None:
        """The dungeon room ``key``'s line is (``KeyResolver.room_of``); none
        for an overridden key or a keyword line."""
        if self._is_synthetic(key):
            return None
        return self.resolver.room_of(key)

    def keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far: how a truncation is named."""
        overridden = [key for key, (known, _) in self._overrides.items() if known == text]
        return self.resolver.keys_of(text) + overridden

    def keyword_of_key(self, key) -> str | None:
        """Which keyword one ability key's line *is*, if any."""
        fragment = keyword_line_fragment(key)
        if fragment is not None:
            return normalize_keyword(fragment)
        line = self.resolve(key)[1]
        return line.bare_keyword if line is not None else None


def _rewrites(edit: TextOverride) -> bool:
    return edit.script is not None or edit.role_spans is not None


def _edited_line(line: SidecarLine | None, edit: TextOverride) -> SidecarLine | None:
    """``line`` with the edit's script and role spans in place of its own,
    where the edit gives them."""
    if line is None or not _rewrites(edit):
        return line
    return replace(
        line,
        script_text=line.script_text if edit.script is None else edit.script,
        role_spans=line.role_spans if edit.role_spans is None else edit.role_spans,
    )


@dataclass(frozen=True, slots=True)
class StagedLines:
    """A batch's lines as the surface reads them: ``e`` as a row of the
    staged encodings, keyed by each key's text on the run's surface."""

    text: TextResolution
    rows: dict[str, int]

    def e_for(self, key) -> int | None:
        text = self.text.text_of(key)
        return None if text is None else self.rows.get(text)

    def has_line(self, key) -> bool:
        # A dropped key gets no token; a line a variant masks keeps its
        # token, zeroed in ``prepare``, so the geometry is the model's.
        return self.text.text_of(key) is not None

    def keyword_of(self, key) -> str | None:
        return self.text.keyword_of_key(key)

    def options_for(self, key) -> tuple:
        return self.text.options_for(key)

    def board_keys(self, key) -> tuple:
        return self.text.board_keys(key)

    def room_of(self, key) -> str | None:
        return self.text.room_of(key)
