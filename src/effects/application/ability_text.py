"""The text an ability is encoded from, resolved once per key.

One definition of the key-to-text join, because every consumer has to agree on
it exactly: ``build-corpus`` keys its rarity table and its per-text caps on it,
the trainer weights records by it, and the batcher encodes it. Two spellings
that disagree miss every lookup, and the model then trains on a zero vector
while every number it reports still moves.
"""

from __future__ import annotations

from collections import defaultdict

from effects.domain.ability_encoder import encoding_text
from effects.domain.records import EffectRecord, acting_keys
from effects.infrastructure.sidecar_io import UnconfiguredTree


def resolve_key(key, sidecars, surface: str) -> tuple[str | None, object]:
    """``(text, line)`` for one provenance key on ``surface``; ``(None, None)``
    where the key names no line.

    Falls back to the key itself where the sidecar carries neither surface, so
    a line that cannot be read is still a distinct unit for rarity weighting
    rather than collapsing into every other unreadable line.

    A key the sidecar does not describe raises ``KeyError``: that is the
    contract's fail-loudly case, and reading it as no text is what hid it.
    """
    try:
        line = sidecars.line_for(key)
    except UnconfiguredTree:
        return None, None
    if line is None:
        return None, None
    text = encoding_text(line, sidecars.prose_for(key), surface)
    return (
        text or f"{key.script_file}:{key.trait_kind}:{key.index_within_kind}",
        line,
    )


def text_for_key(key, sidecars, surface: str) -> str | None:
    """One provenance key's text on ``surface``, or None where it has none."""
    return resolve_key(key, sidecars, surface)[0]


def ability_text_of(
    record: EffectRecord, sidecars, surface: str = "prose",
) -> str | None:
    """The acting line's text on ``surface``, or None where no line acts.

    ``combat`` and the ``attackers``/``blockers`` subkinds have none, which is
    what makes them sample uniformly rather than by rarity.
    """
    for key in acting_keys(record):
        text = text_for_key(key, sidecars, surface)
        if text is not None:
            return text
    return None


class KeyResolver:
    """:func:`resolve_key` and a charm's option keys, memoised for a run.

    ``SidecarCache`` already reads each card once, but a key is still resolved
    several times per step — once for the batch's texts and again for every
    record its entity appears in — and every step meets the same few thousand
    cards. The answer depends only on the key, the sidecars and the surface,
    all fixed for the resolver's life, so it is computed once per run.

    A ``KeyError`` is deliberately not cached: a failure remembered as "no
    text" is exactly the fail-loudly case going quiet.
    """

    def __init__(self, sidecars, surface: str) -> None:
        self.sidecars = sidecars
        self.surface = surface
        self._resolved: dict[object, tuple[str | None, object]] = {}
        self._options: dict[object, tuple] = {}
        #: ``text -> keys resolved to it``: how a truncated text is named.
        self._keys_by_text: dict[str, list] = defaultdict(list)

    def resolve(self, key) -> tuple[str | None, object]:
        cached = self._resolved.get(key)
        if cached is None:
            cached = self._resolved[key] = resolve_key(key, self.sidecars, self.surface)
            if cached[0] is not None:
                self._keys_by_text[cached[0]].append(key)
        return cached

    def text_of(self, key) -> str | None:
        return self.resolve(key)[0]

    def acting_text(self, record: EffectRecord) -> str | None:
        """:func:`ability_text_of`, through the memo."""
        for key in acting_keys(record):
            text = self.text_of(key)
            if text is not None:
                return text
        return None

    def keys_of(self, text: str) -> list:
        """The keys resolved to ``text`` so far."""
        return list(self._keys_by_text.get(text, ()))

    def options_for(self, key) -> tuple:
        """The keys of the charm-mode lines that follow ``key``'s line (FR-063a).

        Empty for every line that is not a charm's root, and for a sidecar
        source that cannot say (a stub, an unconverted script).
        """
        cached = self._options.get(key)
        if cached is not None:
            return cached
        found: tuple = ()
        get = getattr(self.sidecars, "get", None)
        # A mode is never a root: its siblings follow it in the sidecar too,
        # and must not be read as its own options.
        if get is not None and getattr(key, "option", None) is None:
            try:
                sidecar = get(key.script_file)
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
