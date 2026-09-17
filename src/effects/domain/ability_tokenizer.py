"""Tokenizing one ability line, with the offsets and hooks the encoder needs.

Wraps the shared :class:`~price_predictor.domain.tokenizer.MtgTokenizer`'s
vocabulary and token grammar, and adds the three things it does not have:

- **character offsets**, so the sidecar's role spans can be applied per token.
  The shared tokenizer normalizes before splitting and returns bare strings, so
  its output cannot be aligned back to the text the spans index.
- **``[MASK]`` and ``[CLS]``**, which it seeds neither of. ``[CLS]`` is the
  token the bottleneck ``e`` is pooled from and ``[MASK]`` is the MLM
  auxiliary's, so both have to be addressable by id.
- **keyword expansion**, which replaces a keyword token with its definition so
  the model learns "flying" and the sentence it stands for as the same thing.

Tokenization is whole-token only with no subword fallback (FR-066). An unknown
word is ``[UNK]``: unknown *keywords* are covered by forced expansion, and
unknown subtypes and token names by the next vocabulary rebuild. A subword
fallback would hide both behind plausible-looking fragments.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from effects.domain.provenance import RoleSpan
from price_predictor.domain.tokenizer import MtgTokenizer

#: Token grammar, matching ``MtgTokenizer``'s final split: words (which may
#: already carry an underscore), mana symbols, digit runs, and single
#: punctuation characters. Applied to the *original* text so offsets survive.
_TOKEN_RE = re.compile(r"[A-Za-z_]+|\{[^}]+\}|\d+|[^\s\w]")

#: When two role spans cover a token, the more specific one wins. A target
#: phrase sits inside the clause it restricts, so tagging it as that clause
#: would lose the distinction the role embedding exists to make.
_ROLE_PRIORITY: tuple[str, ...] = (
    "target-spec", "cost", "trigger-condition", "effect",
)

#: Keywords whose body is printed on the host card rather than in a reminder
#: template. Expanding one would substitute a generic sentence for the card's
#: own text, which is the opposite of what expansion is for.
HOST_BODIED_KEYWORDS: frozenset[str] = frozenset({
    "chapter", "class", "level up", "saga", "read ahead",
})

_PLACEHOLDER_RE = re.compile(r"%s")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Token:
    """One token, with everything the encoder reads off it.

    ``start``/``end`` index the source text, and are ``None`` for a token that
    came from an expansion rather than from the line — an expanded keyword's
    replacement text has no position in the original.
    """

    text: str
    token_id: int
    start: int | None = None
    end: int | None = None
    role: str | None = None
    #: The keyword this token was expanded from, where it was.
    expanded_from: str | None = None
    #: The numeric value of a digit token, for the monotone number embedding.
    number: float | None = None


class AbilityTokenizer:
    """Tokenizes one ability line into :class:`Token`s."""

    MASK = "[MASK]"
    CLS = "[CLS]"

    #: Bound on :attr:`_tokenize_cache`. A training run tokenizes on the order
    #: of 30,000 unique texts, so this is a guard against something unbounded
    #: (a corpus-construction pass over a whole raw corpus, say) rather than a
    #: policy tuned to a normal run's working set.
    _TOKENIZE_CACHE_CAP = 500_000

    def __init__(
        self,
        vocab: dict[str, int],
        keyword_definitions: dict | None = None,
    ) -> None:
        """
        Args:
            vocab: ``{token: id}``, as written by ``effects build-vocab``.
            keyword_definitions: ``keyword -> KeywordDefinition``, from
                ``extract-keyword-definitions``. Absent, no keyword is
                expandable and unknown keywords stay ``[UNK]``.
        """
        self._vocab = vocab
        self._reverse = {i: t for t, i in vocab.items()}
        self._definitions = keyword_definitions or {}
        # Definitions are looked up by the lowercased, underscore-joined form a
        # token carries, not by Forge's display name.
        self._definition_by_token = {
            self._keyword_token(name): definition
            for name, definition in self._definitions.items()
        }
        # Multi-word vocabulary entries, indexed by first word so
        # _multi_word_at compares a token only against the entries that could
        # possibly match it, rather than every multi-word entry in the
        # vocabulary at every token position. Longest first within a bucket
        # (by word count) so "double strike" is merged before "strike".
        multi_word = sorted(
            (t for t in vocab if "_" in t and not t.startswith("[")),
            key=lambda t: t.count("_"), reverse=True,
        )
        self._multi_word_by_first: dict[str, list[tuple[str, list[str]]]] = {}
        for entry in multi_word:
            parts = entry.split("_")
            self._multi_word_by_first.setdefault(parts[0], []).append(
                (entry, parts),
            )
        #: Memoises :meth:`tokenize` for the (overwhelmingly common)
        #: no-``role_spans`` case. See :meth:`tokenize` for why.
        self._tokenize_cache: dict[str, tuple[Token, ...]] = {}

    # ── ids ─────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def pad_id(self) -> int:
        return self._vocab[MtgTokenizer.PAD]

    @property
    def unk_id(self) -> int:
        return self._vocab[MtgTokenizer.UNK]

    @property
    def mask_id(self) -> int:
        return self._vocab[self.MASK]

    @property
    def cls_id(self) -> int:
        return self._vocab[self.CLS]

    def id_of(self, token: str) -> int:
        """The id of ``token``, or ``[UNK]``'s. No subword fallback (FR-066)."""
        return self._vocab.get(token, self.unk_id)

    def is_known(self, token: str) -> bool:
        return token in self._vocab

    # ── tokenizing ──────────────────────────────────────────────────────

    def tokenize(
        self, text: str, role_spans: tuple[RoleSpan, ...] = (),
    ) -> list[Token]:
        """Split ``text`` into tokens carrying their offsets and roles.

        ``role_spans`` are character ranges over this same text, as the sidecar
        records them for the line.

        With no ``role_spans`` the result depends on ``text`` alone, and
        :class:`~effects.application.surface_batching.SurfaceBatcher` calls
        this once per unique ability text in *every* batch of every epoch —
        the same tens of thousands of lines, retokenized from scratch each
        time. A profile of a training run put 80% of its wall time in this
        method for exactly that reason, so the no-``role_spans`` result is
        cached on the instance (bounded by :attr:`_TOKENIZE_CACHE_CAP`) and a
        **new list** is always handed back: ``Token`` is frozen, so sharing its
        elements is safe, but a caller (``expand_keywords``) builds its own
        list from what it's given, and must never be handed the cached list
        itself to build it from. A call carrying ``role_spans`` bypasses the
        cache: those calls are rare — only the corpus/record-building path
        attaches spans — and the result depends on the spans too, so keying
        the cache on the text alone would silently reuse a role assignment
        from an unrelated span set, and keying it on both text and spans would
        cache something the hot path never repeats.
        """
        if not role_spans:
            cached = self._tokenize_cache.get(text)
            if cached is not None:
                return list(cached)
            tokens = self._merge_multi_word(self._split_with_offsets(text), text)
            if len(self._tokenize_cache) >= self._TOKENIZE_CACHE_CAP:
                self._tokenize_cache.clear()
            self._tokenize_cache[text] = tuple(tokens)
            return list(tokens)
        tokens = self._split_with_offsets(text)
        tokens = self._merge_multi_word(tokens, text)
        tokens = [
            replace(token, role=self._role_at(token, role_spans))
            for token in tokens
        ]
        return tokens

    def _split_with_offsets(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        for match in _TOKEN_RE.finditer(text):
            raw = match.group(0)
            # Mana symbols keep their case; everything else lowercases, matching
            # the shared tokenizer's selective normalization.
            token_text = raw if raw.startswith("{") else raw.lower()
            tokens.append(Token(
                text=token_text,
                token_id=self.id_of(token_text),
                start=match.start(),
                end=match.end(),
                number=float(raw) if raw.isdigit() else None,
            ))
        return tokens

    def _merge_multi_word(self, tokens: list[Token], text: str) -> list[Token]:
        """Join runs of words that spell a multi-word vocabulary entry."""
        if not self._multi_word_by_first:
            return tokens
        merged: list[Token] = []
        index = 0
        while index < len(tokens):
            match = self._multi_word_at(tokens, index)
            if match is None:
                merged.append(tokens[index])
                index += 1
                continue
            entry, length = match
            first, last = tokens[index], tokens[index + length - 1]
            merged.append(Token(
                text=entry,
                token_id=self.id_of(entry),
                start=first.start,
                end=last.end,
            ))
            index += length
        return merged

    def _multi_word_at(
        self, tokens: list[Token], index: int,
    ) -> tuple[str, int] | None:
        # Only entries whose first word matches the token here can possibly
        # match, so this is the whole speedup: no comparison against the rest
        # of the vocabulary's multi-word entries.
        candidates = self._multi_word_by_first.get(tokens[index].text)
        if not candidates:
            return None
        for entry, parts in candidates:
            end = index + len(parts)
            if end > len(tokens):
                continue
            if all(tokens[index + i].text == part for i, part in enumerate(parts)):
                return entry, len(parts)
        return None

    @staticmethod
    def _role_at(token: Token, spans: tuple[RoleSpan, ...]) -> str | None:
        covering = [
            span.role for span in spans
            if token.start is not None
            and span.start <= token.start < span.end
        ]
        if not covering:
            return None
        for role in _ROLE_PRIORITY:
            if role in covering:
                return role
        return covering[0]

    # ── keyword expansion (FR-070) ──────────────────────────────────────

    def expandable(self, token: Token) -> bool:
        """Whether ``token`` names a keyword this tokenizer could expand."""
        if token.expanded_from is not None:
            # A keyword inside an expansion stays a token: expanding
            # recursively would bury the line's own text under definitions.
            return False
        if token.text in HOST_BODIED_KEYWORDS:
            return False
        definition = self._definition_by_token.get(token.text)
        return definition is not None and bool(
            self._definition_text(definition)
        )

    def must_expand(self, token: Token) -> bool:
        """Keywords the vocabulary does not know are always expanded.

        This is what lets a set the vocabulary predates be encoded at all: an
        unknown keyword would otherwise be a bare ``[UNK]``, while its
        definition is ordinary readable text.
        """
        return self.expandable(token) and not self.is_known(token.text)

    def expand_keywords(
        self,
        tokens: list[Token],
        *,
        probability: float = 0.0,
        rng: random.Random | None = None,
        instance_values: dict[str, list[str]] | None = None,
    ) -> list[Token]:
        """Replace some keyword tokens with their definitions.

        Every expandable keyword unknown to the vocabulary is expanded; a known
        one is expanded with probability ``probability``. A parameterized
        template is instantiated with that instance's own values where
        ``instance_values`` supplies them, and otherwise expands with the
        template's generic wording.
        """
        rng = rng or random.Random()
        values = instance_values or {}
        out: list[Token] = []
        for token in tokens:
            if self.must_expand(token) or (
                self.expandable(token) and rng.random() < probability
            ):
                out.extend(self._expansion(token, values.get(token.text)))
            else:
                out.append(token)
        return out

    def _expansion(self, token: Token, values: list[str] | None) -> list[Token]:
        definition = self._definition_by_token[token.text]
        text = self._instantiate(self._definition_text(definition), values)
        expanded = self._merge_multi_word(self._split_with_offsets(text), text)
        return [
            replace(part, start=None, end=None, role=token.role,
                    expanded_from=token.text)
            for part in expanded
        ]

    @staticmethod
    def _definition_text(definition) -> str | None:
        """The text a keyword expands to on the prose surface.

        Stage four overrides this with the captured implementation script on
        the script surface, falling back here where no script exists.
        """
        return getattr(definition, "reminder_template", None)

    @staticmethod
    def _instantiate(template: str | None, values: list[str] | None) -> str:
        """Fill a template's ``%s`` placeholders, or strip them.

        Forge writes parameterized reminder text with ``%s`` — "Cycling %s,
        Discard this card: Draw a card." With the instance's values the
        expansion says what this card actually costs. Without them (a keyword
        named inside another keyword's definition, with no instance of its own)
        the placeholders are dropped rather than tokenized, leaving the
        template's generic wording.
        """
        if not template:
            return ""
        if values:
            filled = template
            for value in values:
                filled = _PLACEHOLDER_RE.sub(value, filled, count=1)
            return _PLACEHOLDER_RE.sub("", filled).strip()
        return _WHITESPACE_RE.sub(
            " ", _PLACEHOLDER_RE.sub("", template),
        ).strip()

    @staticmethod
    def _keyword_token(display_name: str) -> str:
        """Forge's display name as the token form the vocabulary carries."""
        return display_name.lower().replace("'", "").replace(" ", "_")

    # ── the script surface (stage four) ─────────────────────────────────

    def tokenize_script(self, script_text: str) -> list[Token]:
        """Split a Forge script line, decomposing its compound selectors.

        A script selector stacks its restrictions with punctuation:
        ``Creature.nonDragon+OppCtrl`` is a creature, that is not a Dragon, that
        an opponent controls. Read as one token it is a symbol the model has
        seen a handful of times; read as three it is three restrictions the
        model has seen thousands of times each, in every combination.

        That is the whole reason the script surface is worth having — the
        vocabulary is compositional in a way prose is not.
        """
        tokens: list[Token] = []
        for match in _SCRIPT_TOKEN_RE.finditer(script_text):
            raw = match.group(0)
            if _SELECTOR_SPLIT_RE.search(raw):
                tokens.extend(self._split_selector(raw, match.start()))
                continue
            text = raw if raw.startswith("{") else raw.lower()
            tokens.append(Token(
                text=text,
                token_id=self.id_of(text),
                start=match.start(),
                end=match.end(),
                number=float(raw) if raw.isdigit() else None,
            ))
        return tokens

    def _split_selector(self, selector: str, offset: int) -> list[Token]:
        """One compound selector as its parts, each keeping its own offset."""
        tokens: list[Token] = []
        position = 0
        for part in _SELECTOR_SPLIT_RE.split(selector):
            if not part:
                position += 1
                continue
            start = selector.index(part, position)
            text = part.lower()
            tokens.append(Token(
                text=text,
                token_id=self.id_of(text),
                start=offset + start,
                end=offset + start + len(part),
                number=float(part) if part.isdigit() else None,
            ))
            position = start + len(part)
        return tokens


#: Script grammar. A run of selector characters (letters, digits, ``.``, ``+``,
#: ``-``) is one raw token that :meth:`AbilityTokenizer._split_selector` then
#: decomposes; ``$`` and ``|`` are the script's own structure and stay as
#: single-character tokens the model can read as separators.
_SCRIPT_TOKEN_RE = re.compile(r"[A-Za-z0-9_.+\-]+|\{[^}]+\}|[$|]|[^\s\w]")

#: The characters that stack restrictions inside one selector.
_SELECTOR_SPLIT_RE = re.compile(r"[.+]")
