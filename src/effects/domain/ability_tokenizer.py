"""Tokenizing one ability line, with the offsets and hooks the encoder needs.

Wraps the shared :class:`~price_predictor.domain.tokenizer.MtgTokenizer`'s
vocabulary and token grammar, and adds the things it does not have:

- **character offsets**, so the sidecar's role spans can be applied per token.
  The shared tokenizer normalizes before splitting and returns bare strings, so
  its output cannot be aligned back to the text the spans index.
- **``[MASK]``, ``[CLS]`` and ``[SEG]``**, which it seeds none of. ``[CLS]`` is
  the token the bottleneck ``e`` is pooled from, ``[MASK]`` is the MLM
  auxiliary's, and ``[SEG]`` separates the segments of a chained script text.
- **the script surface's rules** (FR-007–009b), applied exactly when the
  vocabulary is a script vocabulary: camel-case compounds split into their
  words, chain labels and counter types kept whole, token script names split
  at their underscores. The prose surface is the shared grammar unchanged.
- **keyword expansion**, which replaces a keyword with its reminder text so the
  model learns "flying" and the sentence it stands for as the same thing. A
  keyword *line* is recognised by its display name and expands whole, with its
  own instance values filled in the way Forge fills them (FR-013, FR-017); a
  keyword word inside any other line expands through the token path (FR-014).

Tokenization is whole-token only with no subword fallback (FR-066). An unknown
word is ``[UNK]``: unknown *keywords* are covered by forced expansion, and
unknown subtypes and token names by the next vocabulary rebuild. A subword
fallback would hide both behind plausible-looking fragments.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from effects.domain.keyword_formatting import (
    format_reminder,
    java_format,
    strip_specifiers,
)
from effects.domain.provenance import RoleSpan
from price_predictor.domain.tokenizer import MtgTokenizer

#: The two encoding surfaces. Spelled here rather than imported from
#: ``ability_encoder``, which loads torch, so the tokenizer stays importable by
#: the corpus tools that never touch a model.
SURFACE_PROSE = "prose"
SURFACE_SCRIPT = "script"

#: The bracketed specials the vocabulary seeds. Each is one token wherever it
#: appears, never ``[``, a word and ``]``.
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEG]")

#: Separator between the segments of a chained script text (FR-002).
SEG = "[SEG]"

_SPECIAL_ALTERNATION = "|".join(re.escape(token) for token in SPECIAL_TOKENS)

#: Token grammar, matching ``MtgTokenizer``'s final split: words (which may
#: already carry an underscore), mana symbols, digit runs, and single
#: punctuation characters, plus the bracketed specials as single tokens.
#: Applied to the *original* text so offsets survive.
_TOKEN_RE = re.compile(
    rf"{_SPECIAL_ALTERNATION}|[A-Za-z_]+|\{{[^}}]+\}}|\d+|[^\s\w]"
)

#: Inside a ``TokenScript$`` value the underscore separates words and is
#: dropped (FR-009a): ``w_1_1_soldier`` reads ``w 1 1 soldier``.
_TOKEN_SCRIPT_GRAMMAR = re.compile(r"[A-Za-z]+|\{[^}]+\}|\d+|[^\sA-Za-z0-9_]")

#: Parameters whose value items stay one token on the script surface (FR-009):
#: the chain references and the counter type. ``Choices$`` is handled apart.
_WHOLE_VALUE_RE = re.compile(
    r"(?<![A-Za-z])"
    r"(?:Execute|SubAbility|RepeatSubAbility|ReplaceWith|CounterType)"
    r"\$[ \t]*([^\s|]+)"
)
#: ``Choices$`` names chain labels only on a chooser API; on every other API
#: (``PutCounter``, ``ChangeZone``, …) it is a selector like
#: ``Creature.YouCtrl`` and splits like any other value.
_CHOICES_RE = re.compile(r"(?<![A-Za-z])Choices\$[ \t]*([^\s|]+)")
_CHOOSER_API_RE = re.compile(
    r"(?<![A-Za-z])(?:SP|DB|AB)\$[ \t]*"
    r"(?:Charm|GenericChoice|AssignGroup|VillainousChoice|Vote)(?![A-Za-z])"
)
#: The label that opens a segment: at the start of the text (an option line's
#: mode) or right after a ``[SEG]``, in its FR-002a form.
_SEGMENT_OPENER_RE = re.compile(r"(?:^|\[SEG\])[ \t]*(SV\d+)(?=:)")
_TOKEN_SCRIPT_VALUE_RE = re.compile(r"(?<![A-Za-z])TokenScript\$[ \t]*([^\s|]+)")

#: When two role spans cover a token, the more specific one wins. A target
#: phrase sits inside the clause it restricts, so tagging it as that clause
#: would lose the distinction the role embedding exists to make.
_ROLE_PRIORITY: tuple[str, ...] = (
    "target-spec", "cost", "trigger-condition", "effect",
)

#: Keywords whose body is printed on the host card rather than in a reminder
#: template, by display name (lowercased). Expanding one would substitute a
#: generic sentence for the card's own text, which is the opposite of what
#: expansion is for. Compared by display name (FR-016): a token-form
#: comparison never matched the two-word names.
HOST_BODIED_KEYWORDS: frozenset[str] = frozenset({
    "chapter", "class", "level up", "saga", "read ahead",
})

#: Known-keyword expansion probability everywhere outside training: validation,
#: evaluation and the ability cache (FR-012). Zero, so a known keyword reads as
#: the token it trained as and a cache is reproducible; an unknown keyword
#: still expands at any probability. Training keeps ``--keyword-expand-p``.
INFERENCE_KEYWORD_EXPAND_P = 0.0

#: The tokenization rules a checkpoint trained under, recorded in its
#: ``training_settings`` under :data:`TOKENIZER_RULES_KEY`. Every training run
#: of this feature records :data:`TOKENIZER_RULES`; a checkpoint recording none
#: (gen-1, feature 023) tokenized every surface with the prose grammar, and the
#: script-surface rules would hand it camel-case parts its vocabulary never
#: held — about one script token in thirty reads as ``[UNK]`` that way, and
#: nearly three texts in four change their token sequence.
TOKENIZER_RULES_KEY = "tokenizer_rules"
TOKENIZER_RULES = "gen-2"


def tokenizer_surface(vocab_surface: str, training_settings) -> str:
    """The surface whose rules a loaded checkpoint's tokenizer applies.

    The vocabulary's surface (FR-010) for a checkpoint trained under the gen-2
    rules; the prose grammar for one that records no rules, because that is
    what it read in training whichever surface its vocabulary was.
    """
    rules = (training_settings or {}).get(TOKENIZER_RULES_KEY)
    return vocab_surface if rules == TOKENIZER_RULES else SURFACE_PROSE


def keyword_token(display_name: str) -> str:
    """Forge's display name as the token form the vocabulary carries."""
    return display_name.lower().replace("'", "").replace(" ", "_")


_HOST_BODIED_TOKENS: frozenset[str] = frozenset(
    keyword_token(name) for name in HOST_BODIED_KEYWORDS
)


def display_name_of(line_text: str) -> str:
    """A keyword line's display name: its text before the first colon (FR-013).

    ``Ward:2`` is ``Ward``, ``TypeCycling:Basic:1 B`` is ``TypeCycling`` and a
    bare ``Flying`` is itself. Whether it names a keyword is the definitions
    table's question, not this function's.
    """
    return line_text.split(":", 1)[0].strip()


@dataclass(frozen=True, slots=True)
class KeywordLine:
    """What makes a line a keyword line: the keyword it names and its values.

    ``name`` is the definition's own display name, ``details`` everything after
    the first colon (the instance values, as Forge hands them to the keyword
    class), and ``name_end`` the offset where the display name ends, so the
    tokens that spell the name can be told from the ones that spell the values.
    """

    name: str
    details: str
    name_end: int


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
    #: Set on the first token of a keyword line only: the line expands whole.
    keyword_line: KeywordLine | None = None


class AbilityTokenizer:
    """Tokenizes one ability line into :class:`Token`s."""

    MASK = "[MASK]"
    CLS = "[CLS]"
    SEG = SEG

    #: Bound on :attr:`_tokenize_cache`. A training run tokenizes on the order
    #: of 30,000 unique texts, so this is a guard against something unbounded
    #: (a corpus-construction pass over a whole raw corpus, say) rather than a
    #: policy tuned to a normal run's working set.
    _TOKENIZE_CACHE_CAP = 500_000

    def __init__(
        self,
        vocab: dict[str, int],
        keyword_definitions: dict | None = None,
        *,
        surface: str = SURFACE_PROSE,
    ) -> None:
        """
        Args:
            vocab: ``{token: id}``, as written by ``effects build-vocab``.
            keyword_definitions: ``keyword -> KeywordDefinition``, from
                ``extract-keyword-definitions``. Absent, no keyword is
                expandable and unknown keywords stay ``[UNK]``.
            surface: ``prose`` or ``script``, from ``surface_of`` on the
                vocabulary's path. Fixed for the tokenizer's life, which is
                what keeps the per-text cache valid (FR-010).
        """
        if surface not in (SURFACE_PROSE, SURFACE_SCRIPT):
            raise ValueError(f"unknown surface {surface!r}")
        self._vocab = vocab
        self._reverse = {i: t for t, i in vocab.items()}
        self._definitions = keyword_definitions or {}
        self.surface = surface
        # Definitions are looked up by the lowercased, underscore-joined form a
        # token carries on the token path, and by the lowercased display name
        # on the keyword-line path.
        self._definition_by_token = {
            keyword_token(name): definition
            for name, definition in self._definitions.items()
        }
        self._name_by_display = {
            name.lower(): name for name in self._definitions
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
            tokens = self._tag_keyword_line(
                self._merge_multi_word(self._split_with_offsets(text), text), text,
            )
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
        return self._tag_keyword_line(tokens, text)

    def _split_with_offsets(self, text: str) -> list[Token]:
        if self.surface == SURFACE_SCRIPT:
            return self._split_script(text)
        return [
            self._token(match.group(0), match.start(), match.end())
            for match in _TOKEN_RE.finditer(text)
        ]

    def _token(self, raw: str, start: int, end: int) -> Token:
        # Mana symbols and the bracketed specials keep their case; everything
        # else lowercases, matching the shared tokenizer's selective
        # normalization.
        token_text = raw if raw.startswith(("{", "[")) else raw.lower()
        return Token(
            text=token_text,
            token_id=self.id_of(token_text),
            start=start,
            end=end,
            number=float(raw) if raw.isdigit() else None,
        )

    # ── the script surface (FR-008, FR-009, FR-009a) ────────────────────

    def _split_script(self, text: str) -> list[Token]:
        """The prose grammar, with the script surface's three rules on top.

        Whole-token values (FR-009) and ``TokenScript$`` values (FR-009a) are
        found by parameter name on the raw text first, because both are
        decided by *which* parameter a value belongs to — something the
        grammar cannot see once the text is split. Everything else is split
        by the grammar, then each word is broken where a lowercase letter
        meets an uppercase one (FR-008), keys and values alike.
        """
        tokens: list[Token] = []
        position = 0
        for start, end, kind in _script_regions(text):
            tokens.extend(self._split_camel_region(text, position, start))
            if kind == "whole":
                whole = text[start:end].lower()
                tokens.append(Token(
                    text=whole, token_id=self.id_of(whole), start=start, end=end,
                ))
            else:
                tokens.extend(
                    self._token(match.group(0), match.start(), match.end())
                    for match in _TOKEN_SCRIPT_GRAMMAR.finditer(text, start, end)
                )
            position = end
        tokens.extend(self._split_camel_region(text, position, len(text)))
        return tokens

    def _split_camel_region(self, text: str, start: int, end: int) -> list[Token]:
        tokens: list[Token] = []
        for match in _TOKEN_RE.finditer(text, start, end):
            raw = match.group(0)
            if not raw[0].isalpha():
                tokens.append(self._token(raw, match.start(), match.end()))
                continue
            for part_start, part_end in camel_parts(raw):
                tokens.append(self._token(
                    raw[part_start:part_end],
                    match.start() + part_start, match.start() + part_end,
                ))
        return tokens

    # ── multi-word entries and roles ────────────────────────────────────

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

    # ── keyword lines (FR-013, FR-016) ──────────────────────────────────

    def keyword_line_of(self, text: str) -> KeywordLine | None:
        """The keyword a line names by its display name, or None.

        Case-insensitive against the definitions table. This is the match
        ``build-corpus`` audits (FR-015): it must only ever fire on a sidecar
        line whose ``script_api_type`` is ``Keyword``.
        """
        if not self._name_by_display:
            return None
        name = self._name_by_display.get(display_name_of(text).lower())
        if name is None:
            return None
        colon = text.find(":")
        details = text[colon + 1:] if colon >= 0 else ""
        return KeywordLine(
            name=name, details=details,
            name_end=colon if colon >= 0 else len(text),
        )

    def _tag_keyword_line(self, tokens: list[Token], text: str) -> list[Token]:
        if not tokens:
            return tokens
        line = self.keyword_line_of(text)
        if line is None:
            return tokens
        tokens[0] = replace(tokens[0], keyword_line=line)
        return tokens

    def _line_expandable(self, line: KeywordLine) -> bool:
        if line.name.lower() in HOST_BODIED_KEYWORDS:
            return False
        return bool(self._definition_text(self._definitions[line.name]))

    # ── keyword expansion (FR-070, FR-012–FR-018) ───────────────────────

    def expandable(self, token: Token) -> bool:
        """Whether ``token`` names a keyword this tokenizer could expand."""
        if token.expanded_from is not None:
            # A keyword inside an expansion stays a token: expanding
            # recursively would bury the line's own text under definitions.
            return False
        if token.text in _HOST_BODIED_TOKENS:
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
        """Replace some keywords with their definitions.

        A **keyword line** — one whose display name names a definition — is
        decided once for the whole line: it expands when any token of its name
        is unknown to the vocabulary, and otherwise with probability
        ``probability``; expanded, every token of the line is replaced by the
        reminder template filled with the line's own values as Forge formats
        them (FR-013, FR-017). Any other line goes token by token: every
        expandable keyword unknown to the vocabulary is expanded, a known one
        with probability ``probability`` (FR-014). ``instance_values`` supplies
        already-formatted values for a keyword, by its token form; without
        them a template's unfilled placeholders are removed.
        """
        rng = rng or random.Random()
        values = instance_values or {}
        line = tokens[0].keyword_line if tokens else None
        if line is not None:
            return self._expand_line(tokens, line, probability, rng, values)
        out: list[Token] = []
        for token in tokens:
            if self.must_expand(token) or (
                self.expandable(token) and rng.random() < probability
            ):
                out.extend(self._expansion(token, values.get(token.text)))
            else:
                out.append(token)
        return out

    def _expand_line(
        self,
        tokens: list[Token],
        line: KeywordLine,
        probability: float,
        rng: random.Random,
        values: dict[str, list[str]],
    ) -> list[Token]:
        if not self._line_expandable(line):
            return list(tokens)
        name_tokens = [
            t for t in tokens if t.start is not None and t.start < line.name_end
        ]
        unknown = not name_tokens or any(
            t.token_id == self.unk_id for t in name_tokens
        )
        if not (unknown or rng.random() < probability):
            return list(tokens)
        form = keyword_token(line.name)
        text = self.expansion_text(line.name, line.details, values.get(form))
        return self._as_expansion(text, role=tokens[0].role, keyword=form)

    def expansion_text(
        self, name: str, details: str = "", values: list[str] | None = None,
    ) -> str:
        """The reminder text one keyword instance expands to, specifiers removed.

        ``details`` are the line's instance values as written after its first
        colon, formatted by the definition's ``formatter``; ``values`` are
        already-formatted values and take precedence. Neither given, the
        template's generic wording.
        """
        definition = self._definitions[name]
        template = self._definition_text(definition) or ""
        if values:
            return strip_specifiers(java_format(template, list(values)))
        return format_reminder(
            getattr(definition, "formatter", None), template, details,
        )

    def _expansion(self, token: Token, values: list[str] | None) -> list[Token]:
        definition = self._definition_by_token[token.text]
        text = self._instantiate(self._definition_text(definition), values)
        return self._as_expansion(text, role=token.role, keyword=token.text)

    def _as_expansion(
        self, text: str, *, role: str | None, keyword: str,
    ) -> list[Token]:
        expanded = self._merge_multi_word(self._split_with_offsets(text), text)
        return [
            replace(part, start=None, end=None, role=role,
                    expanded_from=keyword)
            for part in expanded
        ]

    @staticmethod
    def _definition_text(definition) -> str | None:
        """The text a keyword expands to, on both surfaces: its reminder template.

        The captured implementation script is not used (FR-020). It holds only
        the keyword's root lines, never the effect they execute.
        """
        return getattr(definition, "reminder_template", None)

    @staticmethod
    def _instantiate(template: str | None, values: list[str] | None) -> str:
        """Fill a template's placeholders in order, then strip what is left.

        Forge writes parameterized reminder text with ``%s``, ``%d`` and their
        explicit-index forms — "Cycling %s, Discard this card: Draw a card.",
        "Target a %1$s as you cast this. This card enters attached to that
        %1$s." An explicit index repeats its value at every use. Whatever no
        value fills is removed, so no expansion carries a ``%`` token
        (FR-017), leaving the template's generic wording.
        """
        if not template:
            return ""
        return strip_specifiers(java_format(template, list(values or ())))

    @staticmethod
    def _keyword_token(display_name: str) -> str:
        """Forge's display name as the token form the vocabulary carries."""
        return keyword_token(display_name)


def camel_parts(word: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each part of ``word``, broken at lowercase→uppercase.

    ``nonDragon`` is ``non`` + ``Dragon`` and ``YouCtrl`` is ``You`` +
    ``Ctrl``; an all-caps run (``ETB``) or a lone capital stays with its word,
    because only a lowercase letter followed by an uppercase one ends a word
    (FR-008).
    """
    parts: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(word)):
        if word[index - 1].islower() and word[index].isupper():
            parts.append((start, index))
            start = index
    parts.append((start, len(word)))
    return parts


def _script_regions(text: str) -> list[tuple[int, int, str]]:
    """Ordered, non-overlapping ``(start, end, kind)`` spans the grammar skips.

    ``whole`` spans are FR-009 values, each one token; ``underscore`` spans are
    ``TokenScript$`` values, split at ``_`` (FR-009a).
    """
    regions: list[tuple[int, int, str]] = []
    for match in _WHOLE_VALUE_RE.finditer(text):
        offset = match.start(1)
        for item in match.group(1).split(","):
            if item:
                regions.append((offset, offset + len(item), "whole"))
            offset += len(item) + 1
    for match in _CHOICES_RE.finditer(text):
        if not _CHOOSER_API_RE.search(_segment_at(text, match.start())):
            continue
        offset = match.start(1)
        for item in match.group(1).split(","):
            if item:
                regions.append((offset, offset + len(item), "whole"))
            offset += len(item) + 1
    for match in _SEGMENT_OPENER_RE.finditer(text):
        regions.append((match.start(1), match.end(1), "whole"))
    for match in _TOKEN_SCRIPT_VALUE_RE.finditer(text):
        regions.append((match.start(1), match.end(1), "underscore"))
    regions.sort()
    ordered: list[tuple[int, int, str]] = []
    for region in regions:
        if ordered and region[0] < ordered[-1][1]:
            continue
        ordered.append(region)
    return ordered


def _segment_at(text: str, position: int) -> str:
    """The ``[SEG]``-delimited segment of ``text`` holding ``position``."""
    start = text.rfind(SEG, 0, position)
    start = 0 if start < 0 else start + len(SEG)
    end = text.find(SEG, position)
    return text[start:] if end < 0 else text[start:end]
