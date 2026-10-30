"""Tokenizing one ability line, with the offsets and hooks the encoder needs.

Wraps the shared ``MtgTokenizer``'s vocabulary and grammar, adding what it
lacks: character offsets (it normalizes before splitting, so its output cannot
be aligned to the role spans), the ``[MASK]``/``[CLS]``/``[SEG]`` specials, the
script surface's rules (FR-007–009b) and keyword expansion (FR-013, FR-014,
FR-017). No subword fallback (FR-066): an unknown word is ``[UNK]``, and a
fallback would hide it behind plausible-looking fragments.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import NamedTuple

from effects.domain.keyword_definitions import KeywordDefinition
from effects.domain.keyword_formatting import (
    format_reminder,
    java_format,
    strip_specifiers,
)
from effects.domain.keyword_names import display_name_of
from effects.domain.provenance import Role, RoleSpan
from effects.domain.recorded_version import recorded_version
from effects.domain.script_text import CHAIN_KEYS, SEG
from price_predictor.domain.tokenizer import MtgTokenizer


class Surface(StrEnum):
    """Which text of an ability line is encoded: the converted prose or the
    Forge script. Defined here rather than beside the encoder, which loads torch."""

    PROSE = "prose"
    SCRIPT = "script"


#: Each is one token wherever it appears, never ``[``, a word and ``]``.
SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[UNK]", "[MASK]", "[CLS]", SEG)

_SPECIAL_ALTERNATION = "|".join(re.escape(token) for token in SPECIAL_TOKENS)

#: ``MtgTokenizer``'s final split, applied to the original text so offsets
#: survive.
_TOKEN_RE = re.compile(
    rf"{_SPECIAL_ALTERNATION}|[A-Za-z_]+|\{{[^}}]+\}}|\d+|[^\s\w]"
)

#: Inside a ``TokenScript$`` value the underscore separates words (FR-009a).
_TOKEN_SCRIPT_GRAMMAR = re.compile(r"[A-Za-z]+|\{[^}]+\}|\d+|[^\sA-Za-z0-9_]")

#: Parameters whose value items stay one token on the script surface (FR-009).
_WHOLE_VALUE_RE = re.compile(
    r"(?<![A-Za-z])"
    rf"(?:{'|'.join((*sorted(CHAIN_KEYS), 'CounterType'))})"
    r"\$[ \t]*([^\s|]+)"
)
#: ``Choices$`` names chain labels only on a chooser API; elsewhere it is a
#: selector and splits like any other value.
_CHOICES_RE = re.compile(r"(?<![A-Za-z])Choices\$[ \t]*([^\s|]+)")
_CHOOSER_API_RE = re.compile(
    r"(?<![A-Za-z])(?:SP|DB|AB)\$[ \t]*"
    r"(?:Charm|GenericChoice|AssignGroup|VillainousChoice|Vote)(?![A-Za-z])"
)
#: The label opening a segment: at the start (an option line) or after ``[SEG]``.
_SEGMENT_OPENER_RE = re.compile(rf"(?:^|{re.escape(SEG)})[ \t]*(SV\d+)(?=:)")
_TOKEN_SCRIPT_VALUE_RE = re.compile(r"(?<![A-Za-z])TokenScript\$[ \t]*([^\s|]+)")

#: When two role spans cover a token, the more specific one wins: a target
#: phrase sits inside the clause it restricts.
_ROLE_PRIORITY: tuple[Role, ...] = (
    Role.TARGET_SPEC, Role.COST, Role.TRIGGER_CONDITION, Role.EFFECT,
)

#: Keywords whose body is printed on the host card rather than in a reminder
#: template, by lowercased display name (FR-016). Expanding one would replace
#: the card's own text with a generic sentence.
HOST_BODIED_KEYWORDS: frozenset[str] = frozenset({
    "chapter", "class", "level up", "saga", "read ahead",
})

#: Known-keyword expansion probability outside training (FR-012): zero, so a
#: known keyword reads as the token it trained as and a cache is reproducible.
INFERENCE_KEYWORD_EXPAND_P = 0.0

#: The tokenization rules a checkpoint trained under, in its
#: ``training_settings``. A gen-1 checkpoint records none and tokenized every
#: surface with the prose grammar.
TOKENIZER_RULES_KEY = "tokenizer_rules"
TOKENIZER_RULES = "gen-2"


def tokenizer_surface(vocab_surface: Surface, training_settings) -> Surface:
    """The surface whose rules a loaded checkpoint's tokenizer applies: the
    vocabulary's under the gen-2 rules, else the prose grammar it trained on."""
    rules = recorded_version(training_settings, TOKENIZER_RULES_KEY, (TOKENIZER_RULES,))
    return Surface.PROSE if rules is None else vocab_surface


def keyword_token(display_name: str) -> str:
    """Forge's display name as the token form the vocabulary carries."""
    return display_name.lower().replace("'", "").replace(" ", "_")


def withheld_keyword_tokens(withheld: str | None) -> frozenset[str]:
    """The token form of the keyword ``--withhold-keyword`` names, if any."""
    return frozenset({keyword_token(withheld)}) if withheld else frozenset()


@dataclass(frozen=True, slots=True)
class KeywordLine:
    """The keyword a line names and its values.

    ``details`` is everything after the first colon, as Forge hands it to the
    keyword class; ``name_end`` tells the name's tokens from the values'.
    """

    name: str
    details: str
    name_end: int


def bare_reminder(template: str | None) -> str:
    """A template with every placeholder removed, since a bare keyword token
    carries no values to fill them (FR-017)."""
    if not template:
        return ""
    return strip_specifiers(java_format(template, []))


class KeywordExpander:
    """Which keywords expand, and to what text (FR-013, FR-014, FR-016, FR-017).

    Both surfaces expand to the reminder template: the captured script holds
    only root lines, never their effect (FR-020). Without definitions no
    keyword expands.
    """

    def __init__(self, definitions: Mapping[str, KeywordDefinition] | None = None) -> None:
        self._definitions = definitions or {}
        self._name_by_token = {keyword_token(name): name for name in self._definitions}
        self._name_by_display = {name.lower(): name for name in self._definitions}

    def line_of(self, text: str) -> KeywordLine | None:
        """The keyword a line names by its display name, case-insensitively,
        or None. ``build-corpus`` audits that it fires only on ``Keyword``
        lines (FR-015)."""
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

    def name_of_token(self, token_text: str) -> str | None:
        """The display name a keyword token spells, or None."""
        return self._name_by_token.get(token_text)

    def expands(self, name: str) -> bool:
        """Whether a keyword, by display name, can expand, as a line or as a
        token: it has a reminder template, and its body is not printed on the
        host card (FR-016)."""
        if name.lower() in HOST_BODIED_KEYWORDS:
            return False
        return bool(self._definitions[name].reminder_template)

    def line_text(self, name: str, details: str = "") -> str:
        """The reminder text one keyword instance expands to, ``details``
        formatted by the definition's ``formatter``, specifiers removed."""
        definition = self._definitions[name]
        return format_reminder(definition.formatter, definition.reminder_template or "", details)

    def bare_text(self, name: str) -> str:
        """The reminder text a bare keyword token expands to (FR-014)."""
        return bare_reminder(self._definitions[name].reminder_template)


@dataclass(frozen=True, slots=True)
class Token:
    """One token, with everything the encoder reads off it.

    ``start``/``end`` index the source text, ``None`` for an expansion's token.
    """

    text: str
    token_id: int
    start: int | None = None
    end: int | None = None
    role: Role | None = None
    expanded_from: str | None = None
    number: float | None = None
    #: Set on the first token of a keyword line only: the line expands whole.
    keyword_line: KeywordLine | None = None


class AbilityTokenizer:
    """Tokenizes one ability line into :class:`Token`s."""

    MASK = "[MASK]"
    CLS = "[CLS]"

    #: A guard against an unbounded caller, far above a training run's texts.
    _TOKENIZE_CACHE_CAP = 500_000

    def __init__(
        self,
        vocab: dict[str, int],
        keyword_definitions: Mapping[str, KeywordDefinition] | None = None,
        *,
        surface: Surface = Surface.PROSE,
    ) -> None:
        """``surface`` is fixed for the tokenizer's life, which keeps the
        per-text cache valid (FR-010). Without ``keyword_definitions`` no
        keyword expands."""
        if surface not in (Surface.PROSE, Surface.SCRIPT):
            raise ValueError(f"unknown surface {surface!r}")
        self._vocab = vocab
        self._reverse = {i: t for t, i in vocab.items()}
        self.keywords = KeywordExpander(keyword_definitions)
        self.surface = surface
        # Indexed by first word, longest first within a bucket, so "double
        # strike" is merged before "strike".
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
        self._tokenize_cache: dict[
            tuple[str, tuple[RoleSpan, ...]], tuple[Token, ...]
        ] = {}

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

        Cached by text and spans together, since the batcher retokenizes the
        same lines every batch; a new list is always handed back, so no caller
        can mutate the cached one.
        """
        key = (text, tuple(role_spans))
        cached = self._tokenize_cache.get(key)
        if cached is not None:
            return list(cached)
        tokens = self._merge_multi_word(self._split_with_offsets(text))
        if role_spans:
            tokens = [
                replace(token, role=self._role_at(token, role_spans))
                for token in tokens
            ]
        tokens = self._tag_keyword_line(tokens, text)
        if len(self._tokenize_cache) >= self._TOKENIZE_CACHE_CAP:
            self._tokenize_cache.clear()
        self._tokenize_cache[key] = tuple(tokens)
        return list(tokens)

    def _split_with_offsets(self, text: str) -> list[Token]:
        if self.surface == Surface.SCRIPT:
            return self._split_script(text)
        return [
            self._token(match.group(0), match.start(), match.end())
            for match in _TOKEN_RE.finditer(text)
        ]

    def _token(self, raw: str, start: int, end: int) -> Token:
        # Mana symbols and specials keep their case, as in the shared tokenizer.
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

        Whole and ``TokenScript$`` values (FR-009, FR-009a) are found on the
        raw text first, since which parameter a value belongs to is lost once
        it is split; every other word then splits at camel case (FR-008).
        """
        tokens: list[Token] = []
        position = 0
        for start, end, whole in _script_regions(text):
            tokens.extend(self._split_camel_region(text, position, start))
            if whole:
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

    def _merge_multi_word(self, tokens: list[Token]) -> list[Token]:
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
    def _role_at(token: Token, spans: tuple[RoleSpan, ...]) -> Role | None:
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

    def _tag_keyword_line(self, tokens: list[Token], text: str) -> list[Token]:
        if not tokens:
            return tokens
        line = self.keywords.line_of(text)
        if line is None:
            return tokens
        tokens[0] = replace(tokens[0], keyword_line=line)
        return tokens

    # ── keyword expansion (FR-070, FR-012–FR-018) ───────────────────────

    def expandable(self, token: Token) -> bool:
        """Whether ``token`` names a keyword this tokenizer could expand."""
        if token.expanded_from is not None:
            # Never recursively, which would bury the line under definitions.
            return False
        name = self.keywords.name_of_token(token.text)
        return name is not None and self.keywords.expands(name)

    def must_expand(self, token: Token) -> bool:
        """Keywords the vocabulary does not know are always expanded, so a set
        the vocabulary predates reads as text rather than ``[UNK]``."""
        return self.expandable(token) and not self.is_known(token.text)

    def expand_keywords(
        self,
        tokens: list[Token],
        *,
        probability: float = 0.0,
        rng: random.Random | None = None,
        forced: frozenset[str] = frozenset(),
    ) -> list[Token]:
        """Replace some keywords with their definitions.

        A keyword line is decided once, as a whole: expanded when its name is
        unknown, else with ``probability``, into the template filled with its
        own values (FR-013, FR-017). Any other line goes token by token, the
        template's placeholders removed (FR-014). A keyword in ``forced``
        always expands, as an unknown one does, and an empty ``forced`` takes
        exactly the draws it would without the parameter. Without ``rng`` the
        draws decide nothing, so a ``probability`` they would decide is refused.
        """
        if rng is None:
            if 0.0 < probability < 1.0:
                raise ValueError(f"expanding at probability {probability} needs an rng")
            rng = random.Random(0)
        line = tokens[0].keyword_line if tokens else None
        if line is not None:
            return self._expand_line(tokens, line, probability, rng, forced)
        out: list[Token] = []
        for token in tokens:
            if self.must_expand(token) or (
                self.expandable(token)
                and (token.text in forced or rng.random() < probability)
            ):
                out.extend(self._expansion(token))
            else:
                out.append(token)
        return out

    def encoding_tokens(
        self,
        text: str,
        *,
        role_spans: tuple[RoleSpan, ...] = (),
        withheld: frozenset[str] = frozenset(),
        probability: float = 0.0,
        rng: random.Random | None = None,
    ) -> list[Token]:
        """The tokens one ability text is encoded from, wherever it is encoded.

        The ``withheld`` keyword always expands, and its token is then dropped
        everywhere it is left, inside an expansion included (flying's reminder
        text names flying).
        """
        tokens = self.expand_keywords(
            self.tokenize(text, role_spans),
            probability=probability, rng=rng, forced=withheld,
        )
        if withheld:
            tokens = [token for token in tokens if token.text not in withheld]
        return tokens

    def _expand_line(
        self,
        tokens: list[Token],
        line: KeywordLine,
        probability: float,
        rng: random.Random,
        forced: frozenset[str] = frozenset(),
    ) -> list[Token]:
        if not self.keywords.expands(line.name):
            return list(tokens)
        name_tokens = [
            t for t in tokens if t.start is not None and t.start < line.name_end
        ]
        unknown = not name_tokens or any(
            t.token_id == self.unk_id for t in name_tokens
        )
        form = keyword_token(line.name)
        if not (unknown or form in forced or rng.random() < probability):
            return list(tokens)
        text = self.keywords.line_text(line.name, line.details)
        return self._as_expansion(text, role=tokens[0].role, keyword=form)

    def _expansion(self, token: Token) -> list[Token]:
        name = self.keywords.name_of_token(token.text)
        if name is None:
            raise ValueError(f"{token.text!r} names no keyword definition")
        text = self.keywords.bare_text(name)
        return self._as_expansion(text, role=token.role, keyword=token.text)

    def _as_expansion(
        self, text: str, *, role: Role | None, keyword: str,
    ) -> list[Token]:
        expanded = self._merge_multi_word(self._split_with_offsets(text))
        return [
            replace(part, start=None, end=None, role=role,
                    expanded_from=keyword)
            for part in expanded
        ]


def camel_parts(word: str) -> list[tuple[int, int]]:
    """``(start, end)`` of each part of ``word``, broken at lowercase→uppercase
    only, so an all-caps run (``ETB``) stays whole (FR-008)."""
    parts: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(word)):
        if word[index - 1].islower() and word[index].isupper():
            parts.append((start, index))
            start = index
    parts.append((start, len(word)))
    return parts


class _Region(NamedTuple):
    """A span the script grammar skips: kept ``whole`` (FR-009), or split at
    underscores (FR-009a). Sorts underscore-first at an equal span."""

    start: int
    end: int
    whole: bool


def _item_regions(match: re.Match) -> list[_Region]:
    """One whole region per non-empty item of a comma list in group 1."""
    regions: list[_Region] = []
    offset = match.start(1)
    for item in match.group(1).split(","):
        if item:
            regions.append(_Region(offset, offset + len(item), whole=True))
        offset += len(item) + 1
    return regions


def _script_regions(text: str) -> list[_Region]:
    """The ordered, non-overlapping regions of ``text`` the grammar skips."""
    regions: list[_Region] = []
    for match in _WHOLE_VALUE_RE.finditer(text):
        regions.extend(_item_regions(match))
    for match in _CHOICES_RE.finditer(text):
        if _CHOOSER_API_RE.search(_segment_at(text, match.start())):
            regions.extend(_item_regions(match))
    regions.extend(
        _Region(match.start(1), match.end(1), whole=True)
        for match in _SEGMENT_OPENER_RE.finditer(text)
    )
    regions.extend(
        _Region(match.start(1), match.end(1), whole=False)
        for match in _TOKEN_SCRIPT_VALUE_RE.finditer(text)
    )
    regions.sort()
    ordered: list[_Region] = []
    for region in regions:
        if ordered and region.start < ordered[-1].end:
            continue
        ordered.append(region)
    return ordered


def _segment_at(text: str, position: int) -> str:
    """The ``[SEG]``-delimited segment of ``text`` holding ``position``."""
    start = text.rfind(SEG, 0, position)
    start = 0 if start < 0 else start + len(SEG)
    end = text.find(SEG, position)
    return text[start:] if end < 0 else text[start:end]
