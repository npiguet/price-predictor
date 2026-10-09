"""``build-vocab``: the effects-side tokenizer vocabulary (FR-064, FR-065).

A thin wrapper over ``price_predictor.application.build_vocabulary``, mirroring
``sealed``'s, with the differences that matter:

- the scan covers **three** sources — converted cards, converted token scripts,
  and the keyword-definition file — because a token a card creates and a
  keyword's reminder text are both text the encoder has to read;
- ``[CLS]`` and ``[SEG]`` are seeded alongside ``[MASK]``: the ability encoder
  pools through ``[CLS]`` to the bottleneck ``e``, and ``[SEG]`` separates the
  segments of a chained script text;
- every word of every keyword template is **seeded** (FR-019), so a
  ``--target-size`` truncation can never turn an expansion into ``[UNK]``, and
  the build fails if any definition would still expand to one;
- on the script surface the sidecars' script lines are tokenized by the
  script surface's own rules and counted **before** the prose files, so a
  truncation drops prose tokens first; the chain labels ``sv1…svN`` and every
  part a parameter key splits into are seeded (FR-002a, FR-009b). They cannot
  go through the shared builder, whose tokenizer lowercases before splitting
  and would erase the camel case the script rules split on;
- the prose and script surfaces write **separate files**, so a script rebuild
  never overwrites the vocabulary a prose checkpoint recorded and can still be
  loaded against.

The keyword definitions' generated scripts are not scanned (FR-020): no
expansion substitutes them, on either surface.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain.ability_tokenizer import (
    AbilityTokenizer,
    Token,
    camel_parts,
)
from price_predictor.application.build_vocabulary import (
    MULTI_WORD_KEYWORDS,
    build_vocabulary,
    truncate_to_target_size,
)
from price_predictor.infrastructure.tokenizer_store import save_vocabulary

logger = logging.getLogger(__name__)

SURFACE_PROSE = "prose"
SURFACE_SCRIPT = "script"

DEFAULT_CARDS_FOLDERS: tuple[Path, ...] = (
    Path("output/cardsfolder/"),
    Path("output/tokenscripts/"),
)
DEFAULT_KEYWORD_DEFINITIONS = Path("output/effects/keyword-definitions.json")
DEFAULT_VOCAB_PATH = Path("models/effects/vocab.txt")
DEFAULT_SCRIPT_VOCAB_PATH = Path("models/effects/vocab-script.txt")
DEFAULT_TARGET_SIZE = 5000

#: Seeded into every effects vocabulary regardless of corpus frequency.
#: ``[PAD]``, ``[UNK]`` and ``cardname`` come from the shared builder;
#: ``[MASK]`` is the MLM auxiliary's, ``[CLS]`` is the pooling token the
#: bottleneck ``e`` is read from, and ``[SEG]`` separates script segments.
SEEDED_SPECIALS: tuple[str, ...] = (
    "[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", "[SEG]",
)

#: Script-line frequency floor, the shared builder's production threshold.
_FREQ_THRESHOLD = 2

#: A chain label in its FR-002a form, and the number it carries.
_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])SV(\d+)(?![A-Za-z0-9])")
#: A parameter key, or a ``$``-prefix glued to its value (``Count$Valid``).
_KEY_RE = re.compile(r"([A-Za-z]+)\$")
#: A ``*Description$`` parameter with its value, up to the next separator:
#: free prose, so its words say nothing about how well the vocabulary covers
#: the script syntax (FR-011).
_DESCRIPTION_PARAM_RE = re.compile(r"\w*Description\$[^|\[]*")


@dataclass
class BuildVocabConfig:
    surface: str = SURFACE_PROSE
    cards_folders: tuple[Path, ...] = DEFAULT_CARDS_FOLDERS
    vocab_path: Path | None = None
    keyword_definitions: Path = field(
        default_factory=lambda: DEFAULT_KEYWORD_DEFINITIONS,
    )
    target_size: int = DEFAULT_TARGET_SIZE

    def resolved_vocab_path(self) -> Path:
        """``--vocab-path``, defaulting per surface so the two never collide."""
        if self.vocab_path is not None:
            return Path(self.vocab_path)
        return (
            DEFAULT_SCRIPT_VOCAB_PATH
            if self.surface == SURFACE_SCRIPT
            else DEFAULT_VOCAB_PATH
        )


class EmptyCardsFolderError(RuntimeError):
    """No ``*.txt`` under any ``--cards-folder``. Maps to exit code 1."""


class KeywordExpansionUnknownError(ValueError):
    """A keyword definition would expand to a text holding ``[UNK]`` (FR-019)."""


@dataclass
class SeededTokens:
    """What the build seeds beyond the specials, by kind, for the report."""

    labels: list[str] = field(default_factory=list)
    template_words: list[str] = field(default_factory=list)
    key_parts: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        return [*self.labels, *self.template_words, *self.key_parts]


def _scan_sources(config: BuildVocabConfig) -> list[Path]:
    """Existing directories to scan, in the order they were given."""
    present = [Path(folder) for folder in config.cards_folders if Path(folder).is_dir()]
    if not present or not any(any(p.rglob("*.txt")) for p in present):
        raise EmptyCardsFolderError(
            "No converted cards under "
            + ", ".join(str(f) for f in config.cards_folders)
            + ". Run python -m price_predictor convert first."
        )
    return present


def _keyword_corpus_text(definitions: dict) -> str:
    """What a keyword expands to, as a blob for the frequency scan.

    The reminder template on both surfaces, because that is what expansion
    substitutes in on both. The captured script is never scanned (FR-020).
    """
    lines: list[str] = []
    for definition in definitions.values():
        lines.append(definition.keyword)
        if definition.reminder_template:
            lines.append(definition.reminder_template)
    return "\n".join(lines)


def script_lines(folders: list[Path]) -> list[str]:
    """Every sidecar's script lines and API types, for the script surface."""
    from effects.infrastructure.sidecar_io import SIDECAR_SUFFIX, read_sidecar

    lines: list[str] = []
    for folder in folders:
        for sidecar_path in sorted(folder.rglob(f"*{SIDECAR_SUFFIX}")):
            for line in read_sidecar(sidecar_path).lines:
                if line.script_text:
                    lines.append(line.script_text)
                if line.script_api_type:
                    lines.append(line.script_api_type)
    return lines


def staging_tokenizer(surface: str, definitions: dict | None = None) -> AbilityTokenizer:
    """A tokenizer applying the surface's rules, with no corpus vocabulary yet.

    It knows the multi-word keywords, because the shared builder seeds them and
    the final tokenizer will merge them; anything else it splits exactly as the
    final tokenizer will.
    """
    vocab = {token: index for index, token in enumerate(SEEDED_SPECIALS)}
    for keyword in MULTI_WORD_KEYWORDS:
        vocab.setdefault(keyword, len(vocab))
    return AbilityTokenizer(vocab, definitions or {}, surface=surface)


def template_texts(
    tokenizer: AbilityTokenizer,
    definitions: dict,
    staged: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Every text a keyword can expand to in this corpus.

    Each definition's generic wording, plus — for every staged keyword line —
    the template filled with that line's own values as Forge formats them, so
    the words a formatter writes ("pays", "a basic land card") are as readable
    as the template's own (SC-002).
    """
    texts = [
        tokenizer.expansion_text(name)
        for name, definition in definitions.items()
        if definition.reminder_template
    ]
    for line in staged:
        keyword = tokenizer.keyword_line_of(line)
        if keyword is not None and definitions[keyword.name].reminder_template:
            texts.append(tokenizer.expansion_text(keyword.name, keyword.details))
    return texts


def seed_tokens(
    surface: str, definitions: dict, staged: list[str],
) -> SeededTokens:
    """The tokens seeded ahead of the corpus, so no truncation can drop them.

    Template words on both surfaces (FR-019); on the script surface also the
    chain labels up to the longest chain's count (FR-002a) and every part of
    every parameter key and ``$``-prefix in the staged lines (FR-009b).
    """
    stager = staging_tokenizer(surface, definitions)
    seeded = SeededTokens()
    for text in template_texts(stager, definitions, staged):
        seeded.template_words.extend(t.text for t in stager.tokenize(text))
    if surface == SURFACE_SCRIPT:
        longest = max(
            (int(n) for line in staged for n in _LABEL_RE.findall(line)), default=0,
        )
        seeded.labels = [f"sv{n}" for n in range(1, longest + 1)]
        for line in staged:
            for key in _KEY_RE.findall(line):
                seeded.key_parts.extend(
                    key[start:end].lower() for start, end in camel_parts(key)
                )
    seeded.template_words = list(dict.fromkeys(seeded.template_words))
    seeded.key_parts = list(dict.fromkeys(seeded.key_parts))
    return seeded


def run(config: BuildVocabConfig) -> int:
    """Build and write the vocabulary. Returns the resulting size."""
    if config.surface not in (SURFACE_PROSE, SURFACE_SCRIPT):
        raise ValueError(
            f"--surface must be {SURFACE_PROSE!r} or {SURFACE_SCRIPT!r}, "
            f"got {config.surface!r}"
        )
    folders = _scan_sources(config)
    definitions = _load_definitions(Path(config.keyword_definitions))
    staged = script_lines(folders) if config.surface == SURFACE_SCRIPT else []
    stager = staging_tokenizer(config.surface, definitions)
    staged_tokens = [stager.tokenize(line) for line in staged]
    seeded = seed_tokens(config.surface, definitions, staged)

    # The shared builder scans one directory tree, so the keyword file is
    # staged as a text file in a scratch tree it can walk alongside the corpus.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="effects-vocab-") as scratch:
        extra = Path(scratch)
        if definitions:
            (extra / "keyword_definitions.txt").write_text(
                _keyword_corpus_text(definitions), encoding="utf-8",
            )
        results = [
            # No printings file: the shared builder seeds every set code from
            # it for the price model, whose enriched card texts carry
            # `set: MH3`. The effect model encodes one ability line, which
            # never names a set, so those would be hundreds of rows no text
            # ever reaches.
            build_vocabulary(
                cards_path=source,
                freq_threshold=_FREQ_THRESHOLD,
                printings_path=None,
            )
            for source in [*folders, extra]
        ]
        # Counted inside the scratch tree's lifetime: the staged sources are
        # real sources, and reporting only the card folders understates the
        # scan the docs describe as covering three.
        scanned = len(folders) + len(list(extra.glob("*.txt"))) + bool(staged)

    # Specials first, so they hold the lowest ids; then everything seeded;
    # then the shared builder's fixed domain terms. All of it sits below
    # ``domain_token_count``, which a truncation never drops.
    vocab: dict[str, int] = {}
    for token in [*SEEDED_SPECIALS, *seeded.all()]:
        vocab.setdefault(token, len(vocab))
    for result in results:
        for token, index in result.vocab.items():
            if index < result.domain_token_count:
                vocab.setdefault(token, len(vocab))
    domain_token_count = len(vocab)
    # Script lines before prose: a truncation drops from the tail, so the
    # script syntax the surface encodes outlasts the prose it falls back to.
    for token in _frequent(staged_tokens):
        vocab.setdefault(token, len(vocab))
    for result in results:
        for token in result.vocab:
            vocab.setdefault(token, len(vocab))

    truncated = truncate_to_target_size(vocab, domain_token_count, config.target_size)
    final = AbilityTokenizer(truncated, definitions, surface=config.surface)
    _check_expansions(final, definitions)

    vocab_path = config.resolved_vocab_path()
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(truncated, vocab_path)
    logger.info(
        "Wrote %d tokens to %s (surface=%s, %d sources)",
        len(truncated), vocab_path, config.surface, scanned,
    )
    _report(config.surface, seeded, staged, staged_tokens, final)
    return len(truncated)


def _load_definitions(path: Path) -> dict:
    if not path.exists():
        logger.warning(
            "No keyword-definition file at %s; the vocabulary will not "
            "cover reminder text, so expanding a keyword would produce "
            "[UNK] tokens. Run extract-keyword-definitions first.",
            path,
        )
        return {}
    from effects.application.extract_keyword_definitions import (
        load_keyword_definitions,
    )

    return load_keyword_definitions(path)


def _frequent(staged_tokens: list[list[Token]]) -> list[str]:
    counts = Counter(t.text for tokens in staged_tokens for t in tokens)
    frequent = [
        (token, count) for token, count in counts.items()
        if count >= _FREQ_THRESHOLD
    ]
    frequent.sort(key=lambda pair: (-pair[1], pair[0]))
    return [token for token, _ in frequent]


def _check_expansions(tokenizer: AbilityTokenizer, definitions: dict) -> None:
    """Fail when a definition's generic expansion reads as ``[UNK]`` (FR-019)."""
    broken = []
    for name, definition in definitions.items():
        if not definition.reminder_template:
            continue
        tokens = tokenizer.tokenize(tokenizer.expansion_text(name))
        unknown = sorted({t.text for t in tokens if t.token_id == tokenizer.unk_id})
        if unknown:
            broken.append(f"{name}: {', '.join(unknown)}")
    if broken:
        raise KeywordExpansionUnknownError(
            f"{len(broken)} keyword definition(s) expand to [UNK]: "
            + "; ".join(broken[:20])
        )


def _report(
    surface: str,
    seeded: SeededTokens,
    staged: list[str],
    staged_tokens: list[list[Token]],
    final: AbilityTokenizer,
) -> None:
    """The build log's numbers (FR-006, FR-011): lengths, coverage, seeding."""
    logger.info(
        "Seeded %d specials, %d chain labels, %d template words, %d key parts",
        len(SEEDED_SPECIALS), len(seeded.labels), len(seeded.template_words),
        len(seeded.key_parts),
    )
    if surface != SURFACE_SCRIPT or not staged:
        return
    from effects.domain.ability_encoder import MAX_ABILITY_TOKENS

    lengths = sorted(
        len(tokens) for line, tokens in zip(staged, staged_tokens, strict=True)
        if _KEY_RE.search(line)
    )
    if lengths:
        # [CLS] takes one position, so a line truncates past MAX - 1 tokens.
        over = sum(1 for n in lengths if n > MAX_ABILITY_TOKENS - 1)
        quantiles = statistics.quantiles(lengths, n=100, method="inclusive")
        logger.info(
            "Script-line length in tokens over %d lines: min %d, p50 %.0f, "
            "p90 %.0f, p99 %.0f, max %d; %d over %d (truncated)",
            len(lengths), lengths[0], quantiles[49], quantiles[89],
            quantiles[98], lengths[-1], over, MAX_ABILITY_TOKENS,
        )
    total = unknown = 0
    for line in staged:
        stripped = _DESCRIPTION_PARAM_RE.sub("", line)
        for token in final.tokenize(stripped):
            total += 1
            unknown += token.token_id == final.unk_id
    logger.info(
        "Unknown-token rate over script parameters (descriptions removed): "
        "%.3f%% (%d of %d)", 100.0 * unknown / max(total, 1), unknown, total,
    )
    parts: set[str] = set()
    for line in staged:
        for word in re.findall(r"[A-Za-z]+", line):
            split = camel_parts(word)
            if len(split) > 1:
                parts.update(word[a:b].lower() for a, b in split)
    logger.info("Camel-case split produces %d distinct parts", len(parts))
