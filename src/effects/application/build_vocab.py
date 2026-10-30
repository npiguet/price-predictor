"""``build-vocab``: the effects-side tokenizer vocabulary (FR-064, FR-065).

A wrapper over ``price_predictor.application.build_vocabulary``. Script lines
are tokenized here rather than by the shared builder, whose tokenizer
lowercases before splitting and would erase the camel case the script rules
split on; they are counted before the prose, so a truncation drops prose
first. Every template word is seeded (FR-019), so no truncation can turn an
expansion into ``[UNK]``.
"""

from __future__ import annotations

import logging
import re
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain import default_paths
from effects.domain.ability_tokenizer import (
    AbilityTokenizer,
    Surface,
    Token,
    camel_parts,
)
from effects.domain.script_text import SEG
from effects.infrastructure.tokenizer_loading import load_definitions
from price_predictor.application.build_vocabulary import (
    MULTI_WORD_KEYWORDS,
    build_vocabulary,
    truncate_to_target_size,
)
from price_predictor.infrastructure.tokenizer_store import save_vocabulary

logger = logging.getLogger(__name__)

DEFAULT_KEYWORD_DEFINITIONS = Path(default_paths.DEFAULT_KEYWORD_DEFINITIONS)
DEFAULT_VOCAB_PATH = Path(default_paths.DEFAULT_VOCAB_PATH)
DEFAULT_SCRIPT_VOCAB_PATH = Path(default_paths.DEFAULT_SCRIPT_VOCAB_PATH)
DEFAULT_TARGET_SIZE = 5000

#: Seeded into every effects vocabulary at the lowest ids, whatever their
#: corpus frequency.
SEEDED_SPECIALS: tuple[str, ...] = (
    "[PAD]", "[UNK]", "cardname", "[MASK]", "[CLS]", SEG,
)

#: The shared builder's production threshold.
_FREQ_THRESHOLD = 2

_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])SV(\d+)(?![A-Za-z0-9])")
#: A parameter key, or a ``$``-prefix glued to its value (``Count$Valid``).
_KEY_RE = re.compile(r"([A-Za-z]+)\$")
#: Free prose, so left out of the script coverage figure (FR-011).
_DESCRIPTION_PARAM_RE = re.compile(r"\w*Description\$[^|\[]*")


@dataclass
class BuildVocabConfig:
    surface: Surface = Surface.PROSE
    cards_folders: tuple[Path, ...] = default_paths.DEFAULT_CARDS_FOLDER_PATHS
    vocab_path: Path | None = None
    keyword_definitions: Path = DEFAULT_KEYWORD_DEFINITIONS
    target_size: int = DEFAULT_TARGET_SIZE

    def __post_init__(self) -> None:
        try:
            self.surface = Surface(self.surface)
        except ValueError:
            raise ValueError(
                f"--surface must be one of {', '.join(repr(s.value) for s in Surface)}, "
                f"got {self.surface!r}"
            ) from None

    def resolved_vocab_path(self) -> Path:
        """``--vocab-path``, defaulting per surface so a script rebuild never
        overwrites the vocabulary a prose checkpoint recorded."""
        if self.vocab_path is not None:
            return Path(self.vocab_path)
        return (
            DEFAULT_SCRIPT_VOCAB_PATH
            if self.surface == Surface.SCRIPT
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
    """What a keyword expands to, as a blob for the frequency scan; never the
    captured script, which no expansion substitutes (FR-020)."""
    lines: list[str] = []
    for definition in definitions.values():
        lines.append(definition.keyword)
        if definition.reminder_template:
            lines.append(definition.reminder_template)
    return "\n".join(lines)


def script_lines(folders: list[Path]) -> list[str]:
    """Every sidecar's script lines and API types, for the script surface."""
    from effects.infrastructure.sidecar_io import iter_sidecars

    lines: list[str] = []
    for folder in folders:
        for _path, sidecar in iter_sidecars(folder):
            for line in sidecar.lines:
                if line.script_text:
                    lines.append(line.script_text)
                if line.script_api_type:
                    lines.append(line.script_api_type)
    return lines


def staging_tokenizer(surface: Surface, definitions: dict | None = None) -> AbilityTokenizer:
    """A tokenizer applying the surface's rules before the vocabulary exists,
    knowing the multi-word keywords the final one will merge."""
    vocab = {token: index for index, token in enumerate(SEEDED_SPECIALS)}
    for keyword in MULTI_WORD_KEYWORDS:
        vocab.setdefault(keyword, len(vocab))
    return AbilityTokenizer(vocab, definitions or {}, surface=surface)


def template_texts(
    tokenizer: AbilityTokenizer,
    definitions: dict,
    staged: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Every text a keyword can expand to in this corpus: each generic
    wording, plus each staged keyword line's filled template, so a formatter's
    own words are seeded too (SC-002)."""
    texts = [
        tokenizer.keywords.line_text(name)
        for name, definition in definitions.items()
        if definition.reminder_template
    ]
    for line in staged:
        keyword = tokenizer.keywords.line_of(line)
        if keyword is not None and definitions[keyword.name].reminder_template:
            texts.append(tokenizer.keywords.line_text(keyword.name, keyword.details))
    return texts


def seed_tokens(
    stager: AbilityTokenizer, definitions: dict, staged: list[str],
) -> SeededTokens:
    """The tokens seeded ahead of the corpus, so no truncation can drop them:
    template words (FR-019), and on the script surface the chain labels and
    every key part (FR-002a, FR-009b). ``stager`` is the surface's
    :func:`staging_tokenizer`."""
    seeded = SeededTokens()
    for text in template_texts(stager, definitions, staged):
        seeded.template_words.extend(t.text for t in stager.tokenize(text))
    if stager.surface == Surface.SCRIPT:
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
    folders = _scan_sources(config)
    definitions = _load_definitions(Path(config.keyword_definitions))
    staged = script_lines(folders) if config.surface == Surface.SCRIPT else []
    stager = staging_tokenizer(config.surface, definitions)
    staged_tokens = [stager.tokenize(line) for line in staged]
    seeded = seed_tokens(stager, definitions, staged)

    results, sources = _scan_vocabularies(folders, definitions)
    # The staged script lines count as a source too.
    scanned = sources + bool(staged)
    vocab, domain_token_count = _merge_vocab(results, seeded, staged_tokens)
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


def _scan_vocabularies(folders: list[Path], definitions: dict) -> tuple[list, int]:
    """The shared builder's result over each source folder and the keyword
    file, and how many sources were scanned."""
    # The shared builder scans directory trees, so the keyword file is staged
    # in a scratch one.
    with tempfile.TemporaryDirectory(prefix="effects-vocab-") as scratch:
        extra = Path(scratch)
        if definitions:
            (extra / "keyword_definitions.txt").write_text(
                _keyword_corpus_text(definitions), encoding="utf-8",
            )
        results = [
            # No printings file: its set codes are for the price model's card
            # texts, and an ability line never names a set.
            build_vocabulary(
                cards_path=source,
                freq_threshold=_FREQ_THRESHOLD,
                printings_path=None,
            )
            for source in [*folders, extra]
        ]
        # Inside the scratch tree's lifetime, which the keyword file counts in.
        return results, len(folders) + len(list(extra.glob("*.txt")))


def _merge_vocab(
    results: list, seeded: SeededTokens, staged_tokens: list[list[Token]],
) -> tuple[dict[str, int], int]:
    """One vocabulary from the seeds, the staged script lines and the scans,
    and its ``domain_token_count``."""
    # Specials, seeded tokens and the shared builder's domain terms all sit
    # below ``domain_token_count``, which a truncation never drops.
    vocab: dict[str, int] = {}
    for token in [*SEEDED_SPECIALS, *seeded.all()]:
        vocab.setdefault(token, len(vocab))
    for result in results:
        for token, index in result.vocab.items():
            if index < result.domain_token_count:
                vocab.setdefault(token, len(vocab))
    domain_token_count = len(vocab)
    # Script lines before prose: a truncation drops from the tail.
    for token in _frequent(staged_tokens):
        vocab.setdefault(token, len(vocab))
    for result in results:
        for token in result.vocab:
            vocab.setdefault(token, len(vocab))
    return vocab, domain_token_count


def _load_definitions(path: Path) -> dict:
    if not path.exists():
        logger.warning(
            "No keyword-definition file at %s; the vocabulary will not "
            "cover reminder text, so expanding a keyword would produce "
            "[UNK] tokens. Run extract-keyword-definitions first.",
            path,
        )
    return load_definitions(path)


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
        tokens = tokenizer.tokenize(tokenizer.keywords.line_text(name))
        unknown = sorted({t.text for t in tokens if t.token_id == tokenizer.unk_id})
        if unknown:
            broken.append(f"{name}: {', '.join(unknown)}")
    if broken:
        raise KeywordExpansionUnknownError(
            f"{len(broken)} keyword definition(s) expand to [UNK]: "
            + "; ".join(broken[:20])
        )


def _report(
    surface: Surface,
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
    if surface != Surface.SCRIPT or not staged:
        return
    _report_lengths(staged, staged_tokens)
    _report_unknown_rate(staged, final)
    _report_camel_parts(staged)


def _report_lengths(staged: list[str], staged_tokens: list[list[Token]]) -> None:
    """The script lines' token-length quantiles, and how many truncate."""
    from effects.domain.ability_encoder import MAX_ABILITY_TOKENS

    lengths = sorted(
        len(tokens) for line, tokens in zip(staged, staged_tokens, strict=True)
        if _KEY_RE.search(line)
    )
    if not lengths:
        return
    # [CLS] takes one position, so a line truncates past MAX - 1 tokens.
    over = sum(1 for n in lengths if n > MAX_ABILITY_TOKENS - 1)
    quantiles = statistics.quantiles(lengths, n=100, method="inclusive")
    logger.info(
        "Script-line length in tokens over %d lines: min %d, p50 %.0f, "
        "p90 %.0f, p99 %.0f, max %d; %d over %d (truncated)",
        len(lengths), lengths[0], quantiles[49], quantiles[89],
        quantiles[98], lengths[-1], over, MAX_ABILITY_TOKENS,
    )


def _report_unknown_rate(staged: list[str], final: AbilityTokenizer) -> None:
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


def _report_camel_parts(staged: list[str]) -> None:
    parts: set[str] = set()
    for line in staged:
        for word in re.findall(r"[A-Za-z]+", line):
            split = camel_parts(word)
            if len(split) > 1:
                parts.update(word[a:b].lower() for a, b in split)
    logger.info("Camel-case split produces %d distinct parts", len(parts))
