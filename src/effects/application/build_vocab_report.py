"""The script-line numbers ``build-vocab`` logs (FR-006, FR-011): the token-length
quantiles, the unknown-token rate over script parameters and the camel-case
split's yield. Split from :mod:`effects.application.build_vocab`, which stages
the lines and builds the vocabulary these numbers describe."""

from __future__ import annotations

import logging
import re
import statistics

from effects.domain.ability_tokenizer import AbilityTokenizer, Token, camel_parts

logger = logging.getLogger(__name__)

#: A parameter key, or a ``$``-prefix glued to its value (``Count$Valid``).
KEY_RE = re.compile(r"([A-Za-z]+)\$")
#: Free prose, so left out of the script coverage figure (FR-011).
_DESCRIPTION_PARAM_RE = re.compile(r"\w*Description\$[^|\[]*")


def report_script_lines(
    staged: list[str], staged_tokens: list[list[Token]], final: AbilityTokenizer,
) -> None:
    """The staged script lines' lengths, unknown-token rate and camel parts."""
    _report_lengths(staged, staged_tokens)
    _report_unknown_rate(staged, final)
    _report_camel_parts(staged)


def _report_lengths(staged: list[str], staged_tokens: list[list[Token]]) -> None:
    """The script lines' token-length quantiles, and how many truncate."""
    from effects.domain.ability_encoder import MAX_ABILITY_TOKENS

    lengths = sorted(
        len(tokens) for line, tokens in zip(staged, staged_tokens, strict=True)
        if KEY_RE.search(line)
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
