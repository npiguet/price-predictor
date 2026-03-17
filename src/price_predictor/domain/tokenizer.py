"""MTG domain tokenizer: word-level, no external dependencies."""

from __future__ import annotations

import re

_NAME_LINE_RE = re.compile(r"(?m)^name:.*$")


class MtgTokenizer:
    """Word-level tokenizer for MTG card text.

    Encodes card text into token IDs using a domain vocabulary.
    Mana symbols (e.g. {W}, {R}) are preserved as-is; all other text is
    lowercased. Multi-word keywords (e.g. "first strike") are replaced with
    their underscore form ("first_strike") before splitting.

    Card names on ``name:`` lines are replaced with the ``cardname`` placeholder
    before tokenization. The actual card name is arbitrary and almost always
    produces UNK tokens; stripping it removes noise without losing any
    mechanically useful information.
    """

    PAD = "[PAD]"
    UNK = "[UNK]"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, vocab: dict[str, int]) -> None:
        """Construct from a {token: id} vocabulary mapping.

        Requires [PAD]=0 and [UNK]=1 in the vocab.
        """
        self._vocab = vocab
        self._reverse_vocab = {v: k for k, v in vocab.items()}
        # Multi-word keywords are tokens containing '_', sorted longest-first
        # to handle overlap (e.g. "double strike" before "strike")
        self._multi_word_keywords: list[str] = sorted(
            [t for t in vocab if "_" in t], key=len, reverse=True
        )

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in vocabulary."""
        return len(self._vocab)

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        """Encode text to (input_ids, attention_mask).

        Normalizes text → tokenizes → maps to IDs → truncates → pads.
        attention_mask is 1 for real tokens and 0 for padding.
        """
        tokens = self._tokenize(text)
        ids = [self._vocab.get(t, self.UNK_ID) for t in tokens]

        # Truncate to max_length
        ids = ids[:max_length]

        real_len = len(ids)
        pad_len = max_length - real_len

        ids = ids + [self.PAD_ID] * pad_len
        attention_mask = [1] * real_len + [0] * pad_len

        return ids, attention_mask

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to a space-joined string.

        Stops at the first PAD_ID. Unknown IDs decode to '[UNK]'.
        """
        tokens = []
        for tid in token_ids:
            if tid == self.PAD_ID:
                break
            tokens.append(self._reverse_vocab.get(tid, self.UNK))
        return " ".join(tokens)

    def _tokenize(self, text: str) -> list[str]:
        """Normalize text and split into tokens.

        Steps:
        0. Replace card name values on ``name:`` lines with ``cardname``.
        1. Selective lowercasing: split on brace-enclosed groups, lowercase
           non-brace parts, keep mana symbols (e.g. {W}) as-is.
        2. Replace multi-word keywords (sorted longest-first): "first strike"
           → "first_strike".
        3. Regex split into individual tokens.
        """
        # 0. Strip card names from name: lines — they're arbitrary proper nouns
        #    that almost always produce UNK tokens with no mechanical signal.
        text = _NAME_LINE_RE.sub("name: cardname", text)

        # 1. Selective normalize
        parts = re.split(r"(\{[^}]+\})", text)
        text = "".join(p if p.startswith("{") else p.lower() for p in parts)

        # 2. Replace multi-word keywords (sorted longest-first to avoid partial matches)
        for kw in self._multi_word_keywords:
            display = kw.replace("_", " ")
            text = text.replace(display, kw)

        # 3. Split into tokens: words/underscore-words, mana symbols, digits, punctuation
        return re.findall(r"[a-z_]+|\{[^}]+\}|\d+|[^\s\w]", text)
