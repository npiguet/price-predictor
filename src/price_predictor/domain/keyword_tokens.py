"""The token form of a keyword's display name, and which names are multi-word.

Forge spells a keyword as its card prints it (``Doctor's companion``,
``Jump-start``); the vocabulary carries it in one underscore-joined form that
the tokenizers' merge step rebuilds from the words around it.
"""

from __future__ import annotations

from collections.abc import Iterable


def keyword_token_form(display_name: str) -> str:
    """Lower case, apostrophes dropped, spaces and hyphens as underscores."""
    return display_name.lower().replace("'", "").replace(" ", "_").replace("-", "_")


def multi_word_keywords(display_names: Iterable[str]) -> tuple[str, ...]:
    """The token forms of the names that spell more than one word, sorted."""
    return tuple(sorted({
        keyword_token_form(name) for name in display_names if " " in name or "-" in name
    }))
