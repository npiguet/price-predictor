"""One Forge keyword's expandable definition (FR-060).

``extract-keyword-definitions`` writes the table and
``infrastructure.tokenizer_loading`` reads it; the tokenizer expands a keyword
through it, which is why the value lives in the domain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeywordDefinition:
    """One keyword's expandable definition.

    ``reminder_template`` carries Forge's own ``%s``/``%d`` placeholders.
    ``generated_script`` is read by no expansion (FR-020). ``formatter``
    (FR-018) is None in a file written before it was recorded, which fills the
    raw values instead.
    """

    keyword: str
    reminder_template: str | None
    generated_script: str | None = None
    formatter: str | None = None
