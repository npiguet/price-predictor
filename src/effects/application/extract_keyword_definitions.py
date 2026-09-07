"""``extract-keyword-definitions``: Forge's keyword table as JSON (FR-060).

Keyword → reminder-text template for every keyword, plus (from stage four) the
generated implementation script for the script-generated majority. The encoder's
keyword-expansion dropout reads it, and an unknown keyword is *always* expanded,
so this file is what lets a set the vocabulary predates be encoded at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("output/effects/keyword-definitions.json")


@dataclass
class ExtractKeywordDefinitionsConfig:
    output: Path = field(default_factory=lambda: DEFAULT_OUTPUT)


@dataclass(frozen=True, slots=True)
class KeywordDefinition:
    """One keyword's expandable definition.

    ``reminder_template`` carries Forge's own ``%s`` placeholders; a
    parameterized keyword instantiates them with the instance's values at
    expansion time. ``generated_script`` is None until stage four, and stays
    None for the engine-coded minority that generates no script.
    """

    keyword: str
    reminder_template: str | None
    generated_script: str | None = None


def load_keyword_definitions(path: Path) -> dict[str, KeywordDefinition]:
    """Read the definition file into ``keyword -> KeywordDefinition``.

    Keys are Forge display names, which is what a card's keyword line and the
    converted text both spell.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        keyword: KeywordDefinition(
            keyword=keyword,
            reminder_template=entry.get("reminder_template"),
            generated_script=entry.get("generated_script"),
        )
        for keyword, entry in data.items()
    }


def run(config: ExtractKeywordDefinitionsConfig) -> int:
    """Spawn the Java worker and report what it wrote. Returns an exit code."""
    from effects.infrastructure.keyword_definition_connector import (
        KeywordDefinitionConnector,
    )

    output = Path(config.output)
    try:
        code = KeywordDefinitionConnector().run(output)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    if code != 0:
        logger.error("KeywordDefinitionMain exited with %d", code)
        return code

    definitions = load_keyword_definitions(output)
    with_reminder = sum(1 for d in definitions.values() if d.reminder_template)
    logger.info(
        "Wrote %d keyword definitions to %s (%d with reminder text)",
        len(definitions), output, with_reminder,
    )
    return 0
