"""``extract-keyword-definitions``: Forge's keyword table as JSON (FR-060)."""

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

    ``reminder_template`` carries Forge's own ``%s``/``%d`` placeholders.
    ``generated_script`` is read by no expansion (FR-020). ``formatter``
    (FR-018) is None in a file written before it was recorded, which fills the
    raw values instead.
    """

    keyword: str
    reminder_template: str | None
    generated_script: str | None = None
    formatter: str | None = None


def _normalized(template: str | None) -> str | None:
    """Forge's typographic apostrophe as the ASCII one the vocabulary spells
    (FR-017); normalized at load, never in the file."""
    return template.replace("’", "'") if template else template


def load_keyword_definitions(path: Path) -> dict[str, KeywordDefinition]:
    """Read the definition file into ``display name -> KeywordDefinition``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        keyword: KeywordDefinition(
            keyword=keyword,
            reminder_template=_normalized(entry.get("reminder_template")),
            generated_script=entry.get("generated_script"),
            formatter=entry.get("formatter"),
        )
        for keyword, entry in data.items()
    }


def run(config: ExtractKeywordDefinitionsConfig) -> int:
    """Spawn the Java worker and report what it wrote. Returns an exit code."""
    from effects.infrastructure.one_shot_workers import write_keyword_definitions

    output = Path(config.output)
    try:
        code = write_keyword_definitions(output)
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
