"""``extract-keyword-definitions``: Forge's keyword table as JSON (FR-060)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from effects.domain import default_paths

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(default_paths.DEFAULT_KEYWORD_DEFINITIONS)


@dataclass
class ExtractKeywordDefinitionsConfig:
    output: Path = field(default_factory=lambda: DEFAULT_OUTPUT)


def run(config: ExtractKeywordDefinitionsConfig) -> int:
    """Spawn the Java worker and report what it wrote. Returns an exit code."""
    from effects.infrastructure.one_shot_workers import write_keyword_definitions
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

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
