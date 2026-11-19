"""``extract-keyword-definitions``: Forge's keyword table as JSON (FR-060)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractKeywordDefinitionsConfig:
    output: Path = field(metadata={
        "help": "Output path, inside an inputs run (output/effects/<gen>/inputs/<run>/)",
    })


class KeywordExtractionError(RuntimeError):
    """The worker could not be run, timed out or exited non-zero."""


def run(config: ExtractKeywordDefinitionsConfig) -> int:
    """Spawn the Java worker and report what it wrote; a failure of the worker
    raises :class:`KeywordExtractionError`."""
    from effects.infrastructure.one_shot_workers import write_keyword_definitions
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions

    output = Path(config.output)
    try:
        code = write_keyword_definitions(output)
    except (FileNotFoundError, TimeoutError) as exc:
        raise KeywordExtractionError(str(exc)) from exc
    if code != 0:
        raise KeywordExtractionError(f"KeywordDefinitionMain exited with {code}")

    definitions = load_keyword_definitions(output)
    with_reminder = sum(1 for d in definitions.values() if d.reminder_template)
    logger.info(
        "Wrote %d keyword definitions to %s (%d with reminder text)",
        len(definitions), output, with_reminder,
    )
    return 0
