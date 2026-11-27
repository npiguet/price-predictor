"""The validator's entry points with the CLI's ports already passed.

``effects.application.validate_corpus`` receives its shard reader and record
renderer from the command (``cli.validation_ports``); the validate-corpus tests
call these wrappers, which pass the same ones.
"""

from __future__ import annotations

from effects.application import validate_corpus as _validator
from effects.application.corpus_validation.findings import Thresholds
from effects.infrastructure.cli import validation_ports


def validate_corpus(records, thresholds=Thresholds(), **context_fields):
    """``context_fields`` are the ``CorpusContext`` fields besides the renderer."""
    _source, renderer = validation_ports()
    context = _validator.CorpusContext(record_renderer=renderer, **context_fields)
    return _validator.validate_corpus(records, thresholds, context=context)


def validate_directory(root, **kwargs):
    source, renderer = validation_ports()
    return _validator.validate_directory(
        root, ports=_validator.CorpusPorts(source, renderer),
        options=_validator.DirectoryOptions(**kwargs),
    )


class ShardDamage(_validator.ShardDamage):
    def __init__(self) -> None:
        super().__init__(validation_ports()[0])
