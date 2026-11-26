"""The validator's entry points with the CLI's ports already passed.

``effects.application.validate_corpus`` receives its shard reader and record
renderer from the command (``cli.validation_ports``); the validate-corpus tests
call these wrappers, which pass the same ones.
"""

from __future__ import annotations

from effects.application import validate_corpus as _validator
from effects.infrastructure.cli import validation_ports


def validate_corpus(records, *args, **kwargs):
    _source, renderer = validation_ports()
    return _validator.validate_corpus(records, *args, record_renderer=renderer, **kwargs)


def validate_directory(root, **kwargs):
    source, renderer = validation_ports()
    return _validator.validate_directory(root, source=source, record_renderer=renderer, **kwargs)


class ShardDamage(_validator.ShardDamage):
    def __init__(self) -> None:
        super().__init__(validation_ports()[0])
