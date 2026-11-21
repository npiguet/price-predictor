"""Every CLI help text encodes in cp1252.

On Windows, Python writes a redirected stdout (a pipe, a file) in the ANSI code
page, cp1252; a help character outside it makes ``--help`` end in
``UnicodeEncodeError`` with exit 1. The draft, sealed and effects CLIs are walked
through their ``build_parser``, down every level of subcommands.
"""

from __future__ import annotations

import argparse

import pytest

from draft.infrastructure.cli import build_parser as draft_build_parser
from effects.infrastructure.cli import build_parser as effects_build_parser
from sealed.infrastructure.cli import build_parser as sealed_build_parser

_BUILDERS = {
    "draft": draft_build_parser,
    "sealed": sealed_build_parser,
    "effects": effects_build_parser,
}


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    subparsers = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return {name: sub for action in subparsers for name, sub in action.choices.items()}


def _walk(prefix: str, parser: argparse.ArgumentParser):
    yield prefix, parser
    for name, sub in _subcommands(parser).items():
        yield from _walk(f"{prefix} {name}", sub)


def _every_parser() -> list[tuple[str, argparse.ArgumentParser]]:
    return [found for package, build in _BUILDERS.items() for found in _walk(package, build())]


_PARSERS = _every_parser()


def test_every_draft_command_is_walked():
    draft_commands = {name for name, _ in _PARSERS if name.startswith("draft ")}
    assert "draft train-draft-agent-rl" in draft_commands
    assert len(draft_commands) == 7


@pytest.mark.parametrize("parser", [p for _, p in _PARSERS], ids=[name for name, _ in _PARSERS])
def test_help_encodes_in_cp1252(parser: argparse.ArgumentParser):
    help_text = parser.format_help()
    offending = sorted({c for c in help_text if c.encode("cp1252", errors="ignore") == b""})
    assert offending == [], f"help holds characters cp1252 cannot encode: {offending}"
