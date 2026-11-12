"""Wiring a CLI's subcommands to the run manifest (``run_manifest``).

Each CLI keeps one table, subcommand → :class:`Writes` or :data:`READ_ONLY`,
beside its parser, and its ``main`` hands the parsed arguments to
:func:`dispatch` instead of calling the handler itself. A :class:`Writes`
names the argument holding the command's output — a directory, or a file
whose directory is the run directory — and the arguments it reads; the
handlers and the application code underneath know nothing of the manifest.
:func:`subcommand_names` lets a test hold each table to its parser, so a new
subcommand cannot go undeclared.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from price_predictor.infrastructure.run_manifest import command_line, run_recorded

logger = logging.getLogger(__name__)

#: The attribute :func:`tag_subcommands` sets on a leaf parser's namespace.
SUBCOMMAND_DEST = "run_manifest_subcommand"

ArgumentValue = Callable[[argparse.Namespace], Any]


@dataclass(frozen=True)
class Writes:
    """A subcommand that writes files, and so keeps a run manifest.

    ``output`` is the dest of the argument naming the output, or a function
    of the parsed arguments resolving it where the command derives it (a
    default resolved per variant, a fixed path no flag sets); ``file`` says
    the output is a file, whose directory is the run directory. ``None``
    means this invocation writes nothing worth a manifest. ``inputs`` are
    the dests (or functions) of every path the command reads; a value may be
    a path, a list of paths or None. ``create`` false is for a command whose
    output is a tree it expects to exist (``encode-cards`` writing beside
    the cards): no directory is created for the manifest. ``when``, given,
    decides whether this invocation writes at all (a ``--verify`` mode).
    """

    output: str | ArgumentValue
    file: bool = False
    inputs: tuple[str | ArgumentValue, ...] = ()
    create: bool = True
    when: Callable[[argparse.Namespace], bool] | None = None

    def run_dir(self, args: argparse.Namespace) -> Path | None:
        """The run directory of this invocation, or None when it writes no manifest."""
        if self.when is not None and not self.when(args):
            return None
        value = _value(args, self.output)
        if value is None or str(value) == "":
            return None
        path = Path(value)
        return path.parent if self.file else path

    def input_paths(self, args: argparse.Namespace) -> list[Path]:
        """Every path ``inputs`` names in this invocation, unset ones left out."""
        return [path for spec in self.inputs for path in _paths(_value(args, spec))]


class _ReadOnly:
    """A subcommand that writes nothing (a report, a check, a server)."""

    def __repr__(self) -> str:
        return "READ_ONLY"


READ_ONLY = _ReadOnly()

ManifestTable = Mapping[str, Writes | _ReadOnly]


def _value(args: argparse.Namespace, spec: str | ArgumentValue) -> Any:
    return getattr(args, spec, None) if isinstance(spec, str) else spec(args)


def _paths(value: Any) -> Iterator[Path]:
    if value is None:
        return
    if isinstance(value, (str, os.PathLike)):
        if str(value):
            yield Path(value)
        return
    if isinstance(value, Iterable):
        for item in value:
            yield from _paths(item)


def labelled_paths(specs: Iterable[str] | None) -> list[Path]:
    """The paths of ``LABEL=PATH`` specs (a bare ``PATH`` too)."""
    return [Path(spec.split("=", 1)[-1]) for spec in specs or () if spec]


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]


def _leaves(
    parser: argparse.ArgumentParser, path: tuple[str, ...] = (),
) -> Iterator[tuple[str, argparse.ArgumentParser]]:
    actions = _subparser_actions(parser)
    if not actions:
        if path:
            yield " ".join(path), parser
        return
    for action in actions:
        seen: set[int] = set()
        for name, child in action.choices.items():
            if id(child) in seen:  # an alias names a parser already walked
                continue
            seen.add(id(child))
            yield from _leaves(child, (*path, name))


def subcommand_names(parser: argparse.ArgumentParser) -> list[str]:
    """Every leaf subcommand of ``parser``, nested ones space-joined (``train sklearn``)."""
    return [name for name, _ in _leaves(parser)]


def tag_subcommands(parser: argparse.ArgumentParser) -> None:
    """Have each leaf subcommand record its own name in the parsed arguments."""
    for name, leaf in _leaves(parser):
        leaf.set_defaults(**{SUBCOMMAND_DEST: name})


def dispatch(
    args: argparse.Namespace, table: ManifestTable, *, module: str,
    argv: Sequence[str] | None = None,
) -> int | None:
    """Run the parsed subcommand's handler, inside its run manifest when it writes.

    ``args`` comes from a parser :func:`tag_subcommands` tagged. A manifest
    that cannot be resolved warns and the command runs without one.
    """
    name = getattr(args, SUBCOMMAND_DEST, None)
    entry = table.get(name) if name is not None else None
    if not isinstance(entry, Writes):
        if entry is None:
            logger.warning("subcommand %r declares no run manifest", name)
        return args.func(args)
    try:
        run_dir = entry.run_dir(args)
        inputs = entry.input_paths(args)
    except Exception as error:  # noqa: BLE001 — bookkeeping never stops a run
        logger.warning("run manifest of %s not resolved: %s", name, error)
        return args.func(args)
    if run_dir is None:
        return args.func(args)
    return run_recorded(
        run_dir, lambda: args.func(args),
        command=command_line(sys.argv[1:] if argv is None else argv, module=module),
        inputs=inputs, create=entry.create,
    )
