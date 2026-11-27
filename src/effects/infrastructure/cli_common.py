"""What every ``python -m effects`` command is built from: a config dataclass
read off its flags and its flags off the config (:func:`add_config_flags`,
:func:`config_from_args`), the refusal of an expected failure with an exit
code (:func:`exits_with`) and the exception names its tables share, and the
flags several commands declare alike.
"""

from __future__ import annotations

import argparse
import functools
import logging
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from effects.application.collector_common import HOLDOUT_LIST_NAME
from effects.domain.baselines import BASELINES, Baseline
from effects.domain.default_paths import DEFAULT_CARDS_FOLDER_PATHS, DEFAULT_CARDS_FOLDERS
from effects.domain.split import HOLDOUT_MAX_CARRIERS, HOLDOUT_PERMILLE, HoldoutRule
from price_predictor.domain.dataclass_defaults import field_default
from price_predictor.infrastructure.cli_run_manifest import REFUSAL_DEST

logger = logging.getLogger(__name__)


_Config = TypeVar("_Config")

#: Torch loads with this module, so its exceptions are named rather than imported.
_EFFECT_MODEL_STORE = "effects.infrastructure.effect_model_store"
#: A checkpoint trained on another input layout (FR-045), refused by every loader.
_LAYOUT_REFUSED = "effects.domain.checkpoint_layout.CheckpointLayoutError"
#: A corpus manifest of an older generation, or one missing a member.
_MANIFEST_REFUSED = "effects.domain.corpus_manifest.ManifestError"
#: Converted trees reconverted since the corpus was built.
_SIDECARS_CHANGED = "effects.infrastructure.converted_trees.SidecarsChangedError"

#: Every reader's "nothing to read"; named, as its module is not imported here.
_NO_SHARDS = "effects.infrastructure.shard_io.NoShardsError"
#: A shard damaged before its last gzip member, or a line that does not
#: decode: the read cannot go on, and the message names shard and line or
#: byte. Raised in a pool or prefetch worker too, which hands it over whole.
_SHARD_REFUSED: dict[str, int] = {
    "effects.infrastructure.shard_io.CorruptShardError": 1,
    "effects.infrastructure.shard_io.RecordDecodeError": 1,
}



def config_from_args(cls: type[_Config], args: argparse.Namespace, **overrides: Any) -> _Config:
    """``cls`` built from the parsed flags whose dest names one of its fields.

    A field with no flag keeps the dataclass default. ``overrides`` supplies
    what a flag cannot give as parsed; one naming no field is refused, so a
    typo cannot quietly fall back to the default.
    """
    names = {spec.name for spec in fields(cls) if spec.init}
    unknown = sorted(overrides.keys() - names)
    if unknown:
        raise TypeError(f"{cls.__name__} has no field {', '.join(unknown)}")
    values = {name: getattr(args, name) for name in names if hasattr(args, name)}
    return cls(**{**values, **overrides})


def add_config_flags(
    parser: argparse.ArgumentParser, config: type, *, only: Iterable[str] | None = None,
) -> None:
    """A ``--field-name`` flag per field of ``config``, with the field's own
    default; a field with none is a required flag (artifact-layout rule 2).

    ``only`` picks the fields, in order; without it, every field no flag on
    ``parser`` claims yet. The type comes from ``metadata["type"]``, else the
    annotation, and the help from ``metadata["help"]``; a bool field is a
    ``store_true`` switch, and any other non-scalar without a ``type`` is
    refused, because its flag needs declaring by hand. Reads the class and
    never builds one: a config's ``__post_init__`` may load torch.
    """
    specs = {spec.name: spec for spec in fields(config)}
    hints = typing.get_type_hints(config)
    claimed = {action.dest for action in parser._actions}
    for name in only if only is not None else [n for n in specs if n not in claimed]:
        spec = specs[name]
        kind = spec.metadata.get("type") or _flag_type(config, name, hints[name])
        _add_field_flag(parser, config, spec, kind)


def _add_field_flag(
    parser: argparse.ArgumentParser, config: type, spec: Any, kind: Callable[[str], Any],
) -> None:
    """One field's flag: required where the field has no default."""
    default = field_default(spec, otherwise=_REQUIRED)
    flag, help_text = flag_name(spec.name), spec.metadata.get("help")
    if default is _REQUIRED:
        parser.add_argument(flag, type=kind, required=True, help=help_text)
        return
    if isinstance(default, Path):
        # Spelled as typed in --help; argparse parses a string default too.
        default = default.as_posix()
    if kind is bool and default is False:
        parser.add_argument(flag, action="store_true", help=help_text)
    elif kind is not bool:
        parser.add_argument(flag, type=kind, default=default, help=help_text)
    else:
        raise TypeError(f"{config.__name__}.{spec.name} defaults to True; declare its flag")


def flag_name(field_name: str) -> str:
    return "--" + field_name.replace("_", "-")


def _flag_type(config: type, name: str, hint: Any) -> Callable[[str], Any]:
    """The flag's parser: the scalar itself, or a path, where an optional one
    reads an empty value as None."""
    kinds = [kind for kind in typing.get_args(hint) or (hint,) if kind is not type(None)]
    if len(kinds) == 1 and kinds[0] in (int, float, str, bool):
        return kinds[0]
    if kinds == [Path]:
        return _optional_path if type(None) in typing.get_args(hint) else Path
    raise TypeError(f"{config.__name__}.{name} is {hint}; declare its flag")


_REQUIRED = object()


def exits_with(codes: Mapping[type[Exception] | str, int]) -> Callable:
    """A handler whose expected failures log their message and exit with a code.

    The first entry the exception is an instance of decides the code; anything
    else propagates as a traceback, which is what an internal invariant is
    worth. A string key names a class as ``module.Qualname`` and matches the
    exception's MRO without importing it.
    """
    def decorate(handler: Callable[[argparse.Namespace], int]):
        @functools.wraps(handler)
        def run(args: argparse.Namespace) -> int:
            try:
                return handler(args)
            except Exception as exc:
                code = _exit_code(codes, exc)
                if code is None:
                    raise
                logger.error("%s", exc)
                setattr(args, REFUSAL_DEST, str(exc))
                return code

        run.exit_codes = codes
        return run

    return decorate


def _exit_code(codes: Mapping[type[Exception] | str, int], exc: Exception) -> int | None:
    names = {f"{cls.__module__}.{cls.__qualname__}" for cls in type(exc).__mro__}
    for kind, code in codes.items():
        if kind in names if isinstance(kind, str) else isinstance(exc, kind):
            return code
    return None


def _optional_path(text: str) -> Path | None:
    """A path flag's ``type`` where an empty value means unset, as omitting it does."""
    return Path(text) if text else None


# ── flags several commands share ────────────────────────────────────────


class _AppendCardsFolder(argparse.Action):
    """``--cards-folder``: repeatable, and the first use replaces the default.

    argparse's own ``append`` appends to the default, so both converted trees
    would always be read beside the ones named.
    """

    def __call__(self, parser, namespace, values, option_string=None) -> None:
        chosen = getattr(namespace, self.dest)
        if chosen is self.default:
            chosen = ()
        setattr(namespace, self.dest, (*chosen, Path(values)))


def _add_cards_folder(
    parser: argparse.ArgumentParser, *, recorded: bool = False,
) -> None:
    """``--cards-folder``; ``recorded`` defaults it to the folders the
    checkpoint recorded instead of the converted defaults."""
    default = None if recorded else DEFAULT_CARDS_FOLDER_PATHS
    named = "the folders --checkpoint recorded" if recorded else " and ".join(DEFAULT_CARDS_FOLDERS)
    parser.add_argument(
        "--cards-folder", action=_AppendCardsFolder, dest="cards_folders",
        default=default, help=f"Converted source tree; repeatable (default: {named})",
    )


def _add_effect_records(
    parser: argparse.ArgumentParser, *, help_text: str, required: bool = True,
) -> None:
    """``--effect-records``; a collector passes ``required=False`` so it can tell
    an explicit one beside ``--training-corpus`` from none, and refuses a
    command with neither."""
    parser.add_argument(
        "--effect-records", type=Path, required=required, default=None, help=help_text,
    )


def _add_holdout_sources(
    parser: argparse.ArgumentParser, *, excluded: str = "held-out cards",
) -> None:
    """The three ways a collector names the cards it keeps out of every deck.

    Shared by both collectors, so neither accepts a source the other refuses:
    a corpus depleted against one holdout and split against another is a
    silent corruption.
    """
    parser.add_argument(
        "--split-from", type=_optional_path, default=None,
        help=(
            f"A checkpoint whose {excluded} are excluded from every deck, so "
            "this run cannot contaminate the card-disjoint split it feeds"
        ),
    )
    parser.add_argument(
        "--exclude-cards", type=_optional_path, default=None,
        help=(
            "Card names to keep out of every deck, from 'holdout-cards'; needs "
            "no trained model. Mutually exclusive with --split-from"
        ),
    )
    parser.add_argument(
        "--training-corpus", type=Path, default=None,
        help=(
            f"A corpus directory and the {HOLDOUT_LIST_NAME} it was depleted "
            "against: --effect-records DIR plus --exclude-cards "
            f"DIR/{HOLDOUT_LIST_NAME} in one flag, so the two cannot disagree"
        ),
    )


def _add_holdout_flags(parser: argparse.ArgumentParser) -> None:
    """The holdout rule, on every command that computes it: ``holdout-cards``,
    ``build-corpus``, ``collect-variants`` and ``collect-coverage --only-cards``
    must be passed the same values."""
    parser.add_argument(
        "--holdout-permille", type=int, default=HOLDOUT_PERMILLE,
        help=(
            "A text (or template) is held out when its crc32 %% 1000 is below "
            "this (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--holdout-max-carriers", type=int, default=HOLDOUT_MAX_CARRIERS,
        help=(
            "A text (or template) more cards than this carry is never held out "
            "(default: %(default)s)"
        ),
    )


def holdout_rule(args: argparse.Namespace) -> HoldoutRule:
    """The rule :func:`_add_holdout_flags` parsed."""
    return HoldoutRule(args.holdout_permille, args.holdout_max_carriers)


def _add_variant(parser: argparse.ArgumentParser, *, help_text: str | None = None) -> None:
    parser.add_argument(
        "--variant", type=Baseline, choices=BASELINES, default=Baseline.FULL,
        help=help_text,
    )


def _variant_scripts_help(without: str = "no variant tree is read") -> str:
    """The help of an optional ``--variant-scripts``; ``without`` says what
    leaving it unset costs."""
    return (
        "The perturbed-script tree collect-variants wrote (usually "
        f"an inputs run); unset, {without}"
    )


def _add_variant_scripts(
    parser: argparse.ArgumentParser, *,
    help_text: str | None = None,
    path_type: Callable[[str], object] = _optional_path,
    required: bool = False,
) -> None:
    """``--variant-scripts``, the one declaration every command uses. A command
    that reads the tree to check or count it makes the flag ``required`` (an
    empty value reads none), so a forgotten one cannot read as no variants."""
    parser.add_argument(
        "--variant-scripts", type=path_type, required=required,
        default=None, help=help_text or _variant_scripts_help(),
    )


def _add_recorded_tokenizer_paths(parser: argparse.ArgumentParser) -> None:
    """Default to the paths the checkpoint recorded (FR-097): any other would
    tokenize with a vocabulary the model may never have seen."""
    for flag in ("--vocab-path", "--keyword-definitions"):
        parser.add_argument(
            flag, type=_optional_path, default=None,
            help="Defaults to the path the checkpoint recorded from training",
        )
