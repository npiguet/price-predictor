"""Resume-precedence resolution shared by resumable training CLIs.

``train-scorer`` and ``train-picker`` both register their resumable flags with
``default=None`` (a sentinel) so a value can be resolved with the precedence
**explicit CLI value > resumed checkpoint's ``train_config`` value > dataclass
default** (FR-010, contracts/checkpoint-format.md §Resume Precedence). This
module holds that resolution so the two commands cannot drift.
"""

from __future__ import annotations

import dataclasses
from argparse import Namespace
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def dataclass_default(
    config_cls: type,
    field_name: str,
    required_defaults: Mapping[str, Any] | None = None,
) -> Any:
    """Return a dataclass field's default value.

    Resolves ``default`` then ``default_factory``; for required fields (neither
    is set) falls back to ``required_defaults[field_name]`` when supplied, else
    ``None``. ``required_defaults`` mirrors the help-text defaults a CLI
    advertises for fields that are required on the dataclass itself.
    """
    field = config_cls.__dataclass_fields__[field_name]
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    if required_defaults is not None and field_name in required_defaults:
        return required_defaults[field_name]
    return None


def resolve_resumable_args(
    args: Namespace,
    *,
    flag_names: Iterable[str],
    config_cls: type,
    path_fields: frozenset[str],
    resumed_config: Mapping[str, Any] | None,
    required_defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve every resumable flag and coerce path-typed values to ``Path``.

    For each flag the value is the first of: the explicit CLI value (non-None),
    the resumed checkpoint's stored value (non-None), or the dataclass default.
    Values for flags in ``path_fields`` are wrapped in ``Path`` (``None`` stays
    ``None``); other values pass through unchanged. The returned dict is ready
    to splat into ``config_cls(**resolved)`` alongside the non-resumable flags.
    """
    resolved: dict[str, Any] = {}
    for flag in flag_names:
        cli_value = getattr(args, flag, None)
        if cli_value is not None:
            value = cli_value
        elif resumed_config is not None and resumed_config.get(flag) is not None:
            value = resumed_config[flag]
        else:
            value = dataclass_default(config_cls, flag, required_defaults)
        if value is not None and flag in path_fields:
            value = Path(value)
        resolved[flag] = value
    return resolved
