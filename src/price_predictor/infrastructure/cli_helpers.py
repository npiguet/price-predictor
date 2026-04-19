"""Shared argparse helpers for registering dataclass-backed CLI flags."""

from __future__ import annotations

import argparse
import dataclasses
import typing


def add_dataclass_arg(
    parser: argparse.ArgumentParser,
    dataclass_cls: type,
    field_name: str,
    help_text: str,
    *,
    cli_name: str | None = None,
    type_override: typing.Callable | None = None,
) -> None:
    """Register an argparse option whose default is pulled from a dataclass field.

    The CLI flag name defaults to ``--{field_name.replace('_', '-')}``; pass
    ``cli_name`` to override (e.g. ``--set`` for ``set_code``). Boolean fields
    become ``store_true`` flags and the dataclass default must be ``False``.
    """
    field = dataclass_cls.__dataclass_fields__[field_name]  # type: ignore[attr-defined]
    default = _field_default(field)
    cli_flag = (
        cli_name if cli_name is not None
        else f"--{field_name.replace('_', '-')}"
    )

    if field.type is bool or field.type == "bool":
        if default is not False:
            raise ValueError(
                f"store_true flag {cli_flag} needs dataclass default=False, "
                f"got {default!r}"
            )
        parser.add_argument(
            cli_flag, dest=field_name, action="store_true", help=help_text,
        )
        return

    arg_type = type_override or _infer_type(field.type)
    parser.add_argument(
        cli_flag,
        dest=field_name,
        type=arg_type,
        default=default,
        help=f"{help_text} (default: {default})",
    )


def _field_default(field: dataclasses.Field) -> typing.Any:
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    raise ValueError(
        f"dataclass field {field.name!r} has no default; CLI helper needs one"
    )


def _infer_type(annotation: typing.Any) -> typing.Callable:
    if annotation is int or annotation == "int":
        return int
    if annotation is float or annotation == "float":
        return float
    return str
