"""Shared helpers for parsing and validating power, toughness, and loyalty values."""

from __future__ import annotations

VALID_PT_CHARS = frozenset("0123456789*X+-")


def validate_pt_chars(value: str | None, field_name: str) -> None:
    """Raise ValueError if *value* contains characters outside the P/T alphabet."""
    if value is None:
        return
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty if provided")
    if not all(c in VALID_PT_CHARS for c in stripped):
        raise ValueError(
            f"{field_name} must be a number, '*', or 'X', got '{value}'"
        )


def parse_combat_stat(value: str | None) -> float:
    """Numeric P/T/Loyalty for ML features. None/blank/*/X → 0.0."""
    if value is None:
        return 0.0
    stripped = value.strip()
    if stripped in ("*", "X", "x"):
        return 0.0
    try:
        return float(stripped)
    except ValueError:
        return 0.0


def is_variable_stat(value: str | None) -> bool:
    """True when the stat is variable at runtime (``*`` or ``X``)."""
    if value is None:
        return False
    return value.strip() in ("*", "X", "x")
