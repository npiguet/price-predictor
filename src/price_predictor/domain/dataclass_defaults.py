"""A dataclass field's default, whichever way the field declares it.

Shared because every package reads config defaults off its dataclasses —
for CLI flags, resume precedence, a manifest's absent fields — and each read
must try ``default`` before ``default_factory`` the same way.
"""

from __future__ import annotations

import dataclasses
from typing import Any


def field_default(spec: dataclasses.Field, otherwise: Any = dataclasses.MISSING) -> Any:
    """``spec``'s default, its factory's product, else ``otherwise``.

    Raises:
        ValueError: the field declares no default and ``otherwise`` is not given.
    """
    if spec.default is not dataclasses.MISSING:
        return spec.default
    if spec.default_factory is not dataclasses.MISSING:
        return spec.default_factory()
    if otherwise is not dataclasses.MISSING:
        return otherwise
    raise ValueError(f"dataclass field {spec.name!r} has no default")
