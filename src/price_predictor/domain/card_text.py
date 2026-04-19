"""Typed wrappers for the two card text formats the codebase handles.

The project traffics in two structurally different card text formats:

* **Converted format** (``output/cardsfolder/*.txt``) — lowercase key:value
  lines (``name:``, ``types:``, ``mana cost:``, ``spell[n]:``, ...),
  produced by the Java ``ConvertMain`` tool and ready to feed into the
  price-predictor tokenizer.
* **Raw Forge script** (``../forge/forge-gui/res/cardsfolder/*.txt``) —
  Forge's own syntax with ``Name:``/``ManaCost:``/``A:``/``K:`` lines.
  Only consumed by ``check_convert`` on the Python side.

The two formats share enough surface syntax to be easily confused, so
each gets its own type. :class:`ConvertedCardText` carries format-specific
helpers; :data:`ForgeCardScript` is a NewType alias that merely labels a
string as originating from a raw Forge card script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

ForgeCardScript = NewType("ForgeCardScript", str)

_NAME_LINE_PREFIX = "name:"
_MANA_COST_LINE_PREFIX = "mana cost:"
_ACTIVATED_LINE_PREFIX = "activated"


@dataclass(frozen=True)
class ConvertedCardText:
    """Card text in the tokenizer-ready converted format."""

    text: str

    @classmethod
    def from_file(cls, path: Path) -> "ConvertedCardText":
        return cls(path.read_text(encoding="utf-8", errors="replace"))

    def splitlines(self) -> list[str]:
        return self.text.splitlines()

    def without_name_line(self) -> str:
        """Return the text with the ``name:`` line removed (tokenizer input)."""
        return "\n".join(
            line for line in self.text.splitlines()
            if not line.startswith(_NAME_LINE_PREFIX)
        )

    def mana_cost_line(self) -> str | None:
        """Return the raw ``mana cost:`` value, or None when missing/empty."""
        for line in self.text.splitlines():
            if line.strip().lower().startswith(_MANA_COST_LINE_PREFIX):
                return line.split(":", 1)[1].strip() or None
        return None

    def activated_lines(self) -> list[str]:
        """Return every stripped line starting with ``activated``."""
        return [
            line.strip()
            for line in self.text.splitlines()
            if line.strip().startswith(_ACTIVATED_LINE_PREFIX)
        ]
