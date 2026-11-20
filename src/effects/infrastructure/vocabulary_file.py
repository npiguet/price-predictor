"""The effects vocabulary file: a surface header, then one token per line.

The header names the encoding surface the vocabulary was built for, so every
reader takes the surface from the file rather than from its name: a script
vocabulary passed under another name read every line on the prose surface.
Line ``n + 1`` is token id ``n``. The header takes line 0, where the price and
sealed vocabularies hold ``[PAD]``, so neither loader reads the other's file.
A file without the header is refused: rebuild it with ``build-vocab``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from effects.domain.ability_tokenizer import Surface
from price_predictor.infrastructure.tokenizer_store import vocabulary_of

#: Line 0 of every effects vocabulary, followed by the surface's value.
HEADER = "# effects vocabulary, surface: "


class VocabularyHeaderError(ValueError):
    """A vocabulary file whose line 0 names no surface."""


@dataclass(frozen=True, slots=True)
class Vocabulary:
    surface: Surface
    #: ``token -> id``.
    tokens: dict[str, int]


def save_vocabulary(tokens: dict[str, int], surface: Surface, path: Path) -> None:
    """Write the header, then each token in id order."""
    ordered = [token for token, _ in sorted(tokens.items(), key=lambda item: item[1])]
    Path(path).write_text("\n".join([HEADER + Surface(surface).value, *ordered]),
                          encoding="utf-8")


def read_vocabulary(path: Path) -> Vocabulary:
    """The surface and the tokens of the vocabulary at ``path``."""
    lines = _text(path).splitlines()
    return Vocabulary(_surface(path, lines[0] if lines else ""), vocabulary_of(lines[1:]))


def vocabulary_surface(path: Path) -> Surface:
    """The surface the vocabulary at ``path`` records, from its header alone."""
    with Path(path).open(encoding="utf-8") as file:
        return _surface(path, file.readline().rstrip("\r\n"))


def _text(path: Path) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"Vocabulary file not found: {path}")
    return Path(path).read_text(encoding="utf-8")


def _surface(path: Path, header: str) -> Surface:
    value = header.removeprefix(HEADER) if header.startswith(HEADER) else None
    if value not in {surface.value for surface in Surface}:
        raise VocabularyHeaderError(
            f"{path}: line 0 is {header!r}, not the surface header "
            f"'{HEADER}prose' or '{HEADER}script': rebuild the vocabulary with "
            "`python -m effects build-vocab`"
        )
    return Surface(value)
