"""Parse sealed pools files (`SET_CODE;Card1|Card2|...|CardN` per line)."""

from __future__ import annotations

from pathlib import Path


def parse_pools(pools_file: Path) -> list[tuple[str, list[str]]]:
    """Parse a pools.txt file into ``(set_code, card_names)`` tuples.

    The file format is one pool per line:
    ``SET_CODE;Card1|Card2|...|CardN``. Blank lines are ignored.

    Args:
        pools_file: Path to the pools.txt file.

    Returns:
        One tuple per pool line. Order matches the source file.

    Raises:
        ValueError: If a non-blank line does not contain a ``;`` separator.
    """
    pools: list[tuple[str, list[str]]] = []
    for line in pools_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ";" not in stripped:
            raise ValueError(
                f"Pool line missing ';' set code separator: {stripped!r}"
            )
        set_code, _, names_field = stripped.partition(";")
        card_names = names_field.split("|") if names_field else []
        pools.append((set_code, card_names))
    return pools
