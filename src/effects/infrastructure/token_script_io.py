"""Reading Forge's raw token scripts for the old-token-key remap (FR-151).

The parse of one script is :func:`effects.domain.token_key_remap.parse_token_script`;
this is the directory walk and the file reads around it, kept out of the domain
layer with the rest of the package's IO.
"""

from __future__ import annotations

from pathlib import Path

from effects.domain.token_key_remap import TokenScriptFacts, parse_token_script


def load_token_script_facts(directory: Path) -> dict[str, list[TokenScriptFacts]]:
    """Printed name (lowercase) -> the token scripts that carry it.

    Sorted by path, so same-name scripts keep a stable order between runs.
    """
    out: dict[str, list[TokenScriptFacts]] = {}
    for path in sorted(Path(directory).glob("*.txt")):
        facts = parse_token_script(path.stem, path.read_text(encoding="utf-8", errors="replace"))
        if facts is not None:
            out.setdefault(facts.name, []).append(facts)
    return out
