"""Where the tests that read the sibling Forge checkout's sources find it.

A test pinning a closed list to Forge's source is skipped on a machine with no
``../forge`` checkout, which this repo is often cloned without. Where the
checkout exists, a file the test reads must be there: a wrong path would
otherwise skip the test as if there were no checkout, and the pin would check
nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The sibling Forge checkout, beside this repository.
FORGE = Path(__file__).resolve().parents[2].parent / "forge"
#: ``forge-game``'s package root, ``forge.game``.
FORGE_GAME = FORGE / "forge-game" / "src" / "main" / "java" / "forge" / "game"
#: ``forge-ai``'s package root, ``forge.ai``.
FORGE_AI = FORGE / "forge-ai" / "src" / "main" / "java" / "forge" / "ai"
#: Forge's resources: card scripts, token scripts.
FORGE_RES = FORGE / "forge-gui" / "res"
#: This repository's connector sources, package root ``com.pricepredictor.connector``.
CONNECTOR = (Path(__file__).resolve().parents[2] / "forge-connector" / "src" / "main"
             / "java" / "com" / "pricepredictor" / "connector")


def no_forge_checkout() -> bool:
    """Whether there is no sibling Forge checkout at all."""
    return not FORGE.is_dir()


def forge_source(path: Path) -> Path:
    """``path``, a file or folder of the sibling Forge checkout: the calling
    test is skipped when there is no checkout, and fails when there is one
    and ``path`` is not in it."""
    if no_forge_checkout():
        pytest.skip(f"no Forge checkout at {FORGE}")
    if not path.exists():
        pytest.fail(f"the Forge checkout at {FORGE} holds no {path}: "
                    "the test reads the wrong place")
    return path


def connector_source(relative: str) -> str:
    """The text of ``relative``, a file under :data:`CONNECTOR`: part of this
    repository, so a missing file fails the calling test, never skips it."""
    path = CONNECTOR / relative
    if not path.is_file():
        pytest.fail(f"the connector holds no {path}: the test reads the wrong place")
    return path.read_text(encoding="utf-8")
