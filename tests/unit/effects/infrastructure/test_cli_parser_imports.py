"""Building the parser loads no torch.

``python -m effects --help`` builds every subcommand's parser, and each one
formats its defaults from the modules that own them. Torch costs seconds and
hundreds of megabytes to import, so a default read from a module that drags it
in makes every ``--help`` and every collector start pay for a GPU stack it
never touches. The handlers import torch when they run; the parser must not.
"""

from __future__ import annotations

import subprocess
import sys


def test_building_the_parser_never_loads_torch() -> None:
    probe = (
        "import sys\n"
        "from effects.infrastructure.cli import build_parser\n"
        "build_parser()\n"
        "print('torch' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "False", "building the effects parser loads torch"
