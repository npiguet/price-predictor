"""Where the encoder probes read their inputs and write their artifacts.

Every location derives from ``_lib.paths.REPO`` (and the NAS training-data
root), read once at import. Importing this module creates ``SCRATCH``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = str(Path(__file__).resolve().parents[1])
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from _lib.paths import REPO, SEALED_TRAINING_DATA  # noqa: E402

ENCODER_CKPT = REPO / "models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt"
VOCAB_PATH = REPO / "models/sealed/encoder/vocab.txt"
CARDS_PATH = REPO / "output/cardsfolder-512"
ALLPRINTINGS = REPO / "resources/AllPrintings.json"
FORGE_HINTS = REPO / "output/scorer-probes/forge_hints.csv"
WIN_RATES = SEALED_TRAINING_DATA / "matches-bo1" / "cards-win-rates.txt"
SCRATCH = REPO / "output" / "encoder-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)
CARD_TABLE = SCRATCH / "card_table.pkl"  # written by p0b_card_table.py
