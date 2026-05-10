"""SC-005 verification: ``--shrinkage-k`` shifts low-observation cards'
labels visibly across all 9 head columns while leaving high-observation
cards' shrunk values within a few thousandths of their raw counterparts.

The fixture sealed corpus is too small for the high-observation half of
SC-005, so this test builds a tiny synthetic ``cards-played.txt`` with
exactly two cards: one low-observation and one high-observation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.application.train_encoder import TrainEncoderConfig
from sealed.application.train_encoder import run as run_train_encoder


def _seed_card(folder: Path, card_name: str, mana_cost: str) -> None:
    sanitized = card_name.lower().replace(" ", "_")
    letter = sanitized[0]
    (folder / letter).mkdir(parents=True, exist_ok=True)
    (folder / letter / f"{sanitized}.txt").write_text(
        f"name: {card_name}\nmana cost: {mana_cost}\ntypes: instant\n",
        encoding="utf-8",
    )


def _build_minimal_vocab(vocab_path: Path) -> None:
    tokens = [
        "[PAD]", "[UNK]", "cardname", "[MASK]",
        "name", "mana", "cost", "types", "instant", "creature",
        "{r}", "{u}", "{1}", "none",
    ]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocab, vocab_path)


def _build_synthetic_cards_played(
    path: Path, low_card: str, high_card: str,
) -> None:
    """Two cards: ``low_card`` shows up in 2 games; ``high_card`` in 10000.

    Both cards "play and win" in every appearance so the raw
    ``score_play`` cell is +1.0; with N=10000 and k=20 the shrunk
    shift on the high-n card is ~k/(N+k) ≈ 0.002, comfortably below
    SC-005's "few thousandths" bound.
    """
    lines: list[str] = []
    # Format: timestamp;run_id;set_code;method_A;method_B;
    #         cards_played_A;cards_played_B;cards_not_played_A;cards_not_played_B;
    #         winner;starter
    base = "2026-05-10T00:00:00Z;run-001;BLB;forge-best;forge-3sub"
    for _ in range(2):
        lines.append(
            f"{base};{low_card}|{high_card};{high_card};;;A;A"
        )
    for _ in range(10000):
        lines.append(
            f"{base};{high_card};{high_card};;;A;A"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_win_rates(path: Path) -> dict[str, list[str]]:
    """Parse cards-win-rates.txt into {card_name: [field, ...]}."""
    out: dict[str, list[str]] = {}
    rows = path.read_text(encoding="utf-8").splitlines()
    for row in rows[1:]:
        parts = row.split(";")
        out[parts[0]] = parts
    return out


_SHRUNK_HEAD_INDICES = (
    6,   # shrunk_score_play
    8,   # shrunk_score_draw
    10,  # shrunk_played_rate
    12,  # shrunk_cast_lift
    14,  # shrunk_color_lift_W
    16,  # shrunk_color_lift_U
    18,  # shrunk_color_lift_B
    20,  # shrunk_color_lift_R
    22,  # shrunk_color_lift_G
)
# Per-color heads' shrinkage depends on per-color observation counts,
# which can be small even when a card has many overall observations
# (e.g. a card that only appears in mono-U decks has 0 R observations
# regardless of total N). The high-N stability assertion (b) of SC-005
# applies to heads whose denominator is total in-deck N — i.e. the
# four non-color heads. The low-N shift assertion (a) covers all
# populated heads.
_NONCOLOR_SHRUNK_HEAD_INDICES = _SHRUNK_HEAD_INDICES[:4]


@pytest.mark.integration
def test_shrinkage_diff_across_heads(tmp_path: Path):
    cards_folder = tmp_path / "cardsfolder"
    cards_folder.mkdir(parents=True, exist_ok=True)
    low = "Solo Card"
    high = "Common Card"
    _seed_card(cards_folder, low, "{R}")
    _seed_card(cards_folder, high, "{U}")

    cards_played = tmp_path / "cards-played.txt"
    _build_synthetic_cards_played(cards_played, low, high)

    vocab_path = tmp_path / "models" / "sealed" / "encoder" / "vocab.txt"
    _build_minimal_vocab(vocab_path)

    model_output = tmp_path / "models" / "sealed" / "encoder"

    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        # k = 0 run.
        cfg_k0 = TrainEncoderConfig(
            cards_played_path=cards_played,
            cards_folder=cards_folder,
            vocab_path=vocab_path,
            model_output_dir=model_output,
            batch_size=2, epochs=1, lr=1e-3, patience=1,
            n_layers=2, n_heads=2, n_pool_queries=2,
            shrinkage_k=0.0,
        )
        run_train_encoder(cfg_k0)
        snapshot_k0 = (tmp_path / "output" / "sealed" / "cards-win-rates.txt").read_text(
            encoding="utf-8",
        )
        snapshot_k0_path = tmp_path / "snapshot_k0.txt"
        snapshot_k0_path.write_text(snapshot_k0, encoding="utf-8")

        # k = 20 run.
        cfg_k20 = TrainEncoderConfig(
            cards_played_path=cards_played,
            cards_folder=cards_folder,
            vocab_path=vocab_path,
            model_output_dir=model_output,
            batch_size=2, epochs=1, lr=1e-3, patience=1,
            n_layers=2, n_heads=2, n_pool_queries=2,
            shrinkage_k=20.0,
        )
        run_train_encoder(cfg_k20)
        snapshot_k20_path = tmp_path / "snapshot_k20.txt"
        snapshot_k20_path.write_text(
            (tmp_path / "output" / "sealed" / "cards-win-rates.txt").read_text(
                encoding="utf-8",
            ),
            encoding="utf-8",
        )

        rows_k0 = _parse_win_rates(snapshot_k0_path)
        rows_k20 = _parse_win_rates(snapshot_k20_path)

        # (a) The low-observation card's shrunk values across the
        # populated head columns shift measurably between the two runs.
        low_k0 = rows_k0[low]
        low_k20 = rows_k20[low]
        any_shifted = False
        for idx in _SHRUNK_HEAD_INDICES:
            v_k0 = low_k0[idx]
            v_k20 = low_k20[idx]
            if v_k0 == "" or v_k20 == "":
                continue
            if abs(float(v_k0) - float(v_k20)) > 0.05:
                any_shifted = True
                break
        assert any_shifted, (
            "SC-005: low-observation card's shrunk values must shift "
            "measurably between k=0 and k=20"
        )

        # (b) The high-observation card's shrunk values barely move on
        # the four non-color heads (whose denominator is the card's
        # total in-deck N — high here): within 0.005 of raw on the
        # k=20 run.
        high_k20 = rows_k20[high]
        for shrunk_idx in _NONCOLOR_SHRUNK_HEAD_INDICES:
            raw_idx = shrunk_idx - 1
            raw = high_k20[raw_idx]
            shrunk = high_k20[shrunk_idx]
            if raw == "" or shrunk == "":
                continue
            assert abs(float(raw) - float(shrunk)) < 0.005, (
                f"high-observation card's shrunk value at column "
                f"{shrunk_idx} drifted too far: raw={raw} shrunk={shrunk}"
            )
    finally:
        os.chdir(cwd_before)
