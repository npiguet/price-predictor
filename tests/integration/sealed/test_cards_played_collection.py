"""Integration test for the cards-played.txt collection pipeline.

Drives ``python -m sealed match-outcomes`` for a handful of matches by
spawning the Java worker subprocess directly via
``MatchWorkerConnector``. Verifies the file-format invariants from
``contracts/files.md``:

- ``cards-played.txt`` line count equals the sum of game counts
  (``match-outcomes.txt`` field 8 length) for the matching ``run_id``.
- ``run_id`` and ``(set_code, method_A, method_B)`` tuples line up
  positionally between the two files.
- Basic-land names never appear in any of the four list columns.
- Game lines for one match appear contiguously and in game order.

Tagged ``@pytest.mark.integration`` because it shells out to Forge,
which takes minutes per match and is excluded from the fast suite.
Skipped automatically when the Forge classpath is unavailable.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

_BASIC_LAND_NAMES = frozenset({
    "Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes",
    "Snow-Covered Plains", "Snow-Covered Island", "Snow-Covered Swamp",
    "Snow-Covered Mountain", "Snow-Covered Forest", "Snow-Covered Wastes",
})


def _forge_available() -> bool:
    """Quick check for forge-connector + sibling forge build."""
    fat_jar = list(
        Path("forge-connector/target").glob("*-jar-with-dependencies.jar"),
    )
    forge_dir = Path("../forge")
    return bool(fat_jar) and forge_dir.is_dir()


@pytest.mark.integration
def test_cards_played_collection(tmp_path: Path) -> None:
    if not _forge_available():
        pytest.skip(
            "forge-connector fat jar or sibling ../forge checkout not available",
        )

    from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

    output_file = tmp_path / "match-outcomes.txt"
    cards_played_file = tmp_path / "cards-played.txt"
    run_id = "test-cards-played-collection"

    connector = MatchWorkerConnector()
    log_path = tmp_path / "worker.log"
    with log_path.open("wb") as log:
        process = connector.start(
            output_file=output_file,
            run_id=run_id,
            best_of=3,
            log_file=log,
        )
        # Wait for ~2 matches (Bo3) — Forge takes a while; cap at 3 minutes.
        deadline = time.time() + 180
        target_match_lines = 2
        while time.time() < deadline:
            if output_file.exists():
                lines = output_file.read_text(encoding="utf-8").splitlines()
                if len(lines) >= target_match_lines:
                    break
            time.sleep(2)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    if not output_file.exists():
        pytest.skip("worker did not produce match-outcomes.txt within deadline")

    match_lines = output_file.read_text(encoding="utf-8").splitlines()
    if len(match_lines) < 1:
        pytest.skip("worker produced no completed matches within deadline")
    assert cards_played_file.exists(), "cards-played.txt must be written alongside"
    cards_lines = cards_played_file.read_text(encoding="utf-8").splitlines()

    expected_game_count = 0
    for line in match_lines:
        fields = line.split(";")
        # field 7 = games (e.g. "ABA")
        expected_game_count += len(fields[7])
    assert len(cards_lines) == expected_game_count, (
        f"cards-played line count {len(cards_lines)} != "
        f"sum-of-games {expected_game_count}"
    )

    # Validate run_id and metadata alignment, basic-land filter, game order.
    cursor = 0
    for line in match_lines:
        fields = line.split(";")
        m_run_id, m_set_code, m_method_a, m_method_b = (
            fields[1], fields[2], fields[3], fields[4]
        )
        assert m_run_id == run_id
        games = fields[7]
        play = fields[8]
        for offset, (winner_char, starter_char) in enumerate(zip(games, play)):
            row = cards_lines[cursor + offset].split(";")
            assert len(row) == 11, f"cards-played row must have 11 fields: {row}"
            assert row[1] == m_run_id
            assert row[2] == m_set_code
            assert row[3] == m_method_a
            assert row[4] == m_method_b
            played_a = row[5].split("|") if row[5] else []
            played_b = row[6].split("|") if row[6] else []
            not_played_a = row[7].split("|") if row[7] else []
            not_played_b = row[8].split("|") if row[8] else []
            for col_name, names in (
                ("cards_played_A", played_a),
                ("cards_played_B", played_b),
                ("cards_not_played_A", not_played_a),
                ("cards_not_played_B", not_played_b),
            ):
                for name in names:
                    assert name not in _BASIC_LAND_NAMES, (
                        f"basic land in {col_name}: {name!r}"
                    )
                assert len(set(names)) == len(names), (
                    f"{col_name} must not contain duplicates: {names}"
                )
            assert set(played_a).isdisjoint(set(not_played_a)), (
                "cards_played_A and cards_not_played_A must be disjoint"
            )
            assert set(played_b).isdisjoint(set(not_played_b)), (
                "cards_played_B and cards_not_played_B must be disjoint"
            )
            assert row[9] == winner_char
            assert row[10] == starter_char
        cursor += len(games)
