"""One real steered game (T188, FR-058d).

A steered variant round's deck names the rule it is aimed at in a fourth
deck-file column, and the worker stamps every record of a game with its two
decks' rules as ``aimed_at``, in place of the run's list. Checked here against
a real Forge worker playing a staged variant tree: the sidecar pass stages the
tree and writes its loadable list, and the worker reads the four-field deck
file.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from effects.application.collect_variants import LOADABLE_FILE, generate_variants
from effects.infrastructure.deck_file import write_deck_file
from effects.infrastructure.one_shot_workers import write_variant_sidecars
from effects.infrastructure.shard_io import read_records
from price_predictor.infrastructure.forge_jvm import kill_process_tree
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector
from tests.integration.conftest import POLL_SECONDS, require_jar

pytestmark = pytest.mark.integration

_BOLT = (
    "Name:Steered Bolt\nManaCost:R\nTypes:Instant\n"
    "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 3 | "
    "SpellDescription$ CARDNAME deals 3 damage to any target.\n"
    "Oracle:Steered Bolt deals 3 damage to any target.\n"
)

#: A game of burn spells ends in a few turns; the rest is a cold JVM's startup.
_GAME_SECONDS = 300


def _unstage(tmp_path: Path) -> None:
    """Take this test's variant back out of Forge's custom-cards directory:
    staging an empty tree clears what the last staging listed."""
    empty = tmp_path / "no-variants"
    empty.mkdir()
    write_variant_sidecars(empty)


def test_a_steered_games_records_carry_its_decks_rules(tmp_path: Path) -> None:
    require_jar()
    source = tmp_path / "cards"
    source.mkdir()
    (source / "steered_bolt.txt").write_text(_BOLT, encoding="utf-8")
    variants_dir = tmp_path / "variants"
    variants = generate_variants(source, variants_dir, held_out=frozenset())
    assert variants, "no variant was generated from a perturbable script"
    variant = variants[0].name

    records_dir = tmp_path / "records"
    records_dir.mkdir()
    progress_file = tmp_path / "progress.txt"
    decks_file = tmp_path / "decks.txt"
    write_deck_file(
        [
            [variant] * 23 + ["Mountain"] * 17,
            [variant] * 23 + ["Mountain"] * 16 + ["Forest"],
        ],
        decks_file,
        label="variant",
        rules=["704.5f", "702.19b"],
    )
    assert all(len(line.split(";")) == 4 for line in decks_file.read_text().splitlines())

    process = None
    try:
        assert write_variant_sidecars(variants_dir) == 0, "VariantSidecarMain failed"
        assert variant in (variants_dir / LOADABLE_FILE).read_text(encoding="utf-8").splitlines()

        process = MatchWorkerConnector().start(
            None,
            run_id=str(uuid.uuid4()),
            best_of=1,
            effect_records_dir=records_dir,
            worker_index=0,
            # A variant run aims at no rule of its own: the decks' rules replace it.
            collection_caps={"effect.aimed.at": ""},
            side_a_decks_path=decks_file,
            side_b_decks_path=decks_file,
            decks_only=True,
            progress_file=progress_file,
            variant_scripts=variants_dir,
        )
        deadline = time.monotonic() + _GAME_SECONDS
        while time.monotonic() < deadline:
            time.sleep(POLL_SECONDS)
            if process.poll() is not None:
                pytest.fail(f"worker exited early with {process.returncode}")
            if progress_file.exists() and progress_file.read_text(encoding="utf-8").strip():
                break
        else:
            pytest.fail("no game finished inside the window")
    finally:
        if process is not None:
            kill_process_tree(process)
        _unstage(tmp_path)

    records = list(read_records(records_dir))
    assert records, "the steered game wrote no record"
    assert {record.aimed_at for record in records} == {("702.19b", "704.5f")}
    names = {entity.name for record in records for entity in record.state.entities}
    assert variant in names, f"the variant never reached a record; saw {sorted(names)}"
