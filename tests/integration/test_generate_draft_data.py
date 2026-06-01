"""End-to-end generate-draft-data smoke (Forge-dependent).

Drives the real Java ``DraftWorkerMain`` through the supervisor and the real
picker/scorer labeler, then checks the corpus is parseable and FR-016
reconstructable. Skips when the built JAR, model checkpoints, or .npz cache are
absent (so the fast suite and CI without Forge stay green).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from draft.application.generate_draft_data import (
    GenerateDraftDataConfig,
    GenerateDraftDataSupervisor,
)
from draft.domain.draft_geometry import DraftGeometry
from draft.infrastructure.draft_record_io import read_records
from price_predictor.infrastructure.forge_jvm import resolve_connector_jar

pytestmark = pytest.mark.integration

_SCORER = Path("models/sealed/scorer/latest.pt")
_PICKER = Path("models/sealed/picker/latest.pt")
_CARDS = Path("output/cardsfolder")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        pytest.skip(reason)


def test_generate_two_drafts_end_to_end(tmp_path: Path) -> None:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")
    _require(_SCORER.exists(), f"missing {_SCORER}")
    _require(_PICKER.exists(), f"missing {_PICKER}")
    _require(_CARDS.is_dir(), f"missing {_CARDS}")

    out_path = tmp_path / "drafts.jsonl"
    config = GenerateDraftDataConfig(
        n_drafts=2,
        agent_mix=[("forge-full", 6), ("forge-r30", 1), ("forge-r100", 1)],
        scorer_checkpoint=_SCORER,
        picker_checkpoint=_PICKER,
        cards_path=_CARDS,
        output_path=out_path,
    )

    count = GenerateDraftDataSupervisor(config).run()
    assert count == 2

    records = list(read_records(out_path))
    assert len(records) == 2
    for record in records:
        geo = DraftGeometry.from_record(record)
        assert geo.pod_size == len(record.seats)
        assert len(record.boosters) == geo.pod_size * geo.packs
        # Every seat's drafted pool reconstructs to packs * P cards (SC-002).
        for seat_idx in range(geo.pod_size):
            pool = geo.drafted_pool(record, seat_idx)
            assert len(pool) == geo.packs * geo.pack_size
