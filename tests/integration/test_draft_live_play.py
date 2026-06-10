"""End-to-end live model-pilot smoke (Forge-dependent, US1).

Drives the real Java ``DraftWorkerMain`` with a model-piloted seat: the trained
policy answers every pick over the live side-channel, and the completed draft is
labeled + scored and appended to ``drafts.jsonl``. Skips when the built JAR, the
agent/scorer/picker checkpoints, or the .npz cache are absent (so the fast suite
and CI without Forge stay green).
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

_AGENT = Path("models/draft/agent/latest.pt")
_SCORER = Path("models/sealed/scorer/latest.pt")
_PICKER = Path("models/sealed/picker/latest.pt")
_CARDS = Path("output/cardsfolder")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        pytest.skip(reason)


def test_model_piloted_draft_is_labeled_and_recorded(tmp_path: Path) -> None:
    try:
        resolve_connector_jar()
    except FileNotFoundError:
        pytest.skip("forge-connector JAR not built (mvn package -DskipTests)")
    _require(_AGENT.exists(), f"missing {_AGENT}")
    _require(_SCORER.exists(), f"missing {_SCORER}")
    _require(_PICKER.exists(), f"missing {_PICKER}")
    _require(_CARDS.is_dir(), f"missing {_CARDS}")

    out_path = tmp_path / "drafts.jsonl"
    # An all-draft-agent mix guarantees every seat is model-piloted (≥1 model
    # seat), exercising the full pick side-channel for the whole pod.
    config = GenerateDraftDataConfig(
        n_drafts=1,
        set_code="BLB",
        agent_mix=[("draft-agent", 1)],
        scorer_checkpoint=_SCORER,
        picker_checkpoint=_PICKER,
        cards_path=_CARDS,
        output_path=out_path,
        agent_checkpoints={"draft-agent": _AGENT},
    )

    count = GenerateDraftDataSupervisor(config).run()
    assert count == 1

    records = list(read_records(out_path))
    assert len(records) == 1
    record = records[0]
    geo = DraftGeometry.from_record(record)
    assert geo.pod_size == len(record.seats)
    # Every seat carries the model label and a built + scored deck (US1).
    assert all(s.agent == "draft-agent" for s in record.seats)
    assert all(len(s.deck) == 40 and s.deck_score is not None for s in record.seats)
