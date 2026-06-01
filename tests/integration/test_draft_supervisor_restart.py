"""Crash-recovery: the supervisor restarts a dying worker until --n-drafts (SC-003).

Exercises the FR-011 restart loop with a stub worker process (no live Forge):
each fake worker emits a couple of sentinel transcripts then "crashes" (its
stdout iterator ends), and the supervisor must relaunch until it reaches the
target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from draft.application.generate_draft_data import (
    SENTINEL,
    GenerateDraftDataConfig,
    GenerateDraftDataSupervisor,
)
from draft.infrastructure.draft_record_io import read_records

pytestmark = pytest.mark.integration


class _FakeProc:
    """A stub worker: yields some sentinel lines, then EOF (a 'crash')."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self._returncode = 1

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        return self._returncode


class _StubLabeler:
    def build_and_score(self, pool):
        return ["Plains"] * 40, 1.0


def _transcript_line(draft_id: str) -> str:
    payload = {
        "draft_id": draft_id,
        "boosters": [
            {"set_code": "BLB", "picks": ["A", "B"]},
            {"set_code": "BLB", "picks": ["C", "D"]},
        ],
        "seats": [{"agent": "forge-full"}, {"agent": "forge-full"}],
    }
    return SENTINEL + json.dumps(payload)


def test_supervisor_restarts_until_target(tmp_path: Path) -> None:
    out_path = tmp_path / "drafts.jsonl"
    config = GenerateDraftDataConfig(
        n_drafts=5,
        agent_mix=[("forge-full", 1)],
        output_path=out_path,
    )

    # Each launch yields a worker emitting 2 drafts then crashing. Mix in noise
    # lines that must be ignored. The 3rd+ launches finish the target of 5.
    launches = {"n": 0}

    def launch():
        launches["n"] += 1
        base = launches["n"] * 100
        lines = [
            "Forge: noise line that is not a sentinel\n",
            _transcript_line(f"d{base}") + "\n",
            SENTINEL + "{garbage json\n",          # parse failure -> skipped
            _transcript_line(f"d{base + 1}") + "\n",
        ]
        return _FakeProc(lines)

    supervisor = GenerateDraftDataSupervisor(
        config, labeler=_StubLabeler(), launch_worker=launch,
    )
    count = supervisor.run()

    assert count == 5
    records = list(read_records(out_path))
    assert len(records) == 5
    assert all(r.run_id == supervisor.run_id for r in records)
    # 2 good drafts per launch -> needed 3 launches to reach 5.
    assert launches["n"] == 3
