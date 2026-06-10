"""Supervisor live-pick routing with a fake worker stream (T008).

Drives ``GenerateDraftDataSupervisor`` without a JVM: a fake worker yields a
canned line stream and captures the responses written to its stdin.
"""

from __future__ import annotations

import json

import pytest

from draft.application.agent_pick_service import PickFault
from draft.application.generate_draft_data import (
    PICK_REQUEST_SENTINEL,
    PICK_RESPONSE_SENTINEL,
    SENTINEL,
    GenerateDraftDataConfig,
    GenerateDraftDataSupervisor,
    MaxConsecutiveFaultsError,
    parse_abandoned,
)


class _FakeStdin:
    def __init__(self) -> None:
        self.written: list[str] = []

    def write(self, s: str) -> None:
        self.written.append(s)

    def flush(self) -> None:
        pass


class _FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self.stdin = _FakeStdin()

    def poll(self):
        return 0

    def wait(self):
        return 0


class _FakeLabeler:
    def build_and_score(self, pool):
        return [], None


class _FakeRegistry:
    external_labels = frozenset({"draft-agent"})

    def __init__(self, *, fault: bool = False) -> None:
        self._fault = fault
        self.reset_calls: list[str] = []

    def pick(self, request) -> str:
        if self._fault:
            raise PickFault("forced fault")
        return request.pack[0]

    def reset_draft(self, draft_id: str) -> None:
        self.reset_calls.append(draft_id)


def _pick_request(draft_id="dx", seat=0, pick_number=1) -> str:
    return PICK_REQUEST_SENTINEL + json.dumps({
        "draft_id": draft_id, "seat": seat, "agent": "draft-agent",
        "pod_size": 8, "pack_number": 1, "pick_number": pick_number,
        "set_code": "TST", "pack": ["A", "B", "C"],
    })


def _event(draft_id="d-done") -> str:
    return SENTINEL + json.dumps({
        "draft_id": draft_id,
        "boosters": [
            {"set_code": "TST", "picks": ["A", "B"]},
            {"set_code": "TST", "picks": ["C", "D"]},
        ],
        "seats": [{"agent": "draft-agent"}, {"agent": "forge-full"}],
    })


def _supervisor(tmp_path, lines, *, registry, target=1, max_faults=5):
    config = GenerateDraftDataConfig(
        n_drafts=target,
        agent_mix=[("draft-agent", 1), ("forge-full", 1)],
        output_path=tmp_path / "drafts.jsonl",
        max_consecutive_faults=max_faults,
    )
    proc = _FakeProc(lines)
    sup = GenerateDraftDataSupervisor(
        config, labeler=_FakeLabeler(),
        launch_worker=lambda: proc, registry=registry,
    )
    return sup, proc


def _responses(proc) -> list[dict]:
    return [
        json.loads(w[len(PICK_RESPONSE_SENTINEL):])
        for w in proc.stdin.written if w.startswith(PICK_RESPONSE_SENTINEL)
    ]


def test_pick_request_answered_with_registry_pick(tmp_path) -> None:
    registry = _FakeRegistry()
    sup, proc = _supervisor(
        tmp_path, [_pick_request(), _event()], registry=registry,
    )
    sup.run()
    responses = _responses(proc)
    assert responses == [{
        "draft_id": "dx", "seat": 0, "pack_number": 1, "pick_number": 1,
        "pick": "A",  # registry returns pack[0]
    }]
    # The completed draft's trackers are reset (FR-016).
    assert "d-done" in registry.reset_calls


def test_python_side_fault_sends_abort_and_drops_draft(tmp_path) -> None:
    registry = _FakeRegistry(fault=True)
    sup, proc = _supervisor(
        tmp_path, [_pick_request(draft_id="bad"), _event()], registry=registry,
    )
    sup.run()
    responses = _responses(proc)
    assert responses == [{
        "draft_id": "bad", "seat": 0, "pack_number": 1, "pick_number": 1,
        "abort": True,
    }]
    # SC-002: only the genuinely completed draft is recorded (no substitute).
    lines = (tmp_path / "drafts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["draft_id"] == "d-done"
    assert "bad" in registry.reset_calls


def test_abandoned_notice_counted_and_reset_on_completion(tmp_path) -> None:
    registry = _FakeRegistry()
    abandoned = parse_abandoned  # sanity: importable
    assert abandoned is not None
    lines = [
        '<<DRAFT-ABANDONED>>{"draft_id":"a1","reason":"mismatch"}',
        '<<DRAFT-ABANDONED>>{"draft_id":"a2","reason":"mismatch"}',
        _event(),
    ]
    sup, proc = _supervisor(tmp_path, lines, registry=registry)
    sup.run()
    # Counter incremented on each abandonment, then reset by the completed draft.
    assert sup._consecutive_faults == 0
    assert {"a1", "a2"}.issubset(set(registry.reset_calls))


def test_consecutive_fault_auto_abort(tmp_path) -> None:
    registry = _FakeRegistry(fault=True)
    lines = [_pick_request(draft_id=f"f{i}") for i in range(5)]
    sup, proc = _supervisor(
        tmp_path, lines, registry=registry, target=99, max_faults=3,
    )
    with pytest.raises(MaxConsecutiveFaultsError):
        sup.run()


def test_empty_registry_is_gen1_identical(tmp_path) -> None:
    # No registry: pick lines never appear (worker wouldn't emit them); the loop
    # writes no responses and records the completed draft exactly as gen-1 (SC-004).
    sup, proc = _supervisor(tmp_path, [_event()], registry=None)
    sup.run()
    assert proc.stdin.written == []
    lines = (tmp_path / "drafts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
