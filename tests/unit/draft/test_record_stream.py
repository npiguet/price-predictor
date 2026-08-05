"""The resident rollout stream: ``iter_records`` + its launch/labeler seams.

Covers contracts/rollout-stream.md § 1 and research D1/D2/D14. The generator is
what makes one Forge worker resident across rounds: while the consumer is not
pulling, the generator is suspended and the worker stays alive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from draft.application.generate_draft_data import (
    PICK_REQUEST_SENTINEL,
    POD_SIZE,
    SENTINEL,
    GenerateDraftDataConfig,
    GenerateDraftDataSupervisor,
    MaxConsecutiveFaultsError,
)


class _FakeProc:
    """A stub worker: yields lines, records what was written to its stdin."""

    def __init__(self, lines: list[str]) -> None:
        self.stdout = iter(lines)
        self.terminated = False
        self.written: list[str] = []
        self.stdin = self._Stdin(self.written)

    class _Stdin:
        def __init__(self, sink: list[str]) -> None:
            self._sink = sink

        def write(self, text: str) -> None:
            self._sink.append(text)

        def flush(self) -> None:
            pass

    def poll(self):
        return 1

    def wait(self, timeout=None):
        self.terminated = True
        return 1


class _StubLabeler:
    def build_and_score(self, pool):
        return ["Plains"] * 40, 1.0


def _transcript(draft_id: str, agents: list[str] | None = None) -> str:
    payload = {
        "draft_id": draft_id,
        "boosters": [
            {"set_code": "BLB", "picks": ["A", "B"]},
            {"set_code": "BLB", "picks": ["C", "D"]},
        ],
        "seats": [{"agent": a} for a in (agents or ["forge-full", "forge-full"])],
    }
    return SENTINEL + json.dumps(payload) + "\n"


def _config(**overrides) -> GenerateDraftDataConfig:
    base = {"n_drafts": 0, "agent_mix": [("forge-full", 1)]}
    base.update(overrides)
    return GenerateDraftDataConfig(**base)


def _supervisor(**overrides) -> GenerateDraftDataSupervisor:
    return GenerateDraftDataSupervisor(_config(**overrides), labeler=_StubLabeler())


# --------------------------------------------------------------------------- #
# Worker lifetime + suspension
# --------------------------------------------------------------------------- #

def test_worker_launches_only_on_the_first_pull() -> None:
    launches = {"n": 0}

    def launch():
        launches["n"] += 1
        return _FakeProc([_transcript("d1"), _transcript("d2")])

    records = _supervisor().iter_records(launch, _StubLabeler())
    assert launches["n"] == 0  # constructing the generator starts nothing

    next(records)
    assert launches["n"] == 1
    records.close()


def test_yields_fully_assembled_labeled_records() -> None:
    supervisor = _supervisor()

    def launch():
        return _FakeProc([_transcript("d1")])

    records = supervisor.iter_records(launch, _StubLabeler())
    record = next(records)
    records.close()

    assert record.draft_id == "d1"
    assert record.run_id == supervisor.run_id
    assert record.timestamp
    assert len(record.seats) == 2
    assert all(seat.deck_score == 1.0 for seat in record.seats)
    assert len(record.boosters) == 2


def test_generator_stays_suspended_between_pulls() -> None:
    """Nothing past the yielded record is consumed until the consumer asks."""
    procs: list[_FakeProc] = []

    def launch():
        proc = _FakeProc([_transcript(f"d{i}") for i in range(5)])
        procs.append(proc)
        return proc

    records = _supervisor().iter_records(launch, _StubLabeler())
    first = next(records)

    assert first.draft_id == "d0"
    assert len(procs) == 1
    assert not procs[0].terminated  # the worker is still alive and resident
    # The remaining four transcripts have not been read off stdout.
    assert next(procs[0].stdout).startswith(SENTINEL)

    records.close()


def test_relaunches_after_a_worker_exit_and_keeps_yielding() -> None:
    launches = {"n": 0}

    def launch():
        launches["n"] += 1
        base = launches["n"] * 10
        return _FakeProc([_transcript(f"d{base}"), _transcript(f"d{base + 1}")])

    records = _supervisor().iter_records(launch, _StubLabeler())
    ids = [next(records).draft_id for _ in range(5)]
    records.close()

    assert ids == ["d10", "d11", "d20", "d21", "d30"]
    assert launches["n"] == 3


def test_is_endless_where_run_would_have_stopped() -> None:
    """The generator has no target; stopping is entirely the consumer's business."""
    def launch():
        return _FakeProc([_transcript("d1")])

    records = _supervisor(n_drafts=1).iter_records(launch, _StubLabeler())
    assert len([next(records) for _ in range(4)]) == 4
    records.close()


def test_close_terminates_the_worker() -> None:
    procs: list[_FakeProc] = []

    def launch():
        proc = _FakeProc([_transcript("d1"), _transcript("d2")])
        procs.append(proc)
        return proc

    records = _supervisor().iter_records(launch, _StubLabeler())
    next(records)
    assert not procs[0].terminated

    records.close()
    assert procs[0].terminated


def test_nothing_is_written_to_disk_by_the_generator(tmp_path: Path) -> None:
    """The consumer owns persistence — the stream only yields."""
    out_path = tmp_path / "drafts.jsonl"

    def launch():
        return _FakeProc([_transcript("d1")])

    records = _supervisor(output_path=out_path).iter_records(launch, _StubLabeler())
    next(records)
    records.close()

    assert not out_path.exists()


def test_unparseable_transcripts_and_noise_are_skipped() -> None:
    def launch():
        return _FakeProc([
            "Forge: noise\n",
            SENTINEL + "{garbage json\n",
            _transcript("d1"),
        ])

    records = _supervisor().iter_records(launch, _StubLabeler())
    assert next(records).draft_id == "d1"
    records.close()


# --------------------------------------------------------------------------- #
# Pick routing is unchanged
# --------------------------------------------------------------------------- #

class _StubRegistry:
    def __init__(self, pick: str | None = "A") -> None:
        self._pick = pick
        self.reset_ids: list[str] = []

    def pick(self, request) -> str:
        from draft.application.agent_pick_service import PickFault

        if self._pick is None:
            raise PickFault("no genuine pick")
        return self._pick

    def reset_draft(self, draft_id: str) -> None:
        self.reset_ids.append(draft_id)


def _pick_request(draft_id: str = "d1", seat: int = 0) -> str:
    payload = {
        "draft_id": draft_id, "seat": seat, "agent": "draft-agent",
        "set_code": "BLB", "pack_number": 1, "pick_number": 1,
        "pod_size": POD_SIZE, "pack": ["A", "B"],
    }
    return PICK_REQUEST_SENTINEL + json.dumps(payload) + "\n"


def test_pick_requests_are_answered_from_the_registry() -> None:
    procs: list[_FakeProc] = []
    registry = _StubRegistry(pick="B")

    def launch():
        proc = _FakeProc([_pick_request(), _transcript("d1")])
        procs.append(proc)
        return proc

    supervisor = GenerateDraftDataSupervisor(
        _config(agent_checkpoints={"draft-agent": Path("x.pt")}),
        labeler=_StubLabeler(), registry=registry,
    )
    records = supervisor.iter_records(launch, _StubLabeler())
    next(records)
    records.close()

    assert len(procs[0].written) == 1
    assert '"pick": "B"' in procs[0].written[0] or '"pick":"B"' in procs[0].written[0]


def test_max_consecutive_faults_propagates_out_of_the_generator() -> None:
    def launch():
        return _FakeProc([_pick_request(f"d{i}", 0) for i in range(10)])

    supervisor = GenerateDraftDataSupervisor(
        _config(
            agent_checkpoints={"draft-agent": Path("x.pt")},
            max_consecutive_faults=2,
        ),
        labeler=_StubLabeler(), registry=_StubRegistry(pick=None),
    )
    records = supervisor.iter_records(launch, _StubLabeler())
    with pytest.raises(MaxConsecutiveFaultsError):
        next(records)


# --------------------------------------------------------------------------- #
# The launch seam: required_agent forwarding (research D2, FR-003)
# --------------------------------------------------------------------------- #

class _RecordingConnector:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def start(self, **kwargs):
        self.kwargs = kwargs
        return _FakeProc([])


def test_default_launch_forwards_required_agent(monkeypatch) -> None:
    connector = _RecordingConnector()
    monkeypatch.setattr(
        "draft.infrastructure.draft_worker_connector.DraftWorkerConnector",
        lambda: connector,
    )
    supervisor = _supervisor(required_agent="gen-3")

    supervisor._default_launch_worker(frozenset({"gen-3"}))()

    assert connector.kwargs["required_agent"] == "gen-3"


def test_default_launch_omits_required_agent_when_unset(monkeypatch) -> None:
    """Existing callers must launch exactly the command line they always did."""
    connector = _RecordingConnector()
    monkeypatch.setattr(
        "draft.infrastructure.draft_worker_connector.DraftWorkerConnector",
        lambda: connector,
    )
    supervisor = _supervisor()

    supervisor._default_launch_worker(frozenset())()

    assert "required_agent" not in connector.kwargs


def test_required_agent_defaults_to_none_on_the_config() -> None:
    assert _config().required_agent is None


# --------------------------------------------------------------------------- #
# The labeler seam: one shared locator (research D14)
# --------------------------------------------------------------------------- #

def test_build_labeler_uses_a_supplied_locator(monkeypatch) -> None:
    """One memoizing locator serves labeler + pick services + trainer."""
    import draft.application.generate_draft_data as gdd

    sentinel = object()
    captured: dict = {}

    class _StubScorerStore:
        def load_checkpoint(self, path):
            raise AssertionError("not reached")

    def fake_greedy(scorer, locator):
        captured["locator"] = locator
        return "labeler"

    monkeypatch.setattr(gdd, "_GreedyLabeler", fake_greedy)
    monkeypatch.setattr(
        "sealed.infrastructure.scorer_store.ScorerStore",
        lambda: _FakeScorerStore(),
    )
    monkeypatch.setattr(
        "sealed.domain.scorer_model.SetTransformerScorer", _FakeScorer,
    )

    gdd.build_labeler(_config(build_method="greedy"), locator=sentinel)
    assert captured["locator"] is sentinel


def test_build_labeler_constructs_its_own_locator_when_absent(monkeypatch) -> None:
    import draft.application.generate_draft_data as gdd

    captured: dict = {}

    def fake_greedy(scorer, locator):
        captured["locator"] = locator
        return "labeler"

    monkeypatch.setattr(gdd, "_GreedyLabeler", fake_greedy)
    monkeypatch.setattr(
        "sealed.infrastructure.scorer_store.ScorerStore",
        lambda: _FakeScorerStore(),
    )
    monkeypatch.setattr(
        "sealed.domain.scorer_model.SetTransformerScorer", _FakeScorer,
    )

    gdd.build_labeler(_config(build_method="greedy"))

    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
    assert isinstance(captured["locator"], ConvertedCardLocator)


class _FakeCheckpoint:
    config = object()
    model_state_dict: dict = {}


class _FakeScorerStore:
    def load_checkpoint(self, path):
        return _FakeCheckpoint()


class _FakeScorer:
    def __init__(self, config) -> None:
        pass

    def load_state_dict(self, state) -> None:
        pass

    def eval(self) -> None:
        pass

    def to(self, device) -> None:
        pass
