"""generate-draft-data supervisor: drive a Java draft worker, label + record drafts.

The Java ``DraftWorkerMain`` runs Forge's draft AI for all pod seats and streams
one ``<<DRAFT-EVENT-JSON>>`` transcript line per completed draft (boosters +
per-seat agent, no deck/score). This supervisor reconstructs each seat's drafted
pool from the booster geometry (FR-016), builds a 40-card deck per seat with the
frozen picker (default) or SA greedy builder, scores the non-basic subset with
the frozen scorer, assembles a self-contained record, and appends it to
``drafts.jsonl``. A crashed worker is restarted until ``--n-drafts`` is reached
(FR-011); SIGINT stops cleanly.

Follows ``sealed/application/match_outcomes.py`` (run-id + crash-restart +
status loop), additionally consuming worker stdout (which the match supervisor
discards). Collaborators (deck labeler, worker launcher) are injectable so the
sentinel parsing, record assembly, and restart loop are unit-testable without a
live JVM (T010, T011, T014).
"""

from __future__ import annotations

import json
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from draft.application.agent_mix import format_agent_mix
from draft.domain.draft_geometry import Booster, DraftGeometry, DraftRecord, Seat
from draft.infrastructure.draft_record_io import append_record, count_complete_records

SENTINEL = "<<DRAFT-EVENT-JSON>>"
PICK_REQUEST_SENTINEL = "<<DRAFT-PICK-REQUEST>>"
PICK_RESPONSE_SENTINEL = "<<DRAFT-PICK-RESPONSE>>"
ABANDONED_SENTINEL = "<<DRAFT-ABANDONED>>"

# Live draft geometry (fixed on both sides; the Java worker mirrors these).
POD_SIZE = 8
PACKS = 3


def _format_duration(seconds: float) -> str:
    """Compact h/m/s duration, e.g. '4m12s' or '1h03m'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _progress_line(
    count: int, target: int, rate_per_min: float, elapsed: float,
) -> str:
    """Progress line with a recent-window throughput + ETA.

    ``rate_per_min`` is a rolling rate (drafts over the last window), not the
    lifetime average, so it reflects current speed and a responsive ETA rather
    than being dragged down by one-time startup cost.
    """
    parts = [
        f"  {count}/{target} drafts recorded",
        f"{rate_per_min:.1f}/min",
        f"{_format_duration(elapsed)} elapsed",
    ]
    remaining = target - count
    if rate_per_min > 0 and remaining > 0:
        parts.append(f"ETA ~{_format_duration(remaining / rate_per_min * 60)}")
    return " | ".join(parts)


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class GenerateDraftDataConfig:
    n_drafts: int
    agent_mix: list[tuple[str, int]]
    set_code: str | None = None
    scorer_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/scorer/latest.pt"),
    )
    build_method: str = "picker"
    picker_checkpoint: Path = field(
        default_factory=lambda: Path("models/sealed/picker/latest.pt"),
    )
    cards_path: Path = field(default_factory=lambda: Path("output/cardsfolder/"))
    output_path: Path = field(
        default_factory=lambda: Path("output/draft/drafts.jsonl"),
    )
    resume: bool = False
    # Live model-pilot additions (all optional; defaults preserve gen-1, SC-004).
    agent_checkpoints: dict[str, Path] = field(default_factory=dict)
    pick_mode: str = "argmax"
    temperature: float = 1.0
    seed: int | None = None
    max_consecutive_faults: int = 5


# --------------------------------------------------------------------------- #
# Sentinel parsing + record assembly (pure, Forge-free)
# --------------------------------------------------------------------------- #

def parse_sentinel_line(line: str) -> dict[str, Any] | None:
    """Return the parsed transcript dict for a sentinel line, else ``None``.

    Keeps only lines starting with ``SENTINEL``; defensively ``json.loads`` the
    suffix and returns ``None`` on any parse failure (Forge stdout noise is
    silently dropped — FR-010, worker-protocol.md).
    """
    if not line.startswith(SENTINEL):
        return None
    suffix = line[len(SENTINEL):].strip()
    try:
        parsed = json.loads(suffix)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


# --------------------------------------------------------------------------- #
# Live-pick side-channel messages (contracts/pick-protocol.md, data-model §2.1–2.3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PickRequest:
    """Worker→supervisor request for one external (model-piloted) seat's pick.

    Parsed from a ``<<DRAFT-PICK-REQUEST>>`` line. ``pack`` holds the card names
    remaining in the held pack, in pick (offset) order (order is insignificant
    to the model); ``set_code`` is informational (data-model §2.1).
    """

    draft_id: str
    seat: int
    agent: str
    pod_size: int
    pack_number: int
    pick_number: int
    set_code: str
    pack: list[str]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PickRequest":
        """Build + validate a request from its JSON payload (data-model §2.1).

        Raises ``ValueError`` on a structurally invalid request (a protocol
        desync the supervisor treats as a pick fault).
        """
        try:
            request = cls(
                draft_id=str(payload["draft_id"]),
                seat=int(payload["seat"]),
                agent=str(payload["agent"]),
                pod_size=int(payload["pod_size"]),
                pack_number=int(payload["pack_number"]),
                pick_number=int(payload["pick_number"]),
                set_code=str(payload.get("set_code", "")),
                pack=[str(c) for c in payload["pack"]],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed pick request: {exc}") from exc
        if request.pod_size != POD_SIZE:
            raise ValueError(
                f"pick request pod_size {request.pod_size} != live pod size {POD_SIZE}"
            )
        if not 1 <= request.pack_number <= PACKS:
            raise ValueError(
                f"pick request pack_number {request.pack_number} out of range 1..{PACKS}"
            )
        if request.pick_number < 1:
            raise ValueError(f"pick request pick_number {request.pick_number} < 1")
        if not request.pack:
            raise ValueError("pick request has an empty pack")
        return request


def parse_pick_request(line: str) -> PickRequest | None:
    """Parse a ``<<DRAFT-PICK-REQUEST>>`` line, else ``None`` for any other line.

    Returns ``None`` when the line is not a pick request or its JSON suffix is
    not a dict; raises ``ValueError`` (via :meth:`PickRequest.from_payload`) for
    a pick request whose fields are invalid.
    """
    if not line.startswith(PICK_REQUEST_SENTINEL):
        return None
    suffix = line[len(PICK_REQUEST_SENTINEL):].strip()
    try:
        payload = json.loads(suffix)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return PickRequest.from_payload(payload)


@dataclass(frozen=True)
class PickResponse:
    """Supervisor→worker response for one pick (data-model §2.2).

    Exactly one of ``pick`` (the chosen card) or ``abort`` (supervisor-initiated
    draft abandonment on a Python-side fault) is meaningful.
    """

    draft_id: str
    seat: int
    pack_number: int
    pick_number: int
    pick: str | None = None
    abort: bool = False

    def __post_init__(self) -> None:
        if (self.pick is None) == (not self.abort):
            raise ValueError(
                "PickResponse must carry exactly one of `pick` or `abort=True`"
            )

    def serialize(self) -> str:
        """Render the ``<<DRAFT-PICK-RESPONSE>>`` line (newline-free, UTF-8)."""
        payload: dict[str, Any] = {
            "draft_id": self.draft_id,
            "seat": self.seat,
            "pack_number": self.pack_number,
            "pick_number": self.pick_number,
        }
        if self.abort:
            payload["abort"] = True
        else:
            payload["pick"] = self.pick
        return PICK_RESPONSE_SENTINEL + json.dumps(payload, separators=(",", ":"))


@dataclass(frozen=True)
class AbandonedNotice:
    """Worker→supervisor notice that the worker self-detected a fault (§2.3)."""

    draft_id: str
    reason: str


def parse_abandoned(line: str) -> AbandonedNotice | None:
    """Parse a ``<<DRAFT-ABANDONED>>`` line, else ``None`` for any other line."""
    if not line.startswith(ABANDONED_SENTINEL):
        return None
    suffix = line[len(ABANDONED_SENTINEL):].strip()
    try:
        payload = json.loads(suffix)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return AbandonedNotice(
        draft_id=str(payload.get("draft_id", "")),
        reason=str(payload.get("reason", "")),
    )


class DeckLabeler(Protocol):
    """Builds + scores a 40-card deck from one seat's drafted pool.

    A labeler MAY additionally implement ``build_and_score_many(pools)`` to label
    a whole pod's seats in one batched GPU pass; :func:`_label_pools` uses it when
    present and otherwise falls back to per-pool ``build_and_score``.
    """

    def build_and_score(self, pool: list[str]) -> tuple[list[str], float | None]:
        """Return ``(deck, deck_score)``; ``([], None)`` on a failed build."""
        ...


def _label_pools(
    labeler: DeckLabeler, pools: list[list[str]],
) -> list[tuple[list[str], float | None]]:
    """Label all of a pod's seat pools, batching when the labeler supports it.

    The real picker/greedy labelers expose ``build_and_score_many`` so the 8
    seats of a draft go through one picker forward + one scorer forward instead
    of 16 batch-of-1 calls; simple test doubles only need ``build_and_score``.
    """
    batch = getattr(labeler, "build_and_score_many", None)
    if batch is not None:
        return batch(pools)
    return [labeler.build_and_score(p) for p in pools]


def assemble_record(
    transcript: dict[str, Any], run_id: str, labeler: DeckLabeler,
) -> DraftRecord:
    """Turn a worker transcript into a complete, self-contained ``DraftRecord``.

    Reconstructs every seat's drafted pool via :mod:`draft.domain.draft_geometry`
    and labels each with ``labeler``. A seat whose pool yields no legal deck gets
    ``deck=[]`` / ``deck_score=None`` (FR-007, FR-014).
    """
    boosters = [
        Booster(set_code=b["set_code"], picks=list(b["picks"]))
        for b in transcript["boosters"]
    ]
    agents = [s["agent"] for s in transcript["seats"]]
    # Placeholder seats let the geometry helper reconstruct each drafted pool.
    placeholder = DraftRecord(
        draft_id=transcript["draft_id"],
        run_id=run_id,
        timestamp="",
        seats=[Seat(agent=a, deck=[], deck_score=None) for a in agents],
        boosters=boosters,
    )
    geo = DraftGeometry.from_record(placeholder)
    pools = [geo.drafted_pool(placeholder, i) for i in range(len(agents))]
    labeled = _label_pools(labeler, pools)
    seats = [
        Seat(agent=agent, deck=deck, deck_score=score)
        for agent, (deck, score) in zip(agents, labeled)
    ]
    return DraftRecord(
        draft_id=transcript["draft_id"],
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seats=seats,
        boosters=boosters,
    )


# --------------------------------------------------------------------------- #
# Concrete labelers (build over torch; no Forge dependency)
# --------------------------------------------------------------------------- #

def build_labeler(config: GenerateDraftDataConfig) -> DeckLabeler:
    """Construct the picker or greedy labeler from frozen checkpoints (FR-007/8)."""
    import torch

    from sealed.domain.scorer_model import SetTransformerScorer
    from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
    from sealed.infrastructure.scorer_store import ScorerStore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locator = ConvertedCardLocator(config.cards_path)

    scorer_ckpt = ScorerStore().load_checkpoint(config.scorer_checkpoint)
    scorer = SetTransformerScorer(scorer_ckpt.config)
    scorer.load_state_dict(scorer_ckpt.model_state_dict)
    scorer.eval()
    scorer.to(device)

    if config.build_method == "greedy":
        return _GreedyLabeler(scorer, locator)
    return _PickerLabeler(config.picker_checkpoint, scorer, locator, device)


class _PickerLabeler:
    """Picker forward + § 1.1 pick walk + basics, then scorer (mirrors pick_decks)."""

    def __init__(self, picker_checkpoint, scorer, locator, device) -> None:
        import torch

        from sealed.domain.picker_model import PickerModel
        from sealed.infrastructure.picker_store import PickerStore

        ckpt = PickerStore().load_checkpoint(picker_checkpoint)
        model = PickerModel(ckpt.config)
        model.load_state_dict(ckpt.model_state_dict)
        model.eval()
        model.to(device)
        self._torch = torch
        self._model = model
        self._scorer = scorer
        self._locator = locator
        self._device = device

    def build_and_score(self, pool: list[str]) -> tuple[list[str], float | None]:
        return self.build_and_score_many([pool])[0]

    def build_and_score_many(
        self, pools: list[list[str]],
    ) -> list[tuple[list[str], float | None]]:
        """Label a whole pod in one picker forward + one scorer forward.

        Pools with < 23 embeddable cards fail to ``([], None)`` and are left out
        of both batches; the rest are zero-padded to the pod's longest pool,
        run through the picker once, decomposed into decks, and scored once.
        """
        import numpy as np

        from sealed.application.deck_assembly import (
            assemble_full_deck,
            load_pool_embeddings,
        )
        from sealed.application.evaluate_scorer import score_decks
        from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
        from sealed.domain.picker_model import decompose_picks

        torch = self._torch
        results: list[tuple[list[str], float | None] | None] = [None] * len(pools)
        prepared: list[tuple[int, list[str], list]] = []
        for i, pool in enumerate(pools):
            embeddings, valid = load_pool_embeddings(pool, self._locator)
            if len(valid) < NONLAND_DECK_SIZE:
                results[i] = ([], None)
                continue
            prepared.append((i, valid, [embeddings[n] for n in valid]))

        if prepared:
            b = len(prepared)
            max_n = max(len(embs) for _, _, embs in prepared)
            dim = prepared[0][2][0].shape[0]
            arr = np.zeros((b, max_n, dim), dtype=np.float32)
            mask_np = np.zeros((b, max_n), dtype=bool)
            for row, (_, _, embs) in enumerate(prepared):
                arr[row, : len(embs)] = np.stack(embs)
                mask_np[row, : len(embs)] = True
            cards = torch.from_numpy(arr).to(self._device)
            mask = torch.from_numpy(mask_np).to(self._device)
            with torch.no_grad():
                logits, _ = self._model(cards, mask)
            logits_cpu = logits.cpu()  # single device->host sync for the pod

            decks: list[list[str]] = []
            deck_idx: list[int] = []
            for row, (i, valid, embs) in enumerate(prepared):
                chosen = decompose_picks(logits_cpu[row, : len(embs)], embs, valid)
                decks.append(assemble_full_deck(chosen, self._locator))
                deck_idx.append(i)
            scores = score_decks(self._scorer, decks, self._locator)
            for j, i in enumerate(deck_idx):
                results[i] = (decks[j], float(scores[j]))

        return [r if r is not None else ([], None) for r in results]


class _GreedyLabeler:
    """SA greedy builder + basics, then scorer (mirrors build-decks)."""

    def __init__(self, scorer, locator) -> None:
        self._scorer = scorer
        self._locator = locator

    def build_and_score(self, pool: list[str]) -> tuple[list[str], float | None]:
        return self.build_and_score_many([pool])[0]

    def build_and_score_many(
        self, pools: list[list[str]],
    ) -> list[tuple[list[str], float | None]]:
        """Per-pool SA search (independent), then one batched scorer forward."""
        from sealed.application.deck_assembly import (
            assemble_full_deck,
            load_pool_embeddings,
        )
        from sealed.application.evaluate_scorer import score_decks
        from sealed.domain.greedy_deck_builder import (
            NONLAND_DECK_SIZE,
            GreedyDeckBuilder,
        )

        results: list[tuple[list[str], float | None] | None] = [None] * len(pools)
        decks: list[list[str]] = []
        deck_idx: list[int] = []
        for i, pool in enumerate(pools):
            embeddings, valid = load_pool_embeddings(pool, self._locator)
            if len(valid) < NONLAND_DECK_SIZE:
                results[i] = ([], None)
                continue
            chosen = GreedyDeckBuilder(self._scorer, embeddings).build(valid)
            decks.append(assemble_full_deck(chosen, self._locator))
            deck_idx.append(i)
        if decks:
            scores = score_decks(self._scorer, decks, self._locator)
            for j, i in enumerate(deck_idx):
                results[i] = (decks[j], float(scores[j]))
        return [r if r is not None else ([], None) for r in results]


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #

class MaxConsecutiveFaultsError(RuntimeError):
    """Raised when too many consecutive drafts abandon on a pick fault (FR-016)."""


class _WorkerRestart(Exception):
    """Internal: force the current worker to be terminated + restarted.

    Used for an unanswerable protocol desync (a malformed pick request leaves
    the worker blocked on stdin); restarting is the gen-1 crash path.
    """


class GenerateDraftDataSupervisor:
    """Spawn/restart the draft worker, label transcripts, append records."""

    STATUS_EVERY = 10  # log progress every N completed drafts

    def __init__(
        self,
        config: GenerateDraftDataConfig,
        *,
        labeler: DeckLabeler | None = None,
        launch_worker: Callable[[], Any] | None = None,
        registry: Any | None = None,
    ) -> None:
        self._config = config
        self._run_id = str(uuid.uuid4())
        self._labeler = labeler
        self._launch_worker = launch_worker
        self._registry = registry  # AgentRegistry | None (None ⇒ gen-1 path)
        self._consecutive_faults = 0
        self._shutdown = threading.Event()

    @property
    def run_id(self) -> str:
        return self._run_id

    def _build_registry(self) -> Any:
        from draft.application.agent_registry import AgentRegistry
        from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

        locator = ConvertedCardLocator(self._config.cards_path)
        mix_labels = {label for label, _ in self._config.agent_mix}
        return AgentRegistry.build(
            self._config.agent_checkpoints, mix_labels,
            locator=locator, pick_mode=self._config.pick_mode,
            temperature=self._config.temperature, seed=self._config.seed,
        )

    def _default_launch_worker(self, external_labels: frozenset[str]) -> Callable[[], Any]:
        from draft.infrastructure.draft_worker_connector import DraftWorkerConnector

        connector = DraftWorkerConnector()
        mix = format_agent_mix(self._config.agent_mix)
        run_id = self._run_id

        def launch() -> Any:
            return connector.start(
                agent_mix=mix, set_code=self._config.set_code,
                external_agents=external_labels, run_id=run_id,
            )

        return launch

    def run(self) -> int:
        """Generate drafts until ``--n-drafts`` complete records exist. Returns the count."""
        out_path = self._config.output_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        count = count_complete_records(out_path) if self._config.resume else 0
        target = self._config.n_drafts
        if count >= target:
            _log(f"Already have {count} >= {target} records in {out_path}; nothing to do.")
            return count

        labeler = self._labeler if self._labeler is not None else build_labeler(self._config)

        # Build the agent registry once (fail fast on bad labels/geometry, FR-011/
        # FR-012) before any worker launches. With no bound checkpoints the
        # registry stays None and the path is byte-for-byte gen-1 (SC-004).
        if self._registry is None and self._config.agent_checkpoints:
            self._registry = self._build_registry()
        external_labels = (
            self._registry.external_labels if self._registry is not None
            else frozenset()
        )
        launch = self._launch_worker or self._default_launch_worker(external_labels)

        self._install_signal_handlers()
        _log(
            f"Run {self._run_id}: generating {target - count} more drafts "
            f"(have {count}) -> {out_path}"
        )
        if self._registry is not None:
            for label in sorted(self._config.agent_checkpoints):
                _log(
                    f"  model agent: {label} -> "
                    f"{self._config.agent_checkpoints[label]} "
                    f"(pick-mode={self._config.pick_mode}"
                    + (f", seed={self._config.seed}" if self._config.seed is not None else "")
                    + ")"
                )

        start_count = count
        start_time = time.monotonic()
        open_mode = "a" if (self._config.resume or out_path.exists()) else "w"
        with open(out_path, open_mode, buffering=1, encoding="utf-8") as out:
            count = self._supervise(
                launch, labeler, out, count, target, start_count, start_time,
            )

        elapsed = time.monotonic() - start_time
        produced = count - start_count
        rate = produced / elapsed * 60 if elapsed > 0 else 0.0
        _log(
            f"Done: {count} records in {out_path} "
            f"({produced} this run in {_format_duration(elapsed)}, {rate:.1f}/min)"
        )
        return count

    RATE_WINDOW_SECONDS = 20  # rolling window for the reported drafts/min

    def _supervise(
        self,
        launch: Callable[[], Any],
        labeler: DeckLabeler,
        out: Any,
        count: int,
        target: int,
        start_count: int,
        start_time: float,
    ) -> int:
        from collections import deque

        from price_predictor.infrastructure.forge_jvm import kill_process_tree

        # (monotonic time, count) samples for the rolling-window rate.
        samples: deque[tuple[float, int]] = deque([(start_time, count)])

        def windowed_rate(now: float, current: int) -> float:
            samples.append((now, current))
            # Trim to the window, but always keep at least the previous sample
            # (evict a left sample only when the next one is still in-window).
            # This way a log interval longer than the window — e.g. slow greedy
            # builds logging every ~30s with a 20s window — still yields the
            # inter-log rate instead of a degenerate zero.
            while (
                len(samples) > 2
                and now - samples[1][0] > self.RATE_WINDOW_SECONDS
            ):
                samples.popleft()
            old_t, old_c = samples[0]
            interval = now - old_t
            return (current - old_c) / interval * 60 if interval > 0 else 0.0

        while count < target and not self._shutdown.is_set():
            proc = launch()
            try:
                for line in proc.stdout:
                    if self._shutdown.is_set():
                        break
                    # Live pick side-channel (only when model seats are bound).
                    if self._registry is not None:
                        if self._route_pick_line(line, proc):
                            continue  # answered a request / logged an abandonment
                    transcript = parse_sentinel_line(line)
                    if transcript is None:
                        continue
                    try:
                        record = assemble_record(transcript, self._run_id, labeler)
                    except Exception as exc:  # malformed transcript — skip, keep going
                        _log(f"Skipping unparseable transcript: {exc}")
                        continue
                    append_record(out, record)
                    count += 1
                    # A completed draft resets the consecutive-fault counter and
                    # discards that draft's trackers (FR-016).
                    self._consecutive_faults = 0
                    if self._registry is not None:
                        self._registry.reset_draft(record.draft_id)
                    produced = count - start_count
                    # First draft confirms the pipeline is alive; then every
                    # STATUS_EVERY drafts report throughput + ETA.
                    if produced == 1 or count % self.STATUS_EVERY == 0:
                        now = time.monotonic()
                        _log(_progress_line(
                            count, target, windowed_rate(now, count),
                            now - start_time,
                        ))
                    if count >= target:
                        break
            except _WorkerRestart:
                pass  # terminate + relaunch below (gen-1 crash path)
            finally:
                self._terminate(proc, kill_process_tree)
            if count < target and not self._shutdown.is_set():
                _log(
                    f"Worker exited (code {self._returncode(proc)}) at "
                    f"{count}/{target}; restarting..."
                )
        return count

    def _route_pick_line(self, line: str, proc: Any) -> bool:
        """Handle a live-pick side-channel line; return True if it was one.

        A ``<<DRAFT-PICK-REQUEST>>`` is answered with the policy's pick (or an
        ``abort`` on a Python-side fault); a ``<<DRAFT-ABANDONED>>`` is logged +
        counted. Both kinds of fault feed the consecutive-fault auto-abort
        (FR-016), which raises :class:`MaxConsecutiveFaultsError`.
        """
        try:
            request = parse_pick_request(line)
        except ValueError as exc:
            # A pick-request sentinel with invalid fields is a protocol desync we
            # cannot answer; force a worker restart (the gen-1 crash path) and
            # count it as a fault.
            _log(f"ERROR malformed pick request — dropping draft, restarting worker: {exc}")
            self._register_fault("")
            raise _WorkerRestart from exc
        if request is not None:
            self._answer_pick(request, proc)
            return True
        notice = parse_abandoned(line)
        if notice is not None:
            _log(
                f"ERROR worker abandoned draft {notice.draft_id}: {notice.reason} "
                "— not recorded"
            )
            if self._registry is not None:
                self._registry.reset_draft(notice.draft_id)
            self._register_fault(notice.draft_id)
            return True
        return False

    def _answer_pick(self, request: "PickRequest", proc: Any) -> None:
        from draft.application.agent_pick_service import PickFault

        try:
            card = self._registry.pick(request)
        except PickFault as exc:
            _log(
                f"ERROR pick fault on draft {request.draft_id} seat {request.seat} "
                f"(pack {request.pack_number} pick {request.pick_number}): {exc} "
                "— draft abandoned, not recorded"
            )
            self._send_response(proc, PickResponse(
                request.draft_id, request.seat, request.pack_number,
                request.pick_number, abort=True,
            ))
            self._registry.reset_draft(request.draft_id)
            self._register_fault(request.draft_id)
            return
        self._send_response(proc, PickResponse(
            request.draft_id, request.seat, request.pack_number,
            request.pick_number, pick=card,
        ))

    @staticmethod
    def _send_response(proc: Any, response: PickResponse) -> None:
        proc.stdin.write(response.serialize() + "\n")
        proc.stdin.flush()

    def _register_fault(self, draft_id: str) -> None:
        self._consecutive_faults += 1
        if self._consecutive_faults >= self._config.max_consecutive_faults:
            raise MaxConsecutiveFaultsError(
                f"{self._consecutive_faults} consecutive pick-fault-abandoned "
                f"drafts (--max-consecutive-faults="
                f"{self._config.max_consecutive_faults}); aborting run"
            )

    @staticmethod
    def _returncode(proc: Any) -> Any:
        try:
            return proc.poll()
        except Exception:
            return "?"

    @staticmethod
    def _terminate(proc: Any, kill_process_tree: Callable[[Any], None]) -> None:
        try:
            if hasattr(proc, "pid"):
                kill_process_tree(proc)
            else:  # test doubles
                proc.wait()
        except Exception:
            pass

    def _install_signal_handlers(self) -> None:
        def handler(_signum, _frame) -> None:
            self._shutdown.set()

        try:
            signal.signal(signal.SIGINT, handler)
            signal.signal(signal.SIGTERM, handler)
        except ValueError:
            # Not on the main thread (e.g. under test); skip signal wiring.
            pass
