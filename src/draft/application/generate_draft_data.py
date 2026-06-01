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
    count: int, target: int, done_this_run: int, elapsed: float,
) -> str:
    """Progress with throughput + ETA over the drafts produced this run."""
    rate = done_this_run / elapsed * 60 if elapsed > 0 else 0.0
    parts = [
        f"  {count}/{target} drafts recorded",
        f"{rate:.1f}/min",
        f"{_format_duration(elapsed)} elapsed",
    ]
    remaining = target - count
    if rate > 0 and remaining > 0:
        parts.append(f"ETA ~{_format_duration(remaining / rate * 60)}")
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

class GenerateDraftDataSupervisor:
    """Spawn/restart the draft worker, label transcripts, append records."""

    STATUS_EVERY = 10  # log progress every N completed drafts

    def __init__(
        self,
        config: GenerateDraftDataConfig,
        *,
        labeler: DeckLabeler | None = None,
        launch_worker: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        self._run_id = str(uuid.uuid4())
        self._labeler = labeler
        self._launch_worker = launch_worker
        self._shutdown = threading.Event()

    @property
    def run_id(self) -> str:
        return self._run_id

    def _default_launch_worker(self) -> Callable[[], Any]:
        from draft.infrastructure.draft_worker_connector import DraftWorkerConnector

        connector = DraftWorkerConnector()
        mix = format_agent_mix(self._config.agent_mix)

        def launch() -> Any:
            return connector.start(agent_mix=mix, set_code=self._config.set_code)

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
        launch = self._launch_worker or self._default_launch_worker()

        self._install_signal_handlers()
        _log(
            f"Run {self._run_id}: generating {target - count} more drafts "
            f"(have {count}) -> {out_path}"
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
        from price_predictor.infrastructure.forge_jvm import kill_process_tree

        while count < target and not self._shutdown.is_set():
            proc = launch()
            try:
                for line in proc.stdout:
                    if self._shutdown.is_set():
                        break
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
                    produced = count - start_count
                    # First draft confirms the pipeline is alive; then every
                    # STATUS_EVERY drafts report throughput + ETA.
                    if produced == 1 or count % self.STATUS_EVERY == 0:
                        _log(_progress_line(
                            count, target, produced, time.monotonic() - start_time,
                        ))
                    if count >= target:
                        break
            finally:
                self._terminate(proc, kill_process_tree)
            if count < target and not self._shutdown.is_set():
                _log(
                    f"Worker exited (code {self._returncode(proc)}) at "
                    f"{count}/{target}; restarting..."
                )
        return count

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
