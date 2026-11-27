"""Where the draft commands write and read under ``output/`` and ``models/``:
each path has a flag, and none has a default (025 LAYOUT-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from draft.infrastructure import cli
from draft.infrastructure.cli import build_parser
from draft.infrastructure.cli_train_draft_agent import _resolve_train_agent_config
from draft.infrastructure.cli_train_draft_agent_rl import _resolve_train_agent_rl_config

_RUN = "models/draft/gen3/agent/2026-11-15-online"
_CACHE = "output/draft/gen3/cache/2026-11-15-encoder"
_LOGS = "output/draft/gen3/records/2026-11-15-x/logs"
#: What the offline trainers need on a fresh run (none of these has a default).
_FRESH = ["--drafts-path", "d.jsonl", "--cards-path", "cards"]


def _args(*argv: str):
    return build_parser().parse_args(list(argv))


def _rl(*extra: str):
    return _args("train-draft-agent-rl", "--checkpoint", "ref.pt", "--learner-agents", "gen-2",
                 "--rollout-temperature", "1.0", *_FRESH, *extra)


# --------------------------------------------------------------------------- #
# The offline trainers' checkpoint directory
# --------------------------------------------------------------------------- #

class TestTheOfflineTrainersCheckpointDirectory:
    def test_train_draft_agent_writes_where_the_flag_says(self):
        config = _resolve_train_agent_config(
            _args("train-draft-agent", *_FRESH, "--checkpoint-dir", _RUN), None)
        assert config.checkpoint_dir == Path(_RUN)

    def test_train_draft_agent_rl_writes_where_the_flag_says(self):
        assert _resolve_train_agent_rl_config(
            _rl("--checkpoint-dir", _RUN), None).checkpoint_dir == Path(_RUN)

    def test_a_resumed_run_keeps_the_directory_it_ran_in(self):
        resumed = {"checkpoint_dir": _RUN}
        args = _args("train-draft-agent", "--resume", f"{_RUN}/latest.pt")
        assert _resolve_train_agent_config(args, resumed).checkpoint_dir == Path(_RUN)
        rl = _args("train-draft-agent-rl", "--resume", f"{_RUN}/latest.pt")
        assert _resolve_train_agent_rl_config(rl, resumed).checkpoint_dir == Path(_RUN)

    @pytest.mark.parametrize("command", ["train-draft-agent", "train-draft-agent-rl"])
    def test_the_manifest_is_kept_in_that_directory(self, command):
        args = _args(command, "--checkpoint-dir", _RUN)
        assert cli.RUN_MANIFESTS[command].run_dir(args) == Path(_RUN)

    @pytest.mark.parametrize("command", ["train-draft-agent", "train-draft-agent-rl"])
    def test_unset_no_manifest_is_kept_and_the_trainer_refuses(self, command):
        assert cli.RUN_MANIFESTS[command].run_dir(_args(command)) is None

    def test_a_resume_s_manifest_follows_the_directory_the_run_writes(self, tmp_path, monkeypatch):
        """The resumed checkpoint's recorded directory, as the trainer resolves it."""
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        class _Loaded:
            train_config = {"checkpoint_dir": _RUN}

        monkeypatch.setattr(DraftAgentStore, "load_checkpoint", lambda _self, _path: _Loaded())
        args = _args("train-draft-agent", "--resume", str(tmp_path / "elsewhere.pt"))
        assert cli.RUN_MANIFESTS["train-draft-agent"].run_dir(args) == Path(_RUN)

    def test_a_resume_of_a_run_that_recorded_none_keeps_no_manifest(self, tmp_path, monkeypatch):
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        class _Loaded:
            train_config = {"lr": 1e-4}

        monkeypatch.setattr(DraftAgentStore, "load_checkpoint", lambda _self, _path: _Loaded())
        args = _args("train-draft-agent-rl", "--resume", str(tmp_path / "old.pt"))
        assert cli.RUN_MANIFESTS["train-draft-agent-rl"].run_dir(args) is None


# --------------------------------------------------------------------------- #
# The online trainer's run directory
# --------------------------------------------------------------------------- #

def _online(*extra: str):
    return _args("train-draft-agent-online", "--learner", "gen-3=a.pt", "--frozen", "gen-1=b.pt",
                 "--scorer-checkpoint", "s.pt", "--cards-path", "cards",
                 "--output-path", "d.jsonl", "--agent-temp", "gen-3=2.0", *extra)


class TestTheOnlineTrainersRunDirectory:
    def _config(self, monkeypatch, *extra):
        import draft.application.online_config_validation as validation
        import draft.application.train_draft_agent_online as online

        seen = []
        monkeypatch.setattr(validation, "validate_config", lambda _config: None)
        monkeypatch.setattr(online.TrainDraftAgentOnlineUseCase, "execute",
                            lambda _self, config: seen.append(config) or 0)
        args = _online(*extra)
        assert args.func(args) == 0
        return seen[0]

    def test_checkpoints_and_worker_logs_go_to_the_flag_s_directory(self, monkeypatch):
        config = self._config(monkeypatch, "--checkpoint-dir", _RUN)
        assert config.checkpoint_dir == Path(_RUN)
        assert config.worker_log_dir == Path(_RUN) / "logs"

    def test_the_manifest_is_kept_in_the_run_directory(self):
        args = _online("--checkpoint-dir", _RUN)
        assert cli.RUN_MANIFESTS["train-draft-agent-online"].run_dir(args) == Path(_RUN)

    def test_the_generation_it_drives_logs_its_workers_there(self):
        from draft.application.train_draft_agent_online import (
            TrainDraftAgentOnlineConfig,
            TrainDraftAgentOnlineUseCase,
        )

        config = TrainDraftAgentOnlineConfig(
            learner_label="gen-3", scorer_checkpoint=Path("s.pt"), cards_path=Path("cards"),
            output_path=Path("d.jsonl"), checkpoint_dir=Path(_RUN),
            worker_log_dir=Path(_RUN) / "logs", embeddings_path=Path(_CACHE),
        )
        generation = TrainDraftAgentOnlineUseCase._generation_config(config, 2.0)
        assert generation.worker_log_dir == Path(_RUN) / "logs"
        assert generation.embeddings_path == Path(_CACHE)

    def test_its_checkpoints_are_written_in_the_config_s_directory(self, tmp_path):
        """``_snapshot`` and ``_write_best`` name files in ``checkpoint_dir``."""
        import draft.application.train_draft_agent_online as online
        from draft.application.online_checkpoints import CheckpointSource, CheckpointStamp

        written = []
        config = online.TrainDraftAgentOnlineConfig(
            scorer_checkpoint=Path("s.pt"), cards_path=Path("cards"),
            output_path=Path("d.jsonl"), checkpoint_dir=tmp_path / "run",
            worker_log_dir=tmp_path / "run" / "logs",
        )
        source = CheckpointSource(
            store=None, model=None, optimizer=None, config=config, base=None, generation=0,
        )
        stamp = CheckpointStamp(rounds=1)
        original = online._write_checkpoint
        try:
            online._write_checkpoint = lambda _source, _stamp, path: written.append(path)
            online.TrainDraftAgentOnlineUseCase._snapshot(source, stamp)
            online.TrainDraftAgentOnlineUseCase._write_best(
                source, stamp, round_index=0, margin=None)
        finally:
            online._write_checkpoint = original
        assert [path.parent for path in written] == [tmp_path / "run"] * 2
        assert written[1].name.startswith("best_")


# --------------------------------------------------------------------------- #
# generate-draft-data's worker logs
# --------------------------------------------------------------------------- #

class TestGenerateDraftDataWorkerLogs:
    def _config(self, monkeypatch, *extra):
        from draft.application import generate_draft_data

        seen = []
        monkeypatch.setattr(generate_draft_data.GenerateDraftDataSupervisor, "run",
                            lambda self: seen.append(self._config) or 0)
        args = _args("generate-draft-data", "--n-drafts", "1", "--scorer-checkpoint", "s.pt",
                     "--picker-checkpoint", "p.pt", "--cards-path", "cards",
                     "--output-path", "d.jsonl", "--logs-dir", _LOGS, *extra)
        assert args.func(args) == 0
        return seen[0]

    def test_they_go_to_the_flag_s_directory(self, monkeypatch):
        assert self._config(monkeypatch).worker_log_dir == Path(_LOGS)

    def test_the_worker_is_started_with_that_directory(self, tmp_path, monkeypatch):
        from draft.application.generate_draft_data import (
            GenerateDraftDataConfig,
            GenerateDraftDataSupervisor,
        )
        from draft.infrastructure.draft_worker_connector import DraftWorkerConnector

        started = []
        monkeypatch.setattr(DraftWorkerConnector, "start",
                            lambda _self, launch: started.append(launch))
        config = GenerateDraftDataConfig(
            n_drafts=1, agent_mix=[("draft-agent", 1)], scorer_checkpoint=Path("s.pt"),
            cards_path=Path("cards"), output_path=tmp_path / "d.jsonl",
            worker_log_dir=tmp_path / "logs")
        GenerateDraftDataSupervisor(config)._default_launch_worker(frozenset({"draft-agent"}))()
        assert started[0].log_dir == tmp_path / "logs"


# --------------------------------------------------------------------------- #
# --embeddings-path on every reader of the .npz cache
# --------------------------------------------------------------------------- #

class TestEmbeddingsPath:
    def test_the_offline_trainers_read_them_where_the_flag_says(self):
        assert _resolve_train_agent_config(
            _args("train-draft-agent", *_FRESH, "--checkpoint-dir", _RUN,
                  "--embeddings-path", _CACHE), None,
        ).embeddings_path == Path(_CACHE)
        assert _resolve_train_agent_rl_config(
            _rl("--checkpoint-dir", _RUN, "--embeddings-path", _CACHE), None,
        ).embeddings_path == Path(_CACHE)

    def test_a_resumed_offline_trainer_keeps_them(self):
        args = _args("train-draft-agent", "--resume", "x.pt")
        assert _resolve_train_agent_config(
            args, {"embeddings_path": _CACHE}).embeddings_path == Path(_CACHE)

    def test_unset_they_are_read_beside_the_cards(self):
        fresh = _args("train-draft-agent", *_FRESH, "--checkpoint-dir", _RUN)
        assert _resolve_train_agent_config(fresh, None).embeddings_path is None
        assert _resolve_train_agent_rl_config(
            _rl("--checkpoint-dir", _RUN), None).embeddings_path is None

    def test_generate_draft_data_reads_them_where_the_flag_says(self, monkeypatch):
        config = TestGenerateDraftDataWorkerLogs()._config(
            monkeypatch, "--embeddings-path", _CACHE)
        assert config.embeddings_path == Path(_CACHE)

    def test_the_online_trainer_reads_them_where_the_flag_says(self, monkeypatch):
        config = TestTheOnlineTrainersRunDirectory()._config(
            monkeypatch, "--checkpoint-dir", _RUN, "--embeddings-path", _CACHE)
        assert config.embeddings_path == Path(_CACHE)

    def test_validate_builder_reads_them_where_the_flag_says(self, monkeypatch):
        from draft.application import validate_builder

        seen = []
        monkeypatch.setattr(validate_builder, "run_validate", lambda config: seen.append(config))
        monkeypatch.setattr(validate_builder, "format_diagnostic", lambda _diagnostic: "")
        args = _args("validate-builder", "--pools-from", "d.jsonl", "--scorer-checkpoint", "s.pt",
                     "--picker-checkpoint", "p.pt", "--cards-path", "cards",
                     "--embeddings-path", _CACHE)
        assert args.func(args) == 0
        assert seen[0].embeddings_path == Path(_CACHE)

    def test_the_online_width_probe_reads_the_embeddings_tree(self, tmp_path):
        import numpy as np

        from draft.application.online_config_validation import probe_embedding_width

        (tmp_path / "cache" / "a").mkdir(parents=True)
        np.savez(tmp_path / "cache" / "a" / "c.npz", embedding=np.zeros(6, dtype=np.float32))
        assert probe_embedding_width(tmp_path / "cache") == 6

    @pytest.mark.parametrize("argv", [
        ["generate-draft-data", "--n-drafts", "1", "--scorer-checkpoint", "s.pt",
         "--picker-checkpoint", "p.pt", "--cards-path", "cards", "--output-path", "o/d.jsonl",
         "--logs-dir", _LOGS],
        ["train-draft-agent", *_FRESH, "--checkpoint-dir", _RUN],
        ["train-draft-agent-online", "--learner", "gen-3=a.pt", "--scorer-checkpoint", "s.pt",
         "--cards-path", "cards", "--output-path", "o/d.jsonl", "--checkpoint-dir", _RUN],
    ])
    def test_every_writer_lists_the_embeddings_it_read(self, argv):
        args = _args(*argv, "--embeddings-path", _CACHE)
        assert Path(_CACHE) in cli.RUN_MANIFESTS[argv[0]].input_paths(args)
