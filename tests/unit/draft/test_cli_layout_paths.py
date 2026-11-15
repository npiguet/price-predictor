"""Where the draft commands write and read under ``output/`` and ``models/``:
each path has a flag, and every default is today's path (025 LAYOUT-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from draft.infrastructure import cli
from draft.infrastructure.cli import (
    _resolve_train_agent_config,
    _resolve_train_agent_rl_config,
    build_parser,
)

_RUN = "models/draft/gen3/agent/2026-11-15-online"
_CACHE = "output/draft/gen3/cache/2026-11-15-encoder"
_TODAY = Path("models/draft/agent")


def _args(*argv: str):
    return build_parser().parse_args(list(argv))


def _rl(*extra: str):
    return _args("train-draft-agent-rl", "--checkpoint", "ref.pt", "--learner-agents", "gen-2",
                 "--rollout-temperature", "1.0", *extra)


# --------------------------------------------------------------------------- #
# The offline trainers' checkpoint directory
# --------------------------------------------------------------------------- #

class TestTheOfflineTrainersCheckpointDirectory:
    def test_train_draft_agent_writes_where_the_flag_says(self):
        config = _resolve_train_agent_config(
            _args("train-draft-agent", "--checkpoint-dir", _RUN), None)
        assert config.checkpoint_dir == Path(_RUN)

    def test_train_draft_agent_rl_writes_where_the_flag_says(self):
        assert _resolve_train_agent_rl_config(
            _rl("--checkpoint-dir", _RUN), None).checkpoint_dir == Path(_RUN)

    def test_unset_both_write_where_they_always_did(self):
        fresh = _resolve_train_agent_config(_args("train-draft-agent"), None)
        assert fresh.checkpoint_dir == _TODAY
        assert _resolve_train_agent_rl_config(_rl(), None).checkpoint_dir == _TODAY

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
    def test_unset_the_manifest_is_kept_where_it_always_was(self, command):
        assert cli.RUN_MANIFESTS[command].run_dir(_args(command)) == _TODAY

    def test_a_resume_s_manifest_follows_the_directory_the_run_writes(self, tmp_path, monkeypatch):
        """The resumed checkpoint's recorded directory, as the trainer resolves it."""
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        class _Loaded:
            train_config = {"checkpoint_dir": _RUN}

        monkeypatch.setattr(DraftAgentStore, "load_checkpoint", lambda _self, _path: _Loaded())
        args = _args("train-draft-agent", "--resume", str(tmp_path / "elsewhere.pt"))
        assert cli.RUN_MANIFESTS["train-draft-agent"].run_dir(args) == Path(_RUN)

    def test_a_resume_of_a_run_that_recorded_none_keeps_today_s(self, tmp_path, monkeypatch):
        from draft.infrastructure.draft_agent_store import DraftAgentStore

        class _Loaded:
            train_config = {"lr": 1e-4}

        monkeypatch.setattr(DraftAgentStore, "load_checkpoint", lambda _self, _path: _Loaded())
        args = _args("train-draft-agent-rl", "--resume", str(tmp_path / "old.pt"))
        assert cli.RUN_MANIFESTS["train-draft-agent-rl"].run_dir(args) == _TODAY


# --------------------------------------------------------------------------- #
# The online trainer's run directory
# --------------------------------------------------------------------------- #

def _online(*extra: str):
    return _args("train-draft-agent-online", "--learner", "gen-3=a.pt", "--frozen", "gen-1=b.pt",
                 "--agent-temp", "gen-3=2.0", *extra)


class TestTheOnlineTrainersRunDirectory:
    def _config(self, monkeypatch, *extra):
        import draft.application.train_draft_agent_online as online

        seen = []
        monkeypatch.setattr(online, "validate_config", lambda _config: None)
        monkeypatch.setattr(online.TrainDraftAgentOnlineUseCase, "execute",
                            lambda _self, config: seen.append(config) or 0)
        args = _online(*extra)
        assert args.func(args) == 0
        return seen[0]

    def test_checkpoints_and_worker_logs_go_to_the_flag_s_directory(self, monkeypatch):
        config = self._config(monkeypatch, "--checkpoint-dir", _RUN)
        assert config.checkpoint_dir == Path(_RUN)
        assert config.worker_log_dir == Path(_RUN) / "logs"

    def test_unset_they_go_where_they_always_did(self, monkeypatch):
        config = self._config(monkeypatch)
        assert config.checkpoint_dir == _TODAY
        assert config.worker_log_dir == Path("output/draft")

    def test_the_manifest_is_kept_in_the_run_directory(self):
        args = _online("--checkpoint-dir", _RUN, "--output-path", "output/draft/x/drafts.jsonl")
        assert cli.RUN_MANIFESTS["train-draft-agent-online"].run_dir(args) == Path(_RUN)

    def test_unset_the_manifest_is_kept_beside_the_drafts(self):
        args = _online("--output-path", "output/draft/x/drafts.jsonl")
        assert cli.RUN_MANIFESTS["train-draft-agent-online"].run_dir(args) == Path("output/draft/x")

    def test_the_generation_it_drives_logs_its_workers_there(self):
        from draft.application.train_draft_agent_online import (
            TrainDraftAgentOnlineConfig,
            TrainDraftAgentOnlineUseCase,
        )

        config = TrainDraftAgentOnlineConfig(
            learner_label="gen-3", worker_log_dir=Path(_RUN) / "logs",
            embeddings_path=Path(_CACHE),
        )
        generation = TrainDraftAgentOnlineUseCase._generation_config(config, 2.0)
        assert generation.worker_log_dir == Path(_RUN) / "logs"
        assert generation.embeddings_path == Path(_CACHE)

    def test_its_checkpoints_are_written_in_the_config_s_directory(self, tmp_path):
        """``_snapshot`` and ``_write_best`` name files in ``checkpoint_dir``."""
        import draft.application.train_draft_agent_online as online

        written = []
        config = online.TrainDraftAgentOnlineConfig(checkpoint_dir=tmp_path / "run")
        original = online._write_checkpoint
        try:
            online._write_checkpoint = lambda *args: written.append(args[7])
            online.TrainDraftAgentOnlineUseCase._snapshot(None, None, None, config, None, 0, 1)
            online.TrainDraftAgentOnlineUseCase._write_best(
                None, None, None, config, None, 0, 1, 0, round_index=1, margin=None)
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
        args = _args("generate-draft-data", "--n-drafts", "1", *extra)
        assert args.func(args) == 0
        return seen[0]

    def test_they_go_to_the_flag_s_directory(self, monkeypatch):
        logs = "output/draft/gen3/records/2026-11-15-x/logs"
        assert self._config(monkeypatch, "--logs-dir", logs).worker_log_dir == Path(logs)

    def test_unset_they_go_where_they_always_did(self, monkeypatch):
        assert self._config(monkeypatch).worker_log_dir == Path("output/draft")

    def test_the_worker_is_started_with_that_directory(self, tmp_path, monkeypatch):
        from draft.application.generate_draft_data import (
            GenerateDraftDataConfig,
            GenerateDraftDataSupervisor,
        )
        from draft.infrastructure.draft_worker_connector import DraftWorkerConnector

        started = []
        monkeypatch.setattr(DraftWorkerConnector, "start",
                            lambda _self, **kwargs: started.append(kwargs))
        config = GenerateDraftDataConfig(
            n_drafts=1, agent_mix=[("draft-agent", 1)], worker_log_dir=tmp_path / "logs")
        GenerateDraftDataSupervisor(config)._default_launch_worker(frozenset({"draft-agent"}))()
        assert started[0]["log_dir"] == tmp_path / "logs"


# --------------------------------------------------------------------------- #
# --embeddings-path on every reader of the .npz cache
# --------------------------------------------------------------------------- #

class TestEmbeddingsPath:
    def test_the_offline_trainers_read_them_where_the_flag_says(self):
        assert _resolve_train_agent_config(
            _args("train-draft-agent", "--embeddings-path", _CACHE), None,
        ).embeddings_path == Path(_CACHE)
        assert _resolve_train_agent_rl_config(
            _rl("--embeddings-path", _CACHE), None).embeddings_path == Path(_CACHE)

    def test_a_resumed_offline_trainer_keeps_them(self):
        args = _args("train-draft-agent", "--resume", "x.pt")
        assert _resolve_train_agent_config(
            args, {"embeddings_path": _CACHE}).embeddings_path == Path(_CACHE)

    def test_unset_they_are_read_beside_the_cards(self):
        assert _resolve_train_agent_config(_args("train-draft-agent"), None).embeddings_path is None
        assert _resolve_train_agent_rl_config(_rl(), None).embeddings_path is None

    def test_generate_draft_data_reads_them_where_the_flag_says(self, monkeypatch):
        config = TestGenerateDraftDataWorkerLogs()._config(
            monkeypatch, "--embeddings-path", _CACHE)
        assert config.embeddings_path == Path(_CACHE)

    def test_the_online_trainer_reads_them_where_the_flag_says(self, monkeypatch):
        config = TestTheOnlineTrainersRunDirectory()._config(
            monkeypatch, "--embeddings-path", _CACHE)
        assert config.embeddings_path == Path(_CACHE)

    def test_validate_builder_reads_them_where_the_flag_says(self, monkeypatch):
        from draft.application import validate_builder

        seen = []
        monkeypatch.setattr(validate_builder, "run_validate", lambda config: seen.append(config))
        monkeypatch.setattr(validate_builder, "format_diagnostic", lambda _diagnostic: "")
        args = _args("validate-builder", "--pools-from", "d.jsonl", "--embeddings-path", _CACHE)
        assert args.func(args) == 0
        assert seen[0].embeddings_path == Path(_CACHE)

    def test_the_online_width_probe_reads_the_embeddings_tree(self, tmp_path):
        import numpy as np

        from draft.application.train_draft_agent_online import probe_embedding_width

        (tmp_path / "cache" / "a").mkdir(parents=True)
        np.savez(tmp_path / "cache" / "a" / "c.npz", embedding=np.zeros(6, dtype=np.float32))
        assert probe_embedding_width(tmp_path / "cache") == 6

    @pytest.mark.parametrize("argv", [
        ["generate-draft-data", "--n-drafts", "1"],
        ["train-draft-agent"],
        ["train-draft-agent-online", "--learner", "gen-3=a.pt", "--output-path", "o/d.jsonl"],
    ])
    def test_every_writer_lists_the_embeddings_it_read(self, argv):
        args = _args(*argv, "--embeddings-path", _CACHE)
        assert Path(_CACHE) in cli.RUN_MANIFESTS[argv[0]].input_paths(args)
