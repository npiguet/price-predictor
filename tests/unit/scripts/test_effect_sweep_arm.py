"""One sweep arm keeps its logs, cache and probe reports in its run directory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "effect_sweep_arm.py"


def _load():
    spec = importlib.util.spec_from_file_location("effect_sweep_arm", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm = _load()


def _plan(argv):
    args, train_args = arm.parse_args(argv)
    return arm.plan(args, train_args)


class TestPlan:
    def test_train_flags_after_the_separator_reach_the_trainer(self, tmp_path):
        steps = _plan([
            "--run-dir", str(tmp_path / "a"), "--corpus", "c",
            "--", "--e-dim", "128", "--withhold-keyword", "ward",
        ])
        step, command, log = steps[0]
        assert (step, log) == ("train", "train.log")
        assert command[-4:] == ["--e-dim", "128", "--withhold-keyword", "ward"]
        assert command[command.index("--model-output") + 1] == str(tmp_path / "a")

    def test_every_output_lands_in_the_run_dir(self, tmp_path):
        run_dir = tmp_path / "a"
        steps = _plan(["--run-dir", str(run_dir), "--corpus", "c", "--", "--e-dim", "64"])
        for step, command, _ in steps[1:]:
            assert command[command.index("--checkpoint") + 1] == str(run_dir / "latest.pt")
            assert command[command.index(
                "--output-root" if step == "encode" else "--abilities-root") + 1] == str(
                run_dir / "abilities")
            if step.endswith("probes"):
                assert command[command.index("--reports-dir") + 1] == str(run_dir / "reports")

    def test_training_encoding_and_evaluation_read_the_variant_tree(self, tmp_path):
        steps = {s: c for s, c, _ in _plan(["--run-dir", str(tmp_path), "--corpus", "c"])}
        for step in ("train", "encode", "evaluate"):
            command = steps[step]
            assert command[command.index("--variant-scripts") + 1] == str(
                Path("output/effects/variant-scripts"))
        train = steps["train"]
        assert train[train.index("--vocab-path") + 1] == str(
            Path("models/effects/vocab-script.txt"))

    def test_each_embedding_probe_gets_its_own_log(self, tmp_path):
        logs = [log for step, _, log in _plan(
            ["--run-dir", str(tmp_path), "--corpus", "c"]) if step == "embedding-probes"]
        assert logs == [f"probes-{s}.log" for s in arm.EMBEDDING_PROBES]

    def test_only_the_first_arm_freezes_the_probe_set(self, tmp_path):
        base = ["--run-dir", str(tmp_path), "--corpus", "c"]
        plain = [c for s, c, _ in _plan(base) if s == "knowledge-probes"]
        first = [c for s, c, _ in _plan(base + ["--freeze-probe-set"])
                 if s == "knowledge-probes"]
        assert len(plain) == 1 and "--freeze-probe-set" not in plain[0]
        assert len(first) == 2 and first[0][-1] == "--freeze-probe-set"
        assert first[1] == plain[0]
        assert plain[0][plain[0].index("--probe-set-dir") + 1] == str(arm.SHARED_REPORTS)


class TestLogging:
    def test_the_command_line_is_the_log_s_first_line(self, tmp_path):
        log = tmp_path / "step.log"
        command = [sys.executable, "-c", "print('hello')"]
        assert arm.run_logged(command, log) == 0
        lines = log.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("$ ") and "print" in lines[0]
        assert lines[1].startswith("# commit ")
        assert "hello" in lines
        assert lines[-1].startswith("# exit 0")

    def test_a_rerun_appends_under_its_own_command_line(self, tmp_path):
        log = tmp_path / "step.log"
        arm.run_logged([sys.executable, "-c", "print('one')"], log)
        arm.run_logged([sys.executable, "-c", "raise SystemExit(3)"], log)
        lines = log.read_text(encoding="utf-8").splitlines()
        assert sum(line.startswith("$ ") for line in lines) == 2
        assert lines[-1].startswith("# exit 3")


class TestMain:
    def test_a_trained_arm_is_never_trained_over(self, tmp_path, capsys):
        (tmp_path / "latest.pt").write_bytes(b"")
        code = arm.main(["--run-dir", str(tmp_path), "--corpus", "c", "--", "--e-dim", "64"])
        assert code == 2
        assert "--from-step" in capsys.readouterr().err

    def test_training_needs_its_flags(self, tmp_path):
        assert arm.main(["--run-dir", str(tmp_path / "a"), "--corpus", "c"]) == 2

    def test_a_failed_step_stops_the_arm_and_names_the_resume_step(
        self, tmp_path, monkeypatch,
    ):
        ran = []

        def fake(command, log):
            ran.append(log.name)
            return 1 if log.name == "evaluate.log" else 0

        monkeypatch.setattr(arm, "run_logged", fake)
        code = arm.main(["--run-dir", str(tmp_path), "--corpus", "c", "--from-step", "encode"])
        assert code == 1
        assert ran == ["encode.log", "evaluate.log"]
        assert "continue with --from-step evaluate" in (tmp_path / "arm.log").read_text(
            encoding="utf-8")


class TestResume:
    """An interrupted training step continues from its resume file."""

    def test_resume_reaches_the_trainer(self, tmp_path):
        steps = _plan([
            "--run-dir", str(tmp_path), "--corpus", "c", "--resume",
            "--", "--e-dim", "64",
        ])
        assert steps[0][1][-1] == "--resume"

    def test_a_trained_arm_with_a_resume_file_is_continued(self, tmp_path, monkeypatch):
        (tmp_path / "latest.pt").write_bytes(b"")
        (tmp_path / "resume.pt").write_bytes(b"")
        ran = []
        monkeypatch.setattr(arm, "run_logged", lambda command, log: ran.append(command) or 0)
        code = arm.main([
            "--run-dir", str(tmp_path), "--corpus", "c", "--resume", "--", "--e-dim", "64",
        ])
        assert code == 0
        assert ran[0][-1] == "--resume"

    def test_resume_without_a_resume_file_is_refused(self, tmp_path, capsys):
        code = arm.main([
            "--run-dir", str(tmp_path), "--corpus", "c", "--resume", "--", "--e-dim", "64",
        ])
        assert code == 2
        assert "resume.pt" in capsys.readouterr().err

    def test_resume_continues_training_only(self, tmp_path):
        (tmp_path / "resume.pt").write_bytes(b"")
        assert arm.main([
            "--run-dir", str(tmp_path), "--corpus", "c", "--resume", "--from-step", "encode",
        ]) == 2

    def test_a_failed_training_step_names_resume(self, tmp_path, monkeypatch):
        monkeypatch.setattr(arm, "run_logged", lambda command, log: 1)
        arm.main(["--run-dir", str(tmp_path), "--corpus", "c", "--", "--e-dim", "64"])
        assert "continue with --resume" in (tmp_path / "arm.log").read_text(encoding="utf-8")

    def test_the_arm_log_records_the_arguments_main_was_given(self, tmp_path, monkeypatch):
        """A caller passing argv (a test, a driver script) is what ran, not sys.argv."""
        monkeypatch.setattr(arm, "run_logged", lambda command, log: 0)
        monkeypatch.setattr(sys, "argv", ["pytest", "--something-else"])
        arm.main(["--run-dir", str(tmp_path), "--corpus", "c", "--", "--e-dim", "64"])
        first = (tmp_path / "arm.log").read_text(encoding="utf-8").splitlines()[0]
        assert "--e-dim 64" in first
        assert "--something-else" not in first
