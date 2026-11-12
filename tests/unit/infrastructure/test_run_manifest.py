"""The run manifest every file-writing command keeps in its run directory."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from price_predictor.infrastructure import run_manifest
from price_predictor.infrastructure.run_manifest import (
    MANIFEST_NAME,
    RunManifest,
    command_line,
    read_commits,
    run_recorded,
    run_script,
)

COMMITS = {"repo": "abc1234", "forge": None, "dirty": False}
ZONE = timezone(timedelta(hours=1))


class Clock:
    """A clock that ticks a minute per reading, so start and end differ."""

    def __init__(self) -> None:
        self.now = datetime(2026, 11, 13, 9, 12, tzinfo=ZONE)

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture(autouse=True)
def fixed_commits(monkeypatch):
    monkeypatch.setattr(run_manifest, "read_commits", lambda *_args, **_kw: dict(COMMITS))


def manifest(run_dir: Path, repo: Path) -> RunManifest:
    return RunManifest(run_dir, repo_root=repo, cwd=repo, clock=Clock())


def read(run_dir: Path) -> dict:
    return json.loads((run_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


class TestStart:
    def test_creates_the_manifest_with_every_field(self, repo: Path):
        run_dir = repo / "output" / "effects" / "gen3" / "records" / "2026-11-13-full"
        (repo / "output" / "cardsfolder").mkdir(parents=True)

        manifest(run_dir, repo).start(
            command="python -m sealed match-outcomes --workers 6",
            inputs=[repo / "output" / "cardsfolder", None, "resources/AllPrintings.json"],
        )

        document = read(run_dir)
        assert list(document) == [
            "run", "purpose", "status", "superseded_by", "commands", "commits",
            "inputs", "continues", "started", "ended", "outcome",
        ]
        assert document["run"] == "output/effects/gen3/records/2026-11-13-full"
        assert document["purpose"] is None
        assert document["status"] == "running"
        assert document["superseded_by"] is None
        assert document["commands"] == ["python -m sealed match-outcomes --workers 6"]
        assert document["commits"] == COMMITS
        assert document["inputs"] == ["output/cardsfolder", "resources/AllPrintings.json"]
        assert document["continues"] == []
        assert document["started"] == "2026-11-13T09:13:00+01:00"
        assert document["ended"] is None
        assert document["outcome"] is None

    def test_a_path_outside_the_repository_stays_absolute(self, repo: Path, tmp_path: Path):
        elsewhere = tmp_path / "elsewhere"
        manifest(elsewhere, repo).start(command="c", inputs=[tmp_path / "in.txt"])
        document = read(elsewhere)
        assert document["run"] == elsewhere.resolve().as_posix()
        assert document["inputs"] == [(tmp_path / "in.txt").resolve().as_posix()]

    def test_an_input_given_twice_is_listed_once(self, repo: Path):
        manifest(repo / "run", repo).start(command="c", inputs=["a.txt", "a.txt", "./a.txt"])
        assert read(repo / "run")["inputs"] == ["a.txt"]

    def test_without_create_a_missing_directory_gets_no_manifest(self, repo: Path):
        manifest(repo / "absent", repo).start(command="c", inputs=[], create=False)
        assert not (repo / "absent").exists()

    def test_never_written_at_the_repository_root(self, repo: Path):
        manifest(repo, repo).start(command="c", inputs=[])
        assert not (repo / MANIFEST_NAME).exists()


class TestFinish:
    def test_finished(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=0)
        document = read(repo / "run")
        assert document["status"] == "finished"
        assert document["ended"] == "2026-11-13T09:14:00+01:00"
        assert document["outcome"] == "exit 0"

    def test_a_nonzero_exit_fails(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=2)
        document = read(repo / "run")
        assert (document["status"], document["outcome"]) == ("failed", "exit 2")

    def test_an_exception_fails_with_its_one_line_summary(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=1, error=ValueError("bad shard\nsecond line"))
        document = read(repo / "run")
        assert document["status"] == "failed"
        assert document["outcome"] == "exit 1: ValueError: bad shard"

    def test_an_interrupt_fails_as_interrupted(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=130, error=KeyboardInterrupt())
        document = read(repo / "run")
        assert (document["status"], document["outcome"]) == ("failed", "interrupted")

    def test_exit_130_reads_as_interrupted(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=130)
        assert read(repo / "run")["outcome"] == "exit 130: interrupted"

    def test_keeps_what_the_operator_edited_during_the_run(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        document = read(repo / "run")
        document["purpose"] = "Full-strength collection."
        (repo / "run" / MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")

        run.finish(exit_code=0)

        assert read(repo / "run")["purpose"] == "Full-strength collection."

    def test_an_operator_status_set_during_the_run_stays(self, repo: Path):
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        document = read(repo / "run")
        document["status"] = "scratch"
        (repo / "run" / MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")

        run.finish(exit_code=0)

        document = read(repo / "run")
        assert document["status"] == "scratch"
        assert document["outcome"] == "exit 0"

    def test_a_nested_command_s_finish_is_overwritten_by_the_outer_one(self, repo: Path):
        outer = manifest(repo / "run", repo)
        outer.start(command="outer", inputs=[])
        inner = manifest(repo / "run", repo)
        inner.start(command="inner", inputs=[])
        inner.finish(exit_code=0)

        outer.finish(exit_code=3)

        document = read(repo / "run")
        assert document["commands"] == ["outer", "inner"]
        assert (document["status"], document["outcome"]) == ("failed", "exit 3")


class TestResume:
    def test_appends_the_command_and_runs_again(self, repo: Path):
        first = manifest(repo / "run", repo)
        first.start(command="python -m effects train-effect-model", inputs=["a.txt"])
        first.finish(exit_code=130, error=KeyboardInterrupt())
        document = read(repo / "run")
        document["purpose"] = "Gen-3 arm."
        document["status"] = "keep"
        document["continues"] = ["models/effects/gen3/effect-model/2026-11-01-base"]
        (repo / "run" / MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")

        manifest(repo / "run", repo).start(
            command="python -m effects train-effect-model --resume", inputs=["a.txt", "b.txt"],
        )

        resumed = read(repo / "run")
        assert resumed["commands"] == [
            "python -m effects train-effect-model",
            "python -m effects train-effect-model --resume",
        ]
        assert resumed["status"] == "running"
        assert resumed["purpose"] == "Gen-3 arm."
        assert resumed["continues"] == ["models/effects/gen3/effect-model/2026-11-01-base"]
        assert resumed["started"] == document["started"]
        assert resumed["inputs"] == ["a.txt", "b.txt"]

    def test_an_unreadable_manifest_is_left_alone(self, repo: Path, caplog):
        (repo / "run").mkdir()
        (repo / "run" / MANIFEST_NAME).write_text("{not json", encoding="utf-8")

        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=0)

        assert (repo / "run" / MANIFEST_NAME).read_text(encoding="utf-8") == "{not json"
        assert "run.json" in caplog.text


class TestNeverRaises:
    def test_a_failing_write_warns_and_the_command_goes_on(
        self, repo: Path, monkeypatch, caplog,
    ):
        def refuse(*_args, **_kwargs):
            raise PermissionError("locked")

        monkeypatch.setattr(run_manifest.os, "replace", refuse)
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=0)

        assert not (repo / "run" / MANIFEST_NAME).exists()
        assert list((repo / "run").iterdir()) == []  # no temp file left behind
        assert "locked" in caplog.text

    def test_an_unwritable_run_directory_warns(self, repo: Path, caplog):
        (repo / "taken").write_text("a file, not a directory", encoding="utf-8")
        run = manifest(repo / "taken", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=0)
        assert "run.json" in caplog.text

    def test_a_failing_commit_read_still_writes(self, repo: Path, monkeypatch):
        def broken(*_args, **_kwargs):
            raise OSError("no git")

        monkeypatch.setattr(run_manifest, "read_commits", broken)
        manifest(repo / "run", repo).start(command="c", inputs=[])
        assert read(repo / "run")["commits"] == {"repo": None, "forge": None, "dirty": False}


class TestAtomicWrite:
    def test_the_manifest_is_replaced_never_written_in_place(self, repo: Path, monkeypatch):
        replaced: list[tuple[str, str]] = []
        real_replace = run_manifest.os.replace

        def spy(source, target):
            replaced.append((Path(source).name, Path(target).name))
            real_replace(source, target)

        monkeypatch.setattr(run_manifest.os, "replace", spy)
        run = manifest(repo / "run", repo)
        run.start(command="c", inputs=[])
        run.finish(exit_code=0)

        assert len(replaced) == 2
        assert all(target == MANIFEST_NAME and source != MANIFEST_NAME
                   for source, target in replaced)
        assert sorted(path.name for path in (repo / "run").iterdir()) == [MANIFEST_NAME]


class TestRunRecorded:
    def test_returns_the_body_s_exit_code_and_finishes(self, repo: Path):
        code = run_recorded(
            repo / "run", lambda: 0, command="c", inputs=[], repo_root=repo, clock=Clock(),
        )
        assert code == 0
        assert read(repo / "run")["status"] == "finished"

    def test_an_exception_is_recorded_and_raised(self, repo: Path):
        def body():
            raise RuntimeError("worker pool gave up")

        with pytest.raises(RuntimeError):
            run_recorded(repo / "run", body, command="c", inputs=[], repo_root=repo)
        document = read(repo / "run")
        assert document["status"] == "failed"
        assert document["outcome"] == "exit 1: RuntimeError: worker pool gave up"

    def test_ctrl_c_is_recorded_and_raised(self, repo: Path):
        def body():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            run_recorded(repo / "run", body, command="c", inputs=[], repo_root=repo)
        document = read(repo / "run")
        assert (document["status"], document["outcome"]) == ("failed", "interrupted")

    @pytest.mark.parametrize(("code", "status", "outcome"), [
        (None, "finished", "exit 0"),
        (0, "finished", "exit 0"),
        (2, "failed", "exit 2"),
        ("refused: no corpus", "failed", "exit 1: refused: no corpus"),
    ])
    def test_sys_exit_is_recorded_and_raised(self, repo: Path, code, status, outcome):
        def body():
            raise SystemExit(code)

        with pytest.raises(SystemExit):
            run_recorded(repo / "run", body, command="c", inputs=[], repo_root=repo)
        document = read(repo / "run")
        assert (document["status"], document["outcome"]) == (status, outcome)


class TestRunScript:
    def test_records_the_script_s_command(self, repo: Path, monkeypatch):
        monkeypatch.setattr(run_manifest, "repository_root", lambda: repo)
        code = run_script(repo / "run", lambda: 0, script=repo / "scripts" / "probe.py",
                          argv=["--output", "x"])
        assert code == 0
        assert read(repo / "run")["commands"] == ["python scripts/probe.py --output x"]

    def test_no_run_directory_runs_the_body_bare(self, repo: Path):
        assert run_script(None, lambda: 3, script="s.py", argv=[]) == 3


class TestCommandLine:
    def test_a_module_command(self):
        assert command_line(["build-corpus", "--output", "a b"], module="effects") == (
            'python -m effects build-corpus --output "a b"'
        )

    def test_a_script_command(self, tmp_path: Path):
        script = tmp_path / "scripts" / "rule_breadth.py"
        assert command_line(["--output", "x.json"], script=script, repo_root=tmp_path) == (
            "python scripts/rule_breadth.py --output x.json"
        )


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(repo), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def make_repo(path: Path) -> str:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    (path / "tracked.txt").write_text("one\n", encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "commit", "-q", "-m", "first")
    return git(path, "rev-parse", "--short", "HEAD")


class TestReadCommits:
    """The real ``read_commits``, over throwaway repositories."""

    @pytest.fixture(autouse=True)
    def fixed_commits(self):
        """Overrides the module's stub: these tests read real repositories."""

    def test_both_repositories(self, tmp_path: Path):
        repo_sha = make_repo(tmp_path / "price-predictor")
        forge_sha = make_repo(tmp_path / "forge")
        assert read_commits(tmp_path / "price-predictor") == {
            "repo": repo_sha, "forge": forge_sha, "dirty": False,
        }

    def test_without_forge(self, tmp_path: Path):
        repo_sha = make_repo(tmp_path / "price-predictor")
        assert read_commits(tmp_path / "price-predictor") == {
            "repo": repo_sha, "forge": None, "dirty": False,
        }

    def test_an_untracked_file_is_not_dirty(self, tmp_path: Path):
        make_repo(tmp_path / "price-predictor")
        (tmp_path / "price-predictor" / "new.txt").write_text("x", encoding="utf-8")
        assert read_commits(tmp_path / "price-predictor")["dirty"] is False

    def test_a_changed_tracked_file_is_dirty(self, tmp_path: Path):
        make_repo(tmp_path / "price-predictor")
        (tmp_path / "price-predictor" / "tracked.txt").write_text("two\n", encoding="utf-8")
        assert read_commits(tmp_path / "price-predictor")["dirty"] is True

    def test_a_dirty_forge_makes_the_run_dirty(self, tmp_path: Path):
        make_repo(tmp_path / "price-predictor")
        make_repo(tmp_path / "forge")
        (tmp_path / "forge" / "tracked.txt").write_text("two\n", encoding="utf-8")
        assert read_commits(tmp_path / "price-predictor")["dirty"] is True

    def test_not_a_repository(self, tmp_path: Path):
        (tmp_path / "plain").mkdir()
        assert read_commits(tmp_path / "plain") == {"repo": None, "forge": None, "dirty": False}
