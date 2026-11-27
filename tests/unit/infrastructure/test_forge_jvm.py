"""Unit tests for the shared Forge JVM helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

from price_predictor.infrastructure import (
    custom_card_staging,
    forge_classpath,
    job_objects,
    jvm_processes,
)
from price_predictor.infrastructure.custom_card_staging import claim_staging as real_claim_staging
from price_predictor.infrastructure.custom_card_staging import (
    clear_stale_staging as real_clear_stale_staging,
)
from price_predictor.infrastructure.forge_classpath import (
    build_forge_classpath,
    resolve_connector_jar,
)
from price_predictor.infrastructure.jvm_processes import (
    UNSAFE_MEMORY_ACCESS_OPTION,
    JvmOptions,
    build_jvm_command,
    parse_java_feature_version,
)

# Bound at import, before the unit-test conftest swaps the module attribute
# for a stub, so the probe itself can still be exercised here.
from price_predictor.infrastructure.jvm_processes import (
    java_feature_version as real_java_feature_version,
)
from price_predictor.infrastructure.jvm_processes import (
    kill_process_tree as real_kill_process_tree,
)
from price_predictor.infrastructure.worker_log_files import WorkerLogFiles
from tests.unit.infrastructure.real_processes import (
    SLEEPER,
    OwnedProcess,
    end_tree,
    start_with_grandchild,
    wait_for_file,
)


class TestResolveConnectorJar:
    def test_raises_when_jar_missing(self, tmp_path):
        with patch.object(forge_classpath, "project_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="mvn install"):
                resolve_connector_jar()

    def test_returns_path_when_jar_exists(self, tmp_path):
        jar_dir = tmp_path / "forge-connector" / "target"
        jar_dir.mkdir(parents=True)
        jar = jar_dir / forge_classpath.CONNECTOR_JAR_NAME
        jar.write_bytes(b"")
        with patch.object(forge_classpath, "project_root", return_value=tmp_path):
            assert resolve_connector_jar() == jar


class TestBuildForgeClasspath:
    @pytest.fixture
    def fake_jar(self, tmp_path):
        jar_dir = tmp_path / "forge-connector" / "target"
        jar_dir.mkdir(parents=True)
        (jar_dir / forge_classpath.CONNECTOR_JAR_NAME).write_bytes(b"")
        with patch.object(forge_classpath, "project_root", return_value=tmp_path):
            yield tmp_path

    def test_includes_connector_and_core_modules(self, fake_jar):
        cp = build_forge_classpath()
        parts = cp.split(os.pathsep)
        assert any(forge_classpath.CONNECTOR_JAR_NAME in p for p in parts)
        assert any("forge-game" in p for p in parts)
        assert any("forge-core" in p for p in parts)

    def test_full_runtime_includes_gui_and_ai(self, fake_jar):
        cp = build_forge_classpath(include_full_runtime=True)
        assert "forge-gui" in cp
        assert "forge-ai" in cp

    def test_excludes_gui_and_ai_when_disabled(self, fake_jar):
        cp = build_forge_classpath(include_full_runtime=False)
        assert "forge-gui" not in cp
        assert "forge-ai" not in cp

    def test_dependency_glob_appended_when_requested(self, fake_jar):
        cp = build_forge_classpath(include_dependency_glob=True)
        assert os.path.join("forge-game", "target", "dependency", "*") in cp

    def test_dependency_glob_omitted_by_default(self, fake_jar):
        cp = build_forge_classpath()
        assert os.path.join("dependency", "*") not in cp


class TestRunForgeWorker:
    @pytest.mark.parametrize("glob", [False, True])
    def test_dependency_glob_reaches_the_classpath(self, tmp_path, glob):
        jar_dir = tmp_path / "forge-connector" / "target"
        jar_dir.mkdir(parents=True)
        (jar_dir / forge_classpath.CONNECTOR_JAR_NAME).write_bytes(b"")
        with patch.object(forge_classpath, "project_root", return_value=tmp_path), \
             patch.object(jvm_processes.subprocess, "Popen") as popen:
            popen.return_value.communicate.return_value = (None, None)
            popen.return_value.returncode = 0
            if glob:
                jvm_processes.run_forge_worker(
                    "Main", run=jvm_processes.OneShotRun(include_dependency_glob=True),
                )
            else:
                jvm_processes.run_forge_worker("Main")

        cmd = popen.call_args.args[0]
        classpath = cmd[cmd.index("-cp") + 1]
        assert (os.path.join("dependency", "*") in classpath) is glob


class TestBuildJvmCommand:
    def test_minimal_command_starts_with_java(self):
        cmd = build_jvm_command(main_class="com.example.Main", classpath="cp")
        assert cmd[0] == "java"
        assert "-cp" in cmd
        cp_index = cmd.index("-cp")
        assert cmd[cp_index + 1] == "cp"
        assert cmd[-1] == "com.example.Main"

    def test_xmx_flag_included(self):
        cmd = build_jvm_command(
            main_class="com.example.Main", classpath="cp", options=JvmOptions(xmx="1200m")
        )
        assert "-Xmx1200m" in cmd

    def test_system_properties_precede_classpath(self):
        cmd = build_jvm_command(
            main_class="com.example.Main",
            classpath="cp",
            options=JvmOptions(system_properties={"foo": "bar", "best.of": "3"}),
        )
        assert "-Dfoo=bar" in cmd
        assert "-Dbest.of=3" in cmd
        cp_index = cmd.index("-cp")
        for prop in ("-Dfoo=bar", "-Dbest.of=3"):
            assert cmd.index(prop) < cp_index

    def test_main_args_appended_after_main_class(self):
        cmd = build_jvm_command(
            main_class="com.example.Main",
            classpath="cp",
            main_args=["--set", "RVR", "--size", "10"],
        )
        main_index = cmd.index("com.example.Main")
        assert cmd[main_index + 1 :] == ["--set", "RVR", "--size", "10"]


class TestUnsafeMemoryAccessOption:
    """JDK 24+ warns on every worker's stderr about Guava's Unsafe calls; the
    option silences it, and JDK 17 would refuse to start with it."""

    @pytest.mark.parametrize("version", [24, 25])
    def test_a_jdk_that_warns_gets_the_option_first(self, monkeypatch, version):
        monkeypatch.setattr(jvm_processes, "java_feature_version", lambda: version)
        cmd = build_jvm_command(main_class="com.example.Main", classpath="cp")
        assert cmd[:2] == ["java", UNSAFE_MEMORY_ACCESS_OPTION]

    @pytest.mark.parametrize("version", [17, 21, 23, None])
    def test_an_older_or_unidentified_jdk_gets_no_option(self, monkeypatch, version):
        monkeypatch.setattr(jvm_processes, "java_feature_version", lambda: version)
        cmd = build_jvm_command(main_class="com.example.Main", classpath="cp")
        assert UNSAFE_MEMORY_ACCESS_OPTION not in cmd

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ('java version "25.0.1" 2025-10-21 LTS', 25),
            ('openjdk version "17.0.12" 2024-07-16', 17),
            ('openjdk version "24" 2025-03-18', 24),
            ('java version "1.8.0_421"', 8),
            ("no version here", None),
        ],
    )
    def test_the_feature_release_is_read_off_java_version(self, output, expected):
        assert parse_java_feature_version(output) == expected

    def test_a_missing_java_answers_none(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("java")):
            assert real_java_feature_version.__wrapped__() is None

    def test_the_probe_reads_what_java_prints_on_stderr(self):
        with patch("subprocess.run") as run:
            run.return_value.stderr = 'openjdk version "21.0.4" 2024-07-16'
            assert real_java_feature_version.__wrapped__() == 21
        assert run.call_args[0][0] == ["java", "-version"]


class TestWorkerLogFiles:
    """F4: a real, findable destination for the five latched effect-record
    failure reporters, in place of the DEVNULL every collecting worker wrote
    to before."""

    def test_get_creates_the_directory_and_a_file_named_by_run_and_worker(
        self, tmp_path,
    ):
        logs = WorkerLogFiles(tmp_path / "records", "run-1")
        handle = logs.get(3)
        try:
            assert handle.closed is False
            path = tmp_path / "records" / "run-1.3.log"
            assert path.exists()
            assert logs.path_for(3) == path
        finally:
            logs.close_all()

    def test_a_restart_reuses_the_same_handle(self, tmp_path):
        logs = WorkerLogFiles(tmp_path, "run-1")
        first = logs.get(0)
        second = logs.get(0)
        try:
            assert first is second
        finally:
            logs.close_all()

    def test_different_workers_get_different_files(self, tmp_path):
        logs = WorkerLogFiles(tmp_path, "run-1")
        try:
            assert logs.get(0) is not logs.get(1)
            assert logs.path_for(0) != logs.path_for(1)
        finally:
            logs.close_all()

    def test_a_line_written_by_a_child_process_is_on_disk_after_a_restart(
        self, tmp_path,
    ):
        """Stands in for what a real worker does: write through the handle
        Popen was given, exit, and the next Popen call reuses the slot's log
        rather than losing what was already written to it."""
        import subprocess
        import sys

        logs = WorkerLogFiles(tmp_path, "run-1")
        for lifetime in (1, 2):  # the same slot, a new process each time
            handle = logs.get(0)
            subprocess.run(
                [sys.executable, "-c",
                 f"import sys; print('reporter line from JVM lifetime {lifetime}');"
                 f" print('on stderr {lifetime}', file=sys.stderr)"],
                stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=60,
            )
        logs.close_all()

        content = logs.path_for(0).read_text(encoding="utf-8")
        assert "lifetime 1" in content and "on stderr 1" in content
        assert "lifetime 2" in content and "on stderr 2" in content

    def test_rotates_past_the_cap_and_keeps_one_backup_generation(self, tmp_path):
        logs = WorkerLogFiles(tmp_path, "run-1", max_bytes=10)
        first = logs.get(0)
        first.write(b"0123456789 - more than ten bytes")
        first.flush()

        second = logs.get(0)
        try:
            assert second is not first
            assert first.closed
            backup = tmp_path / "run-1.0.log.1"
            assert backup.exists()
            assert b"more than ten bytes" in backup.read_bytes()
            # The rotated-in replacement starts empty, not appended past the cap.
            assert logs.path_for(0).stat().st_size == 0
        finally:
            logs.close_all()

    def test_a_second_rotation_replaces_the_first_backup_rather_than_accumulating(
        self, tmp_path,
    ):
        # Each iteration's get(0) writes past max_bytes, so the *next*
        # iteration's get(0) is what rotates it out: after 3 iterations,
        # generation 2 is still the live file and generation 1 -- not 0 -- is
        # the backup, replaced rather than joined by an earlier one.
        logs = WorkerLogFiles(tmp_path, "run-1", max_bytes=5)
        for i in range(3):
            handle = logs.get(0)
            handle.write(f"generation {i} is over five bytes long".encode())
            handle.flush()
        logs.close_all()

        # Exactly current + one backup, however many rotations happened.
        matches = sorted(tmp_path.glob("run-1.0.log*"))
        assert [p.name for p in matches] == ["run-1.0.log", "run-1.0.log.1"]
        assert b"generation 2" in (tmp_path / "run-1.0.log").read_bytes()
        backup = (tmp_path / "run-1.0.log.1").read_bytes()
        assert b"generation 1" in backup
        assert b"generation 0" not in backup

    def test_a_rotation_failure_degrades_rather_than_raising(self, tmp_path):
        """On Windows especially, a rename can lose to a concurrent reader
        with the file still open. The failure must not propagate out of
        ``get`` -- which ``spawn_worker`` calls directly -- into
        ``ForgeWorkerPool._monitor_worker``'s bare ``except Exception``, which
        retries with no backoff: a tight respawn/print loop instead of a log
        that just missed one rotation.

        The test also pins that the closed handle is not registered after the
        failure. The ``try/except`` around the rename alone does not
        guarantee that, because ``get`` always reassigns
        ``self._open[worker_id]`` before returning on every path that does
        not raise. What the ``del self._open[worker_id]`` ahead of the rename
        guards is a handle leaking out to a call between the failed close and
        the reassignment, which a single-threaded test cannot observe by
        calling ``get`` again (a second real rotation could kick in once the
        writes below push the file back over the cap, returning a third,
        equally valid handle rather than exposing a stale one). Reading
        ``logs._open`` directly below -- after the patched block, before any
        further write -- is what actually pins "not still the closed one",
        without that confound.
        """
        logs = WorkerLogFiles(tmp_path, "run-1", max_bytes=10)
        first = logs.get(0)
        first.write(b"0123456789 - more than ten bytes")
        first.flush()

        with patch("pathlib.Path.rename", side_effect=OSError("boom")):
            second = logs.get(0)  # must not raise
        try:
            assert second is not first
            assert first.closed
            assert second.closed is False
            # Rotation failed, so nothing was moved to a ".1" backup --
            # the oversized file is still the live one, appended to rather
            # than replaced.
            assert not (tmp_path / "run-1.0.log.1").exists()
            # The registry holds the fresh handle, not a stale reference to
            # the one this method already closed.
            assert logs._open[0] is second
            second.write(b"more, past the cap the failed rotation left it at")
            second.flush()  # a closed-handle bug would raise ValueError here
        finally:
            logs.close_all()

    def test_a_rotation_failure_is_reported(self, tmp_path, capsys):
        logs = WorkerLogFiles(tmp_path, "run-1", max_bytes=10)
        first = logs.get(0)
        first.write(b"0123456789 - more than ten bytes")
        first.flush()

        with patch("pathlib.Path.rename", side_effect=OSError("boom")):
            logs.get(0)
        logs.close_all()

        assert "boom" in capsys.readouterr().out

    def test_a_failed_close_during_rotation_still_degrades_rather_than_raising(
        self, tmp_path,
    ):
        """``handle.close()`` itself, immediately ahead
        of the rename, was still unguarded -- only the rename and the stat
        above it were. A ``close()`` that raises must not stop rotation (or
        the eventual reopen) from completing.

        ``_io.BufferedWriter`` is a C-level, immutable type -- neither
        ``patch.object(type(first), "close", ...)`` nor an instance-level
        patch can override its ``close`` (Python itself raises ``TypeError:
        cannot set 'close' attribute of immutable type``), so this swaps a
        thin wrapper into ``logs``'s own registry instead: a legitimate
        white-box substitution for a real handle whose ``close()`` raises,
        matching real file semantics (the resource is still released even
        when the underlying OS call fails) by closing the real handle before
        raising.
        """
        logs = WorkerLogFiles(tmp_path, "run-1", max_bytes=10)
        first = logs.get(0)
        first.write(b"0123456789 - more than ten bytes")
        first.flush()

        class RaisingCloseHandle:
            def __init__(self, real):
                self._real = real
                self.closed = False

            def close(self):
                self._real.close()
                self.closed = True
                raise OSError("boom")

            def write(self, data):
                return self._real.write(data)

            def flush(self):
                self._real.flush()

        logs._open[0] = RaisingCloseHandle(first)

        second = logs.get(0)  # must not raise, despite close() raising above
        try:
            assert second is not first
            assert second.closed is False
            second.write(b"still writable after a close() that raised")
            second.flush()
        finally:
            logs.close_all()

    def test_close_all_closes_every_open_handle(self, tmp_path):
        logs = WorkerLogFiles(tmp_path, "run-1")
        handles = [logs.get(i) for i in range(3)]

        logs.close_all()

        assert all(h.closed for h in handles)

    def test_close_all_is_safe_to_call_twice(self, tmp_path):
        logs = WorkerLogFiles(tmp_path, "run-1")
        logs.get(0)
        logs.close_all()
        logs.close_all()  # must not raise

    def test_close_all_with_nothing_ever_opened_is_a_noop(self, tmp_path):
        WorkerLogFiles(tmp_path, "run-1").close_all()  # must not raise


class TestKillProcessTree:
    """No test kills a real process; a wait that times out is not swallowed,
    so a survivor is not reported dead."""

    @pytest.mark.real_process_kill
    def test_a_real_child_and_its_grandchild_both_die(self, tmp_path):
        child, grandchild = start_with_grandchild(tmp_path / "grandchild.pid")
        try:
            assert jvm_processes.kill_process_tree(child) is True
            assert child.poll() is not None
            assert not grandchild.is_alive(within=10)
        finally:
            child.kill()
            grandchild.end()

    def test_a_process_already_gone_counts_as_killed(self):
        class Exited:
            pid = 7

            def poll(self):
                return 0

        assert real_kill_process_tree(Exited()) is True

    def _stubborn(self, *, dies_on_kill: bool):
        class Stubborn:
            pid = 4242
            killed = False

            def poll(self):
                return 0 if self.killed and dies_on_kill else None

            def wait(self, timeout=None):
                if self.poll() is None:
                    raise subprocess.TimeoutExpired("java", timeout)
                return 0

            def kill(self):
                self.killed = True

        return Stubborn()

    def test_a_tree_kill_that_times_out_falls_back_to_killing_the_process(self):
        proc = self._stubborn(dies_on_kill=True)
        with patch.object(jvm_processes.subprocess, "run") as taskkill, \
             patch.object(jvm_processes.platform, "system", return_value="Windows"):
            assert real_kill_process_tree(proc, timeout=0.01) is True
        assert taskkill.call_args.args[0][:3] == ["taskkill", "/F", "/T"]
        assert proc.killed

    def test_a_survivor_is_reported_not_swallowed(self, capsys):
        proc = self._stubborn(dies_on_kill=False)
        with patch.object(jvm_processes.subprocess, "run"), \
             patch.object(jvm_processes.platform, "system", return_value="Windows"):
            assert real_kill_process_tree(proc, timeout=0.01) is False
        assert "PID 4242 survived" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "nt", reason="the job object is Windows-only")
class TestWorkersDieWithTheirParent:
    """A hard-killed supervisor (Stop-Process, taskkill without /T)
    left its JVMs running and appending to shards; a worker bound to the
    parent's job object is killed by Windows when the parent's handle closes."""

    _PARENT = (
        "import subprocess, sys, time, os\n"
        "from price_predictor.infrastructure import job_objects\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "bound = job_objects.bind_to_this_process(child)\n"
        "open(sys.argv[1] + '.tmp', 'w').write(f'{child.pid} {bound}')\n"
        "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
        "time.sleep(60)\n"
    )

    def _start_parent(self, tmp_path):
        out = tmp_path / "child.txt"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(["src", "."]))
        parent = subprocess.Popen(
            [sys.executable, "-c", self._PARENT, str(out)],
            env=env, cwd=forge_classpath.project_root(),
        )
        try:
            pid, bound = wait_for_file(out, parent).split()
            worker = OwnedProcess(int(pid))
        except BaseException:
            end_tree(parent)
            raise
        return parent, worker, bound == "True"

    def test_a_hard_killed_parent_takes_its_bound_worker_with_it(self, tmp_path):
        parent, worker, bound = self._start_parent(tmp_path)
        try:
            assert bound
            parent.kill()  # TerminateProcess: no finally, no handler runs
            parent.wait(timeout=10)
            assert not worker.is_alive(within=10)
        finally:
            parent.kill()
            worker.end()

    def test_a_process_without_a_handle_is_left_unbound(self):
        """A test's fake process has a made-up PID: binding by PID could put
        an unrelated process in the job."""

        class Fake:
            pid = 1000

        assert job_objects.bind_to_this_process(Fake()) is False

    def test_a_mock_process_is_left_unbound(self, capsys):
        """A ``MagicMock`` answers ``_handle`` with a made-up value too, which
        could name one of this process's own handles: it is never tried."""
        from unittest.mock import MagicMock

        assert job_objects.bind_to_this_process(MagicMock(pid=1000)) is False
        assert "Could not bind" not in capsys.readouterr().out


class TestOneShotTimeout:
    """A one-shot JVM runs under a timeout, so a hung Forge startup does not hang
    the command."""

    @pytest.mark.real_process_kill
    def test_a_one_shot_worker_past_its_timeout_is_killed_and_raises(self, monkeypatch):
        started: list[subprocess.Popen] = []
        real_popen = subprocess.Popen

        def popen(*args, **kwargs):
            started.append(real_popen(*args, **kwargs))
            return started[-1]

        monkeypatch.setattr(jvm_processes, "build_forge_classpath", lambda **_: "")
        monkeypatch.setattr(jvm_processes, "build_jvm_command", lambda **_: list(SLEEPER))
        monkeypatch.setattr(jvm_processes.subprocess, "Popen", popen)
        begun = time.monotonic()
        try:
            with pytest.raises(jvm_processes.ForgeWorkerTimeout, match="Main"):
                jvm_processes.run_forge_worker("Main", run=jvm_processes.OneShotRun(timeout=1.0))
            assert time.monotonic() - begun < 20
            assert started[0].poll() is not None
        finally:
            for proc in started:
                proc.kill()

    def test_the_timeout_is_an_os_error_a_caller_can_degrade_on(self):
        assert issubclass(jvm_processes.ForgeWorkerTimeout, TimeoutError)


class TestStaleStagingIsClearedAtTheFirstLaunch:
    """A hard-killed collect-variants never unstages its scripts, and
    every later Forge start on the machine then loads them. The first JVM a
    process launches clears what the last staging left, once per process."""

    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        custom = tmp_path / "forge" / "forge-gui" / "data" / "custom" / "cards"
        custom.mkdir(parents=True)
        (custom / "bolt_variant_0a1b2c3d.txt").write_text("Name:x", encoding="utf-8")
        (custom / "my_card.txt").write_text("Name:mine", encoding="utf-8")
        (custom / custom_card_staging.STAGED_MANIFEST).write_text(
            "bolt_variant_0a1b2c3d.txt", encoding="utf-8",
        )
        monkeypatch.setattr(custom_card_staging, "forge_dir", lambda: tmp_path / "forge")
        monkeypatch.setattr(jvm_processes, "clear_stale_staging", real_clear_stale_staging)
        monkeypatch.setattr(custom_card_staging, "_stale_staging_checked", False)
        return custom

    def test_the_first_command_clears_and_says_so(self, staged, capsys):
        build_jvm_command(main_class="Main", classpath="cp")

        assert not (staged / "bolt_variant_0a1b2c3d.txt").exists()
        assert (staged / "my_card.txt").exists(), "a card a person put there stays"
        assert "Removed 1 variant script(s)" in capsys.readouterr().out

    def test_it_is_checked_once_per_process(self, staged, capsys):
        build_jvm_command(main_class="Main", classpath="cp")
        (staged / "late_variant.txt").write_text("Name:y", encoding="utf-8")
        (staged / custom_card_staging.STAGED_MANIFEST).write_text(
            "late_variant.txt", encoding="utf-8",
        )

        build_jvm_command(main_class="Main", classpath="cp")

        assert (staged / "late_variant.txt").exists(), (
            "a staging this run's own workers made is never taken from under them"
        )

    def test_the_manifest_is_the_one_the_staging_writes(self):
        import re

        source = (forge_classpath.project_root() / "forge-connector" / "src" / "main" / "java"
                  / "com" / "pricepredictor" / "connector" / "ForgeEnvironmentInitializer.java")
        declared = re.search(
            r'STAGED_MANIFEST\s*=\s*"([^"]+)"', source.read_text(encoding="utf-8"),
        )
        assert declared is not None and declared.group(1) == custom_card_staging.STAGED_MANIFEST


class TestAStagingInUseIsLeftAlone:
    """A clean-up from any process's first JVM emptied the staging a
    collect-variants still running elsewhere used, so its next recycled
    worker refused every variant deck. The staging run records its owner
    (PID and start time); a live owner keeps its staging."""

    @pytest.fixture
    def staged(self, tmp_path, monkeypatch):
        custom = tmp_path / "forge" / "forge-gui" / "data" / "custom" / "cards"
        custom.mkdir(parents=True)
        (custom / "bolt_variant_0a1b2c3d.txt").write_text("Name:x", encoding="utf-8")
        (custom / custom_card_staging.STAGED_MANIFEST).write_text(
            "bolt_variant_0a1b2c3d.txt", encoding="utf-8",
        )
        monkeypatch.setattr(custom_card_staging, "forge_dir", lambda: tmp_path / "forge")
        monkeypatch.setattr(jvm_processes, "clear_stale_staging", real_clear_stale_staging)
        monkeypatch.setattr(custom_card_staging, "_stale_staging_checked", False)
        return custom

    def _owned_by(self, staged, pid, started):
        import json

        (staged / custom_card_staging.STAGING_OWNER).write_text(
            json.dumps({"pid": pid, "started": started}), encoding="utf-8",
        )

    def test_a_live_owner_keeps_its_staging_and_the_notice_names_it(self, staged, capsys):
        owner = subprocess.Popen(SLEEPER)
        try:
            self._owned_by(staged, owner.pid, custom_card_staging.process_start_time(owner.pid))

            build_jvm_command(main_class="Main", classpath="cp")

            assert (staged / "bolt_variant_0a1b2c3d.txt").exists()
            assert f"PID {owner.pid}) still runs" in capsys.readouterr().out
        finally:
            owner.kill()
            owner.wait()

    def test_a_reused_pid_is_no_owner(self, staged):
        """This process's PID with another start time: the owner is gone."""
        self._owned_by(staged, os.getpid(), 12345)

        build_jvm_command(main_class="Main", classpath="cp")

        assert not (staged / "bolt_variant_0a1b2c3d.txt").exists()
        assert not (staged / custom_card_staging.STAGING_OWNER).exists()

    def test_the_claim_names_this_process(self, tmp_path):
        real_claim_staging(tmp_path)

        assert custom_card_staging.staging_owner(tmp_path) == os.getpid()

    def test_an_unreadable_manifest_is_reported_and_the_command_goes_on(self, staged, capsys):
        (staged / custom_card_staging.STAGED_MANIFEST).write_bytes(b"\xff\xfe\x00bad")

        command = build_jvm_command(main_class="Main", classpath="cp")

        assert command[-1] == "Main"
        assert "Could not clear the variant scripts staged" in capsys.readouterr().out

    def test_an_entry_outside_the_directory_is_never_deleted(self, staged, capsys):
        outside = staged.parent / "outside.txt"
        outside.write_text("Name:keep", encoding="utf-8")
        (staged / custom_card_staging.STAGED_MANIFEST).write_text(
            f"../outside.txt\n{outside}\nbolt_variant_0a1b2c3d.txt\n", encoding="utf-8",
        )

        build_jvm_command(main_class="Main", classpath="cp")

        assert outside.exists()
        assert not (staged / "bolt_variant_0a1b2c3d.txt").exists()
        assert "outside" in capsys.readouterr().out
