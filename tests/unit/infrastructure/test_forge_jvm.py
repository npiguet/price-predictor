"""Unit tests for the shared Forge JVM helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from price_predictor.infrastructure import forge_jvm
from price_predictor.infrastructure.forge_jvm import (
    WorkerLogFiles,
    build_forge_classpath,
    build_jvm_command,
    resolve_connector_jar,
)


class TestResolveConnectorJar:
    def test_raises_when_jar_missing(self, tmp_path):
        with patch.object(forge_jvm, "project_root", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="mvn package"):
                resolve_connector_jar()

    def test_returns_path_when_jar_exists(self, tmp_path):
        jar_dir = tmp_path / "forge-connector" / "target"
        jar_dir.mkdir(parents=True)
        jar = jar_dir / forge_jvm.CONNECTOR_JAR_NAME
        jar.write_bytes(b"")
        with patch.object(forge_jvm, "project_root", return_value=tmp_path):
            assert resolve_connector_jar() == jar


class TestBuildForgeClasspath:
    @pytest.fixture
    def fake_jar(self, tmp_path):
        jar_dir = tmp_path / "forge-connector" / "target"
        jar_dir.mkdir(parents=True)
        (jar_dir / forge_jvm.CONNECTOR_JAR_NAME).write_bytes(b"")
        with patch.object(forge_jvm, "project_root", return_value=tmp_path):
            yield tmp_path

    def test_includes_connector_and_core_modules(self, fake_jar):
        cp = build_forge_classpath()
        parts = cp.split(os.pathsep)
        assert any(forge_jvm.CONNECTOR_JAR_NAME in p for p in parts)
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
            main_class="com.example.Main", classpath="cp", xmx="1200m"
        )
        assert "-Xmx1200m" in cmd

    def test_system_properties_precede_classpath(self):
        cmd = build_jvm_command(
            main_class="com.example.Main",
            classpath="cp",
            system_properties={"foo": "bar", "best.of": "3"},
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

    def test_two_runs_never_share_a_file(self, tmp_path):
        """A fresh run_id per invocation is what keeps a long-lived
        ``effect_records_dir`` from accumulating one never-rotated file
        across every collection ever run into it."""
        a = WorkerLogFiles(tmp_path, "run-a")
        b = WorkerLogFiles(tmp_path, "run-b")
        assert a.path_for(0) != b.path_for(0)

    def test_a_line_written_by_a_child_process_is_on_disk_after_a_restart(
        self, tmp_path,
    ):
        """Stands in for what a real worker does: write through the handle
        Popen was given, exit, and the next Popen call reuses the slot's log
        rather than losing what was already written to it."""
        logs = WorkerLogFiles(tmp_path, "run-1")
        first = logs.get(0)
        first.write(b"reporter line from JVM lifetime 1\n")
        first.flush()

        second = logs.get(0)  # the same slot, a new JVM
        second.write(b"reporter line from JVM lifetime 2\n")
        second.flush()
        logs.close_all()

        content = logs.path_for(0).read_text(encoding="utf-8")
        assert "lifetime 1" in content
        assert "lifetime 2" in content

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
