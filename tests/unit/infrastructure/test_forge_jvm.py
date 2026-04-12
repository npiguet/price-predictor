"""Unit tests for the shared Forge JVM helpers."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from price_predictor.infrastructure import forge_jvm
from price_predictor.infrastructure.forge_jvm import (
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
