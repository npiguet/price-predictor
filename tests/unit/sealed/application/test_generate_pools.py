"""Unit tests for GeneratePoolsUseCase."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sealed.application.generate_pools import GeneratePoolsUseCase


def _mock_connector():
    connector = MagicMock()
    connector.generate = MagicMock(return_value=0)
    return connector


class TestGeneratePoolsDirectoryCreation:
    def test_creates_output_directory_if_absent(self, tmp_path):
        pools_path = tmp_path / "sealed" / "pools" / "RVR"
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute("RVR", 10, pools_path, connector)
        assert pools_path.exists()

    def test_works_when_directory_already_exists(self, tmp_path):
        pools_path = tmp_path / "pools"
        pools_path.mkdir(parents=True)
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute("RVR", 10, pools_path, connector)
        connector.generate.assert_called_once()


class TestGeneratePoolsConnectorArgs:
    def test_connector_called_with_correct_set_code(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute("MH3", 5, pools_path, connector)
        connector.generate.assert_called_once()
        args = connector.generate.call_args[0]
        assert args[0] == "MH3"

    def test_connector_called_with_correct_pool_count(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute("RVR", 42, pools_path, connector)
        args = connector.generate.call_args[0]
        assert args[1] == 42

    def test_connector_called_with_pools_path(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute("RVR", 10, pools_path, connector)
        args = connector.generate.call_args[0]
        assert args[2] == pools_path


class TestGeneratePoolsErrorPropagation:
    def test_connector_error_propagates(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _mock_connector()
        connector.generate.side_effect = RuntimeError("JAR failed")
        use_case = GeneratePoolsUseCase()
        with pytest.raises(RuntimeError, match="JAR failed"):
            use_case.execute("RVR", 10, pools_path, connector)


class TestGeneratePoolsRandomSet:
    def test_set_code_none_passed_through_to_connector(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _mock_connector()
        use_case = GeneratePoolsUseCase()
        use_case.execute(None, 10, pools_path, connector)
        connector.generate.assert_called_once()
        args = connector.generate.call_args[0]
        assert args[0] is None
        assert args[1] == 10
        assert args[2] == pools_path

    def test_cli_default_pools_path_when_no_set(self, tmp_path, monkeypatch):
        """When --set is omitted, the default pools-path is output/sealed/pools/
        (no per-set subdirectory)."""
        from sealed.infrastructure import cli

        monkeypatch.chdir(tmp_path)

        captured: dict = {}

        class FakeConnector:
            def generate(self, set_code, pool_count, pools_path):
                captured["set_code"] = set_code
                captured["pools_path"] = pools_path
                return 0

        monkeypatch.setattr(
            "sealed.infrastructure.pool_connector.PoolConnector",
            FakeConnector,
        )

        parser = cli.build_parser()
        args = parser.parse_args(["generate-pools", "--size", "1"])
        rc = cli.run_generate_pools(args)
        assert rc == 0
        assert captured["set_code"] is None
        assert captured["pools_path"] == Path("output") / "sealed" / "pools"

    def test_cli_default_pools_path_when_set_given(self, tmp_path, monkeypatch):
        from sealed.infrastructure import cli

        monkeypatch.chdir(tmp_path)

        captured: dict = {}

        class FakeConnector:
            def generate(self, set_code, pool_count, pools_path):
                captured["set_code"] = set_code
                captured["pools_path"] = pools_path
                return 0

        monkeypatch.setattr(
            "sealed.infrastructure.pool_connector.PoolConnector",
            FakeConnector,
        )

        parser = cli.build_parser()
        args = parser.parse_args(["generate-pools", "--set", "MH3", "--size", "1"])
        rc = cli.run_generate_pools(args)
        assert rc == 0
        assert captured["set_code"] == "MH3"
        assert captured["pools_path"] == Path("output") / "sealed" / "pools" / "MH3"
