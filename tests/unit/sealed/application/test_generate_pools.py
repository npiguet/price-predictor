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


def _generate(set_code, pool_count, pools_path):
    """Run the use case against a mock connector and return the connector."""
    connector = _mock_connector()
    GeneratePoolsUseCase().execute(set_code, pool_count, pools_path, connector)
    return connector


def _run_cli_generate_pools(monkeypatch, tmp_path, argv):
    """Run ``generate-pools`` with a fake connector; return ``(rc, captured)``."""
    from sealed.infrastructure import cli, cli_generate_pools

    monkeypatch.chdir(tmp_path)

    captured: dict = {}

    class FakeConnector:
        def generate(self, set_code, pool_count, pools_path, exclude_cards=None):
            captured["set_code"] = set_code
            captured["pools_path"] = pools_path
            captured["exclude_cards"] = exclude_cards
            return 0

    monkeypatch.setattr(
        "sealed.infrastructure.pool_connector.PoolConnector",
        FakeConnector,
    )

    parser = cli.build_parser()
    rc = cli_generate_pools.run_generate_pools(parser.parse_args(argv))
    return rc, captured


class TestGeneratePoolsDirectoryCreation:
    def test_creates_output_directory_if_absent(self, tmp_path):
        pools_path = tmp_path / "sealed" / "pools" / "RVR"
        _generate("RVR", 10, pools_path)
        assert pools_path.exists()

    def test_works_when_directory_already_exists(self, tmp_path):
        pools_path = tmp_path / "pools"
        pools_path.mkdir(parents=True)
        connector = _generate("RVR", 10, pools_path)
        connector.generate.assert_called_once()


class TestGeneratePoolsConnectorArgs:
    def test_connector_called_with_correct_set_code(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _generate("MH3", 5, pools_path)
        connector.generate.assert_called_once()
        args = connector.generate.call_args[0]
        assert args[0] == "MH3"

    def test_connector_called_with_correct_pool_count(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _generate("RVR", 42, pools_path)
        args = connector.generate.call_args[0]
        assert args[1] == 42

    def test_connector_called_with_pools_path(self, tmp_path):
        pools_path = tmp_path / "pools"
        connector = _generate("RVR", 10, pools_path)
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
        connector = _generate(None, 10, pools_path)
        connector.generate.assert_called_once()
        args = connector.generate.call_args[0]
        assert args[0] is None
        assert args[1] == 10
        assert args[2] == pools_path

    def test_cli_pools_path_is_used_as_given_when_no_set(self, tmp_path, monkeypatch):
        """When --set is omitted, --pools-path is used as given."""
        rc, captured = _run_cli_generate_pools(
            monkeypatch, tmp_path,
            ["generate-pools", "--size", "1", "--pools-path", "pools/x"],
        )
        assert rc == 0
        assert captured["set_code"] is None
        assert captured["pools_path"] == Path("pools/x")

    def test_cli_pools_path_fills_in_the_set(self, tmp_path, monkeypatch):
        rc, captured = _run_cli_generate_pools(
            monkeypatch, tmp_path, [
                "generate-pools", "--set", "MH3", "--size", "1",
                "--pools-path", "pools/{set}",
            ],
        )
        assert rc == 0
        assert captured["set_code"] == "MH3"
        assert captured["pools_path"] == Path("pools/MH3")
