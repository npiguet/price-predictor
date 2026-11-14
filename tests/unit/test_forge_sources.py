"""The sibling Forge checkout's sources, as the tests reading them find them."""

from __future__ import annotations

import pytest

from tests.unit import forge_sources


def test_a_file_missing_from_an_existing_checkout_fails_the_test(monkeypatch, tmp_path):
    monkeypatch.setattr(forge_sources, "FORGE", tmp_path)
    with pytest.raises(pytest.fail.Exception, match="reads the wrong place"):
        forge_sources.forge_source(tmp_path / "forge-ai" / "GameCopier.java")


def test_no_checkout_skips_the_test(monkeypatch, tmp_path):
    monkeypatch.setattr(forge_sources, "FORGE", tmp_path / "absent")
    with pytest.raises(pytest.skip.Exception):
        forge_sources.forge_source(tmp_path / "absent" / "GameCopier.java")


def test_a_file_in_the_checkout_is_handed_back(monkeypatch, tmp_path):
    monkeypatch.setattr(forge_sources, "FORGE", tmp_path)
    source = tmp_path / "Game.java"
    source.write_text("class Game {}", encoding="utf-8")
    assert forge_sources.forge_source(source) == source


def test_a_connector_file_is_read_and_a_missing_one_fails_the_test():
    assert "class ForkRefusals" in forge_sources.connector_source("effects/ForkRefusals.java")
    with pytest.raises(pytest.fail.Exception, match="reads the wrong place"):
        forge_sources.connector_source("effects/NoSuchClass.java")
