"""The conversion record on disk (finding 11.3, task K03): the sidecar digest,
which ``run.json`` names a tree's conversion, the connector JAR's Forge
commit, and the Forge checkout's state."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from price_predictor.domain.converted_provenance import ConversionRefused
from price_predictor.infrastructure.converted_provenance import (
    collection_conversion,
    connector_forge_commit,
    forge_checkout,
    read_conversion,
    sidecar_digest,
)
from tests.unit.conversion_fixtures import (
    FORGE_COMMIT,
    OTHER_COMMIT,
    conversion_member,
    stand_in_checkout,
    write_conversion,
    write_jar,
)


@pytest.fixture(autouse=True)
def _checkout_at_the_jars_commit(monkeypatch):
    """The checkout stands at the connector's commit; ``real_checkout`` undoes it."""
    stand_in_checkout(monkeypatch)


def _sidecar(root: Path, relative: str, text: str = "{}") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestTheDigest:
    def test_the_same_tree_has_the_same_digest(self, tmp_path):
        _sidecar(tmp_path / "cardsfolder", "s/shock.provenance.json")
        _sidecar(tmp_path / "tokenscripts", "goblin.provenance.json")
        roots = {"cardsfolder": tmp_path / "cardsfolder", "tokenscripts": tmp_path / "tokenscripts"}

        assert sidecar_digest(roots) == sidecar_digest(dict(reversed(roots.items())))

    def test_a_resized_sidecar_moves_it(self, tmp_path):
        shock = _sidecar(tmp_path / "cardsfolder", "s/shock.provenance.json")
        roots = {"cardsfolder": tmp_path / "cardsfolder"}
        before = sidecar_digest(roots)

        shock.write_text('{"lines": []}', encoding="utf-8")

        assert sidecar_digest(roots) != before

    def test_only_sidecars_count(self, tmp_path):
        _sidecar(tmp_path / "cardsfolder", "s/shock.provenance.json")
        roots = {"cardsfolder": tmp_path / "cardsfolder"}
        before = sidecar_digest(roots)

        _sidecar(tmp_path / "cardsfolder", "s/shock.txt", "name: Shock")
        _sidecar(tmp_path / "cardsfolder", "run.json", "{}")

        assert sidecar_digest(roots) == before


class TestATreesConversion:
    def test_the_latest_invocation_that_changed_the_tree_decides(self, tmp_path):
        root = write_conversion(
            tmp_path / "converted", conversion_member(OTHER_COMMIT), conversion_member(),
        )

        assert read_conversion(root / "cardsfolder").forge_commit == FORGE_COMMIT

    def test_a_conversion_that_did_not_finish_whole_is_refused(self, tmp_path):
        root = write_conversion(tmp_path / "converted", conversion_member(), None)

        with pytest.raises(ConversionRefused, match="did not finish whole"):
            read_conversion(root / "cardsfolder")

    def test_a_conversion_naming_no_commit_is_refused(self, tmp_path):
        root = write_conversion(tmp_path / "converted", conversion_member(None))

        with pytest.raises(ConversionRefused, match="names no Forge commit"):
            read_conversion(root / "cardsfolder")

    def test_a_run_json_of_another_command_in_the_tree_is_passed_over(self, tmp_path):
        """``encode-cards`` keeps its manifest in the card tree it encodes beside."""
        root = write_conversion(tmp_path / "converted")
        (root / "cardsfolder" / "run.json").write_text(
            '{"invocations": [{"command": "python -m sealed encode-cards"}]}',
            encoding="utf-8",
        )

        assert read_conversion(root / "cardsfolder").manifest == root / "run.json"

    def test_a_tree_with_no_run_json_is_refused_naming_where_it_looked(self, tmp_path):
        tree = tmp_path / "converted" / "cardsfolder"
        tree.mkdir(parents=True)

        with pytest.raises(ConversionRefused) as refused:
            read_conversion(tree)

        assert str(tree / "run.json") in str(refused.value)
        assert str(tree.parent / "run.json") in str(refused.value)


class TestTheConnectorsCommit:
    def test_the_manifest_names_it(self, tmp_path):
        assert connector_forge_commit(write_jar(tmp_path / "c.jar")) == FORGE_COMMIT

    def test_a_jar_built_before_it_is_refused_naming_the_jar(self, tmp_path):
        jar = write_jar(tmp_path / "c.jar", None)

        with pytest.raises(ConversionRefused) as refused:
            connector_forge_commit(jar)

        assert str(jar) in str(refused.value)
        assert "rebuild the connector" in str(refused.value)

    def test_an_unreadable_jar_is_refused_naming_it(self, tmp_path):
        jar = tmp_path / "c.jar"
        jar.write_text("not a zip", encoding="utf-8")

        with pytest.raises(ConversionRefused, match="rebuild the connector"):
            connector_forge_commit(jar)

    def test_a_matching_jar_lets_the_collection_start(self, tmp_path):
        root = write_conversion(tmp_path / "converted")

        conversion = collection_conversion(
            [root / "cardsfolder", root / "tokenscripts"], jar=write_jar(tmp_path / "c.jar"),
        )

        assert conversion.run == root


def _git(repository: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)


def _repository(path: Path) -> Path:
    """A real git repository holding one committed script, ``shock.txt``."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "test")
    (path / "shock.txt").write_text("Name:Shock\n", encoding="utf-8")
    _git(path, "add", "shock.txt")
    _git(path, "commit", "-q", "-m", "shock")
    return path


def _head(repository: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


@pytest.fixture
def real_checkout(monkeypatch, tmp_path):
    """The real :func:`forge_checkout` over a real repository, and the
    conversion and connector JAR built at its HEAD."""
    monkeypatch.undo()
    checkout = _repository(tmp_path / "forge")
    head = _head(checkout)
    root = write_conversion(tmp_path / "converted", conversion_member(head))
    trees = [root / "cardsfolder", root / "tokenscripts"]
    return checkout, trees, write_jar(tmp_path / "c.jar", head), head


class TestTheCheckoutTheWorkersReadScriptsFrom:
    def test_a_checkout_at_the_jars_commit_lets_the_collection_start(self, real_checkout):
        checkout, trees, jar, _built_at = real_checkout

        conversion = collection_conversion(trees, jar=jar, forge=checkout)

        assert conversion.checkout_dirty is False
        assert conversion.recorded()["checkout_dirty"] is False

    def test_a_checkout_moved_after_the_build_is_refused_naming_both_commits(self, real_checkout):
        checkout, trees, jar, built_at = real_checkout
        (checkout / "bolt.txt").write_text("Name:Bolt\n", encoding="utf-8")
        _git(checkout, "add", "bolt.txt")
        _git(checkout, "commit", "-q", "-m", "bolt")

        with pytest.raises(ConversionRefused) as refused:
            collection_conversion(trees, jar=jar, forge=checkout)

        message = str(refused.value)
        assert _head(checkout) in message[:200]
        assert built_at in message[:200]
        assert str(checkout) in message
        assert str(jar) in message

    def test_a_dirty_checkout_is_warned_about_and_recorded_not_refused(
        self, real_checkout, caplog,
    ):
        checkout, trees, jar, _built_at = real_checkout
        (checkout / "shock.txt").write_text("Name:Shock\nA:SP$ DealDamage\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            conversion = collection_conversion(trees, jar=jar, forge=checkout)

        assert conversion.checkout_dirty is True
        assert conversion.recorded()["checkout_dirty"] is True
        assert "uncommitted changes" in caplog.text
        assert str(checkout) in caplog.text

    def test_a_directory_that_is_no_checkout_is_refused_naming_it(self, real_checkout, tmp_path):
        _checkout, trees, jar, _built_at = real_checkout
        bare = tmp_path / "no-forge"
        bare.mkdir()

        with pytest.raises(ConversionRefused) as refused:
            collection_conversion(trees, jar=jar, forge=bare)

        assert str(bare) in str(refused.value)
        assert "no git checkout" in str(refused.value)


def test_the_checkout_gives_its_full_head_and_whether_it_is_dirty(tmp_path):
    _repository(tmp_path)
    script = tmp_path / "shock.txt"

    head, dirty = forge_checkout(tmp_path)
    assert len(head) == 40
    assert dirty is False

    script.write_text("Name:Shock\nA:SP$ DealDamage\n", encoding="utf-8")
    assert forge_checkout(tmp_path) == (head, True)


def test_no_checkout_gives_no_commit(tmp_path):
    assert forge_checkout(tmp_path / "absent") == (None, False)
