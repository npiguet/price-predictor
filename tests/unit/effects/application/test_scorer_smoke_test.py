"""``scorer-smoke-test`` with both sealed subprocesses mocked (T110, SC-012).

The two sealed commands are the interface (FR-071): what is tested is the
argument vectors they receive, that every file the run writes lands under
``--scratch-dir``, and that ``effects`` reaches the sealed pipeline through no
import of ``sealed.application``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from effects.application import scorer_smoke_test as smoke
from effects.application.scorer_smoke_test import ScorerSmokeTestConfig
from effects.domain.ability_cache_layout import ARRAY_KEY


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A converted tree, an ability cache for one of its two cards, a checkpoint."""
    cards = tmp_path / "output" / "cardsfolder"
    for stem in ("s/shock", "g/grizzly_bears"):
        path = cards / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("name: x\n", encoding="utf-8")
    (cards / "s" / "shock.npz").write_bytes(b"the real sealed vector, never touched")
    abilities = tmp_path / "abilities" / "cardsfolder" / "s"
    abilities.mkdir(parents=True)
    np.savez_compressed(abilities / "shock.npz", **{ARRAY_KEY: np.ones((2, 3), np.float32)})
    np.savez_compressed(
        abilities / "shock.taxonomy.npz", **{ARRAY_KEY: np.ones((2, 5), np.float32)},
    )

    class _Store:
        def __init__(self, directory):
            pass

        def load(self, path):
            return SimpleNamespace(variant="full", encoder_config=SimpleNamespace(e_dim=3))

    monkeypatch.setattr(
        "effects.infrastructure.effect_model_store.EffectModelStore", _Store,
    )
    return SimpleNamespace(root=tmp_path, cards=cards, abilities=tmp_path / "abilities")


def _config(tree) -> ScorerSmokeTestConfig:
    return ScorerSmokeTestConfig(
        scratch_dir=tree.root / "scratch",
        checkpoint=tree.root / "models" / "latest.pt",
        sealed_encoder_checkpoint=tree.root / "sealed" / "encoder" / "latest.pt",
        abilities_root=tree.abilities,
        cards_path=tree.cards,
    )


class _Runner:
    """Stands in for ``subprocess.run``; ``encode-cards`` writes sealed vectors."""

    def __init__(self, config: ScorerSmokeTestConfig, *, width: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.config = config
        self.width = width

    def __call__(self, command, check=False):
        self.calls.append(list(command))
        if "encode-cards" in command:
            from sealed.domain.card_embedding_layout import FEATURE_COUNT

            for text in self.config.scratch_cards.rglob("*.txt"):
                np.savez_compressed(
                    text.with_suffix(".npz"),
                    embedding=np.zeros(self.width + FEATURE_COUNT, np.float32),
                )
        return subprocess.CompletedProcess(command, 0)


def _files(root: Path) -> dict[Path, float]:
    return {p: p.stat().st_mtime_ns for p in root.rglob("*") if p.is_file()}


class TestTheRun:
    def test_both_sealed_commands_get_the_scratch_tree(self, tree):
        config = _config(tree)
        runner = _Runner(config)
        result = smoke.run(config, runner=runner)

        encode, train = runner.calls
        assert encode[:4] == [sys.executable, "-m", "sealed", "encode-cards"]
        assert encode[encode.index("--cards-path") + 1] == str(config.scratch_cards)
        assert encode[encode.index("--encoder-checkpoint") + 1] == str(
            config.sealed_encoder_checkpoint,
        )
        assert encode[encode.index("--vocab-path") + 1] == str(
            config.sealed_encoder_checkpoint.parent / "vocab.txt",
        )
        assert train[:4] == [sys.executable, "-m", "sealed", "train-scorer"]
        assert train[train.index("--cards-path") + 1] == str(config.scratch_cards)
        assert train[train.index("--checkpoint-dir") + 1] == str(config.scorer_dir)
        assert train[train.index("--embedding-lr") + 1] == "0"
        assert (result.cards_written, result.cards_without_e) == (2, 1)

    def test_nothing_is_written_outside_the_scratch_dir(self, tree):
        """SC-012: the converted tree the sealed pipeline reads is untouched."""
        config = _config(tree)
        before = _files(tree.cards) | _files(tree.abilities)
        smoke.run(config, runner=_Runner(config))
        assert _files(tree.cards) | _files(tree.abilities) == before
        written = {p for p in _files(tree.root) if p not in before}
        assert all(config.scratch_dir in p.parents for p in written)

    def test_every_scratch_vector_carries_e_at_the_checkpoint_s_width(self, tree):
        from sealed.domain.card_embedding_layout import FEATURE_COUNT

        config = _config(tree)
        smoke.run(config, runner=_Runner(config, width=4))
        for path in config.scratch_cards.rglob("*.npz"):
            with np.load(path) as data:
                assert data["embedding"].shape == (4 + 6 + FEATURE_COUNT,)

    def test_a_cache_of_another_width_is_refused_before_any_command(self, tree, monkeypatch):
        class _Wider:
            def __init__(self, directory):
                pass

            def load(self, path):
                return SimpleNamespace(
                    variant="full", encoder_config=SimpleNamespace(e_dim=8),
                )

        monkeypatch.setattr(
            "effects.infrastructure.effect_model_store.EffectModelStore", _Wider,
        )
        config = _config(tree)
        runner = _Runner(config)
        with pytest.raises(ValueError, match="another checkpoint"):
            smoke.run(config, runner=runner)
        assert runner.calls == []

    def test_a_missing_cache_is_refused(self, tree):
        config = ScorerSmokeTestConfig(
            scratch_dir=tree.root / "scratch", abilities_root=tree.root / "nowhere",
            cards_path=tree.cards,
        )
        with pytest.raises(FileNotFoundError, match="encode-abilities"):
            smoke.run(config, runner=_Runner(config))


def test_the_module_imports_nothing_from_sealed_application():
    """FR-071, FR-090: Phase A is a subprocess, never an import."""
    tree = ast.parse(Path(smoke.__file__).read_text(encoding="utf-8"))
    modules = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ] + [
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not [m for m in modules if m.startswith("sealed.application")]


def test_the_cli_refuses_a_run_without_a_scratch_dir():
    from effects.infrastructure.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["scorer-smoke-test"])
