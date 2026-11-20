"""``--embeddings-path``: where ``encode-cards`` writes the ``.npz`` card
embeddings and where every reader of them looks; unset, beside the cards."""

from __future__ import annotations

from pathlib import Path

import pytest

from sealed.infrastructure import (
    cli,
    cli_decks,
    cli_encode_cards,
    cli_evaluate_scorer,
    cli_train_picker,
    cli_train_scorer,
)

_CARDS = "output/sealed/gen1/converted/cardsfolder"
_CACHE = "output/sealed/gen1/cache/2026-11-15-encoder"

#: Each reader: its argv (without the flag), the use case run, its run function.
_READERS = {
    "build-decks": (
        ["build-decks", "--pools-path", "p.txt", "--label", "x", "--cards-path", _CARDS,
         "--checkpoint", "s.pt", "--output", "d.txt"],
        "sealed.application.build_decks.BuildDecksUseCase", cli_decks.run_build_decks,
    ),
    "pick-decks": (
        ["pick-decks", "--pools-path", "p.txt", "--label", "x", "--cards-path", _CARDS,
         "--picker-checkpoint", "p.pt", "--output", "d.txt"],
        "sealed.application.pick_decks.PickDecksUseCase", cli_decks.run_pick_decks,
    ),
    "evaluate-scorer": (
        ["evaluate-scorer", "--checkpoint", "s.pt", "--cards-path", _CARDS],
        "sealed.application.evaluate_scorer.EvaluateScorerUseCase",
        cli_evaluate_scorer.run_evaluate_scorer,
    ),
    "train-scorer": (
        ["train-scorer", "--cards-path", _CARDS, "--outcomes-path", "o.txt",
         "--checkpoint-dir", "run"],
        "sealed.application.train_scorer.TrainScorerUseCase", cli_train_scorer.run_train_scorer,
    ),
}


def _config_of(monkeypatch, argv, use_case, run):
    """The config ``run`` hands its use case, which is not run."""
    module, _, name = use_case.rpartition(".")
    seen = []
    monkeypatch.setattr(f"{module}.{name}.execute",
                        lambda _self, config: seen.append(config) or 0)
    assert run(cli.build_parser().parse_args(argv)) == 0
    return seen[0]


@pytest.mark.parametrize("command", sorted(_READERS))
def test_a_reader_looks_for_the_embeddings_where_the_flag_says(monkeypatch, command):
    argv, use_case, run = _READERS[command]
    config = _config_of(monkeypatch, [*argv, "--embeddings-path", _CACHE], use_case, run)
    assert config.embeddings_path == Path(_CACHE)
    assert config.cards_path == Path(_CARDS)


@pytest.mark.parametrize("command", sorted(_READERS))
def test_unset_a_reader_looks_beside_the_cards(monkeypatch, command):
    argv, use_case, run = _READERS[command]
    assert _config_of(monkeypatch, argv, use_case, run).embeddings_path is None


def test_train_picker_reads_the_embeddings_where_the_flag_says(monkeypatch, tmp_path):
    scorer = tmp_path / "scorer.pt"
    scorer.write_bytes(b"")
    config = _config_of(
        monkeypatch,
        ["train-picker", "--pools-path", "p.txt", "--scorer-checkpoint", str(scorer),
         "--cards-path", _CARDS, "--embeddings-path", _CACHE,
         "--checkpoint-dir", "run"],
        "sealed.application.train_picker.TrainPickerUseCase", cli_train_picker.run_train_picker,
    )
    assert config.embeddings_path == Path(_CACHE)


def test_a_resumed_trainer_keeps_the_embeddings_it_ran_on():
    from sealed.application.train_scorer import TrainScorerConfig
    from sealed.infrastructure.cli_resume import resolve_resumable_args

    args = cli.build_parser().parse_args(["train-scorer"])
    resolved = resolve_resumable_args(
        args, flag_names=cli_train_scorer._RESUMABLE_FLAG_NAMES, config_cls=TrainScorerConfig,
        path_fields=cli_train_scorer._PATH_FIELDS, resumed_config={"embeddings_path": _CACHE},
    )
    assert resolved["embeddings_path"] == Path(_CACHE)
    assert "embeddings_path" in cli_train_picker._RESUMABLE_PICKER_FLAG_NAMES
    assert "embeddings_path" in cli_train_picker._PICKER_PATH_FIELDS


class TestEncodeCardsWritesWhereTheFlagSays:
    def _args(self, *extra):
        return cli.build_parser().parse_args([
            "encode-cards", "--cards-path", _CARDS, "--vocab-path", "vocab.txt",
            "--encoder-checkpoint", "e.pt", *extra,
        ])

    def test_the_run_directory_is_the_embeddings_tree_made_for_it(self):
        args = self._args("--embeddings-path", _CACHE)
        writes = cli.RUN_MANIFESTS["encode-cards"]
        assert writes.run_dir(args) == Path(_CACHE)
        assert writes.create(args) is True
        assert Path(_CARDS) in writes.input_paths(args)

    def test_unset_it_records_in_the_cards_tree_and_creates_nothing(self):
        args = self._args()
        writes = cli.RUN_MANIFESTS["encode-cards"]
        assert args.embeddings_path is None
        assert writes.run_dir(args) == Path(_CARDS)
        assert writes.create(args) is False

    @pytest.mark.parametrize("command", [
        "build-decks", "pick-decks", "evaluate-scorer", "train-scorer", "train-picker",
    ])
    def test_every_reader_lists_the_embeddings_it_read(self, command):
        argv = {
            "build-decks": ["--pools-path", "p", "--label", "x", "--checkpoint", "s.pt",
                            "--output", "d.txt"],
            "pick-decks": ["--pools-path", "p", "--label", "x", "--picker-checkpoint",
                           "p.pt", "--output", "d.txt"],
            "evaluate-scorer": ["--checkpoint", "s.pt", "--work-dir", "w"],
            "train-scorer": ["--outcomes-path", "o.txt", "--checkpoint-dir", "run"],
            "train-picker": ["--pools-path", "p", "--scorer-checkpoint", "s.pt",
                             "--checkpoint-dir", "run"],
        }.get(command, [])
        args = cli.build_parser().parse_args(
            [command, *argv, "--cards-path", _CARDS, "--embeddings-path", _CACHE])
        assert Path(_CACHE) in cli.RUN_MANIFESTS[command].input_paths(args)


def test_encode_cards_writes_the_embeddings_into_the_flag_s_tree(tmp_path, monkeypatch):
    """End to end through ``run_encode_cards``: the cards tree gains nothing."""
    from sealed.application import encode_cards

    cards = tmp_path / "cardsfolder"
    (cards / "a").mkdir(parents=True)
    (cards / "a" / "card_a.txt").write_text("type: creature", encoding="utf-8")
    cache = tmp_path / "cache"
    seen = []

    def execute(_self, config, encoder, store, progress=None):
        seen.append(config)
        return encode_cards.EncodeCardsResult(processed=1, skipped=0, errors=[])

    monkeypatch.setattr(encode_cards.EncodeCardsUseCase, "execute", execute)
    monkeypatch.setattr(cli_encode_cards, "_load_encoder_for_encode_cards", _stub_encoder)
    monkeypatch.setattr("price_predictor.infrastructure.tokenizer_store.load_tokenizer",
                        lambda _path: object())
    monkeypatch.setattr("sealed.domain.card_encoder.CardEncoder",
                        lambda *_args, **_kwargs: object())
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n", encoding="utf-8")
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"")
    args = cli.build_parser().parse_args([
        "encode-cards", "--encoder-checkpoint", str(encoder), "--vocab-path", str(vocab),
        "--cards-path", str(cards), "--embeddings-path", str(cache),
    ])
    assert cli_encode_cards.run_encode_cards(args) == 0
    assert seen[0].embeddings_path == cache and seen[0].cards_path == cards


def _stub_encoder(_path):
    class _Model:
        def eval(self):
            return self

    class _Config:
        max_seq_len = 8
        pooled_dim = 4

    return _Model(), _Config()
