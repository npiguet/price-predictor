"""The effects vocabulary (T061).

Three things separate it from the sealed one it wraps: the scan covers converted
cards, token scripts and the keyword-definition file; ``[CLS]`` is seeded
alongside ``[MASK]``; and the prose and script surfaces write different files, so
a stage-four rebuild cannot overwrite the vocabulary a stage-one checkpoint
recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.application.build_vocab import (
    DEFAULT_SCRIPT_VOCAB_PATH,
    DEFAULT_VOCAB_PATH,
    SEEDED_SPECIALS,
    SURFACE_SCRIPT,
    BuildVocabConfig,
    EmptyCardsFolderError,
    run,
)
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar
from price_predictor.infrastructure.tokenizer_store import load_vocabulary

_CARD = """name: serra angel
mana cost: {3}{W}{W}
types: creature angel
power toughness: 4/4
static: flying
static: vigilance
"""

_TOKEN = """name: soldier token
types: creature soldier
power toughness: 1/1
"""


@pytest.fixture
def corpus(tmp_path):
    """A two-tree converted corpus plus a keyword-definition file.

    Every word under test appears at least twice, because the scan uses the
    production ``freq_threshold=2``: a hapax is corpus noise on 33k cards, and
    testing against a lower threshold would test something the command never
    does.
    """
    cards = tmp_path / "cardsfolder"
    (cards / "s").mkdir(parents=True)
    (cards / "s" / "serra_angel.txt").write_text(_CARD, encoding="utf-8")
    (cards / "a").mkdir(parents=True)
    (cards / "a" / "archangel.txt").write_text(
        _CARD.replace("serra angel", "archangel"), encoding="utf-8",
    )

    tokens = tmp_path / "tokenscripts"
    tokens.mkdir()
    (tokens / "soldier.txt").write_text(_TOKEN, encoding="utf-8")
    (tokens / "veteran_soldier.txt").write_text(
        _TOKEN.replace("soldier token", "veteran soldier token"), encoding="utf-8",
    )

    keywords = tmp_path / "keyword-definitions.json"
    keywords.write_text(
        json.dumps({
            "Flying": {
                "reminder_template": (
                    "This creature can't be blocked except by creatures with "
                    "flying or reach."
                ),
                "generated_script": None,
            },
            "Reach": {
                "reminder_template": (
                    "This creature can block creatures with flying, and is "
                    "never blocked by reach."
                ),
                "generated_script": None,
            },
            "Vigilance": {
                "reminder_template": "Attacking doesn't cause this creature to tap.",
                "generated_script": None,
            },
        }),
        encoding="utf-8",
    )
    return cards, tokens, keywords


def _config(corpus, tmp_path, **overrides) -> BuildVocabConfig:
    cards, tokens, keywords = corpus
    defaults = {
        "cards_folders": (cards, tokens),
        "keyword_definitions": keywords,
        "vocab_path": tmp_path / "vocab.txt",
        "printings_path": tmp_path / "absent-printings.json",
    }
    defaults.update(overrides)
    return BuildVocabConfig(**defaults)


class TestTheKeywordScanFollowsTheSurface:
    """A keyword expands to its template on prose, its script from stage four.

    Scanning both put Forge script syntax into a prose vocabulary that can
    never encode it: 65 tokens of it — ``activezones``, ``cantblockby``,
    ``counters_ge``, a bare ``_p`` — sitting alongside the words cards are
    written in. Nothing failed, because a vocabulary with extra entries still
    loads and still tokenizes; the tokens were simply never going to be seen.
    """

    @pytest.fixture
    def scripted(self, corpus, tmp_path):
        """Two keywords whose scripts use words their reminder text does not.

        Two of each word, because the scan runs at the production
        ``freq_threshold=2`` — a hapax is corpus noise on 33k cards, and a
        fixture that tested against a lower threshold would test something the
        command never does.
        """
        cards, tokens, keywords = corpus
        script = (
            "Mode$ CantBlockBy | ValidAttacker$ Creature.Self | "
            "ActiveZones$ Battlefield"
        )
        reminder = "This creature can't be blocked except by creatures with it."
        keywords.write_text(
            json.dumps({
                "Flying": {
                    "reminder_template": reminder,
                    "generated_script": script,
                },
                "Shadow": {
                    "reminder_template": reminder,
                    "generated_script": script,
                },
            }),
            encoding="utf-8",
        )
        return cards, tokens, keywords

    def test_prose_does_not_take_script_parameter_names(self, scripted, tmp_path):
        config = _config(scripted, tmp_path, target_size=5000)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "activezones" not in vocab
        assert "cantblockby" not in vocab
        # The reminder text is still scanned: it is what prose expands to.
        assert "blocked" in vocab

    def test_the_script_surface_takes_both(self, scripted, tmp_path):
        # Its own file, and the engine-coded keywords generate no script and
        # keep their template — so the template is scanned on both surfaces.
        config = _config(
            scripted, tmp_path, surface=SURFACE_SCRIPT,
            vocab_path=tmp_path / "vocab-script.txt", target_size=5000,
        )
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "activezones" in vocab
        assert "blocked" in vocab


class TestSeededSpecials:
    def test_the_five_specials_are_all_present(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        for token in SEEDED_SPECIALS:
            assert token in vocab, f"{token} was not seeded"

    def test_pad_and_unk_keep_the_ids_every_reader_assumes(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert vocab["[PAD]"] == 0
        assert vocab["[UNK]"] == 1

    def test_cls_is_seeded_ahead_of_the_corpus_tokens(self, corpus, tmp_path):
        """It holds a low id so a --target-size truncation cannot drop it."""
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert vocab["[CLS]"] < 8

    def test_the_specials_survive_an_aggressive_target_size(self, corpus, tmp_path):
        config = _config(corpus, tmp_path, target_size=80)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        for token in SEEDED_SPECIALS:
            assert token in vocab


class TestScanSources:
    def test_card_text_reaches_the_vocabulary(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "angel" in vocab

    def test_token_script_text_reaches_it_too(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "soldier" in vocab

    def test_keyword_reminder_text_reaches_it(self, corpus, tmp_path):
        """Expansion substitutes reminder text in; unscanned, it would be [UNK]."""
        config = _config(corpus, tmp_path)
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "blocked" in vocab
        assert "reach" in vocab

    def test_a_missing_keyword_file_warns_rather_than_failing(
        self, corpus, tmp_path, caplog,
    ):
        config = _config(corpus, tmp_path, keyword_definitions=tmp_path / "absent.json")
        with caplog.at_level("WARNING"):
            assert run(config) > 0
        assert "keyword-definition" in caplog.text

    def test_an_empty_corpus_fails_with_the_command_to_run(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        config = BuildVocabConfig(
            cards_folders=(empty,), vocab_path=tmp_path / "vocab.txt",
        )
        with pytest.raises(EmptyCardsFolderError, match="convert"):
            run(config)

    def test_an_absent_folder_is_skipped_rather_than_fatal(self, corpus, tmp_path):
        cards, _tokens, keywords = corpus
        config = _config(
            corpus, tmp_path, cards_folders=(cards, tmp_path / "never-converted"),
        )
        assert run(config) > 0


class TestTargetSize:
    def test_target_size_caps_the_vocabulary(self, corpus, tmp_path):
        config = _config(corpus, tmp_path, target_size=90)
        size = run(config)
        assert size <= 90
        assert len(load_vocabulary(config.vocab_path)) == size

    def test_an_uncapped_build_keeps_everything(self, corpus, tmp_path):
        big = run(_config(corpus, tmp_path, target_size=10_000))
        small = run(_config(corpus, tmp_path, target_size=90))
        assert big > small

    def test_a_target_below_the_seeded_count_is_rejected(self, corpus, tmp_path):
        with pytest.raises(ValueError, match="seeded"):
            run(_config(corpus, tmp_path, target_size=3))


class TestSurfaces:
    def test_the_two_surfaces_default_to_different_files(self):
        assert DEFAULT_VOCAB_PATH != DEFAULT_SCRIPT_VOCAB_PATH
        assert BuildVocabConfig().resolved_vocab_path() == DEFAULT_VOCAB_PATH
        assert BuildVocabConfig(
            surface=SURFACE_SCRIPT,
        ).resolved_vocab_path() == DEFAULT_SCRIPT_VOCAB_PATH

    def test_an_explicit_path_overrides_the_surface_default(self, tmp_path):
        chosen = tmp_path / "elsewhere.txt"
        assert BuildVocabConfig(
            surface=SURFACE_SCRIPT, vocab_path=chosen,
        ).resolved_vocab_path() == chosen

    def test_an_unknown_surface_is_rejected(self, corpus, tmp_path):
        with pytest.raises(ValueError, match="--surface"):
            run(_config(corpus, tmp_path, surface="oracle"))

    def test_the_script_surface_scans_the_sidecars_script_lines(
        self, corpus, tmp_path,
    ):
        cards, _tokens, _keywords = corpus
        for stem, card_name in (("s/serra_angel", "serra angel"),
                                ("a/archangel", "archangel")):
            script_file = f"cardsfolder/{stem}.txt"
            write_sidecar(
                ProvenanceSidecar(
                    card=card_name,
                    script_file=script_file,
                    lines=(
                        SidecarLine(
                            line_index=4, line_kind="static",
                            provenance=(
                                ProvenanceKey(script_file, 0, "static", 0),
                            ),
                            script_api_type="PutCounter",
                            script_text="Mode$ Continuous | AddKeyword$ Flying",
                        ),
                    ),
                ),
                sidecar_path_for(cards / f"{stem}.txt"),
            )
        config = _config(
            corpus, tmp_path, surface=SURFACE_SCRIPT,
            vocab_path=tmp_path / "vocab-script.txt",
        )
        run(config)
        vocab = load_vocabulary(config.vocab_path)
        assert "addkeyword" in vocab
        assert "putcounter" in vocab

    def test_building_the_script_surface_leaves_the_prose_file_untouched(
        self, corpus, tmp_path,
    ):
        prose_path = tmp_path / "vocab.txt"
        run(_config(corpus, tmp_path, vocab_path=prose_path))
        before = prose_path.read_bytes()
        run(_config(
            corpus, tmp_path, surface=SURFACE_SCRIPT,
            vocab_path=tmp_path / "vocab-script.txt",
        ))
        assert prose_path.read_bytes() == before


class TestOutput:
    def test_the_parent_directory_is_created(self, corpus, tmp_path):
        config = _config(corpus, tmp_path, vocab_path=tmp_path / "deep" / "v.txt")
        run(config)
        assert Path(config.vocab_path).exists()

    def test_ids_are_dense_and_start_at_zero(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        size = run(config)
        assert sorted(load_vocabulary(config.vocab_path).values()) == list(range(size))
