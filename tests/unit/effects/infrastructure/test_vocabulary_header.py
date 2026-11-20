"""The effects vocabulary file records its surface in a header (finding 1.5).

The surface was read from the file's name, so a script vocabulary renamed or
passed under another name tokenized every line on the prose surface. A file
without the header is refused rather than read as either surface.
"""

from __future__ import annotations

import json

import pytest

from effects.application.build_vocab import BuildVocabConfig, run
from effects.domain.ability_tokenizer import Surface
from effects.domain.tokenizer_files import TokenizerFiles
from effects.infrastructure.tokenizer_loading import load_tokenizer
from price_predictor.infrastructure.tokenizer_store import load_vocabulary


@pytest.fixture
def corpus(tmp_path):
    cards = tmp_path / "cardsfolder" / "s"
    cards.mkdir(parents=True)
    for name in ("serra_angel", "suntail_hawk"):
        (cards / f"{name}.txt").write_text(
            f"name: {name}\ntypes: creature bird\nstatic: flying\n", encoding="utf-8")
    keywords = tmp_path / "keyword-definitions.json"
    keywords.write_text(json.dumps({}), encoding="utf-8")
    return tmp_path / "cardsfolder", keywords


def _build(corpus, path, surface: Surface):
    cards, keywords = corpus
    run(BuildVocabConfig(cards_folders=(cards,), vocab_path=path,
                         keyword_definitions=keywords, surface=surface))
    return TokenizerFiles.of(path, keywords)


class TestTheSurfaceIsReadFromTheFile:
    @pytest.mark.parametrize("surface", list(Surface))
    def test_whatever_the_file_is_named(self, corpus, tmp_path, surface):
        files = _build(corpus, tmp_path / "renamed.txt", surface)
        assert load_tokenizer(files).surface == surface, (
            f"a {surface} vocabulary named renamed.txt loads as another surface"
        )

    def test_a_script_name_does_not_make_a_prose_vocabulary_script(self, corpus, tmp_path):
        files = _build(corpus, tmp_path / "vocab-script.txt", Surface.PROSE)
        assert load_tokenizer(files).surface == Surface.PROSE


class TestAFileWithoutTheHeaderIsRefused:
    def test_the_refusal_names_the_file(self, corpus, tmp_path):
        _cards, keywords = corpus
        bare = tmp_path / "vocab.txt"
        bare.write_text("[PAD]\n[UNK]\ncardname\n[MASK]\n[CLS]\n[SEG]", encoding="utf-8")
        with pytest.raises(ValueError, match=r"vocab\.txt.*surface"):
            load_tokenizer(TokenizerFiles.of(bare, keywords))


class TestTheSharedLoaderDoesNotReadIt:
    def test_the_price_and_sealed_loader_refuses_an_effects_vocabulary(self, corpus, tmp_path):
        """The header takes line 0, where ``[PAD]`` must be for the shared
        loader: the two vocabularies cannot be read one for the other."""
        files = _build(corpus, tmp_path / "vocab.txt", Surface.PROSE)
        with pytest.raises(ValueError, match=r"\[PAD\]"):
            load_vocabulary(files.vocab)
