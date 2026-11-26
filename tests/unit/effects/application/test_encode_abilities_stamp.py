"""The ability cache is tied to the checkpoint it was encoded from (finding 11.7).

``encode-abilities`` writes a stamp beside each variant's cache (the
checkpoint's content hash, its encoding surface, the vocabulary hash and the
converted trees' sidecar digest); every reader compares it with the
checkpoint it evaluates and refuses a cache stamped for another, or with no
stamp. Without it, two arms sharing a cache directory give a gate-3 verdict on
the other arm's vectors.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from effects.application.encode_abilities import UnencodedVariantError, run
from effects.domain.ability_cache_layout import (
    ARRAY_KEY,
    CacheStamp,
    CacheStampError,
    cache_path_for,
    stamp_path,
)
from effects.domain.baselines import Baseline
from effects.infrastructure.ability_cache_store import (
    AbilityCacheStore,
    cache_stamp,
    read_stamp,
    require_cache_stamp,
    write_stamp,
)
from effects.infrastructure.converted_trees import SourceTrees
from effects.infrastructure.effect_model_store import content_hash, load_effect_checkpoint
from tests.unit.effects.application import test_encode_abilities as _encoding
from tests.unit.effects.application.test_encode_abilities import _config

#: The encoding tests' trees and checkpoint.
corpus = _encoding.corpus


def _encoded(corpus, root: Path, variant: Baseline) -> None:
    """A cache of ``variant`` written and stamped by hand for the fixture's
    checkpoint, which has no encoder worth running."""
    AbilityCacheStore(root, variant=variant).write(
        "cardsfolder/s/serra_angel.txt", np.zeros((3, 8)),
    )
    write_stamp(root, _wanted(corpus), variant=variant)


def _wanted(corpus) -> CacheStamp:
    cards, tokens, checkpoint, _vocab = corpus
    return cache_stamp(checkpoint, load_effect_checkpoint(checkpoint),
                       cards_folders=(cards, tokens), surface="prose")


class TestTheStampWritten:
    def test_it_names_the_checkpoint_surface_vocabulary_and_trees(self, corpus, tmp_path):
        cards, tokens, checkpoint, vocab = corpus
        config = _config(corpus, tmp_path)
        run(config)
        stamp = read_stamp(config.output_root, Baseline.TAXONOMY)
        assert stamp == CacheStamp(
            checkpoint_hash=content_hash(checkpoint), surface="prose",
            vocab_hash=content_hash(vocab),
            sidecar_digest=SourceTrees.of((cards, tokens)).converted_sidecar_digest(),
        )

    def test_a_run_cut_short_leaves_no_stamp(self, corpus, tmp_path, monkeypatch):
        """The stamp goes before the first file is written and comes back after
        the last, so a partial cache is never read as a whole one."""
        config = _config(corpus, tmp_path)
        run(config)

        def cut(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(AbilityCacheStore, "write", cut)
        with pytest.raises(KeyboardInterrupt):
            run(config)
        assert not stamp_path(config.output_root, Baseline.TAXONOMY).exists()


class TestTheVariantsTheEncoderDoesNotFeed:
    @pytest.mark.parametrize("variant", [Baseline.IDENTITY, Baseline.STATE_ONLY])
    def test_it_is_refused_before_anything_is_written(self, corpus, tmp_path, variant):
        with pytest.raises(UnencodedVariantError, match=f"--variant {variant}"):
            run(_config(corpus, tmp_path, variant=variant))
        assert not (tmp_path / "abilities").exists()


class TestAReader:
    def test_a_cache_stamped_for_another_checkpoint_is_refused_naming_both_hashes(
        self, corpus, tmp_path,
    ):
        config = _config(corpus, tmp_path)
        run(config)
        wanted = _wanted(corpus)
        other = CacheStamp(**{**wanted.as_dict(), "checkpoint_hash": "0" * 64})
        with pytest.raises(CacheStampError) as excinfo:
            require_cache_stamp(config.output_root, other, variant=Baseline.TAXONOMY)
        assert wanted.checkpoint_hash in str(excinfo.value)
        assert "0" * 64 in str(excinfo.value)

    def test_the_cache_of_its_own_checkpoint_is_read(self, corpus, tmp_path):
        config = _config(corpus, tmp_path)
        run(config)
        store = require_cache_stamp(
            config.output_root, _wanted(corpus), variant=Baseline.TAXONOMY,
        )
        assert store.read("cardsfolder/s/serra_angel.txt").shape[0] == 3

    def test_a_cache_with_no_stamp_is_refused(self, tmp_path):
        shipping = cache_path_for("cardsfolder/s/serra_angel.txt", root=tmp_path)
        shipping.parent.mkdir(parents=True)
        np.savez_compressed(shipping, **{ARRAY_KEY: np.zeros((3, 8))})
        wanted = CacheStamp("c", "prose", "v", "s")
        with pytest.raises(CacheStampError, match="no stamp"):
            require_cache_stamp(tmp_path, wanted)

    def test_a_reader_holding_no_checkpoint_of_the_variant_is_refused(self, corpus, tmp_path):
        config = _config(corpus, tmp_path, variant=Baseline.NO_STATE)
        AbilityCacheStore(config.output_root, variant=Baseline.NO_STATE).write(
            "cardsfolder/s/serra_angel.txt", np.zeros((3, 8)),
        )
        with pytest.raises(CacheStampError, match="no-state=PATH"):
            require_cache_stamp(config.output_root, None, variant=Baseline.NO_STATE)

    def test_no_cache_needs_no_stamp(self, tmp_path):
        store = require_cache_stamp(tmp_path / "none", None, variant=Baseline.NO_STATE)
        assert store.written_files() == []


class TestTheEvaluatorsReads:
    """``evaluate-effect-model`` reads each variant's cache through its stamp."""

    def _context(self, corpus, tmp_path, *, checkpoint=None):
        cards, tokens, main_path, vocab = corpus
        main = load_effect_checkpoint(main_path)
        return SimpleNamespace(
            config=SimpleNamespace(
                checkpoint=checkpoint or main_path, abilities_root=tmp_path / "abilities",
                baseline_checkpoints={}, cards_folders=(cards, tokens),
            ),
            main=main, tokenizer_files=SimpleNamespace(vocab=vocab),
            manifest=SimpleNamespace(
                sidecar_digest=SourceTrees.of((cards, tokens)).converted_sidecar_digest(),
            ),
        ), main

    def test_a_cache_another_checkpoint_wrote_is_refused(self, corpus, tmp_path):
        from effects.application.evaluate_effect_model import cache_store

        _encoded(corpus, tmp_path / "abilities", Baseline.FULL)
        other = tmp_path / "other.pt"
        other.write_bytes(Path(corpus[2]).read_bytes() + b"another arm")
        context, _main = self._context(corpus, tmp_path, checkpoint=other)
        with pytest.raises(CacheStampError, match=content_hash(other)):
            cache_store(context, Baseline.FULL, {})

    def test_the_taxonomy_cache_is_read_without_its_checkpoint(self, corpus, tmp_path):
        """No weights enter it: the rest of the stamp is still compared."""
        from effects.application.evaluate_effect_model import cache_store

        run(_config(corpus, tmp_path))
        context, _main = self._context(corpus, tmp_path)
        assert cache_store(context, Baseline.TAXONOMY, {}).written_files()

    def test_a_baseline_cache_is_compared_with_its_own_checkpoint(self, corpus, tmp_path):
        from effects.application.evaluate_effect_model import cache_store

        _encoded(corpus, tmp_path / "abilities", Baseline.NO_STATE)
        context, main = self._context(corpus, tmp_path)
        context.config.baseline_checkpoints = {Baseline.NO_STATE: corpus[2]}
        store = cache_store(context, Baseline.NO_STATE, {Baseline.NO_STATE: main})
        assert store.written_files()
