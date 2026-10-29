"""The evaluation context's fail-fast rows: the corpus a checkpoint trained on,
and the baseline checkpoints named beside it.

Each is a case where continuing would score the gates partly on games the model
trained on, or against a baseline that never ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.evaluation_context import parse_baseline_checkpoint
from effects.infrastructure.effect_model_store import (
    SplitProvenance,
)


class TestCorpusMismatch:
    """``evaluate-effect-model`` refuses a corpus rebuilt since training
    (FR-147) — the same discipline as the vocabulary-hash guard, for the
    dataset rather than the vocabulary file."""

    def test_evaluate_refuses_a_corpus_rebuilt_since_training(self, tmp_path):
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError, match="rebuilt"):
            check_corpus(provenance, actual_digest="something-else")

    def test_matching_digests_pass(self, tmp_path):
        from effects.application.evaluation_context import check_corpus

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="same",
        )
        check_corpus(provenance, actual_digest="same")  # no raise

    def test_the_error_names_the_corpus_path_and_both_digests(self, tmp_path):
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            check_corpus,
        )

        provenance = SplitProvenance(
            corpus_path=str(tmp_path), corpus_digest="trained-against-this",
        )
        with pytest.raises(CorpusMismatchError) as excinfo:
            check_corpus(provenance, actual_digest="something-else")
        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert "trained-against-this" in message
        assert "something-else" in message


class TestCorpusPathResolution:
    """``--corpus`` on ``evaluate-effect-model`` defaults to what the
    checkpoint recorded, an explicit value overriding it (FR-147), and a
    checkpoint that recorded no corpus is refused."""

    _CHECKPOINT = Path("models/effects/effect-model/latest.pt")

    def test_defaults_to_the_checkpoints_recorded_path(self):
        from effects.application.evaluation_context import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=None, checkpoint=self._CHECKPOINT,
        ) == Path("output/effects/corpus")

    def test_an_explicit_corpus_overrides(self, tmp_path):
        from effects.application.evaluation_context import resolve_corpus_path

        provenance = SplitProvenance(
            corpus_path="output/effects/corpus", corpus_digest="abc123",
        )
        assert resolve_corpus_path(
            provenance, corpus=tmp_path, checkpoint=self._CHECKPOINT,
        ) == tmp_path

    def test_a_checkpoint_that_recorded_no_corpus_is_refused(self, tmp_path):
        """Nothing else says which games it trained on, --corpus or not."""
        from effects.application.evaluation_context import (
            CorpusMismatchError,
            resolve_corpus_path,
        )

        for corpus in (None, tmp_path):
            with pytest.raises(CorpusMismatchError, match="records no curated corpus"):
                resolve_corpus_path(
                    SplitProvenance(), corpus=corpus, checkpoint=self._CHECKPOINT,
                )


class TestVariantCheckpointFlag:
    def test_it_parses_name_equals_path(self):
        name, path = parse_baseline_checkpoint("identity=models/x/latest.pt")
        assert name == "identity"
        assert path.name == "latest.pt"

    def test_a_bare_path_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("models/x/latest.pt")

    def test_an_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="NAME=PATH"):
            parse_baseline_checkpoint("=models/x/latest.pt")

    def test_a_misspelt_baseline_is_rejected(self):
        """Otherwise gate 1 skips for want of an identity checkpoint."""
        with pytest.raises(ValueError, match="unknown baseline 'identiy'"):
            parse_baseline_checkpoint("identiy=models/x/latest.pt")

    def test_the_shipping_model_is_not_a_baseline(self):
        with pytest.raises(ValueError, match="unknown baseline 'full'"):
            parse_baseline_checkpoint("full=models/x/latest.pt")
