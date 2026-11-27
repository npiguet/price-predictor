"""The build refuses what it cannot build faithfully, and verify reports drift."""

from __future__ import annotations

import pytest

from effects.application.build_corpus import build
from effects.domain.corpus_plan import BuildCorpusError
from effects.domain.split import HoldoutRule
from tests.unit.effects.application.build_corpus_support import (
    _BOLT_KEY,
    _BOLT_TEXT,
    _BUDGET_FREE_RULES,
    _dataset,
    _resolution,
)


def test_verify_reports_drift_and_writes_nothing(tmp_path, a_corpus):
    out = tmp_path / "curated"
    a_corpus.build(out)
    before = (out / "manifest.json").read_text(encoding="utf-8")

    (a_corpus.records / "extra.0-zz.jsonl.gz").write_bytes(b"")
    code = build(a_corpus.config(out, verify=True))

    assert code == 1
    assert (out / "manifest.json").read_text(encoding="utf-8") == before


def test_verify_is_quiet_when_nothing_moved(tmp_path, a_corpus):
    out = tmp_path / "curated"
    a_corpus.build(out)
    a_corpus.build(out, verify=True)


# ── a dataset that cannot score gate 1 is refused, not written ────────


def test_build_refuses_a_cards_folder_that_resolved_to_nothing(
    tmp_path, a_corpus, monkeypatch,
):
    """Launched from the wrong directory, the relative --cards-folder paths
    miss, the holdout selects nothing, and the whole card-disjoint stratum —
    the entire basis of gate 1's margins — comes out empty. The builder used
    to log that as an INFO line, pay for two full parallel passes, write the
    dataset and exit 0; the trainer then refuses it in its first minute.

    ``run_survey`` is replaced with a bomb, so this also pins that the refusal
    comes *before* the survey rather than after it.
    """
    from effects.application import build_corpus as module

    def no_survey(*args, **kwargs):
        raise AssertionError("the survey must not run on an empty holdout")

    monkeypatch.setattr(module, "run_survey", no_survey)

    with pytest.raises(ValueError, match="no held-out card") as raised:
        a_corpus.build(
            tmp_path / "curated",
            cards_folders=(tmp_path / "not-here", tmp_path / "nor-here"),
        )

    message = str(raised.value)
    assert "not-here" in message and "nor-here" in message
    assert "missing" in message
    assert "0 converted card" in message


def test_build_refuses_a_holdout_that_selects_no_card(tmp_path, a_corpus):
    """The same unusable dataset by the other route: the cards folder is real
    and full, and ``--holdout-permille 0`` holds nothing out of it. Refusing
    only on an empty folder would let this one through.
    """
    with pytest.raises(ValueError, match="no held-out card") as raised:
        a_corpus.build(
            tmp_path / "curated", holdout=HoldoutRule(permille=0),
        )

    assert "4 converted card" in str(raised.value)


def test_build_refuses_a_corpus_no_game_of_which_names_a_held_out_card(
    tmp_path, a_corpus,
):
    """A perfectly good holdout against shards that hold none of it.

    This repo keeps depleted and full-strength shards in sibling
    subdirectories, so a ``--records-dir`` pointed one level too deep surveys
    only the depleted half: every game is clean, the card-disjoint stratum is
    empty, and the dataset cannot score gate 1 at all. Nothing is written.
    """
    import shutil

    depleted = tmp_path / "depleted-only"
    depleted.mkdir()
    shutil.copy(
        a_corpus.records / "depleted" / "run.0-b.jsonl.gz",
        depleted / "run.0-b.jsonl.gz",
    )
    out = tmp_path / "curated"

    with pytest.raises(ValueError, match="card-disjoint"):
        a_corpus.build(out, records_dir=depleted)

    assert not (out / "manifest.json").exists()


def test_build_refuses_an_empty_corpus(tmp_path, a_corpus):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no shards"):
        a_corpus.build(tmp_path / "out", records_dir=empty)


# ── a build stops before it clears a dataset it could not replace ─────


def test_a_build_with_no_rules_catalogue_refuses_before_clearing_the_dataset(
    tmp_path, a_corpus,
):
    """A rules directory with no scenario or no heads line would tag and budget
    nothing and only fail at the precision exit, after the dataset was
    replaced: it is refused before the survey, the earlier dataset intact."""
    out = tmp_path / "curated"
    a_corpus.build(out)
    before = _dataset(out)
    with pytest.raises(BuildCorpusError, match="rules catalogue"):
        build(a_corpus.config(out, rules_dir=tmp_path / "nowhere"))
    assert _dataset(out) == before


def test_a_display_name_match_on_a_non_keyword_line_fails_the_build(tmp_path, a_corpus):
    """Spec Story 4 scenario 8 / FR-015."""
    import json

    definitions = tmp_path / "keyword-definitions.json"
    definitions.write_text(json.dumps({
        _BOLT_TEXT.upper(): {"reminder_template": "a reminder"},
    }), encoding="utf-8")

    with pytest.raises(BuildCorpusError, match="FR-015"):
        a_corpus.build(tmp_path / "out", rules=_BUDGET_FREE_RULES, keyword_definitions=definitions)


def test_a_keyword_definitions_path_with_no_file_is_refused_naming_it(tmp_path, a_corpus):
    """A typo is not "no table": its manifest digest would be empty and the
    trainer would then accept any keyword file."""
    missing = tmp_path / "no-such-keywords.json"
    with pytest.raises(FileNotFoundError, match="no-such-keywords.json"):
        a_corpus.build(tmp_path / "out", keyword_definitions=missing)
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_a_gen1_shard_among_gen3_ones_is_refused_naming_it(tmp_path, a_corpus):
    """FR-033: before any record is read."""
    import gzip
    import json

    from effects.infrastructure.record_codec import record_to_dict

    gen1 = record_to_dict(_resolution("old.1", "g-old", ability=(_BOLT_KEY,)))
    del gen1["random_seat"], gen1["generation"]
    path = a_corpus.records / "full-strength" / "old.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(gen1) + "\n")

    with pytest.raises(BuildCorpusError, match=r"old\.jsonl\.gz, line 1: the record is gen-1 "):
        a_corpus.build(tmp_path / "out", rules=_BUDGET_FREE_RULES)
