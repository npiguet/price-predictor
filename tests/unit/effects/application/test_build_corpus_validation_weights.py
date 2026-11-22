"""A validation stratum is written as the training stratum is (D9 (b), FR-059a,
FR-068a): each record's natural copy once, plus the copies the rule budgets
plan for it, up to the reuse cap; the class mix is a weight each row carries
(``curation.class_weight``), never a repetition. A protection copy counts
toward its drawn rule alone (finding 14.2)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from effects.application.build_corpus import build
from effects.domain.sampling import DEFAULT_KIND_MIX, sampling_class
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import read_records
from tests.unit.effects.application.test_build_corpus import (
    _NO_RULES,
    _gen3_config,
    _gen3_records,
)

REUSE_CAP = 4
#: The real catalogue, whose first-strike scenario one test copies.
_RULES = Path(__file__).parents[4] / "rules"


@pytest.fixture(scope="module")
def weighted_build(tmp_path_factory) -> CorpusStore:
    """The gen-3 fixture under the real catalogue at the default reuse cap."""
    root = tmp_path_factory.mktemp("weighted-build")
    assert build(_gen3_config(_gen3_records(root), root / "out", reuse_cap=REUSE_CAP)) == 0
    return CorpusStore(root / "out")


def _validation(store: CorpusStore):
    return {"card-disjoint": list(read_records(store.card_disjoint_dir)),
            "game-disjoint": list(read_records(store.game_disjoint_dir))}


def _rules_dir(root: Path, scenario: Path, heads: list[dict]) -> Path:
    """A rules directory holding one scenario file and one heads file."""
    rules = root / "rules"
    (rules / "scenarios").mkdir(parents=True)
    (rules / "heads").mkdir()
    (rules / "scenarios" / scenario.name).write_bytes(scenario.read_bytes())
    (rules / "heads" / "heads.jsonl").write_text(
        "".join(json.dumps(head) + "\n" for head in heads), encoding="utf-8")
    return rules


def _weighted_natural(rows) -> dict[str, float]:
    """``class -> summed class_weight`` of the natural copies."""
    weighted: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        if not row.curation.is_added:
            weighted[str(sampling_class(row))] += row.curation.class_weight
    return weighted


def _added_per_rule(by_class) -> Counter[str]:
    """``rule -> added copies`` over one stratum's manifest rows."""
    added: Counter[str] = Counter()
    for rows in by_class.values():
        added.update({rule: sum(sum(row.added.values()) for row in by_polarity.values())
                      for rule, by_polarity in rows.items()})
    return added


def test_a_validation_record_is_written_once_and_its_copies_stay_under_the_reuse_cap(
        weighted_build):
    """Each record id holds one natural copy, natural share 1, and its added
    copies; none is written more often than the reuse cap."""
    for name, rows in _validation(weighted_build).items():
        assert rows, name
        ids = Counter(r.record_id for r in rows)
        assert max(ids.values()) <= REUSE_CAP, name
        natural = Counter(r.record_id for r in rows if not r.curation.is_added)
        assert set(natural.values()) == {1}, f"{name}: a natural copy repeated"
        assert {r.curation.natural_share for r in rows if not r.synthetic} == {1}, name
        assert all(r.synthetic or r.curation.copies == ids[r.record_id] for r in rows), name


def test_a_training_row_weighs_1(weighted_build):
    rows = list(read_records(weighted_build.training_dir))
    assert rows and {r.curation.class_weight for r in rows} == {1.0}


def test_a_class_weighed_4_carries_its_weight_and_takes_added_copies(tmp_path):
    """Finding 14.2: a class the mix weighs at the reuse cap was written four
    times a record, which left a short rule no room for a copy. Here the mix
    asks far more combat than the fixture holds, so the card-disjoint
    stratum's combat class weighs 4, and first strike (702.7b, tagged on one
    of its records) is the one trained rule, short of its half: each combat
    row weighs 4 and the first-strike record takes added copies for it, up
    to the reuse cap."""
    rules = _rules_dir(tmp_path, _RULES / "scenarios" / "702.7.json", [
        {"id": "702.7b", "output": "trained", "outputs": ["damage_taken", "zone_outcome"]}])
    mix = {**{str(k): v for k, v in DEFAULT_KIND_MIX.items()}, "combat": 10.0}
    output = tmp_path / "out"
    assert build(_gen3_config(_gen3_records(tmp_path), output, rules_dir=rules,
                              reuse_cap=REUSE_CAP, class_mix=mix)) == 0
    rows = [r for r in _validation(CorpusStore(output))["card-disjoint"]
            if str(sampling_class(r)) == "combat"]
    assert rows and {r.curation.class_weight for r in rows} == {4.0}
    drawn = Counter(r.curation.drawn_rule for r in rows if r.curation.is_added)
    assert set(drawn) == {"702.7b"}
    assert max(Counter(r.record_id for r in rows).values()) == REUSE_CAP


def test_the_weighted_class_shares_of_a_validation_stratum_are_the_class_mix(tmp_path):
    """FR-059a: each record's natural copy weighed by its ``class_weight``
    gives the stratum the training stratum's class mix over the classes it
    holds. The reuse cap is lifted so no class's weight is capped on the
    fixture's few records, and the game-disjoint share raised so that stratum
    draws games enough to hold several classes."""
    output = tmp_path / "out"
    assert build(_gen3_config(_gen3_records(tmp_path), output, reuse_cap=100,
                              game_disjoint_share=0.1)) == 0
    for name, rows in _validation(CorpusStore(output)).items():
        weighted = _weighted_natural(rows)
        mixed = {str(k): share for k, share in DEFAULT_KIND_MIX.items() if weighted.get(str(k))}
        assert len(mixed) >= 3, "the fixture must hold several classes or this proves nothing"
        total = sum(weighted[k] for k in mixed)
        total_share = sum(mixed.values())
        shares = {k: weighted[k] / total for k in mixed}
        assert shares == pytest.approx(
            {k: share / total_share for k, share in mixed.items()}, rel=1e-9), name


def test_the_manifest_adds_to_a_protection_rule_the_copies_written_for_it(tmp_path):
    """A record offering several protection rules enters the budgets once;
    a copy drawn for one of them adds to that rule's row alone, so each
    rule's ``added`` is the protection copies written for it (702.16j's
    protection from everything is offered beside 702.16e and 702.16f here,
    and before counted every draw)."""
    from effects.domain.protection_copies import PROTECTION_RULES

    # A scenario no record can tag: a build refuses a catalogue with none.
    rules = _rules_dir(tmp_path, _NO_RULES / "scenarios" / "103.json", [
        {"id": rule, "output": "trained", "outputs": []} for rule in sorted(PROTECTION_RULES)])
    assert build(_gen3_config(_gen3_records(tmp_path), tmp_path / "out", rules_dir=rules)) == 0
    store = CorpusStore(tmp_path / "out")
    manifest = store.load()
    planned = {"training": manifest.rules, **manifest.validation_rules}
    directories = {"training": store.training_dir, "card-disjoint": store.card_disjoint_dir,
                   "game-disjoint": store.game_disjoint_dir}
    written = {name: Counter(r.curation.drawn_rule for r in read_records(d) if r.synthetic)
               for name, d in directories.items()}
    assert any(written.values()), "the fixture draws no protection copy: this proves nothing"
    for name in directories:
        assert +_added_per_rule(planned[name]) == written[name], name
