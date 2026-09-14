"""The write pass and ``build()``: turn a survey and its decisions into a
fixed training corpus and two fixed validation strata on disk.

``a_corpus`` is the load-bearing fixture. It builds a small but real raw
corpus: a converted card tree with sidecars (a held-out card, plus two
ordinary ones), and two raw shards covering four games —

- ``g-tainted`` carries an entity named after the held-out card *and* a
  resolvable acting ability, so the survey marks it held-out *and* admits it
  to the card-disjoint stratum (FR-142's cap needs at least one admissible
  text to ever let a held-out game in); it also carries an observed ``combat``
  record and the probe record whose ``mirror_of`` names it, so the pair is
  guaranteed to land together in validation rather than possibly being
  scattered by the game-disjoint sample.
- ``g-clean-1`` and ``g-clean-2`` each repeat one ability three times (six
  copies total) — under ``--text-cap 2`` the survey's ``CapHeap`` admits only
  two of the six *no matter which game each copy sits in or which one the
  seeded game-disjoint draw removes*, so at least one training-eligible copy
  is always cap-dropped.
- ``g-clean-3`` is a small filler so each raw shard holds more than one game.

The held-out text itself is not asserted by inspection — it is *constructed*
by searching for a string that ``text_is_held_out`` actually selects at the
default ``--holdout-permille 20``, per the task brief.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from effects.application.build_corpus import BuildCorpusConfig, _output_name, build
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.record_io import read_records, write_shard
from effects.infrastructure.sidecar_io import sidecar_path_for, write_sidecar

_HELD_OUT_CARD = "Held Out Bears"
#: Found by brute force over ``text_is_held_out(..., permille=20)`` (the
#: default), the way the brief asks: constructed, not guessed at.
_HELD_OUT_TEXT = "deals 10 damage to any target held out effect"
_HELD_OUT_FILE = "cardsfolder/h/held_out_bears.txt"
_HELD_OUT_KEY = ProvenanceKey(_HELD_OUT_FILE, 0, "spell", 0)

_BOLT_CARD = "Common Bolt"
_BOLT_TEXT = "deals 3 damage to any target"
_BOLT_FILE = "cardsfolder/c/common_bolt.txt"
_BOLT_KEY = ProvenanceKey(_BOLT_FILE, 0, "spell", 0)

_SHOCK_CARD = "Common Shock"
_SHOCK_TEXT = "deals 2 damage to any target"
_SHOCK_FILE = "cardsfolder/c/common_shock.txt"
_SHOCK_KEY = ProvenanceKey(_SHOCK_FILE, 0, "spell", 0)


def _write_card(root: Path, script_file: str, card: str, text: str) -> None:
    """A minimal converted ``.txt`` plus a sidecar whose one line carries ``text``.

    ``line_index=999`` is deliberately past the converted file's last line, so
    ``prose_for`` always misses and ``text_for_key`` falls back to
    ``script_text`` — what matters for these tests is that every record
    sharing one provenance key resolves to the same text, not which surface
    supplied it.
    """
    relative = script_file.split("/", 1)[1]
    txt_path = root / "cardsfolder" / relative
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(f"name: {card}\ntypes: creature\n", encoding="utf-8")
    key = ProvenanceKey(script_file, 0, "spell", 0)
    write_sidecar(
        ProvenanceSidecar(
            card=card,
            script_file=script_file,
            lines=(
                SidecarLine(
                    line_index=999, line_kind="spell", provenance=(key,),
                    script_text=text,
                ),
            ),
        ),
        sidecar_path_for(txt_path),
    )


def _snapshot(entity_names: tuple[str, ...] = ()) -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        players=(),
        entities=tuple(
            EntityState(id=f"E{index}", name=name, zone="battlefield", controller="P0")
            for index, name in enumerate(entity_names)
        ),
    )


def _resolution(
    record_id: str,
    game_id: str,
    *,
    ability: tuple[ProvenanceKey, ...],
    entity_names: tuple[str, ...] = (),
) -> EffectRecord:
    return EffectRecord(
        record_id=record_id, run_id="run", timestamp="2026-09-14T00:00:00.000000Z",
        game_id=game_id, kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION,
        actor_player="P0", ability=ability, state=_snapshot(entity_names),
        payload=ResolutionPayload(),
    )


def _combat(
    record_id: str,
    game_id: str,
    *,
    fork: bool = False,
    mirror_of: str | None = None,
    probed_keyword: str | None = None,
    probed_entity: str | None = None,
) -> EffectRecord:
    return EffectRecord(
        record_id=record_id, run_id="run", timestamp="2026-09-14T00:00:00.000000Z",
        game_id=game_id, kind=RecordKind.COMBAT, actor_player="P0",
        fork=fork, mirror_of=mirror_of, state=_snapshot(),
        payload=CombatPayload(
            attackers=("E0",), probed_keyword=probed_keyword,
            probed_entity=probed_entity,
        ),
    )


def _rewrite(
    record_id: str, game_id: str, *, ability: tuple[ProvenanceKey, ...],
) -> EffectRecord:
    return EffectRecord(
        record_id=record_id, run_id="run", timestamp="2026-09-14T00:00:00.000000Z",
        game_id=game_id, kind=RecordKind.REWRITE, actor_player="P0",
        ability=ability, state=_snapshot(),
        payload=RewritePayload(incoming=Event(type=EventType.ZONE_CHANGE)),
    )


@dataclass(frozen=True, slots=True)
class CorpusFixture:
    """A small real raw corpus: the raw shard root and the converted trees."""

    records: Path
    cards: tuple[str, ...]


@pytest.fixture
def a_corpus(tmp_path: Path) -> CorpusFixture:
    root = tmp_path / "source"
    _write_card(root, _HELD_OUT_FILE, _HELD_OUT_CARD, _HELD_OUT_TEXT)
    _write_card(root, _BOLT_FILE, _BOLT_CARD, _BOLT_TEXT)
    _write_card(root, _SHOCK_FILE, _SHOCK_CARD, _SHOCK_TEXT)

    shard_a = [
        # Both held-out (names Held Out Bears) *and* carries a resolvable
        # acting ability, so the card-disjoint cap (FR-142) has a text to
        # admit the game against.
        _resolution(
            "g-tainted.1", "g-tainted", ability=(_BOLT_KEY,),
            entity_names=(_HELD_OUT_CARD,),
        ),
        # The held-out ability actually resolving, which is what the cap
        # counts: FR-142 balances the stratum per *held-out* text, so a game
        # where the held-out card only sat on the battlefield carries nothing
        # the cap is about and earns no place in the stratum.
        _resolution("g-tainted.4", "g-tainted", ability=(_HELD_OUT_KEY,)),
        _combat("g-tainted.2", "g-tainted"),
        _combat(
            "g-tainted.3", "g-tainted", fork=True, mirror_of="g-tainted.2",
            probed_keyword="flying", probed_entity="E0",
        ),
        *(
            _resolution(f"g-clean-1.{n}", "g-clean-1", ability=(_SHOCK_KEY,))
            for n in range(3)
        ),
    ]
    shard_b = [
        *(
            _resolution(f"g-clean-2.{n}", "g-clean-2", ability=(_SHOCK_KEY,))
            for n in range(3)
        ),
        _resolution("g-clean-3.1", "g-clean-3", ability=(_BOLT_KEY,)),
    ]

    records_dir = root / "records"
    write_shard(records_dir / "run.0-a.jsonl.gz", shard_a)
    write_shard(records_dir / "run.0-b.jsonl.gz", shard_b)

    return CorpusFixture(records=records_dir, cards=(str(root / "cardsfolder"),))


def test_build_writes_the_three_strata_and_a_manifest(tmp_path, a_corpus):
    out = tmp_path / "curated"
    assert build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    )) == 0

    store = CorpusStore(out)
    assert store.manifest_path.exists()
    assert list(store.training_dir.glob("*.jsonl.gz"))
    assert store.load().digest()


def test_no_training_record_belongs_to_a_withheld_game(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    store = CorpusStore(out)
    manifest = store.load()
    withheld = set(manifest.card_disjoint_games) | set(manifest.game_disjoint_games)
    assert withheld
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert not trained & withheld


def test_no_training_record_names_a_held_out_card(tmp_path, a_corpus):
    """FR-088: the exclusion is by game, and every held-out game is dropped."""
    from effects.application.train_effect_model import (
        HeldOutCards, record_names_held_out_card,
    )

    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    held_out = HeldOutCards(
        names=frozenset(manifest.held_out_cards), script_files=frozenset(),
    )
    for record in read_records(CorpusStore(out).training_dir):
        assert not record_names_held_out_card(record, held_out)


def test_the_manifest_records_the_held_out_texts(tmp_path, a_corpus):
    """Fix round 1, Finding 1: ``build()`` must populate ``held_out_texts``.

    A ``--corpus`` training run's stratum-sizing guard (``check_holdout``)
    reads ``HeldOutCards.texts`` alone, built from this field — not from
    ``held_out_cards``. A manifest that recorded only the held-out cards would
    make every ``--corpus`` run see an empty text set and report the
    card-disjoint stratum as empty regardless of the dataset's actual health.
    """
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    assert manifest.held_out_texts == (_HELD_OUT_TEXT,)


def test_a_validation_stratum_keeps_whole_games(tmp_path, a_corpus):
    """A probe and the combat record it mirrors land together (FR-136)."""
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    store = CorpusStore(out)
    mirrored = 0
    for directory in (store.card_disjoint_dir, store.game_disjoint_dir):
        records = list(read_records(directory))
        by_id = {r.record_id for r in records}
        for record in records:
            if record.mirror_of is not None:
                mirrored += 1
                assert record.mirror_of in by_id
    assert mirrored, "the fixture must contain a mirrored probe or this proves nothing"


def test_the_cap_trims_a_text_that_exceeds_it(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, text_cap=2, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) > 0


def test_two_builds_of_one_corpus_agree(tmp_path, a_corpus):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        build(BuildCorpusConfig(
            records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
            workers=1, game_disjoint_target=1,
        ))

    assert CorpusStore(first).load().digest() == CorpusStore(second).load().digest()
    assert (
        [r.record_id for r in read_records(CorpusStore(first).training_dir)]
        == [r.record_id for r in read_records(CorpusStore(second).training_dir)]
    )


def test_verify_reports_drift_and_writes_nothing(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    before = (out / "manifest.json").read_text(encoding="utf-8")

    (a_corpus.records / "extra.0-zz.jsonl.gz").write_bytes(b"")
    code = build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, verify=True,
    ))

    assert code == 1
    assert (out / "manifest.json").read_text(encoding="utf-8") == before


def test_verify_is_quiet_when_nothing_moved(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    assert build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, verify=True,
    )) == 0


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
        build(BuildCorpusConfig(
            records_dir=a_corpus.records,
            cards_folders=(str(tmp_path / "not-here"), str(tmp_path / "nor-here")),
            output=tmp_path / "curated", workers=1,
        ))

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
        build(BuildCorpusConfig(
            records_dir=a_corpus.records, cards_folders=a_corpus.cards,
            output=tmp_path / "curated", workers=1, holdout_permille=0,
        ))

    assert "3 converted card" in str(raised.value)


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
        a_corpus.records / "run.0-b.jsonl.gz", depleted / "run.0-b.jsonl.gz",
    )
    out = tmp_path / "curated"

    with pytest.raises(ValueError, match="card-disjoint"):
        build(BuildCorpusConfig(
            records_dir=depleted, cards_folders=a_corpus.cards, output=out,
            workers=1, game_disjoint_target=1,
        ))

    assert not (out / "manifest.json").exists()


def test_build_refuses_an_empty_corpus(tmp_path, a_corpus):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no shards"):
        build(BuildCorpusConfig(
            records_dir=empty, cards_folders=a_corpus.cards,
            output=tmp_path / "out", workers=1,
        ))


# ── fix round 1: _output_name must be injective (review Finding 1) ─────


def test_output_name_is_injective_on_the_reviewers_counterexample():
    """``"/" -> "__"`` collided: ``"a_/b"`` and ``"a/_b"`` both became
    ``"a___b"``. Two source shards writing the same output filename is a
    silent overwrite (``write_shard`` truncates), so this has to hold.
    """
    assert _output_name("a_/b") != _output_name("a/_b")


def test_output_name_flattens_a_subdirectory_path():
    assert _output_name("depleted/run.0-a.jsonl.gz") == "depleted_srun.0-a.jsonl.gz"


# ── fix round 1: per-stratum counts and unique texts (review Finding 2) ─


def test_the_manifest_records_per_stratum_counts_and_unique_texts(tmp_path, a_corpus):
    """FR-143/FR-145: per-class counts alone leave both validation strata
    with no unique-text figure anywhere in the manifest. A test that only
    checked the two fields exist would not catch them being populated empty,
    so this checks actual values against strata ``a_corpus`` is built to
    populate: the tainted game (card-disjoint) and whichever clean game the
    seeded game-disjoint draw takes both carry a resolvable ability.
    """
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    assert manifest.per_stratum["training"] > 0
    assert manifest.unique_texts["training"] > 0
    assert "card-disjoint" in manifest.per_stratum
    assert "game-disjoint" in manifest.per_stratum
    assert manifest.unique_texts["card-disjoint"] > 0
    assert manifest.unique_texts["game-disjoint"] > 0


# ── the delivered mixture is the mixture the manifest records ─────────

#: Shards, records per class per shard, and the cap — chosen so the per-shard
#: estimate of what ``rewrite`` can supply is four times the truth. Each shard
#: holds ``_MIX_PER_SHARD`` records of ONE rewrite text, so a shard-local heap
#: of ``_MIX_TEXT_CAP`` admits that many *per shard* while the corpus-wide cap
#: admits that many in total.
_MIX_SHARDS = 4
_MIX_PER_SHARD = 800
_MIX_TEXT_CAP = 400


@pytest.fixture
def a_mixture_corpus(tmp_path: Path) -> CorpusFixture:
    """Four shards, one text repeated far past the cap, and an uncapped class.

    ``rewrite`` carries one ability text and is what the cap bites; ``combat``
    has no acting line at all (FR-087), so nothing caps it and its
    availability is exact either way. Asked for half and half, the two come
    out half and half only if availability is estimated corpus-wide.

    One further game holds the held-out ability resolving, which is what the
    card-disjoint stratum is made of; its records are ``resolution-effect``,
    a class outside this corpus's ``--class-mix``, so nothing it holds reaches
    the training mixture under test.
    """
    root = tmp_path / "mixture-source"
    _write_card(root, _HELD_OUT_FILE, _HELD_OUT_CARD, _HELD_OUT_TEXT)
    _write_card(root, _BOLT_FILE, _BOLT_CARD, _BOLT_TEXT)

    records_dir = root / "records"
    for shard in range(_MIX_SHARDS):
        rows: list[EffectRecord] = [
            _rewrite(f"rw.{shard}.{n}", f"g-rw-{shard}", ability=(_BOLT_KEY,))
            for n in range(_MIX_PER_SHARD)
        ]
        rows += [
            _combat(f"cb.{shard}.{n}", f"g-cb-{shard}")
            for n in range(_MIX_PER_SHARD)
        ]
        if shard == 0:
            rows.append(
                _resolution("g-tainted.1", "g-tainted", ability=(_HELD_OUT_KEY,)),
            )
        write_shard(records_dir / f"run.0-{shard}.jsonl.gz", rows)

    return CorpusFixture(records=records_dir, cards=(str(root / "cardsfolder"),))


def _mixture_config(output: Path, records: Path, cards: tuple[str, ...]):
    return BuildCorpusConfig(
        records_dir=records, cards_folders=cards, output=output, workers=1,
        text_cap=_MIX_TEXT_CAP, class_mix={"rewrite": 0.5, "combat": 0.5},
        game_disjoint_target=0,
    )


def _delivered_shares(training_dir: Path) -> dict[str, float]:
    from effects.application.train_effect_model import sampling_class

    counts: Counter[str] = Counter()
    for record in read_records(training_dir):
        counts[sampling_class(record)] += 1
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def test_the_delivered_class_mixture_is_the_one_the_manifest_records(
    tmp_path, a_mixture_corpus,
):
    """FR-139: a class is written at its share or its shortfall is recorded.

    ``class_capped_records`` was summed from each shard's own ``CapHeap``
    thresholds while the real cap is corpus-wide, so availability was
    over-estimated by a factor of the shard count for exactly the classes the
    cap touches. ``build()`` divides each class's target by that availability
    to get its admission rate, so every capped class under-delivered against
    the mixture the manifest went on to record.
    """
    out = tmp_path / "curated"
    assert build(_mixture_config(
        out, a_mixture_corpus.records, a_mixture_corpus.cards,
    )) == 0

    delivered = _delivered_shares(CorpusStore(out).training_dir)
    requested = CorpusStore(out).load().class_mix
    for name, share in requested.items():
        assert abs(delivered.get(name, 0.0) - share) < 0.05, (
            f"{name}: delivered {delivered.get(name, 0.0):.3f} against a "
            f"recorded share of {share:.3f}"
        )


def test_the_manifest_records_what_was_delivered_beside_what_was_asked_for(
    tmp_path, a_mixture_corpus,
):
    """The requested mixture is a claim; the delivered one is a fact, and the
    dataset's whole value is that its manifest is true."""
    out = tmp_path / "curated"
    build(_mixture_config(out, a_mixture_corpus.records, a_mixture_corpus.cards))

    manifest = CorpusStore(out).load()
    delivered = _delivered_shares(CorpusStore(out).training_dir)
    assert set(manifest.delivered_mix) == set(delivered)
    for name, share in delivered.items():
        assert manifest.delivered_mix[name] == pytest.approx(share, abs=1e-6)
