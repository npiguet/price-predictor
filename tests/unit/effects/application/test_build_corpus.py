"""The write pass and ``build()``: turn a survey and its decisions into a
fixed training corpus and two fixed validation strata on disk.

``a_corpus`` is the load-bearing fixture. It builds a small but real raw
corpus: a converted card tree with sidecars (a held-out card, plus two
ordinary ones), and two raw shards — one under ``full-strength/`` and one
under ``depleted/``, the two source directories FR-149 reports against —
covering four games, no game spanning both —

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
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from effects.application.build_corpus import (
    BuildCorpusConfig,
    BuildCorpusError,
    _check_samples,
    _output_name,
    build,
    source_of,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
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
from effects.infrastructure.record_io import iter_shards, read_records, read_shard, write_shard
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

    @staticmethod
    def resolution(
        record_id: str,
        *,
        game: str,
        ability: tuple[ProvenanceKey, ...] = (_BOLT_KEY,),
        events: tuple[Event, ...] = (),
    ) -> EffectRecord:
        """A resolution record built the way the fixture builds ``g-tainted``'s.

        ``ability=()`` is normalized to ``None`` so the record round-trips
        through the shard writer as one with no acting line at all, which is
        what ``quality_defect`` refuses as ``no-ability``.
        """
        record = _resolution(record_id, game, ability=ability or None)
        if events:
            record = replace(record, payload=ResolutionPayload(events=events))
        return record

    def with_extra_records(self, records: list[EffectRecord]) -> CorpusFixture:
        """Append ``records`` as one more raw shard under the same root.

        Under ``full-strength/`` because every game these tests add records to
        already lives there, and a game split across two source directories
        would be counted once per directory in ``games_by_source``.
        """
        write_shard(self.records / "full-strength" / "run.0-extra.jsonl.gz", records)
        return self

    @property
    def source_dirs(self) -> set[str]:
        """The top-level directory of every raw shard currently on disk."""
        return {
            source_of(shard.relative_to(self.records).as_posix())
            for shard in iter_shards(self.records)
        }

    @property
    def game_count(self) -> int:
        return len({record.game_id for record in read_records(self.records)})


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

    # Two source directories, the way a real corpus keeps a full-strength
    # collection run apart from a depleted one (FR-149): the tainted game is
    # full-strength by construction, and no game spans the two.
    records_dir = root / "records"
    write_shard(records_dir / "full-strength" / "run.0-a.jsonl.gz", shard_a)
    write_shard(records_dir / "depleted" / "run.0-b.jsonl.gz", shard_b)

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
        HeldOutCards,
        record_names_held_out_card,
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
        a_corpus.records / "depleted" / "run.0-b.jsonl.gz",
        depleted / "run.0-b.jsonl.gz",
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
    # ``validation_sample=0``: this corpus's card-disjoint stratum holds one
    # ``resolution-effect`` record and the mixture under test names neither
    # resolution class, so its sample is empty by construction and the
    # empty-sample guard would refuse the build. That guard is exercised
    # against this same fixture below; here the subject is the mixture.
    return BuildCorpusConfig(
        records_dir=records, cards_folders=cards, output=output, workers=1,
        text_cap=_MIX_TEXT_CAP, class_mix={"rewrite": 0.5, "combat": 0.5},
        game_disjoint_target=0, validation_sample=0,
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


# ── a rebuild replaces the dataset rather than layering over it ────────


def test_a_rebuild_removes_a_shard_the_new_build_does_not_write(tmp_path, a_corpus):
    """FR-144: a grown corpus is rebuilt whole, never extended in place.

    Same-named source shards are truncate-overwritten, so the common case
    survived; a source shard renamed, removed, or a rebuild pointed at a
    different --records-dir left files behind that the trainer reads as part
    of the dataset while the manifest and its digest describe only the second
    build.
    """
    out = tmp_path / "curated"
    config = BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    )
    build(config)

    store = CorpusStore(out)
    orphans = [
        store.training_dir / "orphan.jsonl.gz",
        store.card_disjoint_dir / "orphan.jsonl.gz",
        store.game_disjoint_dir / "orphan.jsonl.gz",
    ]
    for orphan in orphans:
        write_shard(orphan, [_resolution("orphan.1", "g-orphan", ability=(_BOLT_KEY,))])

    build(config)

    assert not any(orphan.exists() for orphan in orphans)
    assert store.manifest_path.exists()
    assert "g-orphan" not in {r.game_id for r in read_records(store.training_dir)}


def test_verify_leaves_the_written_dataset_alone(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    store = CorpusStore(out)
    before = sorted(p.name for p in store.training_dir.glob("*"))

    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, verify=True,
    ))

    assert sorted(p.name for p in store.training_dir.glob("*")) == before


def test_no_empty_output_shard_is_written(tmp_path, a_corpus):
    """One output shard per source shard per stratum, written unconditionally,
    turned 701 source shards into 2,103 files, most of them empty.

    They are not free. `train-effect-model --corpus` lists both validation
    directories and reads every shard in them before the first training step,
    logging a line each; an empty *training* shard drawn into an epoch
    forfeits its share of the step budget, because `_train_on_shard` returns
    early on a shard with nothing in it.
    """
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))

    store = CorpusStore(out)
    written = [
        shard
        for directory in (
            store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir,
        )
        for shard in directory.glob("*.jsonl.gz")
    ]
    from effects.infrastructure.record_io import read_shard

    assert written, "the fixture must write something or this proves nothing"
    empty = [shard for shard in written if not any(True for _ in read_shard(shard))]
    assert not empty, f"empty shards written: {[str(s) for s in empty]}"


def test_the_trainer_refuses_a_real_dataset_built_on_the_other_surface(
    tmp_path, a_corpus,
):
    """The manifest's ``surface`` reaches the trainer (FR-141).

    A real ``build-corpus`` output, read back by ``train_effect_model.run``
    under a vocabulary on the other surface. The guard fires before the
    training loop is imported, so this needs no torch and no GPU — and it is
    the only thing in the suite that round-trips a written dataset into the
    trainer at all.
    """
    from effects.application.train_effect_model import (
        SurfaceMismatchError,
        TrainEffectModelConfig,
    )
    from effects.application.train_effect_model import run as train

    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, game_disjoint_target=1,
    ))
    assert CorpusStore(out).load().surface == "prose"

    with pytest.raises(SurfaceMismatchError, match="prose"):
        train(TrainEffectModelConfig(
            corpus=str(out), vocab_path=Path("models/effects/vocab-script.txt"),
        ))


# ── what a count may be, and what zero means for each ─────────────────


@pytest.mark.parametrize("field,flag", [
    ("text_cap", "--text-cap"),
    ("card_disjoint_text_cap", "--card-disjoint-text-cap"),
    ("game_disjoint_target", "--game-disjoint-games"),
    ("training_records", "--training-records"),
])
def test_a_negative_count_is_refused(tmp_path, a_corpus, field, flag):
    with pytest.raises(ValueError, match=flag):
        BuildCorpusConfig(
            records_dir=a_corpus.records, cards_folders=a_corpus.cards,
            output=tmp_path / "curated", workers=1, **{field: -1},
        )


def test_a_text_cap_of_zero_means_no_cap_rather_than_keep_nothing(
    tmp_path, a_corpus,
):
    """Zero admits everything, the way ``CapHeap`` reads it. Documented here
    because the other reading — keep nothing — is the one the word "cap"
    suggests, and it would silently empty the training corpus."""
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, text_cap=0, game_disjoint_target=1,
    ))

    manifest = CorpusStore(out).load()
    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) == 0
    assert manifest.per_stratum["training"] > 0


def test_a_card_disjoint_text_cap_of_zero_means_no_cap_either(tmp_path, a_corpus):
    out = tmp_path / "curated"
    build(BuildCorpusConfig(
        records_dir=a_corpus.records, cards_folders=a_corpus.cards, output=out,
        workers=1, card_disjoint_text_cap=0, game_disjoint_target=1,
    ))

    assert CorpusStore(out).load().card_disjoint_games == ("g-tainted",)


# ── stage-four variants (FR-141: they are ability texts like any other) ──

_VARIANT_FILE = "variant-scripts/bolt_numdmg_2.txt"
_VARIANT_KEY = ProvenanceKey(_VARIANT_FILE, 0, "spell", 0)
_VARIANT_TEXT = "SP$ DealDamage | NumDmg$ 2"


def _write_variant(root: Path, script_file: str, text: str) -> None:
    """A sidecar in the flat variant tree, which carries no prose surface.

    `collect-variants` writes the perturbed *source* script and a sidecar
    beside it; nothing converts it, so `script_text` is the only surface the
    line has.
    """
    relative = script_file.split("/", 1)[1]
    path = root / "variant-scripts" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card="Bolt Variant",
            script_file=script_file,
            lines=(
                SidecarLine(
                    line_index=0, line_kind="spell",
                    provenance=(_VARIANT_KEY,), script_text=text,
                ),
            ),
        ),
        sidecar_path_for(path),
    )


@pytest.fixture
def a_corpus_with_variants(tmp_path: Path, a_corpus: CorpusFixture):
    """The ordinary fixture plus a variant tree and a shard that uses it."""
    fixture = a_corpus
    root = tmp_path / "source"
    _write_variant(root, _VARIANT_FILE, _VARIANT_TEXT)
    write_shard(
        fixture.records / "variants" / "run.0-v.jsonl.gz",
        [
            _resolution(f"g-var.{i}", "g-var", ability=(_VARIANT_KEY,))
            for i in range(6)
        ],
    )
    return fixture, str(root / "variant-scripts")


def test_a_variant_text_is_capped_and_weighted_like_any_other(
    tmp_path: Path, a_corpus_with_variants,
):
    """FR-138/FR-141 do not exempt the variant tree.

    Without `--variant-scripts` a variant line resolves to no text at all, so
    it lands in neither the rarity table nor the per-text cap while every real
    ability is in both — uncapped records competing against capped ones.
    """
    fixture, variant_root = a_corpus_with_variants
    out = tmp_path / "curated"

    build(BuildCorpusConfig(
        records_dir=fixture.records, cards_folders=fixture.cards,
        variant_scripts=variant_root, output=out, workers=1,
        game_disjoint_target=1,
    ))

    assert _VARIANT_TEXT in CorpusStore(out).load().rarity


def test_without_the_variant_tree_a_variant_text_resolves_to_nothing(
    tmp_path: Path, a_corpus_with_variants,
):
    """The state this flag exists to end; pinned so the fix cannot regress."""
    fixture, _ = a_corpus_with_variants
    out = tmp_path / "curated"

    build(BuildCorpusConfig(
        records_dir=fixture.records, cards_folders=fixture.cards,
        output=out, workers=1, game_disjoint_target=1,
    ))

    assert _VARIANT_TEXT not in CorpusStore(out).load().rarity


def test_the_manifest_records_the_variant_tree_it_read(
    tmp_path: Path, a_corpus_with_variants,
):
    """Which trees resolved decides which texts the rarity table names."""
    fixture, variant_root = a_corpus_with_variants
    out = tmp_path / "curated"

    build(BuildCorpusConfig(
        records_dir=fixture.records, cards_folders=fixture.cards,
        variant_scripts=variant_root, output=out, workers=1,
        game_disjoint_target=1,
    ))

    assert CorpusStore(out).load().variant_scripts == variant_root


def test_the_store_names_the_new_outputs(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    assert store.gate_one_dir == tmp_path / "corpus" / "validation" / "gate-one"
    assert store.samples_dir == tmp_path / "corpus" / "validation" / "samples"
    assert store.sample_path("card-disjoint") == (
        tmp_path / "corpus" / "validation" / "samples" / "card-disjoint.jsonl.gz"
    )
    assert store.parts_dir_for("training") == tmp_path / "corpus" / ".parts" / "training"


def test_clear_outputs_removes_the_new_outputs_too(tmp_path):
    store = CorpusStore(tmp_path / "corpus")
    for directory in (store.gate_one_dir, store.samples_dir, store.parts_dir_for("training")):
        directory.mkdir(parents=True)
        (directory / "x").write_text("x")
    store.clear_outputs()
    assert not store.gate_one_dir.exists()
    assert not store.samples_dir.exists()
    assert not store.parts_dir.exists()


# ── quality, the gate-one slice, per-source reporting, repacking ───────


def test_source_of_is_the_first_path_component():
    assert source_of("depleted/run.0-a.jsonl.gz") == "depleted"
    assert source_of("run.0-a.jsonl.gz") == "."


def test_defective_records_reach_no_output_and_are_counted(tmp_path, a_corpus):
    """A junk resolution record in a training game and one in a held-out game both vanish."""
    from effects.domain.record_quality import NO_ABILITY, UNATTRIBUTED
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("junk-train", game="g-clean-1", ability=()),
        a_corpus.resolution(
            "junk-held", game="g-tainted",
            events=(Event(
                type=EventType.DAMAGE_DEALT, subjects=("P0",),
                attributed_to="unresolved",
            ),),
        ),
    ])
    assert build(BuildCorpusConfig(records_dir=corpus.records, output=tmp_path / "out",
                                   cards_folders=corpus.cards, workers=1)) == 0
    store = CorpusStore(tmp_path / "out")
    ids = {r.record_id for d in (store.training_dir, store.card_disjoint_dir,
                                  store.game_disjoint_dir, store.gate_one_dir)
           for r in read_records(d)}
    assert "junk-train" not in ids and "junk-held" not in ids
    assert store.load().quality_dropped == {NO_ABILITY: 1, UNATTRIBUTED: 1}


def test_the_gate_one_slice_holds_held_out_resolutions_of_card_disjoint_games(tmp_path, a_corpus):
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards, workers=1)) == 0
    store = CorpusStore(tmp_path / "out")
    manifest = store.load()
    slice_ = list(read_records(store.gate_one_dir))
    assert slice_, "g-tainted resolves the held-out text, so the slice is not empty"
    assert all(r.kind is RecordKind.RESOLUTION for r in slice_)
    assert all(r.game_id in manifest.card_disjoint_games for r in slice_)
    assert all(_HELD_OUT_KEY in (r.ability or ()) for r in slice_)
    card_disjoint_ids = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert {r.record_id for r in slice_} <= card_disjoint_ids
    assert manifest.per_stratum["gate-one"] == len(slice_)


def test_games_are_reported_per_source_directory(tmp_path, a_corpus):
    """Shards live under depleted/ and full-strength/ in the fixture."""
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards, workers=1)) == 0
    manifest = CorpusStore(tmp_path / "out").load()
    assert set(manifest.games_by_source) == set(a_corpus.source_dirs)
    assert sum(manifest.games_by_source.values()) == a_corpus.game_count
    assert all(manifest.held_out_games_by_source.get(s, 0) <= n
               for s, n in manifest.games_by_source.items())


def test_output_shards_are_repacked_and_the_parts_removed(tmp_path, a_corpus):
    # ``game_disjoint_target=1`` as everywhere else in this file: the default
    # 1000 takes all three clean games into the game-disjoint stratum, leaving
    # the training output — the one this test reads back — empty.
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, output=tmp_path / "out",
                                   cards_folders=a_corpus.cards, workers=1,
                                   game_disjoint_target=1, shard_records=2)) == 0
    store = CorpusStore(tmp_path / "out")
    assert not store.parts_dir.exists()
    names = sorted(p.name for p in store.training_dir.glob("*.jsonl.gz"))
    assert names and names[0] == "shard-00001.jsonl.gz"
    assert store.load().shard_records == 2


def test_the_build_writes_a_validation_sample_per_stratum(tmp_path, a_corpus):
    assert build(BuildCorpusConfig(records_dir=a_corpus.records, cards_folders=a_corpus.cards,
                                   output=tmp_path / "out", workers=1,
                                   validation_sample=4)) == 0
    store = CorpusStore(tmp_path / "out")
    for stratum in ("card-disjoint", "game-disjoint"):
        assert store.sample_path(stratum).exists()
        assert 0 < sum(1 for _ in read_shard(store.sample_path(stratum))) <= 4 * 8
    assert store.load().validation_sample == 4


def test_the_cli_exposes_the_rework_flags():
    from effects.infrastructure.cli import build_parser
    args = build_parser().parse_args([
        "build-corpus", "--shard-records", "500", "--max-events-per-record", "32",
        "--validation-sample", "64",
    ])
    assert (args.shard_records, args.max_events_per_record, args.validation_sample) == (500, 32, 64)
    defaults = build_parser().parse_args(["build-corpus"])
    assert (
        defaults.shard_records,
        defaults.max_events_per_record,
        defaults.validation_sample,
    ) == (2000, 64, 2048)

    # The zero reading for each of the three flags must survive in --help,
    # not just in the docstring: this is what would have caught it silently
    # vanishing from --max-events-per-record's help text.
    subparsers_action = next(
        action for action in build_parser()._actions
        if getattr(action, "choices", None) and "build-corpus" in action.choices
    )
    max_events_action = next(
        action for action in subparsers_action.choices["build-corpus"]._actions
        if "--max-events-per-record" in action.option_strings
    )
    assert "0 disables" in max_events_action.help


# ── C1: combat is exempt from the unattributed rule ───────────────────


def _unresolved_event() -> Event:
    return Event(
        type=EventType.DAMAGE_DEALT, subjects=("P0",), attributed_to="unresolved",
    )


def test_a_combat_record_with_an_unresolved_event_reaches_training(tmp_path, a_corpus):
    """Nothing resolves in a damage step, so the collector stamps every combat
    event ``unresolved`` by design and the cause lives in ``cause``. Judged by
    the unattributed rule the class was 98.9% refused on the real corpus —
    a statement about the rule, not about the records.
    """
    corpus = a_corpus.with_extra_records([
        replace(
            _combat("g-clean-1.combat", "g-clean-1"),
            payload=CombatPayload(attackers=("E0",), events=(_unresolved_event(),)),
        ),
    ])
    assert build(BuildCorpusConfig(
        records_dir=corpus.records, cards_folders=corpus.cards,
        output=tmp_path / "out", workers=1, game_disjoint_target=1,
    )) == 0

    store = CorpusStore(tmp_path / "out")
    trained = {r.record_id for r in read_records(store.training_dir)}
    assert "g-clean-1.combat" in trained


# ── C1 guard: a whole class refused stops the build ───────────────────


def test_a_class_refused_past_the_share_stops_the_build(tmp_path, a_corpus):
    """A class the quality rules refuse wholesale is a rule bug or an
    unpatched checkout, not a corpus to train on. Written silently it is a
    dataset missing a sampling class, which the trainer reports as a class the
    corpus happens not to hold.
    """
    corpus = a_corpus.with_extra_records([
        replace(
            _combat(f"g-clean-1.flood-{n}", "g-clean-1"),
            payload=CombatPayload(
                attackers=("E0",),
                events=tuple(
                    Event(type=EventType.DAMAGE_DEALT, subjects=("P0",),
                          attributed_to="root")
                    for _ in range(100)
                ),
            ),
        )
        for n in range(5)
    ])

    with pytest.raises(BuildCorpusError, match="combat") as raised:
        build(BuildCorpusConfig(
            records_dir=corpus.records, cards_folders=corpus.cards,
            output=tmp_path / "out", workers=1, game_disjoint_target=1,
        ))

    message = str(raised.value)
    assert "unresolved" in message and "degraded" in message


# ── I3: a stratum with an empty sample is a build that failed quietly ──


def test_an_empty_stratum_sample_is_refused(tmp_path):
    with pytest.raises(BuildCorpusError, match="card-disjoint"):
        _check_samples({"card-disjoint": 0, "game-disjoint": 12}, size=2048)


def test_check_samples_passes_when_every_stratum_has_records():
    _check_samples({"card-disjoint": 4, "game-disjoint": 12}, size=2048)


def test_check_samples_says_nothing_when_no_sample_was_asked_for():
    _check_samples({"card-disjoint": 0, "game-disjoint": 0}, size=0)


def test_a_stratum_whose_sample_comes_out_empty_stops_the_build(
    tmp_path, a_mixture_corpus,
):
    """The mixture names neither resolution class and the card-disjoint
    stratum holds nothing else, so its sample is empty. Shipped, the trainer
    validates ``nan`` on the stratum that selects the best checkpoint.
    """
    with pytest.raises(BuildCorpusError, match="card-disjoint"):
        build(replace(
            _mixture_config(
                tmp_path / "curated", a_mixture_corpus.records,
                a_mixture_corpus.cards,
            ),
            validation_sample=8,
        ))
