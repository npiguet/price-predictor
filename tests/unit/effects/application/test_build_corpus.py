"""The write pass and ``build()``: turn a survey and its decisions into a
fixed training corpus and two fixed validation strata on disk.

``a_corpus`` is the load-bearing fixture. It builds a small but real raw
corpus: a converted card tree with sidecars (a held-out card, plus three
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
- ``g-clean-1`` also carries one record acting through a token's ability,
  keyed under the converted ``tokenscripts/`` tree beside the card tree.

The held-out text itself is not asserted by inspection — it is *constructed*
by searching for a string that ``text_is_held_out`` actually selects at the
default ``--holdout-permille 20``, per the task brief.

A test that adds to the corpus takes its own ``a_corpus``; one that only
builds it reads ``shared_corpus``, and one that only reads the default build
reads ``default_build``, built once for the module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from effects.application.build_corpus import (
    BuildCorpusConfig,
    BuildCorpusError,
    build,
)
from effects.application.corpus_report import check_refusals as _check_refusals
from effects.application.corpus_report import check_samples as _check_samples
from effects.application.corpus_survey import RefusalTally
from effects.domain.corpus_manifest import SourceShard
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
)
from effects.domain.split import HoldoutRule
from effects.domain.state_snapshot import StateSnapshot
from effects.domain.strata import Stratum
from effects.domain.text_holdout import HoldoutUnit
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    iter_shards,
    list_shards,
    read_records,
    read_shard,
    write_shard,
)
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.builders import make_entity, make_record, make_snapshot, write_sidecar

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

#: A vanilla creature: its sidecar declares one rendered spell line at index
#: 1, and lists index 0 — Forge's implicit permanent spell, the one every
#: permanent carries — as dropped. That dropped key is what a record acting
#: through nothing but the creature spell itself names (FR-148).
_BEAR_FILE = "cardsfolder/g/grizzly_bears.txt"
_BEAR_CARD = "Grizzly Bears"
_BEAR_TEXT = "grizzly bears static text"
_BEAR_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 1)
_BEAR_CAST_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 0)
#: Past every spell index this face declared, which is the shape a level-up
#: or bestow trait has: runtime-only, and kept.
_BEAR_LEVEL_UP_KEY = ProvenanceKey(_BEAR_FILE, 0, "spell", 5)

_SHOCK_CARD = "Common Shock"
_SHOCK_TEXT = "deals 2 damage to any target"
_SHOCK_FILE = "cardsfolder/c/common_shock.txt"
_SHOCK_KEY = ProvenanceKey(_SHOCK_FILE, 0, "spell", 0)

#: A token, converted into the flat ``tokenscripts/`` tree beside the card
#: tree — where a token's abilities have always belonged.
_FOOD_CARD = "Food Token"
_FOOD_TEXT = "sacrifice this artifact: you gain 3 life"
_FOOD_FILE = "tokenscripts/c_a_food_sac.txt"
#: ``index_within_kind=1``: a token's printed spell keys start at 1, index 0
#: being the permanent's own cast spell.
_FOOD_KEY = ProvenanceKey(_FOOD_FILE, 0, "spell", 1)


def _write_card(
    root: Path, script_file: str, card: str, text: str, *,
    index_within_kind: int = 0, dropped: tuple[ProvenanceKey, ...] = (),
) -> None:
    """A minimal converted ``.txt`` plus a sidecar whose one line carries ``text``.

    ``dropped`` fills the sidecar's ``dropped_keys``: traits the converter
    rendered no line for, which is what a resolution record has to act only
    through to be refused as ``no-acting-text`` (FR-148).

    ``line_index=999`` is deliberately past the converted file's last line, so
    ``prose_for`` always misses and ``text_for_key`` falls back to
    ``script_text`` — what matters for these tests is that every record
    sharing one provenance key resolves to the same text, not which surface
    supplied it.

    The tree comes from ``script_file`` rather than being fixed, because a
    converted token lives in the flat ``tokenscripts/`` tree beside the
    letter-keyed card tree and the build reads both.
    """
    tree, relative = script_file.split("/", 1)
    txt_path = root / tree / relative
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(f"name: {card}\ntypes: creature\n", encoding="utf-8")
    key = ProvenanceKey(script_file, 0, "spell", index_within_kind)
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
            dropped_keys=dropped,
        ),
        sidecar_path_for(txt_path),
    )


def _snapshot(entity_names: tuple[str, ...] = ()) -> StateSnapshot:
    return make_snapshot(
        entities=tuple(
            make_entity(f"E{index}", name) for index, name in enumerate(entity_names)
        ),
    )


_TIMESTAMP = "2026-09-14T00:00:00.000000Z"


def _resolution(
    record_id: str,
    game_id: str,
    *,
    ability: tuple[ProvenanceKey, ...],
    entity_names: tuple[str, ...] = (),
) -> EffectRecord:
    return make_record(
        record_id=record_id, game_id=game_id, timestamp=_TIMESTAMP,
        ability=ability, state=_snapshot(entity_names),
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
    return make_record(
        RecordKind.COMBAT, record_id=record_id, game_id=game_id, timestamp=_TIMESTAMP,
        fork=fork, mirror_of=mirror_of,
        payload=CombatPayload(
            attackers=("E0",), probed_keyword=probed_keyword,
            probed_entity=probed_entity,
        ),
    )


def _rewrite(
    record_id: str, game_id: str, *, ability: tuple[ProvenanceKey, ...],
) -> EffectRecord:
    return make_record(
        RecordKind.REWRITE, record_id=record_id, game_id=game_id, timestamp=_TIMESTAMP,
        ability=ability,
        payload=RewritePayload(incoming=Event(type=EventType.ZONE_CHANGE)),
    )


#: Feature 023's rules, which most of this file was written against: the held-out
#: text above is held out under the ``text`` unit, ``0.16`` places exactly one
#: game (``g-clean-3``) in the game-disjoint stratum, and one copy per record.
_LEGACY_RULES = dict(
    holdout=HoldoutRule(unit=HoldoutUnit.TEXT),
    game_disjoint_share=0.16, game_disjoint_keyword_share=0.16,
    reuse_cap=1,
)


@dataclass(frozen=True, slots=True)
class CorpusFixture:
    """A small real raw corpus: the raw shard root and the converted trees."""

    records: Path
    cards: tuple[Path, ...]

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

    def config(
        self, output: Path, *, rules: dict = _LEGACY_RULES, **overrides,
    ) -> BuildCorpusConfig:
        """A one-worker build of this corpus under ``rules``; ``overrides``
        replace any field, a rule included."""
        return BuildCorpusConfig(**{
            "records_dir": self.records, "cards_folders": self.cards,
            "output": output, "workers": 1, **rules, **overrides,
        })

    def build(self, output: Path, **overrides) -> CorpusStore:
        """Build :meth:`config` and return the dataset it wrote."""
        assert build(self.config(output, **overrides)) == 0
        return CorpusStore(output)

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
            shard.source for shard in list_shards(self.records)
        }

    @property
    def game_count(self) -> int:
        return len({record.game_id for record in read_records(self.records)})


def _write_corpus(root: Path) -> CorpusFixture:
    _write_card(root, _HELD_OUT_FILE, _HELD_OUT_CARD, _HELD_OUT_TEXT)
    _write_card(root, _BOLT_FILE, _BOLT_CARD, _BOLT_TEXT)
    _write_card(root, _SHOCK_FILE, _SHOCK_CARD, _SHOCK_TEXT)
    _write_card(root, _FOOD_FILE, _FOOD_CARD, _FOOD_TEXT, index_within_kind=1)
    _write_card(
        root, _BEAR_FILE, _BEAR_CARD, _BEAR_TEXT,
        index_within_kind=_BEAR_KEY.index_within_kind, dropped=(_BEAR_CAST_KEY,),
    )

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
        # One training record acting through a token's ability. Its own
        # ability text, so it neither joins nor disturbs the capped six
        # copies of the shock text above.
        _resolution("g-clean-1.food", "g-clean-1", ability=(_FOOD_KEY,)),
    ]
    shard_b = [
        *(
            _resolution(f"g-clean-2.{n}", "g-clean-2", ability=(_SHOCK_KEY,))
            for n in range(3)
        ),
        _resolution("g-clean-3.1", "g-clean-3", ability=(_BOLT_KEY,)),
        # The creature spell itself resolving: its key is the implicit permanent
        # spell, which the sidecar lists as dropped. Refused (FR-148).
        _resolution("g-clean-3.cast", "g-clean-3", ability=(_BEAR_CAST_KEY,)),
        # A runtime-only key (past every spell index the face declared): kept.
        _resolution("g-clean-3.level", "g-clean-3", ability=(_BEAR_LEVEL_UP_KEY,)),
    ]

    # Two source directories, the way a real corpus keeps a full-strength
    # collection run apart from a depleted one (FR-149): the tainted game is
    # full-strength by construction, and no game spans the two.
    records_dir = root / "records"
    write_shard(records_dir / "full-strength" / "run.0-a.jsonl.gz", shard_a)
    write_shard(records_dir / "depleted" / "run.0-b.jsonl.gz", shard_b)

    return CorpusFixture(
        records=records_dir,
        cards=(root / "cardsfolder", root / "tokenscripts"),
    )


@pytest.fixture
def a_corpus(tmp_path: Path) -> CorpusFixture:
    return _write_corpus(tmp_path / "source")


@pytest.fixture(scope="module")
def shared_corpus(tmp_path_factory) -> CorpusFixture:
    """``a_corpus`` written once, for the tests that never add to it."""
    return _write_corpus(tmp_path_factory.mktemp("shared") / "source")


@pytest.fixture(scope="module")
def default_build(tmp_path_factory, shared_corpus) -> CorpusStore:
    """``shared_corpus`` built once under the defaults, for the tests that only
    read what the build wrote."""
    return shared_corpus.build(tmp_path_factory.mktemp("default-build") / "curated")


def test_build_writes_the_three_strata_and_a_manifest(default_build):
    store = default_build
    assert store.manifest_path.exists()
    assert list(store.training_dir.glob("*.jsonl.gz"))
    assert store.load().digest()


def test_a_build_writes_the_legacy_token_key_fields_empty(default_build):
    """Older manifests carry them and a digest hashes them, so they stay, unset."""
    manifest = default_build.load()
    assert manifest.token_keys_remapped == 0
    assert manifest.token_keys_ambiguous == {}
    assert manifest.forge_tokenscripts == ""


def test_no_training_record_belongs_to_a_withheld_game(default_build):
    store = default_build
    manifest = store.load()
    withheld = set(manifest.card_disjoint_games) | set(manifest.game_disjoint_games)
    assert withheld
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert not trained & withheld


def test_no_training_record_names_a_held_out_card(default_build):
    """FR-088: the exclusion is by game, and every held-out game is dropped."""
    from effects.domain.split import (
        HeldOutCards,
        record_names_held_out_card,
    )

    manifest = default_build.load()
    held_out = HeldOutCards(
        names=frozenset(manifest.held_out_cards), script_files=frozenset(),
    )
    for record in read_records(default_build.training_dir):
        assert not record_names_held_out_card(record, held_out)


def test_the_manifest_records_the_held_out_texts(default_build):
    """Fix round 1, Finding 1: ``build()`` must populate ``held_out_texts``.

    A ``--corpus`` training run's stratum-sizing guard (``check_holdout``)
    reads ``HeldOutCards.texts`` alone, built from this field — not from
    ``held_out_cards``. A manifest that recorded only the held-out cards would
    make every ``--corpus`` run see an empty text set and report the
    card-disjoint stratum as empty regardless of the dataset's actual health.
    """
    assert default_build.load().held_out_texts == (_HELD_OUT_TEXT,)


def test_a_validation_stratum_keeps_whole_games(default_build):
    """A probe and the combat record it mirrors land together (FR-136)."""
    store = default_build
    mirrored = 0
    for directory in (store.card_disjoint_dir, store.game_disjoint_dir):
        records = list(read_records(directory))
        by_id = {r.record_id for r in records}
        for record in records:
            if record.mirror_of is not None:
                mirrored += 1
                assert record.mirror_of in by_id
    assert mirrored, "the fixture must contain a mirrored probe or this proves nothing"


def test_the_cap_trims_a_text_that_exceeds_it(tmp_path, shared_corpus):
    manifest = shared_corpus.build(tmp_path / "curated", text_cap=2).load()
    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) > 0


def test_two_builds_of_one_corpus_agree(tmp_path, shared_corpus, default_build):
    second = shared_corpus.build(tmp_path / "again")

    assert default_build.load().digest() == second.load().digest()
    assert (
        [r.record_id for r in read_records(default_build.training_dir)]
        == [r.record_id for r in read_records(second.training_dir)]
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
            tmp_path / "curated", holdout=HoldoutRule(permille=0, unit=HoldoutUnit.TEXT),
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


# ── fix round 1: _output_name must be injective (review Finding 1) ─────


def test_output_name_is_injective_on_the_reviewers_counterexample():
    """``"/" -> "__"`` collided: ``"a_/b"`` and ``"a/_b"`` both became
    ``"a___b"``. Two source shards writing the same output filename is a
    silent overwrite (``write_shard`` truncates), so this has to hold.
    """
    assert _part_name("a_/b") != _part_name("a/_b")


def test_output_name_flattens_a_subdirectory_path():
    assert _part_name("depleted/run.0-a.jsonl.gz") == "depleted_srun.0-a.jsonl.gz"


def _part_name(name: str) -> str:
    return SourceShard(name=name, size=0).part_name


# ── fix round 1: per-stratum counts and unique texts (review Finding 2) ─


def test_the_manifest_records_per_stratum_counts_and_unique_texts(default_build):
    """FR-143/FR-145: per-class counts alone leave both validation strata
    with no unique-text figure anywhere in the manifest. A test that only
    checked the two fields exist would not catch them being populated empty,
    so this checks actual values against strata ``a_corpus`` is built to
    populate: the tainted game (card-disjoint) and whichever clean game the
    seeded game-disjoint draw takes both carry a resolvable ability.
    """
    manifest = default_build.load()
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
#: ``validation_sample=0``: this corpus's card-disjoint stratum holds one
#: ``resolution-effect`` record and the mixture under test names neither
#: resolution class, so its sample is empty by construction and the
#: empty-sample guard would refuse the build. That guard is exercised against
#: this same fixture below; here the subject is the mixture.
_MIXTURE = dict(
    text_cap=_MIX_TEXT_CAP, class_mix={"rewrite": 0.5, "combat": 0.5},
    game_disjoint_share=0.0, game_disjoint_keyword_share=0.0, validation_sample=0,
)


@pytest.fixture(scope="module")
def a_mixture_corpus(tmp_path_factory) -> CorpusFixture:
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
    root = tmp_path_factory.mktemp("mixture-source")
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

    return CorpusFixture(
        records=records_dir, cards=(root / "cardsfolder",),
    )


@pytest.fixture(scope="module")
def mixture_build(tmp_path_factory, a_mixture_corpus) -> CorpusStore:
    return a_mixture_corpus.build(tmp_path_factory.mktemp("mixture") / "curated", **_MIXTURE)


def _delivered_shares(training_dir: Path) -> dict[str, float]:
    from effects.domain.sampling import sampling_class

    counts: Counter[str] = Counter()
    for record in read_records(training_dir):
        counts[sampling_class(record)] += 1
    total = sum(counts.values())
    return {name: count / total for name, count in counts.items()}


def test_the_delivered_class_mixture_is_the_one_the_manifest_records(mixture_build):
    """FR-139: a class is written at its share or its shortfall is recorded.

    ``class_capped_records`` was summed from each shard's own ``CapHeap``
    thresholds while the real cap is corpus-wide, so availability was
    over-estimated by a factor of the shard count for exactly the classes the
    cap touches. ``build()`` divides each class's target by that availability
    to get its admission rate, so every capped class under-delivered against
    the mixture the manifest went on to record.
    """
    delivered = _delivered_shares(mixture_build.training_dir)
    requested = mixture_build.load().class_mix
    for name, share in requested.items():
        assert abs(delivered.get(name, 0.0) - share) < 0.05, (
            f"{name}: delivered {delivered.get(name, 0.0):.3f} against a "
            f"recorded share of {share:.3f}"
        )


def test_the_manifest_records_what_was_delivered_beside_what_was_asked_for(mixture_build):
    """The requested mixture is a claim; the delivered one is a fact, and the
    dataset's whole value is that its manifest is true."""
    manifest = mixture_build.load()
    delivered = _delivered_shares(mixture_build.training_dir)
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
    store = a_corpus.build(out)

    orphans = [
        store.training_dir / "orphan.jsonl.gz",
        store.card_disjoint_dir / "orphan.jsonl.gz",
        store.game_disjoint_dir / "orphan.jsonl.gz",
    ]
    for orphan in orphans:
        write_shard(orphan, [_resolution("orphan.1", "g-orphan", ability=(_BOLT_KEY,))])

    a_corpus.build(out)

    assert not any(orphan.exists() for orphan in orphans)
    assert store.manifest_path.exists()
    assert "g-orphan" not in {r.game_id for r in read_records(store.training_dir)}


def test_verify_leaves_the_written_dataset_alone(tmp_path, a_corpus):
    out = tmp_path / "curated"
    store = a_corpus.build(out)
    before = sorted(p.name for p in store.training_dir.glob("*"))

    a_corpus.build(out, verify=True)

    assert sorted(p.name for p in store.training_dir.glob("*")) == before


def test_no_empty_output_shard_is_written(default_build):
    """One output shard per source shard per stratum, written unconditionally,
    turned 701 source shards into 2,103 files, most of them empty.

    They are not free. `train-effect-model --corpus` lists both validation
    directories and reads every shard in them before the first training step,
    logging a line each; an empty *training* shard drawn into an epoch
    forfeits its share of the step budget, because `_train_on_shard` returns
    early on a shard with nothing in it.
    """
    store = default_build
    written = [
        shard
        for directory in (
            store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir,
        )
        for shard in directory.glob("*.jsonl.gz")
    ]
    assert written, "the fixture must write something or this proves nothing"
    empty = [shard for shard in written if not any(True for _ in read_shard(shard))]
    assert not empty, f"empty shards written: {[str(s) for s in empty]}"


def test_the_trainer_refuses_a_real_dataset_built_on_the_other_surface(default_build):
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

    assert default_build.load().surface == "prose"

    with pytest.raises(SurfaceMismatchError, match="prose"):
        train(TrainEffectModelConfig(
            corpus=str(default_build.directory),
            vocab_path=Path("models/effects/vocab-script.txt"),
        ))


# ── what a count may be, and what zero means for each ─────────────────


@pytest.mark.parametrize("field,flag", [
    ("text_cap", "--text-cap"),
    ("card_disjoint_text_cap", "--card-disjoint-text-cap"),
    ("reuse_cap", "--reuse-cap"),
    ("training_records", "--training-records"),
])
def test_a_negative_count_is_refused(tmp_path, shared_corpus, field, flag):
    with pytest.raises(ValueError, match=flag):
        shared_corpus.config(tmp_path / "curated", **{field: -1})


def test_a_text_cap_of_zero_means_no_cap_rather_than_keep_nothing(
    tmp_path, shared_corpus,
):
    """Zero admits everything, the way ``CapHeap`` reads it. Documented here
    because the other reading — keep nothing — is the one the word "cap"
    suggests, and it would silently empty the training corpus."""
    manifest = shared_corpus.build(
        tmp_path / "curated", text_cap=0,
        # One class, so the class mixture asks for every record the class
        # holds: whatever is dropped then is the text cap's doing.
        class_mix={"resolution-effect": 1.0},
    ).load()

    assert sum(c.dropped_by_cap for c in manifest.per_class.values()) == 0
    assert manifest.per_stratum["training"] > 0


def test_a_card_disjoint_text_cap_of_zero_means_no_cap_either(tmp_path, shared_corpus):
    store = shared_corpus.build(tmp_path / "curated", card_disjoint_text_cap=0)

    assert store.load().card_disjoint_games == ("g-tainted",)


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


@pytest.fixture(scope="module")
def a_corpus_with_variants(tmp_path_factory):
    """The ordinary fixture plus a variant tree and a shard that uses it."""
    root = tmp_path_factory.mktemp("variants") / "source"
    fixture = _write_corpus(root)
    _write_variant(root, _VARIANT_FILE, _VARIANT_TEXT)
    write_shard(
        fixture.records / "variants" / "run.0-v.jsonl.gz",
        [
            _resolution(f"g-var.{i}", "g-var", ability=(_VARIANT_KEY,))
            for i in range(6)
        ],
    )
    return fixture, str(root / "variant-scripts")


@pytest.fixture(scope="module")
def variant_build(tmp_path_factory, a_corpus_with_variants) -> CorpusStore:
    """``a_corpus_with_variants`` built with its variant tree."""
    fixture, variant_root = a_corpus_with_variants
    output = tmp_path_factory.mktemp("variant-build") / "curated"
    return fixture.build(output, variant_scripts=variant_root)


def test_a_variant_text_is_capped_and_weighted_like_any_other(variant_build):
    """FR-138/FR-141 do not exempt the variant tree.

    Without `--variant-scripts` a variant line resolves to no text at all, so
    it lands in neither the rarity table nor the per-text cap while every real
    ability is in both — uncapped records competing against capped ones.
    """
    assert _VARIANT_TEXT in variant_build.load().rarity


def test_without_the_variant_tree_a_variant_text_resolves_to_nothing(
    tmp_path: Path, a_corpus_with_variants,
):
    """The state this flag exists to end; pinned so the fix cannot regress."""
    fixture, _ = a_corpus_with_variants

    store = fixture.build(tmp_path / "curated")

    assert _VARIANT_TEXT not in store.load().rarity


def test_the_manifest_records_the_variant_tree_it_read(a_corpus_with_variants, variant_build):
    """Which trees resolved decides which texts the rarity table names."""
    _, variant_root = a_corpus_with_variants
    assert variant_build.load().variant_scripts == variant_root


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
    assert SourceShard(name="depleted/run.0-a.jsonl.gz", size=0).source == "depleted"
    assert SourceShard(name="run.0-a.jsonl.gz", size=0).source == "."


def test_defective_records_reach_no_output_and_are_counted(tmp_path, a_corpus):
    """A junk resolution record in a training game and one in a held-out game both vanish.

    The second is an event flood -- a whole game that landed on one record --
    rather than the unattributed record it used to be: an unattributed event
    is a watch statistic now, not a refusal (FR-148).
    """
    from effects.domain.record_quality import QualityDefect
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("junk-train", game="g-clean-1", ability=()),
        a_corpus.resolution(
            "junk-held", game="g-tainted",
            events=tuple(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("P0",),
                    attributed_to="root",
                )
                for _ in range(65)
            ),
        ),
    ])
    store = corpus.build(tmp_path / "out")
    ids = {r.record_id for d in (store.training_dir, store.card_disjoint_dir,
                                  store.game_disjoint_dir, store.gate_one_dir)
           for r in read_records(d)}
    assert "junk-train" not in ids and "junk-held" not in ids
    # ``no-acting-text`` is the fixture's own ``g-clean-3.cast``, refused by
    # the third rule in every build this file runs.
    assert store.load().quality_dropped == {
        QualityDefect.NO_ABILITY: 1, QualityDefect.EVENT_FLOOD: 1,
        QualityDefect.NO_ACTING_TEXT: 1,
    }


def test_a_resolution_record_acting_only_through_a_dropped_key_is_refused(default_build):
    """FR-148's third rule, end to end through both passes.

    ``g-clean-3.cast`` acts through the implicit permanent spell alone — a
    ``dropped_keys`` entry — so it would reach the head with an empty acting
    slot. ``g-clean-3.level`` acts through a key past everything the face
    declared, which is a real ability whose text the join cannot reach yet,
    and is kept.
    """
    manifest = default_build.load()
    assert manifest.quality_dropped.get("no-acting-text") == 1
    assert manifest.no_acting_text_scripts == {_BEAR_FILE: 1}
    written = {r.record_id for r in read_records(default_build.training_dir)}
    written |= {r.record_id for r in read_records(default_build.card_disjoint_dir)}
    written |= {r.record_id for r in read_records(default_build.game_disjoint_dir)}
    assert "g-clean-3.cast" not in written
    assert "g-clean-3.level" in written


def test_a_record_acting_through_an_unjoinable_key_is_refused_not_fatal(tmp_path, a_corpus):
    """FR-148's fourth rule, end to end through both passes.

    A charm cast through Flashback keys its mode as a trait plus an ``option``
    no converted line carries, inside a range the sidecar declares. The join
    raises on that key; the build refuses the one record and writes the rest.
    """
    from effects.domain.record_quality import QualityDefect

    mode_key = ProvenanceKey(_BEAR_FILE, 0, "spell", 1, option=0)
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution("flashback-mode", game="g-clean-1", ability=(mode_key,)),
    ])
    store = corpus.build(tmp_path / "curated")
    assert store.load().quality_dropped.get(QualityDefect.UNJOINABLE_KEY) == 1
    written = {
        r.record_id
        for d in (store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir)
        for r in read_records(d)
    }
    assert "flashback-mode" not in written
    assert "g-clean-3.level" in written


def _tallies(
    *, read: Counter, refused_by_class: Counter, quality_dropped: Counter,
    unjoinable_scripts: Counter | None = None,
) -> tuple[RefusalTally, Counter]:
    """A survey's refusals and kept counts, the two halves the guard reads."""
    refusals = RefusalTally(
        by_reason=quality_dropped, by_class=refused_by_class,
        unjoinable_scripts=unjoinable_scripts or Counter(),
    )
    return refusals, read


def test_unjoinable_keys_past_the_share_stop_the_build():
    """Many records failing the join is a reconversion, not a collector gap."""
    tallies = _tallies(
        read=Counter({"resolution-effect": 990}),
        refused_by_class=Counter({"resolution-effect": 30}),
        quality_dropped=Counter({"unjoinable-key": 30}),
        unjoinable_scripts=Counter({_BEAR_FILE: 30}),
    )

    with pytest.raises(BuildCorpusError) as raised:
        _check_refusals(*tallies)

    message = str(raised.value)
    assert "reconverted" in message
    assert _BEAR_FILE in message


def test_unjoinable_keys_under_the_share_raise_nothing():
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 99_970}),
        refused_by_class=Counter({"resolution-effect": 30}),
        quality_dropped=Counter({"unjoinable-key": 30}),
    ))


def test_a_few_unjoinable_keys_in_a_small_corpus_raise_nothing():
    """A pilot of a few hundred records holding one Flashback cast builds."""
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 14}),
        refused_by_class=Counter({"resolution-effect": 1}),
        quality_dropped=Counter({"unjoinable-key": 1}),
    ))


def test_the_gate_one_slice_holds_held_out_resolutions_of_card_disjoint_games(default_build):
    store = default_build
    manifest = store.load()
    slice_ = list(read_records(store.gate_one_dir))
    assert slice_, "g-tainted resolves the held-out text, so the slice is not empty"
    assert all(r.kind is RecordKind.RESOLUTION for r in slice_)
    assert all(r.game_id in manifest.card_disjoint_games for r in slice_)
    assert all(_HELD_OUT_KEY in (r.ability or ()) for r in slice_)
    card_disjoint_ids = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert {r.record_id for r in slice_} <= card_disjoint_ids
    assert manifest.per_stratum["gate-one"] == len(slice_)


def test_games_are_reported_per_source_directory(shared_corpus, default_build):
    """Shards live under depleted/ and full-strength/ in the fixture."""
    manifest = default_build.load()
    assert set(manifest.games_by_source) == set(shared_corpus.source_dirs)
    assert sum(manifest.games_by_source.values()) == shared_corpus.game_count
    assert all(manifest.held_out_games_by_source.get(s, 0) <= n
               for s, n in manifest.games_by_source.items())


def test_output_shards_are_repacked_and_the_parts_removed(tmp_path, shared_corpus):
    # ``game_disjoint_target=1`` as everywhere else in this file: the default
    # 1000 takes all three clean games into the game-disjoint stratum, leaving
    # the training output — the one this test reads back — empty.
    store = shared_corpus.build(tmp_path / "out", shard_records=2)
    assert not store.parts_dir.exists()
    names = sorted(p.name for p in store.training_dir.glob("*.jsonl.gz"))
    assert names and names[0] == "shard-00001.jsonl.gz"
    assert store.load().shard_records == 2


def test_the_build_writes_a_validation_sample_per_stratum(tmp_path, shared_corpus):
    store = shared_corpus.build(tmp_path / "out", validation_sample=4)
    for stratum in ("card-disjoint", "game-disjoint"):
        assert store.sample_path(stratum).exists()
        assert 0 < sum(1 for _ in read_shard(store.sample_path(stratum))) <= 4 * 8
    assert store.load().validation_sample == 4


def test_the_max_events_help_says_zero_disables_the_rule():
    """The zero reading must survive in --help, not just in the docstring:
    this is what would have caught it silently vanishing."""
    from effects.infrastructure.cli import build_parser

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
    the withdrawn unattributed rule the class was 98.9% refused on the real
    corpus — a statement about the rule, not about the records.
    """
    corpus = a_corpus.with_extra_records([
        replace(
            _combat("g-clean-1.combat", "g-clean-1"),
            payload=CombatPayload(attackers=("E0",), events=(_unresolved_event(),)),
        ),
    ])
    store = corpus.build(tmp_path / "out")

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
        corpus.build(tmp_path / "out")

    message = str(raised.value)
    assert "unresolved" in message and "degraded" in message


def test_a_refused_rebuild_leaves_the_previous_dataset_in_place(tmp_path, a_corpus):
    """The survey sees every refusal the write pass would, so a corpus the
    rules reject is refused before the old strata are cleared."""
    out = tmp_path / "out"
    a_corpus.build(out)
    before = CorpusStore(out).load().digest()
    flooded = a_corpus.with_extra_records([
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

    with pytest.raises(BuildCorpusError, match="combat"):
        flooded.build(out)

    store = CorpusStore(out)
    assert store.load().digest() == before
    assert any(store.output_dir(Stratum.TRAINING).iterdir())


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
        a_mixture_corpus.build(tmp_path / "curated", **{**_MIXTURE, "validation_sample": 8})


def test_a_kept_record_with_an_unresolved_event_is_counted_not_refused(
    tmp_path, a_corpus,
):
    """FR-148: unattributed events are a watch statistic in the manifest.

    The refusal that used to be here threw away an eighth of the resolution
    outcomes on the real corpus, the "died" zone outcome among them. The
    count is what replaces it: a rising share is a statement about the
    collector's attribution, not about the corpus.
    """
    # In ``g-tainted`` (card-disjoint), which is admitted whole: the training
    # output is subsampled to the eight-class mixture and this fixture holds
    # two of the eight, so a training record is not a reliable place to look
    # for one particular id.
    corpus = a_corpus.with_extra_records([
        a_corpus.resolution(
            "unattributed-1", game="g-tainted",
            events=(_unresolved_event(),),
        ),
    ])
    store = corpus.build(tmp_path / "out")

    manifest = store.load()
    assert manifest.unattributed_records >= 1
    assert "unattributed-events" not in manifest.quality_dropped
    kept = {r.record_id for r in read_records(store.card_disjoint_dir)}
    assert "unattributed-1" in kept


def test_a_class_lost_to_no_acting_text_is_told_to_reconvert():
    """The third cause the refusal guard names, beside a degraded checkout and
    a rule that does not fit: sidecars written before the converter claimed
    the traits whose text its lines carry. An operator reading only "unpatched
    checkout" would go looking at Forge, and the corpus is the thing to fix.
    """
    tallies = _tallies(
        read=Counter({"resolution-effect": 10}),
        refused_by_class=Counter({"resolution-effect": 90}),
        quality_dropped=Counter({"no-acting-text": 88, "no-ability": 2}),
    )

    with pytest.raises(BuildCorpusError) as raised:
        _check_refusals(*tallies)

    message = str(raised.value)
    assert "no-acting-text" in message
    assert "reconvert" in message


def test_a_class_that_keeps_most_of_its_records_raises_nothing():
    """The guard is for a class lost whole, not for the few percent the rules
    are there to take."""
    _check_refusals(*_tallies(
        read=Counter({"resolution-effect": 95}),
        refused_by_class=Counter({"resolution-effect": 5}),
        quality_dropped=Counter({"no-acting-text": 5}),
    ))


# ── gen-2 selection, placement and refusals (T036, T069, T080, T083) ─────


#: Gen-2's copies up to the default ``--reuse-cap``, on the legacy holdout and
#: game split, with no keyword table.
_GEN2_RULES = {
    **{rule: value for rule, value in _LEGACY_RULES.items() if rule != "reuse_cap"},
    "keyword_definitions": None,
}


def _training_ids(store: CorpusStore) -> list[str]:
    return sorted(r.record_id for r in read_records(store.training_dir))


def test_a_display_name_match_on_a_non_keyword_line_fails_the_build(tmp_path, a_corpus):
    """Spec Story 4 scenario 8 / FR-015."""
    import json

    definitions = tmp_path / "keyword-definitions.json"
    definitions.write_text(json.dumps({
        _BOLT_TEXT.upper(): {"reminder_template": "a reminder", "generated_script": None},
    }), encoding="utf-8")

    with pytest.raises(BuildCorpusError, match="FR-015"):
        a_corpus.build(tmp_path / "out", rules=_GEN2_RULES, keyword_definitions=definitions)


def test_a_records_set_mixing_generations_is_refused(tmp_path, a_corpus):
    """FR-033: before any record is read."""
    import gzip
    import json

    from effects.infrastructure.record_codec import record_to_dict

    gen1 = record_to_dict(_resolution("old.1", "g-old", ability=(_BOLT_KEY,)))
    del gen1["random_seat"]
    path = a_corpus.records / "full-strength" / "old.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(gen1) + "\n")

    with pytest.raises(BuildCorpusError, match="mixes gen-1 shards"):
        a_corpus.build(tmp_path / "out", rules=_GEN2_RULES)


def test_the_retired_game_count_flag_is_rejected_by_argparse(capsys):
    """Spec Story 4 scenario 10: ``--game-disjoint-share`` replaced it."""
    from effects.infrastructure.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["build-corpus", "--game-disjoint-games", "1000"])
    assert "--game-disjoint-games" in capsys.readouterr().err


def test_the_game_disjoint_keywords_parse_as_a_comma_separated_list():
    from effects.infrastructure.cli import build_parser

    args = build_parser().parse_args([
        "build-corpus", "--game-disjoint-keywords", "trample,infect",
    ])
    assert args.game_disjoint_keywords == ("trample", "infect")


def test_an_unscored_game_disjoint_keyword_is_refused(tmp_path, shared_corpus):
    with pytest.raises(BuildCorpusError, match="flying"):
        shared_corpus.config(tmp_path, rules=_GEN2_RULES, game_disjoint_keywords=("flying",))


@pytest.fixture(scope="module")
def gen2_build(tmp_path_factory, shared_corpus) -> CorpusStore:
    """``shared_corpus`` built once under :data:`_GEN2_RULES`."""
    output = tmp_path_factory.mktemp("gen2-build") / "curated"
    return shared_corpus.build(output, rules=_GEN2_RULES)


def test_two_builds_with_one_seed_write_the_same_training_records(
    tmp_path, shared_corpus, gen2_build,
):
    """FR-052."""
    again = shared_corpus.build(tmp_path / "two", rules=_GEN2_RULES)

    assert _training_ids(gen2_build) == _training_ids(again)


def test_random_seat_changes_no_selection(tmp_path, shared_corpus, gen2_build):
    """FR-050: the off-policy flag is reported, never selected on."""
    flipped_root = tmp_path / "flipped"
    for shard in iter_shards(shared_corpus.records):
        records = [replace(r, random_seat=True) for r in read_shard(shard)]
        write_shard(flipped_root / shard.relative_to(shared_corpus.records), records)
    flipped = shared_corpus.build(
        tmp_path / "flipped-out", rules=_GEN2_RULES, records_dir=flipped_root,
    )

    assert _training_ids(gen2_build) == _training_ids(flipped)
    counts = flipped.load().policy_counts
    assert counts and all(c.get("on_policy", 0) == 0 for c in counts.values())


def test_a_rebuild_over_a_grown_corpus_keeps_every_game_disjoint_game(tmp_path, a_corpus):
    """SC-007."""
    first = set(
        a_corpus.build(tmp_path / "first", rules=_GEN2_RULES).load().game_disjoint_games
    )
    assert first

    a_corpus.with_extra_records([
        CorpusFixture.resolution(f"g-more-{i}.1", game=f"g-more-{i}") for i in range(40)
    ])
    grown = set(
        a_corpus.build(tmp_path / "grown", rules=_GEN2_RULES).load().game_disjoint_games
    )

    assert first <= grown


def test_the_manifest_records_families_policy_and_the_flags(tmp_path, shared_corpus):
    """FR-053."""
    manifest = shared_corpus.build(tmp_path / "out", rules=_GEN2_RULES, reuse_cap=3).load()

    assert manifest.holdout_unit == "text"
    assert manifest.reuse_cap == 3
    assert manifest.game_disjoint_share == 0.16
    assert manifest.game_disjoint_target is None
    effect = manifest.families["resolution-effect"]
    assert all(
        set(row.as_dict()) == {
            "available", "share", "planned", "written", "repeats", "shortfall",
        }
        for row in effect.values()
    )
    written = sum(row.written for row in effect.values())
    assert written == manifest.per_class["resolution-effect"].kept
    assert manifest.signatures["resolution-effect"]
    assert manifest.policy_counts["resolution-effect"]["on_policy"] == written


def test_a_short_family_is_repeated_up_to_the_reuse_cap(tmp_path, a_corpus):
    """FR-049 step 1 through the whole build: a family below its share repeats."""
    from effects.domain.records import PlayabilityBlockersPayload, PlayabilitySubkind

    legality = [
        make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
            record_id=f"g-clean-1.leg{i}", game_id="g-clean-1", timestamp=_TIMESTAMP,
            what_if=i >= 2, payload=PlayabilityBlockersPayload(anchor_attacker=f"E{i}"),
        )
        for i in range(22)
    ]
    a_corpus.with_extra_records(legality)
    manifest = a_corpus.build(
        tmp_path / "out", rules=_GEN2_RULES,
        class_mix={"playability-legality": 1.0}, reuse_cap=4, validation_sample=0,
    ).load()

    # 22 distinct records set the class budget. Half of it, 11, is the real
    # half's share; its two records can supply at most 8 with four copies
    # each, and the what-if half is planned the 14 left over.
    families = manifest.families["playability-legality"]
    assert families["real:none"].written == 8
    assert families["real:none"].repeats == 6
    assert families["real:none"].shortfall == 3
    # Records with no acting text are sampled by hash rate: 14 in expectation.
    what_if = families["what-if:none"].written
    assert 8 <= what_if <= 20
    assert manifest.legality_counts["blockers"] == {"real": 8, "what-if": what_if}


# ── what the plan counts, and what it may train on (G4, G13, G14) ──────


def test_a_held_back_game_takes_no_training_slot(tmp_path, a_corpus):
    """A text's capped slots go to its training games' records (G4).

    Counted over every kept record, a text's smallest hashes could all sit in
    a held-out game: the plan's threshold admitted those, the write pass then
    routed the game away, and the text's slots went unwritten. Gen-2 wrote
    82% of the records it planned.
    """
    write_shard(a_corpus.records / "full-strength" / "run.0-held.jsonl.gz", [
        _resolution(
            "g-held-2.0", "g-held-2", ability=(_BOLT_KEY,),
            entity_names=(_HELD_OUT_CARD,),
        ),
        *(
            _resolution(f"g-held-2.shock{n}", "g-held-2", ability=(_SHOCK_KEY,))
            for n in range(60)
        ),
    ])
    store = a_corpus.build(tmp_path / "curated", text_cap=2)

    shock = [r for r in read_records(store.training_dir) if r.ability == (_SHOCK_KEY,)]
    assert len(shock) == 2


def test_every_class_delivers_its_target(tmp_path, shared_corpus):
    """The manifest shows targeted beside delivered, so a shortfall is visible."""
    manifest = shared_corpus.build(tmp_path / "curated", text_cap=2).load()

    assert manifest.class_targets
    for klass, target in manifest.class_targets.items():
        assert manifest.per_class[klass].kept == target


def test_a_held_out_game_resolving_no_held_out_text_is_dropped(tmp_path, a_corpus):
    """Only a resolution counts a game toward its held-out text (G13, FR-142).

    Gate 1 and the gate-one slice score resolution records, so a game whose
    held-out text appears only on a rewrite holds nothing gate 1 measures.
    """
    write_shard(a_corpus.records / "full-strength" / "run.0-rewrite.jsonl.gz", [
        _rewrite("g-held-3.0", "g-held-3", ability=(_HELD_OUT_KEY,)),
        _resolution(
            "g-held-3.1", "g-held-3", ability=(_SHOCK_KEY,),
            entity_names=(_HELD_OUT_CARD,),
        ),
    ])
    store = a_corpus.build(tmp_path / "curated")
    manifest = store.load()

    assert "g-held-3" not in manifest.card_disjoint_games
    assert "g-tainted" in manifest.card_disjoint_games
    trained = {r.game_id for r in read_records(store.training_dir)}
    assert "g-held-3" not in trained


def test_a_token_printing_a_held_out_text_holds_its_game_out(tmp_path, a_corpus):
    """A token script is a carrier, so a game it appears in is held out (G14)."""
    token_file = "tokenscripts/r_2_2_goblin_held.txt"
    token_key = ProvenanceKey(token_file, 0, "spell", 1)
    _write_card(
        tmp_path / "source", token_file, "Goblin Held", _HELD_OUT_TEXT,
        index_within_kind=1,
    )
    write_shard(a_corpus.records / "full-strength" / "run.0-token.jsonl.gz", [
        replace(
            _resolution("g-token.0", "g-token", ability=(_SHOCK_KEY,)),
            state=make_snapshot(entities=(
                make_entity("E1", "Goblin Token", printed=(token_key,)),
            )),
        ),
    ])
    store = a_corpus.build(tmp_path / "curated")

    trained = {r.game_id for r in read_records(store.training_dir)}
    assert "g-token" not in trained
    assert "g-token" not in store.load().game_disjoint_games


def test_a_variant_printing_a_held_out_text_trains_on_nothing(tmp_path, a_corpus):
    """A training game's record acting through a held-out text is dropped (G14).

    The game names no held-out card, so it stays in training; the one record
    would teach the very text the card-disjoint stratum measures.
    """
    variant_file = "variant-scripts/held_out_bears_variant_1.txt"
    variant_key = ProvenanceKey(variant_file, 0, "spell", 0)
    root = tmp_path / "source"
    path = root / "variant-scripts" / "held_out_bears_variant_1.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HELD_OUT_TEXT + "\n", encoding="utf-8")
    write_sidecar(
        ProvenanceSidecar(
            card="Held Out Bears Variant", script_file=variant_file,
            lines=(SidecarLine(
                line_index=0, line_kind="spell", provenance=(variant_key,),
                script_text=_HELD_OUT_TEXT,
            ),),
        ),
        sidecar_path_for(path),
    )
    write_shard(a_corpus.records / "variants" / "run.0-v.jsonl.gz", [
        _resolution("g-var.0", "g-var", ability=(variant_key,)),
        _resolution("g-var.1", "g-var", ability=(_SHOCK_KEY,)),
    ])
    # Every resolution record the class can supply is wanted, so only the
    # held-out text can keep the variant's record out.
    store = a_corpus.build(
        tmp_path / "curated", variant_scripts=str(root / "variant-scripts"), text_cap=0,
        class_mix={"resolution-effect": 1.0},
    )

    training = list(read_records(store.training_dir))
    assert not [r for r in training if r.ability == (variant_key,)]
    assert any(r.game_id == "g-var" for r in training)
    assert store.load().per_stratum["dropped-held-out-text"] == 1
