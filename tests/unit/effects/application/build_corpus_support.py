"""The raw corpus the build-corpus tests share: its fixtures and record builders.

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

import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from effects.application.build_corpus import build
from effects.application.build_corpus_config import BuildCorpusConfig
from effects.application.corpus_survey import RefusalTally
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar, SidecarLine
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    ForkKind,
    RecordKind,
    RewritePayload,
)
from effects.domain.split import HoldoutRule
from effects.domain.state_snapshot import StateSnapshot
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import (
    list_shards,
    read_records,
)
from effects.infrastructure.sidecar_io import sidecar_path_for
from tests.unit.effects.builders import (
    make_entity,
    make_snapshot,
    write_ended_shard,
    write_sidecar,
    write_vocabulary,
)
from tests.unit.effects.builders import make_record as _any_generation_record


def make_record(kind: RecordKind = RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    """A gen-3 record: ``build-corpus`` reads gen-3 shards alone (FR-020)."""
    return _any_generation_record(kind, **{"generation": 3, **overrides})


_HELD_OUT_CARD = "Held Out Bears"


#: Found by brute force over ``text_is_held_out(masked_template(...),
#: permille=20)`` (the default rule): constructed, not guessed at.
_HELD_OUT_TEXT = "deals 10 damage to any target held out effect ac"


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
                    script_text=text, prose_offset=0,
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
        fork=fork, fork_kind=ForkKind.PROBE if fork else None, mirror_of=mirror_of,
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


#: A catalogue of one scenario declared without a detector and one heads line
#: the map does not mark ``trained``: no scenario tags a record and no rule is
#: budgeted, so these hand-built corpora show the curation alone (a build
#: refuses a catalogue with no scenario or no heads line); the gen-3
#: fixture's tests below read the real ``rules/``.
_NO_RULES = Path(__file__).parents[3] / "fixtures" / "effects" / "minimal-rules"


#: The rules most of this file was written against: the held-out text above is
#: held out under the ``template`` unit, ``0.16`` places exactly one
#: game (``g-clean-3``) in the game-disjoint stratum, and one copy per record.
#: Placement's precision thresholds are off (research E5): no half needs a
#: minimum of games, so nothing is promoted past the base draw, and no
#: component a precision target, so a build of four games exits 0.
_LEGACY_RULES = dict(
    holdout=HoldoutRule(),
    game_disjoint_share=0.16, game_disjoint_keyword_share=0.16,
    reuse_cap=1, rules_dir=_NO_RULES, min_games_per_half=0,
    max_component_error=math.inf,
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
            record = replace(record, payload=replace(record.payload, events=events))
        return record

    def config(
        self, output: Path, *, rules: dict = _LEGACY_RULES, **overrides,
    ) -> BuildCorpusConfig:
        """A one-worker build of this corpus under ``rules``; ``overrides``
        replace any field, a rule included."""
        return BuildCorpusConfig(**{
            "records_dir": self.records, "cards_folders": self.cards,
            "output": output, "workers": 1,
            "vocab_path": str(write_vocabulary(self.records.parent / "vocab.txt")),
            "keyword_definitions": None, **rules, **overrides,
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
        write_ended_shard(self.records / "full-strength" / "run.0-extra.jsonl.gz", records)
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
    write_ended_shard(records_dir / "full-strength" / "run.0-a.jsonl.gz", shard_a)
    write_ended_shard(records_dir / "depleted" / "run.0-b.jsonl.gz", shard_b)

    return CorpusFixture(
        records=records_dir,
        cards=(root / "cardsfolder", root / "tokenscripts"),
    )


def _dataset(out: Path) -> dict[str, bytes]:
    """Every file of a built dataset, by its path under ``out``."""
    return {str(path.relative_to(out)): path.read_bytes()
            for path in sorted(out.rglob("*")) if path.is_file()}


def _line(record: EffectRecord) -> bytes:
    import json

    from effects.infrastructure.record_codec import record_to_dict

    return (json.dumps(record_to_dict(record)) + "\n").encode("utf-8")


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


#: The copies up to the default ``--reuse-cap``, on the legacy holdout and
#: game split, with no keyword table.
_BUDGET_FREE_RULES = {
    **{rule: value for rule, value in _LEGACY_RULES.items() if rule != "reuse_cap"},
    "keyword_definitions": None,
}


def _training_ids(store: CorpusStore) -> list[str]:
    return sorted(r.record_id for r in read_records(store.training_dir))


#: One rule the budget tests' own catalogue marks ``trained``: shown by a
#: resolution holding two damage events, which no a_corpus record holds.
_BUDGET_RULE = "120.3"


def _budget_rules(root: Path) -> Path:
    """A rules directory of one scenario and its heads line (FR-060a)."""
    import json

    (root / "scenarios").mkdir(parents=True)
    (root / "heads").mkdir()
    (root / "scenarios" / "120.json").write_text(json.dumps({"unit": "120", "scenarios": [{
        "id": "120.3-1", "rules": [_BUDGET_RULE], "record_kind": "resolution",
        "variants": [{"label": "base", "polarity": "fires"}],
        "detector": {"family": "outcome_event", "record_kind": "resolution",
                     "params": {"event": "damage_dealt", "min_count": 2}, "needs": []},
    }]}), encoding="utf-8")
    (root / "heads" / "120.jsonl").write_text(json.dumps({
        "id": _BUDGET_RULE, "output": "trained", "outputs": ["damage_taken"],
    }) + "\n", encoding="utf-8")
    return root


def _written(store: CorpusStore) -> list[EffectRecord]:
    return [r for d in (store.training_dir, store.card_disjoint_dir, store.game_disjoint_dir,
                        store.gate_one_dir) for r in read_records(d)]
