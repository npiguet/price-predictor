"""Selection sets, control scores, gate scores and cached vectors for the
evaluation and control-score tests."""

from __future__ import annotations

#: The strata a fixture selection set halves.
FIXTURE_STRATA: tuple[str, ...] = ("card-disjoint", "game-disjoint")


def _creature(record):
    """The record's first creature on the battlefield with a size, or None."""
    return next((e for e in record.state.entities
                 if e.pt is not None and e.is_creature and e.on_battlefield), None)


def _with_entities(held) -> list:
    return [r for r in held if r.state.entities]


def _probes(held) -> dict:
    """Two binary probe types: ``dies`` on a board's first entity, and
    ``affordable`` on the record."""
    from effects.domain.selection_set import ProbeRecord

    return {
        "dies": tuple(ProbeRecord(r.record_id, r.game_id, r.state.entities[0].id, float(i % 2))
                      for i, r in enumerate(_with_entities(held)[:12])),
        "affordable": tuple(ProbeRecord(r.record_id, r.game_id, None, float(i % 2))
                            for i, r in enumerate(held[:10])),
    }


def _damage_grid(held) -> dict:
    """The real damage grid's 64 cells over the first two records with an
    acting line and a creature."""
    from effects.domain.grid_edits import VALUES
    from effects.domain.selection_set import FrozenGrid, GridCell, TaggedRecord

    damaged = [(r, _creature(r)) for r in held if r.ability and _creature(r) is not None]
    kept = damaged[:2]
    cells = tuple(GridCell(f"{a}x{b}", (float(a >= b),) * len(kept))
                  for a in VALUES for b in VALUES)
    return {"damage": FrozenGrid(cells, tuple(
        TaggedRecord(r.record_id, r.game_id, (entity.id,)) for r, entity in kept))}


def _rule_test(held) -> dict:
    """One rule test: the first eight boards fire it, the next eight do not."""
    from effects.domain.selection_set import TaggedRecord

    tagged = [TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
              for r in _with_entities(held)[:16]]
    return {"r.board": {"fires": tuple(tagged[:8]), "does_not_fire": tuple(tagged[8:])}}


def _deviance_sample(held) -> tuple:
    from effects.domain.selection_set import SampleRecord

    return tuple(SampleRecord(r.record_id, r.game_id, 0) for r in held[:20])


def _component(build, by_half) -> dict:
    """``build`` of each half's records, in both strata."""
    return {stratum: {half: build(held) for half, held in by_half.items()}
            for stratum in FIXTURE_STRATA}


def _standard_errors(halves) -> dict:
    """The deviance sample's kept terms in each stratum and half: the gate,
    three per-entity fields, a verdict field and two created-objects terms."""
    from effects.domain.effect_head_layout import GATE_TERM
    from effects.domain.selection_set import DEVIANCE, item_key

    terms = (GATE_TERM, "zone_outcome", "damage_taken", "pt_change",
             "verdict.can_play", "created_objects.present", "created_objects.count")
    return {item_key(stratum, half, DEVIANCE, name): {"hard": 0.05, "soft": 0.05}
            for stratum in FIXTURE_STRATA for half in halves for name in terms}


def fixture_selection_set(records, *, corpus_digest: str = "fixture"):
    """A selection set over ``records`` with every component holding items in
    both strata: both strata hold every game of ``records``, split in two by
    game, each half's items over its own games' records: two probe types, the
    real damage grid (its 64 cells, edited by ``grid_edits.GridEdits``) over
    records with an acting line and a creature, a rule test, and a deviance
    sample whose gate, three per-entity fields, a verdict field and two
    created-objects terms the freeze kept."""
    from effects.domain.selection_set import SelectionSet

    games = sorted({r.game_id for r in records})
    halves = {"selection": games[::2], "reporting": games[1::2]}
    by_half = {half: [r for r in records if r.game_id in set(held)]
               for half, held in halves.items()}
    return SelectionSet(
        corpus_digest=corpus_digest, r=4,
        halves={s: {h: tuple(g) for h, g in halves.items()} for s in FIXTURE_STRATA},
        probes=_component(_probes, by_half), probe_kinds={"dies": "binary", "affordable": "binary"},
        grids=_component(_damage_grid, by_half), rule_tests=_component(_rule_test, by_half),
        deviance_samples=_component(_deviance_sample, by_half),
        standard_errors=_standard_errors(halves), left_out={}, settings={"max_item_error": 0.1},
    ).sealed()


#: The rule map's outputs of :func:`fixture_selection_set`'s rule test.
FIXTURE_RULE_OUTPUTS: dict[str, tuple[str, ...]] = {
    "r.board": ("gate", "zone_outcome", "damage_taken", "can_play", "trigger_fired",
                "created_objects"),
}


def fixture_selection(records):
    """A ``FrozenSelection`` of :func:`fixture_selection_set` whose strata
    both read ``records`` as one shard."""
    from effects.application.control_score import FrozenSelection

    records = list(records)
    return FrozenSelection(fixture_selection_set(records),
                           dict.fromkeys(FIXTURE_STRATA, lambda: [records]))


def control_score_of(value: float, weight: float = 0.5):
    """A ``ControlScore`` whose every component of both strata is ``value``."""
    from effects.application.control_score import ControlScore, StratumScore
    from effects.domain.selection_set import COMPONENTS

    return ControlScore({stratum: StratumScore(dict.fromkeys(COMPONENTS, value), {})
                         for stratum in FIXTURE_STRATA}, weight)


def keyword_score(qualifying: int, agreeing: int):
    """Gate 2's score with one observation per qualifying record: a combat
    with a single carrier."""
    from effects.application.gate_two import KeywordScore

    return KeywordScore(
        qualifying=qualifying, agreeing=agreeing, observations=qualifying,
    )


def read_cache(abilities_root, sidecar_roots, *, surface, variant: str = "full"):
    """One variant's cache over freshly read sidecars, the way the evaluator
    reads it: :func:`cached_vectors` over a ``SidecarIndex``, its stamp left
    unchecked (the stamp's tests are ``test_encode_abilities_stamp.py``)."""
    from effects.application.geometry_checks import cached_vectors
    from effects.infrastructure.ability_cache_store import AbilityCacheStore, SidecarIndex

    return cached_vectors(
        SidecarIndex(sidecar_roots, surface=surface),
        AbilityCacheStore(abilities_root, variant=variant),
    )
