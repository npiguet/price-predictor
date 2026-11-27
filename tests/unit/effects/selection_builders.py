"""Selection sets, control scores, gate scores and cached vectors for the
evaluation and control-score tests."""

from __future__ import annotations

#: The strata a fixture selection set halves.
FIXTURE_STRATA: tuple[str, ...] = ("card-disjoint", "game-disjoint")


def fixture_selection_set(records, *, corpus_digest: str = "fixture"):
    """A selection set over ``records`` with every component holding items in
    both strata: both strata hold every game of ``records``, split in two by
    game, each half's items over its own games' records: two probe types, the
    real damage grid (its 64 cells, edited by ``grid_edits.GridEdits``) over
    records with an acting line and a creature, a rule test, and a deviance
    sample whose gate, three per-entity fields, a verdict field and two
    created-objects terms the freeze kept."""
    from effects.domain.effect_head_layout import GATE_TERM
    from effects.domain.grid_edits import VALUES
    from effects.domain.selection_set import (
        DEVIANCE,
        FrozenGrid,
        GridCell,
        ProbeRecord,
        SampleRecord,
        SelectionSet,
        TaggedRecord,
        item_key,
    )

    def creature(record):
        return next((e for e in record.state.entities
                     if e.pt is not None and e.is_creature and e.on_battlefield), None)

    games = sorted({r.game_id for r in records})
    halves = {"selection": games[::2], "reporting": games[1::2]}
    by_half = {half: [r for r in records if r.game_id in set(held)]
               for half, held in halves.items()}
    probes, grids, rules, samples, errors = {}, {}, {}, {}, {}
    for stratum in FIXTURE_STRATA:
        probes[stratum], grids[stratum], rules[stratum], samples[stratum] = {}, {}, {}, {}
        for half, held in by_half.items():
            with_entities = [r for r in held if r.state.entities]
            probes[stratum][half] = {
                "dies": tuple(ProbeRecord(r.record_id, r.game_id, r.state.entities[0].id,
                                          float(i % 2)) for i, r in enumerate(with_entities[:12])),
                "affordable": tuple(ProbeRecord(r.record_id, r.game_id, None, float(i % 2))
                                    for i, r in enumerate(held[:10])),
            }
            damaged = [(r, creature(r)) for r in held if r.ability and creature(r) is not None]
            kept = damaged[:2]
            cells = tuple(GridCell(f"{a}x{b}", (float(a >= b),) * len(kept))
                          for a in VALUES for b in VALUES)
            grids[stratum][half] = {"damage": FrozenGrid(cells, tuple(
                TaggedRecord(r.record_id, r.game_id, (entity.id,)) for r, entity in kept))}
            tagged = [TaggedRecord(r.record_id, r.game_id, (r.state.entities[0].id,))
                      for r in with_entities[:16]]
            rules[stratum][half] = {
                "r.board": {"fires": tuple(tagged[:8]), "does_not_fire": tuple(tagged[8:])},
            }
            samples[stratum][half] = tuple(SampleRecord(r.record_id, r.game_id, 0)
                                           for r in held[:20])
            for name in (GATE_TERM, "zone_outcome", "damage_taken", "pt_change",
                         "verdict.can_play", "created_objects.present", "created_objects.count"):
                errors[item_key(stratum, half, DEVIANCE, name)] = {"hard": 0.05, "soft": 0.05}
    return SelectionSet(
        corpus_digest=corpus_digest, r=4,
        halves={s: {h: tuple(g) for h, g in halves.items()} for s in FIXTURE_STRATA},
        probes=probes, probe_kinds={"dies": "binary", "affordable": "binary"},
        grids=grids, rule_tests=rules, deviance_samples=samples,
        standard_errors=errors, left_out={}, settings={"max_item_error": 0.1},
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
