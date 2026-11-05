"""The frozen rule tests of one half, scored for a probing run's scorecard
(FR-073, FR-074; research E6; story 8 acceptance 6).

A rule test ranks its ``fires`` records above its ``does_not_fire`` records
by the model's prediction of each output the manifest's map names for the
rule, read through ``probe_readouts`` at the entities the tag names (their
mean), or at ``[ACT]`` for a verdict output. An output read here is one a
probe readout answers:

==================  ==================================================
output              read as
==================  ==================================================
``gate``            ``affected``
``zone_outcome``    each zone outcome's probability (``HeadReading``)
``damage_taken``    ``damage_taken``
``target_legal``    ``target_legal``
``blocker_legal``   ``may_block``
``attacker_legal``  ``may_attack``
``trigger_fired``   ``fires``
``can_play``        ``can_play``
``affordable``      ``affordable``
``timing``          ``timing``
==================  ==================================================

The selection set names a rule's records and polarities, not the scenario's
category or direction, so both are read off the records' own outcomes, as the
control score reads them: the category (of ``zone_outcome``) whose observed
share differs most between the polarities, and the sign of that difference,
which turns the reading round where the scenario's outcome is an absence. An
output no ``fires`` and no ``does_not_fire`` record observes, or whose
polarities observe the same, is left out; so is an output no readout answers,
and a rule with no output left. Each rule scores the mean over its outputs of
the AUC, and ``2·AUC − 1`` beside it, the control score's scale.

Nothing here does work at import time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from effect_knowledge_probes import common

FIRES, DOES_NOT_FIRE = "fires", "does_not_fire"


# ── what each output reads and observes ─────────────────────────────────


def _targets(record):
    from effects.domain.effect_targets import derive_targets

    try:
        return derive_targets(record)
    except ValueError:
        return None


def _entry(record, entity):
    targets = _targets(record)
    if targets is None or not targets.gated:
        return None, False
    return targets.get(entity), True


def _observed_gate(record, entity):
    entry, gated = _entry(record, entity)
    return [float(bool(entry is not None and entry.affected))] if gated else None


def _observed_zone(record, entity):
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField

    entry, gated = _entry(record, entity)
    if not gated:
        return None
    index = 0 if entry is None else entry.fields.get(HeadField.ZONE_OUTCOME) or 0
    hot = [0.0] * len(ZONE_OUTCOMES)
    hot[int(index)] = 1.0
    return hot


def _observed_damage(record, entity):
    from effects.domain.effect_head_layout import HeadField

    entry, gated = _entry(record, entity)
    if not gated:
        return None
    return [float((entry.fields.get(HeadField.DAMAGE_TAKEN) or 0) if entry else 0)]


def _candidate(record):
    from effects.domain.records import PlayabilityDecisionPayload

    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload) and payload.candidates:
        return payload.candidates[0]
    return None


def _observed_target_legal(record, entity):
    candidate = _candidate(record)
    if candidate is None or not candidate.legal_targets:
        return None
    return [float(entity in candidate.legal_targets)]


def _observed_listed(payload_type: str, member: str):
    def observe(record, entity):
        from effects.domain import records

        payload = record.payload
        if not isinstance(payload, getattr(records, payload_type)):
            return None
        return [float(entity in getattr(payload, member))]

    return observe


def _observed_candidate(member: str):
    def observe(record, entity):
        candidate = _candidate(record)
        value = None if candidate is None else getattr(candidate, member)
        return None if value is None else [float(value)]

    return observe


def _observed_fired(record, entity):
    from effects.domain.records import TriggerPayload

    payload = record.payload
    return [float(payload.fired)] if isinstance(payload, TriggerPayload) else None


def _zone_candidates(outputs, verdict, row: int, slot: int) -> list[float]:
    from effects.domain.head_decoding import HeadReading

    return [float(v) for v in HeadReading(outputs[row, slot]).zone_probabilities()]


def _readout(name: str) -> Callable[..., list[float]]:
    from effects.domain.probe_readouts import read

    return lambda outputs, verdict, row, slot: [read(name, outputs, verdict, row, slot)]


@dataclass(frozen=True)
class OutputReader:
    """How one map output is read off the model and observed on a record."""

    per_entity: bool
    candidates: Callable[..., list[float]]
    observed: Callable[[object, str | None], list[float] | None]


def readers() -> dict[str, OutputReader]:
    """Every map output a probe readout answers, by output name."""
    from effects.domain.rule_mask import GATE_OUTPUT

    return {
        GATE_OUTPUT: OutputReader(True, _readout("affected"), _observed_gate),
        "zone_outcome": OutputReader(True, _zone_candidates, _observed_zone),
        "damage_taken": OutputReader(True, _readout("damage_taken"), _observed_damage),
        "target_legal": OutputReader(True, _readout("target_legal"), _observed_target_legal),
        "blocker_legal": OutputReader(True, _readout("may_block"), _observed_listed(
            "PlayabilityBlockersPayload", "legal_blockers")),
        "attacker_legal": OutputReader(True, _readout("may_attack"), _observed_listed(
            "PlayabilityAttackersPayload", "legal_attackers")),
        "trigger_fired": OutputReader(False, _readout("fires"), _observed_fired),
        "can_play": OutputReader(False, _readout("can_play"), _observed_candidate("can_play")),
        "affordable": OutputReader(False, _readout("affordable"),
                                   _observed_candidate("affordable")),
        "timing": OutputReader(False, _readout("timing"), _observed_candidate("timing")),
    }


# ── reading the model ───────────────────────────────────────────────────


@dataclass
class _Ask:
    """One record's reading for one rule output, at its entities or ``[ACT]``."""

    rule: str
    output: str
    polarity: str
    entities: tuple[str | None, ...]


def _read(probe_model, records: list, asks: Mapping[str, list[_Ask]],
          by_name: Mapping[str, OutputReader], *, chunk: int = 32) -> dict:
    """``(record_id, output, entity) -> candidates`` for every ask."""
    from effects.domain.probe_readouts import entity_slots

    out: dict = {}
    for start in range(0, len(records), chunk):
        part = records[start : start + chunk]

        def read(outputs, verdict, surface, row, part=part):
            record = part[row]
            slots = entity_slots(surface)
            found = {}
            for ask in asks[record.record_id]:
                reader = by_name[ask.output]
                for entity in ask.entities:
                    slot = slots.get(entity) if reader.per_entity else 0
                    if slot is not None:
                        found[(ask.output, entity)] = reader.candidates(
                            outputs, verdict, row, slot)
            return found

        for record, found in zip(part, common.predict(probe_model, part, read)):
            for (output, entity), values in found.items():
                out[(record.record_id, output, entity)] = values
    return out


# ── scoring ─────────────────────────────────────────────────────────────


def _settle(observed: Mapping[str, list[np.ndarray]]) -> tuple[int, float] | None:
    """The candidate and sign the polarities' observed outcomes choose; None
    when a polarity observed nothing or both observed the same."""
    if not observed[FIRES] or not observed[DOES_NOT_FIRE]:
        return None
    difference = np.mean(observed[FIRES], axis=0) - np.mean(observed[DOES_NOT_FIRE], axis=0)
    candidate = int(np.argmax(np.abs(difference)))
    if difference[candidate] == 0:
        return None
    return candidate, -1.0 if difference[candidate] < 0 else 1.0


def score(probe_model, read_rules: Mapping[str, Mapping[str, Sequence[tuple]]],
          rule_outputs: Mapping[str, Sequence[str]]) -> dict:
    """``{"rules": {rule: {...}}, "left_out": {rule: why}, "component": mean}``
    for one half's rule tests: each rule's records per polarity, its outputs'
    AUCs, their mean (``auc``) and ``2·AUC − 1`` (``score``); ``component``
    is the mean over rules of ``score`` clipped below at −1."""
    from effects.domain.score_scales import auc, clipped, rescaled_auc

    by_name = readers()
    asks: dict[str, list[_Ask]] = defaultdict(list)
    records: dict[str, object] = {}
    unread: dict[str, list[str]] = {}
    for rule, sides in read_rules.items():
        outputs = [o for o in rule_outputs.get(rule, ()) if o in by_name]
        unread[rule] = sorted(set(rule_outputs.get(rule, ())) - set(outputs))
        for polarity in (FIRES, DOES_NOT_FIRE):
            for record, entities in sides.get(polarity, ()):
                for output in outputs:
                    at = tuple(entities) if by_name[output].per_entity else (None,)
                    if at:
                        records[record.record_id] = record
                        asks[record.record_id].append(_Ask(rule, output, polarity, at))
    values = _read(probe_model, list(records.values()), asks, by_name)

    # rule -> output -> polarity -> [(reading candidates, observed)] per record.
    gathered: dict = defaultdict(lambda: defaultdict(lambda: {FIRES: [], DOES_NOT_FIRE: []}))
    for record_id, record_asks in asks.items():
        record = records[record_id]
        for ask in record_asks:
            reader = by_name[ask.output]
            read = [values[(record_id, ask.output, e)] for e in ask.entities
                    if (record_id, ask.output, e) in values]
            seen = [o for o in (reader.observed(record, e) for e in ask.entities)
                    if o is not None]
            if read:
                gathered[ask.rule][ask.output][ask.polarity].append(
                    (np.mean(read, axis=0), np.mean(seen, axis=0) if seen else None))

    rules: dict[str, dict] = {}
    left_out: dict[str, str] = {}
    for rule, sides in sorted(read_rules.items()):
        counts = {p: len(sides.get(p, ())) for p in (FIRES, DOES_NOT_FIRE)}
        if not all(counts.values()):
            left_out[rule] = "one polarity only"
            continue
        by_output: dict[str, float] = {}
        for output, polarities in sorted(gathered.get(rule, {}).items()):
            settled = _settle({p: [seen for _, seen in rows if seen is not None]
                               for p, rows in polarities.items()})
            if settled is None:
                continue
            candidate, sign = settled
            fires = [sign * read[candidate] for read, _ in polarities[FIRES]]
            does_not = [sign * read[candidate] for read, _ in polarities[DOES_NOT_FIRE]]
            value = auc([True] * len(fires) + [False] * len(does_not), [*fires, *does_not])
            if not math.isnan(value):
                by_output[output] = float(value)
        if not by_output:
            left_out[rule] = ("no output a probe readout answers"
                              if not set(rule_outputs.get(rule, ())) - set(unread[rule])
                              else "no output whose polarities' outcomes differ")
            continue
        mean = float(np.mean(list(by_output.values())))
        rules[rule] = {"records": counts, "outputs": by_output, "auc": mean,
                       "score": rescaled_auc(mean), "unread": unread[rule]}
    scores = [clipped(body["score"]) for body in rules.values()]
    return {"rules": rules, "left_out": left_out,
            "component": float(np.mean(scores)) if scores else float("nan")}
