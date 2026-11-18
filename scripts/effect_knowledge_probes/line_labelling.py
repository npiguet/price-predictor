"""What the line and pair labels observe: effect profiles, mana, triggers and their causes.

Nothing here does work at import time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from effect_knowledge_probes import common, labels


class ObservedOutcomes:
    """What each text was seen to do: its effect profiles and the mana it made.

    Observed one record at a time and kept as per-text lists of small arrays,
    so the probe games and the training shards feed one set of labels without
    either being held. A record is observed once, however many rows of it a
    stratum holds: the training stratum repeats each copy curation wrote, and
    a probe of training records drops the repeats (FR-070).
    """

    def __init__(self, acting_text) -> None:
        self._acting_text = acting_text
        self._seen: set[str] = set()
        self.profiles: dict[str, list[np.ndarray]] = defaultdict(list)
        self.mana: dict[str, list[float]] = defaultdict(list)

    def _resolution_text(self, record) -> str | None:
        """The acting text of a resolution record's ability, or None."""
        from effects.domain.records import Moment, RecordKind

        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return None
        return self._acting_text(record.ability[0]) if record.ability else None

    def observe(self, record) -> None:
        if record.record_id in self._seen:
            return
        self._seen.add(record.record_id)
        text = self._resolution_text(record)
        if not text:
            return
        profile = labels.effect_profile(record)
        if profile is not None:
            self.profiles[text].append(profile)
        produced = labels.mana_observed(record)
        if produced is not None:
            self.mana[text].append(produced)


@dataclass
class ProbeGames:
    """The trigger join and the triggers of every distinct probe-game record,
    for the interaction pairs."""

    join: labels.TriggerJoin = field(default_factory=labels.TriggerJoin)
    #: ``trigger record id -> (trigger key, fired)``.
    triggers: dict[str, tuple] = field(default_factory=dict)

    def observe(self, record) -> None:
        from effects.domain.records import TriggerPayload

        self.join.observe(record)
        if isinstance(record.payload, TriggerPayload) and record.ability:
            self.triggers[record.record_id] = (record.ability[0], record.payload.fired)


@dataclass
class RecordObservers:
    """The two observers every record of a half is fed to."""

    games: ProbeGames
    outcomes: ObservedOutcomes

    def observe(self, record) -> None:
        self.games.observe(record)
        self.outcomes.observe(record)


@dataclass(frozen=True, slots=True)
class LineLabels:
    """One ability line's labels, keyed by the provenance of its first carrier.

    ``weights`` holds a per-label weight where a label is an estimate over few
    records (the state-dependence spread).
    """

    key: object
    labels: dict[str, float]
    weights: dict[str, float] = field(default_factory=dict)


def _type_line(item, sidecars) -> str:
    """The lower-cased type line of the card carrying ``item``'s script, or ``""``."""
    from effects.infrastructure.sidecar_io import converted_text_path

    try:
        path = sidecars.path_for(item.script_file)
    except KeyError:
        path = None
    return common.card_facts(converted_text_path(path))["type_line"].lower() if path else ""


def _item_labels(item, spread, outcomes: ObservedOutcomes, sidecars) -> LineLabels | None:
    """One line's script labels plus what its records were seen to do, or None."""
    found = labels.line_labels(item.script_text, item.line_kind, item.api,
                               _type_line(item, sidecars))
    weights = {}
    if item.text in spread:
        found["spread"], n = spread[item.text]
        weights["spread"] = labels.state_weight(n)
    if outcomes.mana.get(item.text):
        found["observed_mana"] = float(np.mean(outcomes.mana[item.text]))
    return LineLabels(item.key, found, weights) if found else None


def label_lines(line_items, outcomes: ObservedOutcomes, sidecars) -> list[LineLabels]:
    """Every encoded line's script labels, plus what its records were seen to do."""
    spread = labels.state_dependence(outcomes.profiles)
    every = (_item_labels(item, spread, outcomes, sidecars) for item in line_items)
    return [entry for entry in every if entry is not None]


def _signature(a, b) -> tuple[str, str]:
    """A pair of provenance keys as the two canonical JSON strings."""
    return (json.dumps(a.as_dict(), sort_keys=True), json.dumps(b.as_dict(), sort_keys=True))


def _joined_labels(games: ProbeGames, joined: labels.JoinResult) -> dict[tuple, float]:
    """``(trigger, cause) signature -> fired`` for every trigger the join found a cause for."""
    pair_labels: dict[tuple, float] = {}
    for record_id, (trigger_key, fired) in games.triggers.items():
        cause = joined.cause.get(record_id)
        if cause is None:
            continue
        pair = _signature(trigger_key, cause)
        pair_labels[pair] = max(pair_labels.get(pair, 0.0), float(fired))
    return pair_labels


def _mined(line_items, pair_labels: dict[tuple, float], limit: int) -> list[dict]:
    """The first ``limit`` mined pairs, by stable hash, that the join did not find."""
    mined = []
    for a, b in labels.mined_pairs(line_items):
        signature = _signature(a, b)
        if signature not in pair_labels:
            mined.append((common.stable_hash("|".join(signature)), a, b))
    return [{"a": a.as_dict(), "b": b.as_dict(), "label": 1.0, "source": "mined"}
            for _, a, b in sorted(mined, key=lambda row: row[0])[:limit]]


def interaction_pairs(games: ProbeGames, line_items) -> tuple[list[dict], labels.JoinResult]:
    """Trigger lines with their causes: joined from the games, then mined from scripts.

    Mined pairs are positives by construction and far outnumber the joined
    ones, so they are capped at the joined count: uncapped, a probe could score
    by telling a mined pair from a joined one rather than reading the pair.
    """
    joined = games.join.result()
    pair_labels = _joined_labels(games, joined)
    pairs = [{"a": json.loads(a), "b": json.loads(b), "label": label, "source": "join"}
             for (a, b), label in sorted(pair_labels.items())]
    return pairs + _mined(line_items, pair_labels, len(pairs)), joined
