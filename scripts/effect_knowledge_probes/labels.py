"""The ten probe families' label extractors (FR-079).

Every label comes from a source that exists without the model: the parsed
script, or an outcome observed in the effect-record corpus. The family table and
the section on assembled labels of
``experiments/2026-09-19-effect-knowledge-probes-design.md`` are the
specification; this module is that table in code.

Two kinds of item:

* a **line** item is one ability text, keyed by provenance, labelled from its
  script (and, for the state-dependence and observed-mana labels, from the
  records the text resolved in). It does not depend on the board.
* a **record** item is one record — and, for a target about one entity, one
  entity of its board — labelled from what the record observed. It depends on
  the board, so the ladder reads it at every rung.

The interaction family adds **pair** items: a trigger line with the ability
whose event fired (or failed to fire) it.

Nothing here does work at import time.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from effects.domain.script_text import api_of, params
from effects.domain.script_text import segments as script_segments

FAMILIES: tuple[str, ...] = (
    "magnitudes", "mana-production", "mana-usage", "timing-costs", "interactions",
    "side", "evasion-blocking", "target-legality", "duration-repeatability",
    "state-dependence",
)

BINARY = "binary"
AMOUNT = "amount"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One probe target: its family, what it is read at, and how it is scored.

    ``scope`` is ``line`` (one text), ``pair`` (two texts), ``act`` (a record,
    read at ``[ACT]``) or ``entity`` (one entity of a record's board, read at
    its ``[CARD]`` slot). ``kind`` picks AUC (``binary``) or R² (``amount``).
    """

    family: str
    name: str
    scope: str
    kind: str


_COLOURS = ("w", "u", "b", "r", "g")

TARGETS: tuple[TargetSpec, ...] = (
    # ── magnitudes and thresholds ──
    *(TargetSpec("magnitudes", name, "line", AMOUNT) for name in (
        "damage", "power_change", "toughness_change", "counters_placed",
        "life_amount", "damage_per_mana",
    )),
    TargetSpec("magnitudes", "dies", "entity", BINARY),
    TargetSpec("magnitudes", "damage_taken", "entity", AMOUNT),
    # ── mana production ──
    *(TargetSpec("mana-production", f"produces_{c}", "line", BINARY)
      for c in (*_COLOURS, "c")),
    TargetSpec("mana-production", "any_colour", "line", BINARY),
    TargetSpec("mana-production", "mana_amount", "line", AMOUNT),
    TargetSpec("mana-production", "restricted", "line", BINARY),
    TargetSpec("mana-production", "conditional", "line", BINARY),
    TargetSpec("mana-production", "observed_mana", "line", AMOUNT),
    # ── mana usage ──
    *(TargetSpec("mana-usage", f"cost_{c}", "line", AMOUNT)
      for c in (*_COLOURS, "c", "generic")),
    TargetSpec("mana-usage", "cost_total", "line", AMOUNT),
    TargetSpec("mana-usage", "cost_reducer", "line", BINARY),
    TargetSpec("mana-usage", "cost_tax", "line", BINARY),
    TargetSpec("mana-usage", "affordable", "act", BINARY),
    # ── timing and non-mana costs ──
    TargetSpec("timing-costs", "instant_speed", "line", BINARY),
    TargetSpec("timing-costs", "tap_cost", "line", BINARY),
    TargetSpec("timing-costs", "sacrifice_cost", "line", BINARY),
    TargetSpec("timing-costs", "life_cost", "line", BINARY),
    # ── interactions ──
    TargetSpec("interactions", "fires", "act", BINARY),
    TargetSpec("interactions", "fires_pair", "pair", BINARY),
    # ── side ──
    TargetSpec("side", "affects_you", "line", BINARY),
    TargetSpec("side", "affects_opponent", "line", BINARY),
    TargetSpec("side", "affects_each", "line", BINARY),
    TargetSpec("side", "affected", "entity", BINARY),
    # ── evasion and blocking ──
    TargetSpec("evasion-blocking", "may_block", "entity", BINARY),
    # ── target legality ──
    TargetSpec("target-legality", "legal_target", "entity", BINARY),
    TargetSpec("target-legality", "legal_target_protected", "entity", BINARY),
    # ── duration and repeatability ──
    TargetSpec("duration-repeatability", "until_end_of_turn", "line", BINARY),
    TargetSpec("duration-repeatability", "as_counter", "line", BINARY),
    TargetSpec("duration-repeatability", "repeatable", "line", BINARY),
    TargetSpec("duration-repeatability", "pt_until_end_of_turn", "entity", BINARY),
    # ── state dependence ──
    TargetSpec("state-dependence", "spread", "line", AMOUNT),
)

TARGETS_BY_NAME: dict[str, TargetSpec] = {spec.name: spec for spec in TARGETS}


def capped(value: float) -> float:
    """``value`` with its magnitude at most the model's ``MAGNITUDE_CAP``, sign kept.

    The model is trained and scored on amounts capped there, so an amount label
    above it asks a probe for a number no training target ever held. A handful
    of script literals do exactly that — Jumbo Cactuar's +9999 power, Overkill's
    −9999 toughness, The Millennium Calendar's 1000 life — and a few items that
    far out dominate an R² taken over thousands.
    """
    from effects.domain.input_layout import MAGNITUDE_CAP

    return max(-MAGNITUDE_CAP, min(MAGNITUDE_CAP, float(value)))


# ── script parsing ──────────────────────────────────────────────────────

_INT_RE = re.compile(r"^[+-]?\d+$")


def segments(script: str | None) -> list[dict[str, str]]:
    """``Key$ value`` pairs of every segment of a (possibly chained) script.

    A one-segment text gives one entry; a chained text is ``seg [SEG] SV1: seg …``.
    """
    if not script or "$" not in script:
        return []
    return [params(segment) for segment in script_segments(script)]


def _literal(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    return float(int(value)) if _INT_RE.match(value) else None


def _sum_amount(parsed: list[dict[str, str]], keys: tuple[str, ...]) -> float | None:
    """A literal amount summed over segments; None where any is variable.

    0 where no segment states it, so "deals no damage" is a label like any
    other.
    """
    values = [_literal(found[key]) for found in parsed for key in keys if key in found]
    if None in values:
        return None
    total = 0.0
    for value in values:
        total += value
    return total


def _cost_shards(cost: str | None) -> list[str]:
    return re.sub(r"<[^>]*>", lambda m: m.group(0).replace(" ", "_"), cost or "").split()


def mana_of_cost(cost: str | None) -> dict[str, float] | None:
    """A ``Cost$``'s mana by colour, colourless and generic; None for ``X``.

    A hybrid pip counts for its first colour and a ``2/W`` pip as generic, so
    the colours sum to the mana the cost asks for.
    """
    out = {f"cost_{c}": 0.0 for c in (*_COLOURS, "c", "generic")}
    for shard in _cost_shards(cost):
        if shard == "X":
            return None
        if shard.isdigit():
            out["cost_generic"] += int(shard)
        elif shard in ("W", "U", "B", "R", "G", "C"):
            out[f"cost_{shard.lower()}"] += 1
        elif re.fullmatch(r"[WUBRG2]/[WUBRGP]", shard):
            first = shard[0]
            out["cost_generic" if first == "2" else f"cost_{first.lower()}"] += 1
    return out


_PT_APIS = frozenset({
    "Pump", "PumpAll", "PutCounter", "PutCounterAll", "Animate", "AnimateAll",
    "Debuff",
})
_COUNTER_APIS = frozenset({"PutCounter", "PutCounterAll"})
_SIDE_SKIP = ("SpellDescription", "TriggerDescription", "Description",
              "StackDescription", "TgtPrompt")
#: The stated amounts, by label. A static's ``AddPower$``/``AddToughness$`` is
#: not among them: reading it would relabel every probe set already frozen.
_AMOUNT_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("damage", ("NumDmg",)),
    ("power_change", ("NumAtt",)),
    ("toughness_change", ("NumDef",)),
    ("counters_placed", ("CounterNum",)),
    ("life_amount", ("LifeAmount",)),
)


@dataclass(frozen=True, slots=True)
class _Script:
    """One line's parsed script, as every family's labels read it."""

    parsed: list[dict[str, str]]
    line_kind: str
    #: The root segment's API, else the sidecar's.
    root_api: str | None
    cost: str | None
    #: The cost's mana by colour; None without a ``Cost$`` or for ``X``.
    cost_mana: dict[str, float] | None

    @property
    def root(self) -> dict[str, str]:
        return self.parsed[0]

    @property
    def apis(self) -> set[str]:
        return {api_of(found) for found in self.parsed} - {None}

    @property
    def total_cost(self) -> float:
        return sum(self.cost_mana.values()) if self.cost_mana else 0.0


def _is_charm_root(script: _Script) -> bool:
    return script.root_api == "Charm" or (
        "Choices" in script.root and script.root_api in {"Charm", "GenericChoice"})


def _stated_amounts(script: _Script) -> dict[str, float]:
    labels: dict[str, float] = {}
    for name, keys in _AMOUNT_KEYS:
        value = _sum_amount(script.parsed, keys)
        if value is not None:
            labels[name] = capped(value)
    return labels


def _has_damage_per_mana(labels: dict[str, float], script: _Script) -> bool:
    """Damage stated on a line with a mana cost of its own (never a spell's)."""
    return bool(labels.get("damage")) and script.total_cost > 0 and script.line_kind != "spell"


def _magnitudes(script: _Script) -> dict[str, float]:
    labels = {} if _is_charm_root(script) else _stated_amounts(script)
    if _has_damage_per_mana(labels, script):
        labels["damage_per_mana"] = labels["damage"] / script.total_cost
    return labels


def _is_conditional_mana(parsed: list[dict[str, str]]) -> bool:
    return any(k.startswith("Condition") or k == "ActivationLimit"
               for found in parsed for k in found)


def _mana_production(script: _Script) -> dict[str, float]:
    if "Mana" not in script.apis:
        return {}
    mana = next(found for found in script.parsed if api_of(found) == "Mana")
    produced = mana.get("Produced", "").split()
    labels = {f"produces_{colour}": float(colour.upper() in produced)
              for colour in (*_COLOURS, "c")}
    labels["any_colour"] = float("Any" in produced)
    amount = _literal(mana.get("Amount", "1"))
    if amount is not None:
        labels["mana_amount"] = amount
    labels["restricted"] = float(any(k.startswith("Restrict") for k in mana))
    labels["conditional"] = float(_is_conditional_mana(script.parsed))
    return labels


def _mana_usage(script: _Script) -> dict[str, float]:
    """Never a spell line's cost: its mana cost is its card's."""
    labels: dict[str, float] = {}
    if script.line_kind != "spell" and script.cost_mana is not None:
        labels.update(script.cost_mana)
        labels["cost_total"] = script.total_cost
    mode = script.root.get("Mode")
    if mode in ("ReduceCost", "RaiseCost"):
        labels["cost_reducer"] = float(mode == "ReduceCost")
        labels["cost_tax"] = float(mode == "RaiseCost")
    elif script.line_kind == "static":
        labels["cost_reducer"] = 0.0
        labels["cost_tax"] = 0.0
    return labels


def _timing_costs(script: _Script, types: str) -> dict[str, float]:
    labels: dict[str, float] = {}
    root = script.root
    if script.line_kind == "spell":
        labels["instant_speed"] = float("instant" in types)
    elif script.line_kind == "activated":
        labels["instant_speed"] = float(
            "SorcerySpeed" not in root and "ActivationPhases" not in root)
    if script.cost is not None:
        shards = _cost_shards(script.cost)
        labels["tap_cost"] = float("T" in shards)
        labels["sacrifice_cost"] = float(any(s.startswith("Sac<") for s in shards))
        labels["life_cost"] = float(any(s.startswith("PayLife<") for s in shards))
    return labels


def _side_values(script: _Script) -> str:
    """Every parameter value of the script but its descriptions, space-joined."""
    return " ".join(
        v for found in script.parsed for k, v in found.items() if k not in _SIDE_SKIP
    )


def _affects_you(defined: str, values: str) -> bool:
    return (defined in ("You", "Self") or "YouCtrl" in values
            or bool(re.search(r"\bYou\b", values)))


def _affects_each(script: _Script, defined: str) -> bool:
    root_api = script.root_api
    return (root_api.endswith("All") or root_api.endswith("EachPlayer")
            or defined == "Player" or "ValidCards" in script.root)


def _side(script: _Script) -> dict[str, float]:
    if script.root_api is None:
        return {}
    values = _side_values(script)
    defined = script.root.get("Defined", "")
    return {
        "affects_you": float(_affects_you(defined, values)),
        "affects_opponent": float("Opponent" in values or "OppCtrl" in values),
        "affects_each": float(_affects_each(script, defined)),
    }


def _duration_repeatability(script: _Script) -> dict[str, float]:
    labels: dict[str, float] = {}
    apis = script.apis
    if apis & _PT_APIS:
        durations = {found.get("Duration") for found in script.parsed
                     if api_of(found) in _PT_APIS}
        counter = bool(apis & _COUNTER_APIS)
        labels["as_counter"] = float(counter)
        labels["until_end_of_turn"] = float(
            not counter and not (durations & {"Permanent", "UntilHostLeavesPlay"}))
    if script.line_kind == "spell":
        labels["repeatable"] = 0.0
    elif script.line_kind == "activated":
        shards = _cost_shards(script.cost)
        labels["repeatable"] = float(not any(
            s.startswith(("Sac<1/CARDNAME", "Exile<1/CARDNAME")) for s in shards))
    return labels


def line_labels(
    script: str | None, line_kind: str, api: str | None, types: str = "",
) -> dict[str, float]:
    """Every script-derived label of one line, by target name.

    A target the line says nothing about is absent rather than zero where zero
    would be a claim: a line with no ``Cost$`` has no cost colours, a line that
    produces no mana has no production colours. Every amount is :func:`capped`,
    and damage per mana is read off the capped damage.
    """
    parsed = segments(script)
    if not parsed:
        return {}
    cost = parsed[0].get("Cost")
    parsed_script = _Script(
        parsed=parsed, line_kind=line_kind, root_api=api_of(parsed[0]) or api,
        cost=cost, cost_mana=mana_of_cost(cost) if cost is not None else None,
    )
    labels = {
        **_magnitudes(parsed_script),
        **_mana_production(parsed_script),
        **_mana_usage(parsed_script),
        **_timing_costs(parsed_script, types),
        **_side(parsed_script),
        **_duration_repeatability(parsed_script),
    }
    return {
        name: capped(value) if TARGETS_BY_NAME[name].kind == AMOUNT else value
        for name, value in labels.items()
    }


# ── observed outcomes: effect profiles, state dependence, mana ──────────

PROFILE_NAMES: tuple[str, ...] = (
    "log_affected", "died", "damaged", "zone_changed", "pt_changed", "countered",
    "life", "drawn",
)


def effect_profile(record) -> np.ndarray | None:
    """One resolution's outcome summary, the embedding probes' profile shares.

    None for a record that is not an effect half.
    """
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import Moment, RecordKind

    if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
        return None
    targets = derive_targets(record)
    affected = [t for t in targets.values() if t.affected]
    n = len(affected)
    died = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
    stayed = ZONE_OUTCOMES.index(ZoneOutcome.STAYED)

    def share(test) -> float:
        return sum(1 for t in affected if test(t.fields)) / n if n else 0.0

    return np.array([
        math.log1p(n),
        share(lambda f: f.get(HeadField.ZONE_OUTCOME) == died),
        share(lambda f: (f.get(HeadField.DAMAGE_TAKEN) or 0) > 0),
        share(lambda f: f.get(HeadField.ZONE_OUTCOME) not in (None, stayed)),
        share(lambda f: bool(f.get(HeadField.POWER_DELTA) or f.get(HeadField.TOUGHNESS_DELTA))),
        share(lambda f: any(k.startswith("counters_delta") for k in f)),
        share(lambda f: bool(f.get(HeadField.LIFE_DELTA))),
        share(lambda f: bool(f.get(HeadField.CARDS_DRAWN))),
    ], dtype=np.float64)


def state_dependence(profiles: dict[str, list[np.ndarray]]) -> dict[str, tuple[float, int]]:
    """``text -> (spread, n)``: how much a text's outcome varies across boards.

    The spread is the mean standard deviation of the profile's components over
    the text's resolutions. A text seen once has no spread and is left out; the
    probe weights the rest by ``n / (n + 5)``.
    """
    out: dict[str, tuple[float, int]] = {}
    for text, rows in profiles.items():
        if len(rows) < 2:
            continue
        matrix = np.stack(rows)
        out[text] = (float(matrix.std(axis=0).mean()), len(rows))
    return out


def state_weight(n: int) -> float:
    return n / (n + 5.0)


def mana_observed(record) -> float | None:
    """Mana a mana resolution produced, :func:`capped`; None where no event says.

    Each ``mana_produced`` event carries ``mana_by_color``, the collector's
    reading of the mana that actually arrived (Forge's produced-mana string,
    after any replacement): a count per colour letter, so ``{"C": 2}`` is a Sol
    Ring's two and ``{"W": 1, "B": 1}`` an Orzhov Basilica's two. A land that
    taps for one colour of several names the one it made. An empty map is
    mana the collector could not read, so it counts for nothing, and a record
    whose every event is empty has no label.
    """
    from effects.domain.event_schema import EventType
    from effects.domain.records import events_of

    total, seen = 0.0, False
    for event in events_of(record):
        if event.type is not EventType.MANA_PRODUCED:
            continue
        mana = event.params.get("mana_by_color") or {}
        counts = [v for v in mana.values() if isinstance(v, int | float)]
        if counts:
            seen = True
            total += sum(counts)
    return capped(total) if seen else None


# ── interactions: the trigger → resolution join ─────────────────────────


def _event_signature(event) -> tuple:
    zone = event.params.get("to_zone") if event.params else None
    return (str(event.type), tuple(sorted(event.subjects)), zone)


@dataclass(frozen=True, slots=True)
class JoinResult:
    """Each trigger record's cause, and how often the join found one.

    ``cause`` maps a trigger record id to the acting key of the resolution whose
    events contain the trigger's pending event.
    """

    cause: dict[str, object]
    fired: int
    fired_joined: int

    @property
    def join_rate(self) -> float:
        return self.fired_joined / self.fired if self.fired else float("nan")


class TriggerJoin:
    """Pair each trigger record with the resolution that produced its event (FR-078).

    No trigger record names the ability behind its event, so the cause is found
    by the event itself: a resolution record of the same game whose events
    include one with the same type, subjects and destination zone. A trigger is
    evaluated while the resolution is still running, and the resolution's
    record is written when it ends, so the first matching resolution at or
    after the trigger is preferred, then the latest one before it. The join
    rate is over fired triggers.

    Records are observed one at a time and only their event signatures are
    kept, so a freeze over hundreds of games never holds the records
    themselves.
    """

    def __init__(self) -> None:
        #: game -> [(timestamp, order, signatures, key, record_id, fired)]
        self._games: dict[str, list[tuple]] = defaultdict(list)

    def observe(self, record) -> None:
        from effects.domain.records import TriggerPayload, events_of

        rows = self._games[record.game_id]
        if _is_acting_resolution(record):
            signatures = tuple(_event_signature(e) for e in events_of(record))
            rows.append((record.timestamp, _RESOLUTION_ROW, signatures, record.ability[0],
                         None, None))
        elif isinstance(record.payload, TriggerPayload):
            rows.append((record.timestamp, _TRIGGER_ROW,
                         (_event_signature(record.payload.event),),
                         None, record.record_id, record.payload.fired))

    def result(self) -> JoinResult:
        tally = _JoinTally()
        for rows in self._games.values():
            ordered = sorted(rows, key=_row_order)
            producers = _producers(ordered)
            triggers = [row for row in ordered if row[1] != _RESOLUTION_ROW]
            for timestamp, _, signatures, _, record_id, was_fired in triggers:
                found = _cause_of(producers.get(signatures[0], ()), timestamp)
                tally.add(record_id, was_fired, found)
        return JoinResult(cause=tally.cause, fired=tally.fired, fired_joined=tally.joined)


#: A join row's kind: a resolution sorts before a trigger at the same timestamp.
_RESOLUTION_ROW = 0
_TRIGGER_ROW = 1


def _is_acting_resolution(record) -> bool:
    """A resolution record's effect half with an acting ability."""
    from effects.domain.records import Moment, RecordKind

    return (record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
            and bool(record.ability))


def _row_order(row: tuple) -> tuple:
    return (row[0], row[1])


def _producers(ordered: list[tuple]) -> dict[tuple, list[tuple[str, object]]]:
    """``signature ->`` the resolutions producing it, in time order."""
    producers: dict[tuple, list[tuple[str, object]]] = defaultdict(list)
    resolutions = [row for row in ordered if row[1] == _RESOLUTION_ROW]
    for timestamp, _, signatures, key, _, _ in resolutions:
        for signature in signatures:
            producers[signature].append((timestamp, key))
    return producers


def _cause_of(candidates, timestamp):
    """The first producer at or after ``timestamp``, else the latest before it."""
    after = [k for t, k in candidates if t >= timestamp]
    if after:
        return after[0]
    before = [k for t, k in candidates if t < timestamp]
    return before[-1] if before else None


@dataclass(slots=True)
class _JoinTally:
    """The join's running result: each trigger's cause, and the fired counts."""

    cause: dict[str, object] = field(default_factory=dict)
    fired: int = 0
    joined: int = 0

    def add(self, record_id: str, was_fired: bool, found) -> None:
        if was_fired:
            self.fired += 1
            self.joined += found is not None
        if found is not None:
            self.cause[record_id] = found


def mined_pairs(items: Iterable) -> list[tuple[object, object]]:
    """Positive pairs read off the scripts: what the join cannot reach.

    A death trigger with a sacrifice outlet, and a lifegain trigger with a
    lifegain ability, fire one another by construction. Each trigger gets one
    partner, chosen by ``crc32`` of the two texts: a fixed partner list would
    pair every death trigger with the same few outlets, and a probe could then
    recognise a mined pair by its partner alone.
    """
    lines = _MinedLines()
    for item in items:
        lines.add(item)
    return [*_partnered(lines.death, lines.sac),
            *_partnered(lines.lifegain_trigger, lines.lifegain)]


def _is_death_trigger(root: dict[str, str]) -> bool:
    return (root.get("Mode") == "ChangesZone" and root.get("Destination") == "Graveyard"
            and root.get("Origin") == "Battlefield")


def _has_sacrifice_cost(root: dict[str, str]) -> bool:
    return "Sac<" in (root.get("Cost") or "")


@dataclass(slots=True)
class _MinedLines:
    """The lines a mined pair is made of, by role, in the order seen."""

    death: list = field(default_factory=list)
    sac: list = field(default_factory=list)
    lifegain_trigger: list = field(default_factory=list)
    lifegain: list = field(default_factory=list)

    def add(self, item) -> None:
        parsed = segments(item.script_text)
        if not parsed:
            return
        root = parsed[0]
        if item.line_kind == "triggered":
            self._add_trigger(item, root)
        if _has_sacrifice_cost(root) and item.line_kind == "activated":
            self.sac.append(item)
        if any(api_of(found) == "GainLife" for found in parsed):
            self.lifegain.append(item)

    def _add_trigger(self, item, root: dict[str, str]) -> None:
        if _is_death_trigger(root):
            self.death.append(item)
        elif root.get("Mode") == "LifeGained":
            self.lifegain_trigger.append(item)


def _text_of(item) -> str:
    return item.text


def _partner(trigger, causes):
    """The cause whose ``crc32`` with ``trigger``'s text is least."""
    return min(causes, key=lambda c: zlib.crc32(f"{trigger.text}|{c.text}".encode("utf-8")))


def _partnered(triggers: list, causes: list) -> list[tuple[object, object]]:
    if not causes:
        return []
    return [(trigger.key, _partner(trigger, causes).key)
            for trigger in sorted(triggers, key=_text_of)]


# ── record items ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RecordItem:
    """One board-dependent probe item: a record, its target, and its entity.

    ``entity`` is the slot an entity target is read at, and whose own abilities
    rung 1 pools beside the acting ``e``. Two targets have a second party the
    acting line does not name: a trigger's cause, whose acting key the join
    found (``partner``), and a blocker's anchored attacker
    (``partner_entity``), whose pooled abilities stand where an acting line
    would.
    """

    record_id: str
    target: str
    label: float
    entity: str | None = None
    partner: object | None = None
    partner_entity: str | None = None
    #: The fold group: the acting text for a target read at ``[ACT]``, the
    #: entity's card for a target that pools that card's lines (FR-082).
    group: str = ""


_PROTECTION = ("hexproof", "shroud", "protection", "ward")


def _is_protected(entity, sidecars) -> bool:
    if any(k in _PROTECTION for k in entity.granted_temporary.keywords):
        return True
    for key in entity.ability_keys:
        line = sidecars.line_or_none(key)
        text = (line.script_text or "") if line is not None else ""
        if text.split(":", 1)[0].strip().lower() in _PROTECTION:
            return True
    return False


def card_of(entity) -> str:
    """The card an entity's pooled lines come from: its script file, or its name."""
    return entity.printed[0].script_file if entity.printed else entity.name


def record_items(record, sidecars, join: JoinResult, acting_text) -> list[RecordItem]:
    """Every board-dependent item one record yields.

    ``acting_text`` is ``key -> text or None``, the checkpoint-independent text
    the fold groups key on.
    """
    from effects.domain.records import (
        Moment,
        PlayabilityBlockersPayload,
        PlayabilityDecisionPayload,
        RecordKind,
        TriggerPayload,
    )

    payload = record.payload
    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        return _resolution_items(record)
    if isinstance(payload, TriggerPayload):
        return [RecordItem(record.record_id, "fires", float(payload.fired),
                           partner=join.cause.get(record.record_id),
                           group=_group_of(acting_text, record.ability))]
    if isinstance(payload, PlayabilityDecisionPayload) and payload.candidates:
        return _decision_items(record, sidecars, acting_text)
    if isinstance(payload, PlayabilityBlockersPayload):
        return _blocker_items(record)
    return []


def _group_of(acting_text, keys) -> str:
    """The first of ``keys``' acting texts, the fold group of an ``[ACT]`` item."""
    for key in keys or ():
        text = acting_text(key)
        if text:
            return text
    return ""


def _battlefield(record) -> list:
    return [entity for entity in record.state.entities if entity.zone == "battlefield"]


def _resolution_items(record) -> list[RecordItem]:
    """Each battlefield entity's outcome of a resolution's effect half."""
    from effects.domain.effect_targets import derive_targets

    targets = derive_targets(record)
    items: list[RecordItem] = []
    for entity in _battlefield(record):
        items += _outcome_items(record.record_id, entity, targets.get(entity.id))
    return items


def _outcome_items(rid: str, entity, entry) -> list[RecordItem]:
    """Whether ``entity`` was affected, and what its target ``entry`` says it took."""
    from effects.domain.effect_head_layout import DURATIONS, Duration, HeadField

    group = card_of(entity)
    items = [RecordItem(rid, "affected", float(bool(entry and entry.affected)),
                        entity=entity.id, group=group)]
    if entry is None:
        return items
    items += _damage_items(rid, entity, entry)
    if entry.fields.get(HeadField.PT_DURATION) is not None:
        eot = DURATIONS.index(Duration.END_OF_TURN)
        items.append(RecordItem(
            rid, "pt_until_end_of_turn", float(entry.fields[HeadField.PT_DURATION] == eot),
            entity=entity.id, group=group))
    return items


def _damage_items(rid: str, entity, entry) -> list[RecordItem]:
    """A damaged creature's death and the damage it took."""
    from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField, ZoneOutcome

    damage = entry.fields.get(HeadField.DAMAGE_TAKEN) or 0
    if not ("creature" in entity.types and damage > 0):
        return []
    died = ZONE_OUTCOMES.index(ZoneOutcome.DIED)
    group = card_of(entity)
    return [
        RecordItem(rid, "dies", float(entry.fields.get(HeadField.ZONE_OUTCOME) == died),
                   entity=entity.id, group=group),
        RecordItem(rid, "damage_taken", capped(damage), entity=entity.id, group=group),
    ]


def _decision_items(record, sidecars, acting_text) -> list[RecordItem]:
    """A decision's first candidate: affordable, and which entities it may target."""
    candidate = record.payload.candidates[0]
    group = _group_of(acting_text, candidate.ability)
    items = [RecordItem(record.record_id, "affordable", float(candidate.affordable),
                        group=group)]
    if candidate.legal_targets:
        items += _target_items(record, set(candidate.legal_targets), sidecars)
    return items


def _target_items(record, legal: set, sidecars) -> list[RecordItem]:
    """Each battlefield entity's legality as a target, and again where it is protected."""
    rid = record.record_id
    items: list[RecordItem] = []
    for entity in _battlefield(record):
        label = float(entity.id in legal)
        group = card_of(entity)
        items.append(RecordItem(rid, "legal_target", label, entity=entity.id, group=group))
        if _is_protected(entity, sidecars):
            items.append(RecordItem(rid, "legal_target_protected", label,
                                    entity=entity.id, group=group))
    return items


def _blocker_items(record) -> list[RecordItem]:
    """Whether each listed creature may block the anchored attacker."""
    payload = record.payload
    entities = {e.id: e for e in record.state.entities}
    legal = set(payload.legal_blockers)
    candidates = legal | {f.entity for f in payload.forbidden}
    listed = [entity_id for entity_id in sorted(candidates) if entity_id in entities]
    return [RecordItem(record.record_id, "may_block", float(entity_id in legal),
                       entity=entity_id, partner_entity=payload.anchor_attacker,
                       group=card_of(entities[entity_id]))
            for entity_id in listed]
