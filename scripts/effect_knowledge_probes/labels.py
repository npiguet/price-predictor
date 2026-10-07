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
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

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


def target_id(spec: TargetSpec) -> str:
    return f"{spec.family}/{spec.name}"


# ── script parsing ──────────────────────────────────────────────────────

_LABEL_RE = re.compile(r"^\s*SV\d+:\s*")
_INT_RE = re.compile(r"^[+-]?\d+$")
_PIP_RE = re.compile(r"\{([^}]+)\}")


def segments(script: str | None) -> list[dict[str, str]]:
    """``Key$ value`` pairs of every segment of a (possibly chained) script.

    A gen-1 text is one segment; a gen-2 text is ``seg [SEG] SV1: seg …``.
    """
    if not script or "$" not in script:
        return []
    out = []
    for part in script.split("[SEG]"):
        found: dict[str, str] = {}
        for piece in _LABEL_RE.sub("", part).split("|"):
            key, separator, value = piece.partition("$")
            key = key.strip()
            if separator and key and key not in found:
                found[key] = value.strip()
        out.append(found)
    return out


def _api(found: dict[str, str]) -> str | None:
    for key in ("SP", "AB", "DB"):
        if key in found:
            return found[key]
    return None


def _literal(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    return float(int(value)) if _INT_RE.match(value) else None


def _sum_amount(parsed: list[dict[str, str]], keys: tuple[str, ...],
                apis: frozenset[str] | None = None) -> float | None:
    """A literal amount summed over segments; None where any is variable.

    0 where no segment states it, so "deals no damage" is a label like any
    other. Restricted to segments of ``apis`` when given.
    """
    total = 0.0
    for found in parsed:
        if apis is not None and _api(found) not in apis:
            continue
        for key in keys:
            if key in found:
                value = _literal(found[key])
                if value is None:
                    return None
                total += value
    return total


def _cost_shards(cost: str | None) -> list[str]:
    return re.sub(r"<[^>]*>", lambda m: m.group(0).replace(" ", "_"), cost or "").split()


def mana_of_cost(cost: str | None) -> dict[str, float] | None:
    """A ``Cost$``'s mana by colour, colourless and generic; None for ``X``."""
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


def line_labels(
    script: str | None, line_kind: str, api: str | None, types: str = "",
) -> dict[str, float]:
    """Every script-derived label of one line, by target name.

    A target the line says nothing about is absent rather than zero where zero
    would be a claim: a line with no ``Cost$`` has no cost colours, a line that
    produces no mana has no production colours.
    """
    parsed = segments(script)
    labels: dict[str, float] = {}
    if not parsed:
        return labels
    root = parsed[0]
    apis = {_api(found) for found in parsed} - {None}
    root_api = _api(root) or api
    is_charm_root = root_api == "Charm" or ("Choices" in root and root_api in {
        "Charm", "GenericChoice"})

    # ── magnitudes ──
    if not is_charm_root:
        for name, keys, only in (
            ("damage", ("NumDmg",), None),
            ("power_change", ("NumAtt", "PowerBonus"), None),
            ("toughness_change", ("NumDef", "ToughnessBonus"), None),
            ("counters_placed", ("CounterNum",), None),
            ("life_amount", ("LifeAmount",), None),
        ):
            value = _sum_amount(parsed, keys, only)
            if value is not None:
                labels[name] = value
    cost = root.get("Cost")
    cost_mana = mana_of_cost(cost) if cost is not None else None
    total_cost = sum(cost_mana.values()) if cost_mana else 0.0
    if labels.get("damage") and total_cost > 0 and line_kind != "spell":
        labels["damage_per_mana"] = labels["damage"] / total_cost

    # ── mana production ──
    if "Mana" in apis:
        mana = next(found for found in parsed if _api(found) == "Mana")
        produced = mana.get("Produced", "").split()
        for colour in (*_COLOURS, "c"):
            labels[f"produces_{colour}"] = float(colour.upper() in produced)
        labels["any_colour"] = float("Any" in produced)
        amount = _literal(mana.get("Amount", "1"))
        if amount is not None:
            labels["mana_amount"] = amount
        labels["restricted"] = float(any(
            k.startswith("Restrict") for k in mana))
        labels["conditional"] = float(any(
            k.startswith("Condition") or k == "ActivationLimit"
            for found in parsed for k in found))

    # ── mana usage ── (never a spell line: its mana cost is its card's)
    if cost is not None and line_kind != "spell" and cost_mana is not None:
        labels.update(cost_mana)
        labels["cost_total"] = total_cost
    if root.get("Mode") in ("ReduceCost", "RaiseCost"):
        labels["cost_reducer"] = float(root["Mode"] == "ReduceCost")
        labels["cost_tax"] = float(root["Mode"] == "RaiseCost")
    elif line_kind == "static":
        labels["cost_reducer"] = 0.0
        labels["cost_tax"] = 0.0

    # ── timing and non-mana costs ──
    if line_kind == "spell":
        labels["instant_speed"] = float("instant" in types)
    elif line_kind == "activated":
        labels["instant_speed"] = float(
            "SorcerySpeed" not in root and "ActivationPhases" not in root)
    if cost is not None:
        shards = _cost_shards(cost)
        labels["tap_cost"] = float("T" in shards)
        labels["sacrifice_cost"] = float(any(s.startswith("Sac<") for s in shards))
        labels["life_cost"] = float(any(s.startswith("PayLife<") for s in shards))

    # ── side ──
    values = " ".join(
        v for found in parsed for k, v in found.items() if k not in _SIDE_SKIP
    )
    defined = root.get("Defined", "")
    if root_api is not None:
        labels["affects_you"] = float(
            defined in ("You", "Self") or "YouCtrl" in values
            or bool(re.search(r"\bYou\b", values)))
        labels["affects_opponent"] = float("Opponent" in values or "OppCtrl" in values)
        labels["affects_each"] = float(
            root_api.endswith("All") or root_api.endswith("EachPlayer")
            or defined == "Player" or "ValidCards" in root)

    # ── duration and repeatability ──
    if apis & _PT_APIS:
        durations = {found.get("Duration") for found in parsed if _api(found) in _PT_APIS}
        counter = bool(apis & _COUNTER_APIS)
        labels["as_counter"] = float(counter)
        labels["until_end_of_turn"] = float(
            not counter and not (durations & {"Permanent", "UntilHostLeavesPlay"}))
    if line_kind == "spell":
        labels["repeatable"] = 0.0
    elif line_kind == "activated":
        shards = _cost_shards(cost)
        labels["repeatable"] = float(not any(
            s.startswith(("Sac<1/CARDNAME", "Exile<1/CARDNAME")) for s in shards))
    return labels


# ── observed outcomes: effect profiles, state dependence, mana ──────────

PROFILE_NAMES: tuple[str, ...] = (
    "log_affected", "died", "damaged", "zone_changed", "pt_changed", "countered",
    "life", "drawn",
)


def effect_profile(record) -> np.ndarray | None:
    """One resolution's outcome summary, the embedding probes' profile shares.

    None for a record that is not an effect half.
    """
    from effects.domain.effect_model import ZONE_OUTCOMES
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import Moment, RecordKind

    if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
        return None
    targets = derive_targets(record)
    affected = [t for t in targets.values() if t.affected]
    n = len(affected)
    died = ZONE_OUTCOMES.index("died")
    stayed = ZONE_OUTCOMES.index("stayed")

    def share(test) -> float:
        return sum(1 for t in affected if test(t.fields)) / n if n else 0.0

    return np.array([
        math.log1p(n),
        share(lambda f: f.get("zone_outcome") == died),
        share(lambda f: (f.get("damage_taken") or 0) > 0),
        share(lambda f: f.get("zone_outcome") not in (None, stayed)),
        share(lambda f: bool(f.get("power_delta") or f.get("toughness_delta"))),
        share(lambda f: any(k.startswith("counters_delta") for k in f)),
        share(lambda f: bool(f.get("life_delta"))),
        share(lambda f: bool(f.get("cards_drawn"))),
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
    """Mana a mana resolution produced, from its ``mana_produced`` events."""
    from effects.domain.event_schema import EventType
    from effects.domain.records import events_of

    total, seen = 0.0, False
    for event in events_of(record):
        if event.type is EventType.MANA_PRODUCED:
            seen = True
            amount = event.params.get("amount")
            if isinstance(amount, int | float):
                total += amount
            elif isinstance(amount, dict):
                total += sum(v for v in amount.values() if isinstance(v, int | float))
            else:
                total += 1
    return total if seen else None


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
        from effects.domain.records import Moment, RecordKind, TriggerPayload, events_of

        rows = self._games[record.game_id]
        if (record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION
                and record.ability):
            signatures = tuple(_event_signature(e) for e in events_of(record))
            rows.append((record.timestamp, 0, signatures, record.ability[0], None, None))
        elif isinstance(record.payload, TriggerPayload):
            rows.append((record.timestamp, 1, (_event_signature(record.payload.event),),
                         None, record.record_id, record.payload.fired))

    def result(self) -> JoinResult:
        cause: dict[str, object] = {}
        fired = joined = 0
        for rows in self._games.values():
            ordered = sorted(rows, key=lambda row: (row[0], row[1]))
            # signature -> the resolutions producing it, in time order.
            producers: dict[tuple, list[tuple[str, object]]] = defaultdict(list)
            for timestamp, kind, signatures, key, _, _ in ordered:
                if kind == 0:
                    for signature in signatures:
                        producers[signature].append((timestamp, key))
            for timestamp, kind, signatures, _, record_id, was_fired in ordered:
                if kind == 0:
                    continue
                candidates = producers.get(signatures[0], ())
                after = [k for t, k in candidates if t >= timestamp]
                before = [k for t, k in candidates if t < timestamp]
                found = after[0] if after else (before[-1] if before else None)
                if was_fired:
                    fired += 1
                    joined += found is not None
                if found is not None:
                    cause[record_id] = found
        return JoinResult(cause=cause, fired=fired, fired_joined=joined)


def join_triggers(records: Iterable) -> JoinResult:
    """:class:`TriggerJoin` over a whole iterable of records."""
    join = TriggerJoin()
    for record in records:
        join.observe(record)
    return join.result()


def mined_pairs(items: Iterable) -> list[tuple[object, object]]:
    """Positive pairs read off the scripts: what the join cannot reach.

    A death trigger with a sacrifice outlet, and a lifegain trigger with a
    lifegain ability, fire one another by construction. Each trigger gets one
    partner, chosen by ``crc32`` of the two texts: a fixed partner list would
    pair every death trigger with the same few outlets, and a probe could then
    recognise a mined pair by its partner alone.
    """
    death, sac, lifegain_trigger, lifegain = [], [], [], []
    for item in items:
        parsed = segments(item.script_text)
        if not parsed:
            continue
        root = parsed[0]
        if item.line_kind == "triggered":
            if (root.get("Mode") == "ChangesZone" and root.get("Destination") == "Graveyard"
                    and root.get("Origin") == "Battlefield"):
                death.append(item)
            elif root.get("Mode") == "LifeGained":
                lifegain_trigger.append(item)
        if "Sac<" in (root.get("Cost") or "") and item.line_kind == "activated":
            sac.append(item)
        if any(_api(found) == "GainLife" for found in parsed):
            lifegain.append(item)
    import zlib

    pairs: list[tuple[object, object]] = []
    for triggers, causes in ((death, sac), (lifegain_trigger, lifegain)):
        if not causes:
            continue
        for trigger in sorted(triggers, key=lambda i: i.text):
            partner = min(causes, key=lambda c: zlib.crc32(
                f"{trigger.text}|{c.text}".encode("utf-8")))
            pairs.append((trigger.key, partner.key))
    return pairs


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
    for key in (*entity.printed, *entity.granted_attached):
        try:
            line = sidecars.line_for(key)
        except KeyError:
            line = None
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
    from effects.domain.effect_model import DURATIONS, ZONE_OUTCOMES
    from effects.domain.effect_targets import derive_targets
    from effects.domain.records import (
        Moment,
        PlayabilityBlockersPayload,
        PlayabilityDecisionPayload,
        RecordKind,
        TriggerPayload,
    )

    items: list[RecordItem] = []
    rid = record.record_id
    payload = record.payload
    entities = {e.id: e for e in record.state.entities}

    def group_of(keys) -> str:
        for key in keys or ():
            text = acting_text(key)
            if text:
                return text
        return ""

    if record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION:
        targets = derive_targets(record)
        died = ZONE_OUTCOMES.index("died")
        eot = DURATIONS.index("end_of_turn")
        for entity in record.state.entities:
            if entity.zone != "battlefield":
                continue
            entry = targets.get(entity.id)
            items.append(RecordItem(rid, "affected", float(bool(entry and entry.affected)),
                                    entity=entity.id, group=card_of(entity)))
            if entry is None:
                continue
            damage = entry.fields.get("damage_taken") or 0
            if "creature" in entity.types and damage > 0:
                items.append(RecordItem(
                    rid, "dies", float(entry.fields.get("zone_outcome") == died),
                    entity=entity.id, group=card_of(entity)))
                items.append(RecordItem(rid, "damage_taken", float(damage),
                                        entity=entity.id, group=card_of(entity)))
            if entry.fields.get("pt_duration") is not None:
                items.append(RecordItem(
                    rid, "pt_until_end_of_turn",
                    float(entry.fields["pt_duration"] == eot),
                    entity=entity.id, group=card_of(entity)))
    elif isinstance(payload, TriggerPayload):
        items.append(RecordItem(rid, "fires", float(payload.fired),
                                partner=join.cause.get(rid),
                                group=group_of(record.ability)))
    elif isinstance(payload, PlayabilityDecisionPayload) and payload.candidates:
        candidate = payload.candidates[0]
        group = group_of(candidate.ability)
        items.append(RecordItem(rid, "affordable", float(candidate.affordable),
                                group=group))
        if candidate.legal_targets:
            legal = set(candidate.legal_targets)
            for entity in record.state.entities:
                if entity.zone != "battlefield":
                    continue
                label = float(entity.id in legal)
                items.append(RecordItem(rid, "legal_target", label,
                                        entity=entity.id, group=card_of(entity)))
                if _is_protected(entity, sidecars):
                    items.append(RecordItem(rid, "legal_target_protected", label,
                                            entity=entity.id, group=card_of(entity)))
    elif isinstance(payload, PlayabilityBlockersPayload):
        legal = set(payload.legal_blockers)
        candidates = legal | {f.entity for f in payload.forbidden}
        for entity_id in sorted(candidates):
            entity = entities.get(entity_id)
            if entity is None:
                continue
            items.append(RecordItem(rid, "may_block", float(entity_id in legal),
                                    entity=entity_id,
                                    partner_entity=payload.anchor_attacker,
                                    group=card_of(entity)))
    return items
