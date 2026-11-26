"""Predicted effects on zones other than the battlefield: draw, discard, mill, exile,
return to hand, return to play.

    python scripts/effect_knowledge_probes/zone_effects.py CHECKPOINT [--records 2000]
        [--shards 200] [--per-game 3] [--per-sweep 40]

The other probes read what an ability does to a permanent on the battlefield. This
one reads what it does to a player's hand, library and graveyard, and to cards that
sit in them. It takes real resolution records from the validation games whose
acting chain names a zone-moving API (``Draw``, ``Discard``, ``Mill``, ``ChangeZone``,
``ChangeZoneAll``, ``Dig``, ``Destroy``, ``DestroyAll``, ``Sacrifice``, ``SacrificeAll``;
the destroy and sacrifice APIs are there so exile has something to be told apart
from) and reports four things:

1. **Card counts at each player.** Every player slot of every chosen record is an
   item, labelled by the cards that player actually drew, discarded or milled. The
   model's answer is the affected gate times the field's predicted rate. AUC says
   whether the model knows a player is affected at all; on records where exactly
   one player was, the share where it predicts more for that player than for the
   other says whether it knows which.
2. **Where a moved card goes, by the zone it starts in.** Every card the
   resolution moved is an item, grouped by its zone before the move (battlefield,
   graveyard, hand, exile, stack) and the outcome it got. The model's answer is
   its zone-outcome distribution at that card's slot, times the affected gate as
   every count here is; the table gives the mean
   probability of the right outcome and how often it is the most probable outcome
   other than ``stayed``. The outcome vocabulary is the head's: a card put into a
   graveyard from anywhere is ``died``, and one put onto the battlefield is
   ``blinked``.
3. **The amount.** For ``Draw``, ``Discard`` and ``Mill`` chains with one literal
   ``NumCards$``, the acting text's number is set to 1 through 6 and re-encoded by
   the checkpoint's own encoder, and the predicted count at the affected player is
   read, as the damage sweep does for ``NumDmg$``.
4. **Who.** For those chains with one player target, where that player is the one
   affected, the target is handed to the other player. The rule moves the effect
   with it; the probe reads the predicted count at both players before and after.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    # Run by path: put scripts/ on the path so the suite imports as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import print_table  # noqa: E402

from effect_knowledge_probes import common, record_walk, sweeps  # noqa: E402
from effect_knowledge_probes.common import auc  # noqa: E402
from effects.domain.script_text import api_of, params, segments  # noqa: E402

ZONE_APIS = frozenset({
    "Draw", "Discard", "Mill", "ChangeZone", "ChangeZoneAll", "Dig",
    "Destroy", "DestroyAll", "Sacrifice", "SacrificeAll",
})
#: The player count field each card-count API writes.
COUNT_FIELDS = {"Draw": "cards_drawn", "Discard": "cards_discarded", "Mill": "cards_milled"}
AMOUNTS = tuple(range(1, 7))
ORIGIN_ZONES = ("battlefield", "graveyard", "hand", "exile", "stack", "library")
_NUMCARDS_RE = re.compile(r"(NumCards\$\s*)(\d+)")


# ── the chain ───────────────────────────────────────────────────────────


def chain_apis(script_text: str | None) -> set[str]:
    """Every API a chain's segments name, ``SP$``, ``AB$`` and ``DB$`` alike."""
    return {api_of(params(segment)) for segment in segments(script_text or "")} - {None}


def player_targets(record) -> list[str]:
    players = {p.id for p in record.state.players}
    return [t for t in record.state.refs.targets if t in players]


def retargeted(record, old: str, new: str):
    """``record`` with its player target ``old`` replaced by ``new``."""
    refs = record.state.refs
    targets = tuple(new if t == old else t for t in refs.targets)
    return replace(record, state=replace(record.state, refs=replace(refs, targets=targets)))


# ── which records qualify ───────────────────────────────────────────────


def _is_plain_resolution(record) -> bool:
    """Whether ``record`` is a real (unforked) resolution that names an ability."""
    from effects.domain.records import Moment, RecordKind

    if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
        return False
    return not record.fork and bool(record.ability)


def chosen_record(record, sidecars):
    """``(record, apis, script text, targets)`` for a qualifying resolution, or None."""
    from effects.domain.effect_targets import derive_targets

    if not _is_plain_resolution(record):
        return None
    line = sidecars.line_or_none(record.ability[0])
    text = line.script_text if line is not None else None
    apis = chain_apis(text) & ZONE_APIS
    if not apis:
        return None
    return record, apis, text, derive_targets(record)


# ── reading the model ───────────────────────────────────────────────────


def _read(outputs, surface, row) -> dict:
    """``{"players": {id: {field: count}}, "cards": {id: outcome probabilities}}``.

    Both gated (:class:`HeadReading`): a count is the gate times the predicted
    rate, and each outcome's probability the gate times the class's, so the
    ``stayed``-or-unaffected mass is ``1 - sum`` of the moved outcomes.
    """
    from effects.domain.head_decoding import HeadReading

    players, cards = {}, {}
    for index, slot in enumerate(surface.slots):
        reading = HeadReading(outputs[row, index])
        if slot.kind.name == "PLAYER":
            players[slot.player_id] = {
                field: float(reading.expected(field)) for field in COUNT_FIELDS.values()
            }
        elif slot.kind.name == "CARD" and slot.entity_id is not None:
            cards[slot.entity_id] = reading.zone_probabilities().cpu().numpy()
    return {"players": players, "cards": cards}


def predict_all(probe_model, records) -> list[dict]:
    return common.predict_all(probe_model, records, lambda o, v, s, r: _read(o, s, r))


# ── the four reports ────────────────────────────────────────────────────


def _count_of(targets, entity_id, field):
    """The count ``field`` of ``entity_id``'s target entry, 0 where it has none."""
    entry = targets.get(entity_id)
    return (entry.fields.get(field) or 0) if entry is not None else 0


def _lone_hit_of_two(hit, counts, players) -> bool:
    """Whether exactly one of two players was affected and the model read both."""
    if len(hit) != 1 or len(counts) != 2:
        return False
    return all(p in players for p in counts)


class CountTally:
    """What :func:`report_counts` reads for one API and its count field."""

    def __init__(self, api: str, field: str) -> None:
        self.api = api
        self.field = field
        self.labels: list[bool] = []
        self.scores: list[float] = []
        self.naming = self.single = self.right = 0
        self.actual: list = []
        self.predicted: list[float] = []

    def add(self, record, apis, targets, reading) -> None:
        counts = {}
        for player in record.state.players:
            value = _count_of(targets, player.id, self.field)
            counts[player.id] = value
            if player.id in reading["players"]:
                self.labels.append(value > 0)
                self.scores.append(reading["players"][player.id][self.field])
        if self.api not in apis:
            return
        self.naming += 1
        hit = [p for p, v in counts.items() if v > 0]
        if _lone_hit_of_two(hit, counts, reading["players"]):
            self._add_lone_hit(counts, hit[0], reading["players"])

    def _add_lone_hit(self, counts, hit, players) -> None:
        self.single += 1
        other = next(p for p in counts if p != hit)
        self.right += players[hit][self.field] > players[other][self.field]
        self.actual.append(counts[hit])
        self.predicted.append(players[hit][self.field])

    def row(self) -> list:
        share = f"{self.right / self.single:.2f}" if self.single else "n/a"
        return [self.field, len(self.labels), sum(self.labels),
                f"{auc(self.labels, self.scores):.3f}", self.naming, self.single, share,
                common.fmt_mean(self.actual, 2), common.fmt_mean(self.predicted, 2)]


def report_counts(chosen, readings) -> None:
    table = []
    for api, field in COUNT_FIELDS.items():
        tally = CountTally(api, field)
        for (record, apis, _, targets), reading in zip(chosen, readings):
            tally.add(record, apis, targets, reading)
        table.append(tally.row())
    print("\ncard counts at each player: every player slot of every chosen record")
    print_table(["field", "player items", "affected", "AUC", "records naming the API",
                 "one player affected", "more predicted for that player", "actual, mean",
                 "predicted, mean"], table, "l" + "r" * 8)


def _zone_outcome(targets, entity_id):
    from effects.domain.effect_head_layout import HeadField

    entry = targets.get(entity_id)
    return entry.fields.get(HeadField.ZONE_OUTCOME) if entry is not None else None


class ZoneTally:
    """What :func:`report_zones` reads: whether each card moved, and where it went."""

    def __init__(self) -> None:
        from effects.domain.effect_head_layout import ZONE_OUTCOMES, ZoneOutcome

        self.outcomes = ZONE_OUTCOMES
        self.stayed = ZONE_OUTCOMES.index(ZoneOutcome.STAYED)
        self.rows = defaultdict(lambda: {"right": [], "top": [], "instead": defaultdict(int)})
        self.moved = defaultdict(lambda: ([], []))

    def add(self, record, targets, reading) -> None:
        for entity in record.state.entities:
            if entity.id in reading["cards"]:
                self._add_card(entity, reading["cards"][entity.id],
                               _zone_outcome(targets, entity.id))

    def _add_card(self, entity, probs, outcome) -> None:
        went = outcome is not None and outcome != self.stayed
        self.moved[entity.zone][0].append(went)
        self.moved[entity.zone][1].append(float(probs.sum() - probs[self.stayed]))
        if went:
            self._add_move(entity.zone, probs, outcome)

    def _add_move(self, zone, probs, outcome) -> None:
        others = probs.copy()
        others[self.stayed] = -1.0
        top = int(np.argmax(others))
        row = self.rows[(zone, self.outcomes[outcome])]
        row["right"].append(float(probs[outcome]))
        row["top"].append(top == outcome)
        if top != outcome:
            row["instead"][self.outcomes[top]] += 1

    def print_moved(self) -> None:
        print("\nwhether a card moves, by the zone it starts in: every card slot of every "
              "chosen record")
        print_table(["zone before", "cards", "moved", "AUC"],
                    [[zone, len(self.moved[zone][0]), sum(self.moved[zone][0]),
                      f"{auc(*self.moved[zone]):.3f}"]
                     for zone in ORIGIN_ZONES if zone in self.moved], "lrrr")

    def print_destinations(self) -> None:
        print("\nwhere a moved card goes, by the zone it starts in")
        order = {z: i for i, z in enumerate(ORIGIN_ZONES)}
        table = []
        for (zone, outcome), row in sorted(self.rows.items(),
                                           key=lambda kv: (order.get(kv[0][0], 99),
                                                           -len(kv[1]["right"]))):
            instead = max(row["instead"].items(), key=lambda kv: kv[1])[0] \
                if row["instead"] else ""
            table.append([zone, outcome, len(row["right"]), common.fmt_mean(row["right"], 2),
                          common.fmt_mean(row["top"], 2), instead])
        print_table(["zone before", "outcome", "cards", "P(outcome), mean",
                     "top outcome other than stayed is right", "most often predicted instead"],
                    table, "llrrrl")


def report_zones(chosen, readings) -> None:
    tally = ZoneTally()
    for (record, _, _, targets), reading in zip(chosen, readings):
        tally.add(record, targets, reading)
    tally.print_moved()
    tally.print_destinations()


def _affected_player(record, targets, field):
    hit = [p.id for p in record.state.players if _count_of(targets, p.id, field) > 0]
    return hit[0] if len(hit) == 1 else None


def _amount_curve(record, targets, field, probe_model):
    """The predicted count at the affected player for each amount, or None where the
    record has no single affected player or its acting text has no literal count."""
    player = _affected_player(record, targets, field)
    if player is None:
        return None
    batcher = probe_model.batcher
    key = record.ability[0]
    original = batcher.text_of(key)
    if original is None or not _NUMCARDS_RE.search(original):
        return None
    curve = []
    for amount in AMOUNTS:
        with batcher.reading_as(key, sweeps.set_amount(original, "NumCards", amount)):
            reading = common.predict(probe_model, [record],
                                     lambda o, v, s, r: _read(o, s, r))[0]
        curve.append(reading["players"].get(player, {}).get(field, float("nan")))
    return curve


@dataclass(frozen=True)
class CountApi:
    """A card-count API and the player count field it writes."""

    api: str
    field: str


def _one_literal_count(api, apis, text) -> bool:
    """Whether the chain names ``api`` and its text has exactly one ``NumCards$``."""
    if api not in apis:
        return False
    return len(_NUMCARDS_RE.findall(text or "")) == 1


def _amount_curves(count: CountApi, chosen, probe_model, per_sweep: int) -> list:
    rows = []
    for record, apis, text, targets in chosen:
        if not _one_literal_count(count.api, apis, text):
            continue
        curve = _amount_curve(record, targets, count.field, probe_model)
        if curve is None:
            continue
        rows.append(curve)
        if len(rows) >= per_sweep:
            break
    return rows


def report_amounts(chosen, probe_model, per_sweep: int) -> None:
    table = []
    for api, field in COUNT_FIELDS.items():
        rows = _amount_curves(CountApi(api, field), chosen, probe_model, per_sweep)
        means = np.nanmean(np.asarray(rows), axis=0) if rows else [None] * len(AMOUNTS)
        table.append([api, len(rows), *("n/a" if m is None else f"{m:.2f}" for m in means)])
    print("\nthe amount: NumCards$ set to each value, predicted count at the affected "
          "player, mean over records")
    print_table(["API", "records", *AMOUNTS], table, "lr" + "r" * len(AMOUNTS))


def _handed_player(record, apis, targets, count: CountApi):
    """The affected player of a two-player ``api`` record whose one player target is
    that player, or None."""
    if count.api not in apis or len(record.state.players) != 2:
        return None
    aimed = player_targets(record)
    player = _affected_player(record, targets, count.field)
    if len(aimed) != 1 or player != aimed[0]:
        return None
    return player


def _handed_edits(count: CountApi, chosen, readings):
    """``(edited records, [(reading before, player, other)])`` of the records the
    target is handed over in."""
    edited, index = [], []
    for (record, apis, _, targets), reading in zip(chosen, readings):
        player = _handed_player(record, apis, targets, count)
        if player is None:
            continue
        other = next(p.id for p in record.state.players if p.id != player)
        edited.append(retargeted(record, player, other))
        index.append((reading, player, other))
    return edited, index


def _who_row(count: CountApi, edited, index, probe_model) -> list:
    after = predict_all(probe_model, edited)
    cells = defaultdict(list)
    for (before, player, other), reading in zip(index, after):
        cells["tb"].append(before["players"][player][count.field])
        cells["ta"].append(reading["players"][player][count.field])
        cells["ob"].append(before["players"][other][count.field])
        cells["oa"].append(reading["players"][other][count.field])
    return [count.api, len(edited),
            *(common.fmt_mean(cells[k], 2) for k in ("tb", "ta", "ob", "oa"))]


def report_who(chosen, readings, probe_model) -> None:
    table = []
    for api, field in COUNT_FIELDS.items():
        count = CountApi(api, field)
        edited, index = _handed_edits(count, chosen, readings)
        if not edited:
            table.append([api, 0, "n/a", "n/a", "n/a", "n/a"])
            continue
        table.append(_who_row(count, edited, index, probe_model))
    print("\nwho: the player target handed to the other player (rule: the count moves "
          "with it)")
    print_table(["API", "records", "targeted player, before", "after", "other player, before",
                 "after"], table, "lrrrrr")


def main() -> None:
    args, context = common.grid_run(
        __doc__, records=2000, per_game=3,
        options=lambda parser: parser.add_argument("--per-sweep", type=int, default=40))
    sidecars = context.sidecars()
    chosen = record_walk.qualifying(
        context.validation_shards(args.shards), lambda record: chosen_record(record, sidecars),
        args.records, per_game=args.per_game)
    probe_model = context.model([c[0] for c in chosen])
    common.heading(args, context,
                   f"{len(chosen)} validation resolutions naming a zone-moving API")

    readings = predict_all(probe_model, [c[0] for c in chosen])
    report_counts(chosen, readings)
    report_zones(chosen, readings)
    report_amounts(chosen, probe_model, args.per_sweep)
    report_who(chosen, readings, probe_model)


if __name__ == "__main__":
    main()
