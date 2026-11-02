"""Which envelope, snapshot and payload members each shard's raw lines carry.

Read from the raw lines rather than the parsed records, since a parsed record
cannot say whether a field was on the wire: three findings, one for gen-2's
envelope fields (FR-033), one for the members every gen-3 record owes (root
§ 8.13 and § 8.4), and one refusing the ``curation`` member a collected shard
never carries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
    histogram,
)
from effects.domain.records import (
    LEGALITY_SUBKINDS,
    Moment,
    PlayabilitySubkind,
    RecordKind,
)
from effects.domain.shard_generation import ShardGeneration

#: The member only ``build-corpus`` writes, on every record of a curated shard.
CURATION_MEMBER = "curation"

#: What every gen-3 record owes where it applies, by where it sits: the
#: members of root § 8.13 and the snapshot facts of § 8.4, all written on
#: every record of their site (null or empty where there is nothing to say).
GEN3_STATE_MEMBERS = ("links",)
GEN3_GLOBAL_MEMBERS = ("day_night", "floating_effects", "delayed_triggers", "dungeons")
GEN3_PLAYER_MEMBERS = ("max_land_plays", "speed", "city_blessing", "rad", "monarch",
                       "initiative", "turn_counts")
GEN3_ENTITY_MEMBERS = (
    "owner", "face_state", "mana_cost", "base_pt", "pt_boosts", "printed_loyalty",
    "printed_defense", "real_face", "designations", "x_paid", "chosen", "costs_paid",
    "history", "order_rank", "lines", "role",
)
GEN3_REFS_MEMBERS = ("random",)
GEN3_ATTACKERS_MEMBERS = ("anchor_attacker", "legal_defenders", "must_attack")
GEN3_BLOCKERS_MEMBERS = ("must_block",)
GEN3_CANDIDATE_MEMBERS = ("failed_checks", "land_play")
GEN3_VERDICT_MEMBERS = ("timing",)
#: A resolution's effect half: every one, its game-end rescue and mana ones too.
GEN3_EFFECT_HALF_MEMBERS = ("conditions_met",)


def _owed(data: Mapping) -> Iterator[tuple[str, bool]]:
    """Each member site a gen-3 record owes, and whether it carries it."""
    state = data.get("state") or {}
    yield from _members("state", state, GEN3_STATE_MEMBERS)
    yield from _members("state.global", state.get("global") or {}, GEN3_GLOBAL_MEMBERS)
    yield from _members("state.refs", state.get("refs") or {}, GEN3_REFS_MEMBERS)
    for player in state.get("players") or ():
        yield from _members("state.players[]", player, GEN3_PLAYER_MEMBERS)
    for entity in state.get("entities") or ():
        yield from _members("state.entities[]", entity, GEN3_ENTITY_MEMBERS)
        for line in entity.get("lines") or ():
            if line.get("kind") == "static":
                yield "state.entities[].lines[static].active", "active" in line
    payload = data.get("payload") or {}
    kind, subkind = data.get("kind"), data.get("subkind")
    if kind == RecordKind.RESOLUTION and data.get("moment") == Moment.RESOLUTION:
        yield from _members("payload", payload, GEN3_EFFECT_HALF_MEMBERS)
    elif kind == RecordKind.PLAYABILITY and subkind == PlayabilitySubkind.ATTACKERS:
        yield from _members("payload", payload, GEN3_ATTACKERS_MEMBERS)
    elif kind == RecordKind.PLAYABILITY and subkind == PlayabilitySubkind.BLOCKERS:
        yield from _members("payload", payload, GEN3_BLOCKERS_MEMBERS)
    elif kind == RecordKind.PLAYABILITY and subkind == PlayabilitySubkind.DECISION:
        for candidate in payload.get("candidates") or ():
            yield from _members("payload.candidates[]", candidate, GEN3_CANDIDATE_MEMBERS)
            yield from _members("payload.candidates[].verdict",
                                candidate.get("verdict") or {}, GEN3_VERDICT_MEMBERS)


def _members(site: str, block: Mapping, members: tuple[str, ...]) -> Iterator[tuple[str, bool]]:
    for member in members:
        yield f"{site}.{member}", member in block


def _owes_what_if(data: Mapping, generation: ShardGeneration) -> bool:
    """Whether a record is of the kinds that carry ``what_if``; one that is
    not must not carry it.

    Both legality subkinds carry it up to gen-2 (a gen-1 shard on none of
    them, which the shard's own ``random_seat`` tells); gen-3 writes it on
    blockers alone, an attackers record being one creature of the
    declaration made.
    """
    subkind = data.get("subkind")
    if data.get("kind") != RecordKind.PLAYABILITY or subkind not in LEGALITY_SUBKINDS:
        return False
    if generation is ShardGeneration.GEN_3:
        return subkind == PlayabilitySubkind.BLOCKERS
    return True


@dataclass(slots=True)
class _ShardPresence:
    """One shard's raw lines, and how many carry each member."""

    records: int = 0
    with_random_seat: int = 0
    #: The legality records that owe ``what_if``, and how many carry it.
    legality: int = 0
    legality_with_what_if: int = 0
    #: Records carrying a ``what_if`` they must not: any non-legality record,
    #: a gen-3 attackers record.
    stray_what_if: int = 0
    curated: int = 0


@dataclass
class FieldPresence:
    """Which members each shard's raw lines carry (FR-033, root § 8.13).

    Per shard for the gen-2 envelope, because a shard is one collector build:
    one where only some records carry ``random_seat`` came from a collector
    stamping it on some paths only.
    """

    shards: dict[str, _ShardPresence] = field(default_factory=dict)
    gen3_records: int = 0
    #: Per gen-3 member site: how often it was owed, and how often missing.
    gen3_owed: Counter[str] = field(default_factory=Counter)
    gen3_missing: Counter[str] = field(default_factory=Counter)
    #: Records that lack at least one member they owe.
    gen3_short: int = 0
    short_examples: list[str] = field(default_factory=list)

    def observe(self, shard: str, data: dict) -> None:
        counts = self.shards.setdefault(shard, _ShardPresence())
        counts.records += 1
        counts.with_random_seat += "random_seat" in data
        counts.curated += CURATION_MEMBER in data
        generation = ShardGeneration.of(data)
        owes = _owes_what_if(data, generation)
        if owes:
            counts.legality += 1
            counts.legality_with_what_if += "what_if" in data
        elif "what_if" in data:
            counts.stray_what_if += 1
        if generation is ShardGeneration.GEN_3:
            self._observe_gen3(data)

    def _observe_gen3(self, data: dict) -> None:
        self.gen3_records += 1
        short = False
        for site, present in _owed(data):
            self.gen3_owed[site] += 1
            if not present:
                self.gen3_missing[site] += 1
                if not short and len(self.short_examples) < EXAMPLES:
                    self.short_examples.append(f"{data.get('record_id')}: no {site}")
                short = True
        self.gen3_short += short

    def findings(self) -> Iterator[Finding]:
        yield self._envelope_finding()
        yield self._gen3_finding()
        yield self._curation_finding()

    def _envelope_finding(self) -> Finding:
        partial_seat, partial_what_if, stray = [], [], []
        for shard, counts in sorted(self.shards.items()):
            records, seats = counts.records, counts.with_random_seat
            legality, what_ifs = counts.legality, counts.legality_with_what_if
            if 0 < seats < records:
                partial_seat.append(f"{shard}: random_seat on {seats} of {records} records")
            # A gen-2 or gen-3 shard owes what_if on every legality record
            # that carries it; a gen-1 one on none, so only a partial set is
            # wrong there.
            if what_ifs < legality and (seats or what_ifs):
                partial_what_if.append(
                    f"{shard}: what_if on {what_ifs} of {legality} legality records"
                )
            if counts.stray_what_if:
                stray.append(
                    f"{shard}: what_if on {counts.stray_what_if} record(s) "
                    "that carry none (another kind, or a gen-3 attackers record)"
                )
        seated = sum(1 for counts in self.shards.values() if counts.with_random_seat)
        bad = partial_seat + partial_what_if + stray
        return Finding(
            name="gen-2 envelope fields are on every record of the kinds that carry them",
            ok=not bad,
            measured=(
                f"{len(self.shards)} shard(s), {seated} carrying random_seat; "
                f"{len(partial_seat)} with random_seat on some records only, "
                f"{len(partial_what_if)} with what_if on some legality records "
                f"only, {len(stray)} with what_if where none belongs"
            ),
            detail=tuple(bad[:EXAMPLES]),
        )

    def _gen3_finding(self) -> Finding:
        missing = sum(self.gen3_missing.values())
        owed = sum(self.gen3_owed.values())
        return Finding(
            name="gen-3 fields are on every record that owes them",
            ok=not missing,
            measured=(
                f"{self.gen3_short}/{self.gen3_records} gen-3 records lack a member they "
                f"owe; {missing} of {owed} owed members missing over "
                f"{len(self.gen3_owed)} sites"
            ),
            detail=tuple(
                [histogram(self.gen3_missing.most_common(EXAMPLES))] if missing else []
            ) + tuple(self.short_examples),
        )

    def _curation_finding(self) -> Finding:
        curated = {shard: counts.curated for shard, counts in sorted(self.shards.items())
                   if counts.curated}
        return Finding(
            name="no collected shard carries a curation member",
            ok=not curated,
            measured=(
                f"{sum(curated.values())} record(s) in {len(curated)} of "
                f"{len(self.shards)} shard(s) carry {CURATION_MEMBER!r}"
            ),
            detail=tuple(
                f"{shard}: {count} record(s); build-corpus writes it, so this is "
                "a curated shard, not a collected one"
                for shard, count in list(curated.items())[:EXAMPLES]
            ),
        )
