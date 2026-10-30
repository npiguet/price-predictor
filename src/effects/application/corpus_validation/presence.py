"""Which gen-2 envelope fields each shard's raw lines carry (FR-033).

Read from the raw lines rather than the parsed records, since a parsed record
cannot say whether a field was on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
)
from effects.domain.records import (
    LEGALITY_SUBKINDS,
    RecordKind,
)
from effects.domain.shard_generation import ShardGeneration


@dataclass(slots=True)
class _ShardPresence:
    """One shard's raw lines, and how many carry each gen-2 envelope field."""

    records: int = 0
    with_random_seat: int = 0
    #: The legality subkinds, which must carry ``what_if`` on a gen-2 shard.
    legality: int = 0
    legality_with_what_if: int = 0
    others_with_what_if: int = 0


@dataclass
class FieldPresence:
    """Which gen-2 envelope fields each shard's raw lines carry (FR-033).

    Per shard, because a shard is one collector build: one where only some
    records carry ``random_seat`` came from a collector stamping it on some
    paths only.
    """

    shards: dict[str, _ShardPresence] = field(default_factory=dict)

    def observe(self, shard: str, data: dict) -> None:
        counts = self.shards.setdefault(shard, _ShardPresence())
        counts.records += 1
        counts.with_random_seat += ShardGeneration.of(data) is ShardGeneration.GEN_2
        legality = (
            data.get("kind") == RecordKind.PLAYABILITY.value
            and data.get("subkind") in LEGALITY_SUBKINDS
        )
        if legality:
            counts.legality += 1
            counts.legality_with_what_if += "what_if" in data
        else:
            counts.others_with_what_if += "what_if" in data

    def finding(self) -> Finding:
        partial_seat, partial_what_if, stray = [], [], []
        for shard, counts in sorted(self.shards.items()):
            records, seats = counts.records, counts.with_random_seat
            legality, what_ifs = counts.legality, counts.legality_with_what_if
            if 0 < seats < records:
                partial_seat.append(f"{shard}: random_seat on {seats} of {records} records")
            # A gen-2 shard owes what_if on every legality record; a gen-1 one
            # on none, so only a partial set is wrong there.
            if what_ifs < legality and (seats or what_ifs):
                partial_what_if.append(
                    f"{shard}: what_if on {what_ifs} of {legality} legality records"
                )
            if counts.others_with_what_if:
                stray.append(
                    f"{shard}: what_if on {counts.others_with_what_if} "
                    "non-legality record(s)"
                )
        gen2 = sum(1 for counts in self.shards.values() if counts.with_random_seat)
        bad = partial_seat + partial_what_if + stray
        return Finding(
            name="gen-2 envelope fields are on every record of the kinds that carry them",
            ok=not bad,
            measured=(
                f"{len(self.shards)} shard(s), {gen2} carrying random_seat; "
                f"{len(partial_seat)} with random_seat on some records only, "
                f"{len(partial_what_if)} with what_if on some legality records "
                f"only, {len(stray)} with what_if on another kind"
            ),
            detail=tuple(bad[:EXAMPLES]),
        )
