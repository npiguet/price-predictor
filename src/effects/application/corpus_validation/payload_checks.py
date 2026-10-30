"""The checks that read one payload type: activation outcomes and costs,
trigger fired ratios, and the rewrite channel."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, fields

from effects.application.corpus_validation.findings import (
    EXAMPLES,
    Finding,
    Share,
    Thresholds,
    ceiling_finding,
    floor_finding,
    histogram,
    keep_example,
    share_of,
)
from effects.domain.records import (
    REWRITE_RESULTS_RUNNING_AN_ABILITY,
    REWRITE_RESULTS_WITHOUT_ABILITY,
    ActivationPayload,
    Costs,
    EffectRecord,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)

#: The six cost channels, in the order the schema lists them.
_COST_FIELDS = tuple(spec.name for spec in fields(Costs))


#: The channels a big enough window must have exercised: every deck pays mana
#: and taps many times a game, while sacrifice, discard, exile and life costs
#: are genuinely rare in a sealed pool.
_COST_FIELDS_ALWAYS_PAID = ("mana_by_color", "tapped")


#: How the rewrite histogram prints a rewrite written before the hook passed a
#: ``result``.
_NO_RESULT_LABEL = "absent"


class Activations:
    """The activation records, read once: whether their outcomes vary, as a
    real game's do, and which cost channels they ever populate."""

    def __init__(self) -> None:
        self.activations = 0
        self.outcomes: Counter[str] = Counter()
        self.cost_fields: Counter[str] = Counter()
        self.populated = 0

    def add(self, record: EffectRecord) -> None:
        if not isinstance(record.payload, ActivationPayload):
            return
        self.activations += 1
        self.outcomes[record.payload.outcome.value] += 1
        paid = {
            name: bool(getattr(record.payload.costs, name)) for name in _COST_FIELDS
        }
        for name, filled in paid.items():
            self.cost_fields[name] += filled
        self.populated += any(paid.values())

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        yield self._outcome_finding()
        yield self._cost_finding(limits)

    def _outcome_finding(self) -> Finding:
        shown = histogram(sorted(self.outcomes.items()))
        return Finding(
            name="outcome takes more than one value",
            # A window with no activation records holds, as an absent kind does.
            ok=len(self.outcomes) > 1 or self.activations == 0,
            measured=(
                f"{len(self.outcomes)} distinct over {self.activations} "
                f"activation records: {shown}"
            ),
            detail=() if len(self.outcomes) > 1 else (
                "a real game counters or fizzles something, so one value means "
                "the collector is writing a literal",
            ),
        )

    def _cost_finding(self, limits: Thresholds) -> Finding:
        shown = ", ".join(
            f"{name} {self.cost_fields[name]}" for name in _COST_FIELDS
        )
        dead = [
            name for name in _COST_FIELDS_ALWAYS_PAID
            if not self.cost_fields[name]
        ]
        enough = self.activations >= limits.min_cost_evidence
        # Nothing paid anywhere is a defect at any size; one dead channel only
        # once the window is big enough to blame the collector, not the pool.
        broken = bool(self.activations and (
            not self.populated or (enough and dead)
        ))
        detail: tuple[str, ...] = ()
        if broken:
            detail = (
                f"never populated over {self.activations} activation records: "
                + ", ".join(dead or _COST_FIELDS),
                "reading a Cost object whose parts were never the ones paid "
                "looks exactly like this",
            )
        elif dead and not enough:
            detail = (
                f"{', '.join(dead)} unexercised, but {self.activations} "
                f"activation records is under the {limits.min_cost_evidence} "
                "this would need to mean anything",
            )
        return Finding(
            name="cost fields are not all empty",
            ok=not broken,
            measured=(
                f"{self.populated}/{self.activations} activation records "
                f"paid something ({shown})"
            ),
            detail=detail,
        )


class Triggers:
    """Fired against not-fired, aggregate and per evaluated mode, because a
    mode drawing no negatives hides behind every other mode's balance."""

    def __init__(self) -> None:
        #: Per evaluated mode: how often it fired, of how often it was asked.
        self.modes: dict[str, Share] = defaultdict(Share)

    def add(self, record: EffectRecord) -> None:
        payload = record.payload
        if not isinstance(payload, TriggerPayload):
            return
        # Forge's own mode name where written: the event-type vocabulary is
        # far coarser than Forge's trigger modes.
        mode = payload.event.params.get("mode") or payload.event.type.value
        self.modes[str(mode)].note(payload.fired)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        fired = Share.total(self.modes.values())
        skewed = sorted(
            ((counts.rate, mode, counts) for mode, counts in self.modes.items()),
            reverse=True,
        )
        yield ceiling_finding(
            name="trigger records draw negatives against positives",
            ceiling=limits.max_trigger_fired_share,
            rate=fired.rate,
            measured=lambda limit: (
                f"{fired.hit}:{fired.seen - fired.hit} fired:not-fired over "
                f"{fired.seen} trigger records ({fired.rate:.1%} fired, {limit})"
            ),
            detail=(
                f"{mode}: {counts.hit}:{counts.seen - counts.hit} "
                f"({rate:.1%} fired)"
                for rate, mode, counts in skewed[:EXAMPLES]
            ),
        )


@dataclass(slots=True)
class _ResultCounts:
    """One replacement result's rewrite records, and what they carry."""

    with_outgoing: int = 0
    with_replaced_by: int = 0
    seen: int = 0


class Rewrites:
    """The rewrite channel's four findings, all read off one result histogram."""

    def __init__(self) -> None:
        self.rewrites = 0
        #: Keyed by result; None for a rewrite written before the hook passed one.
        self.results: dict[RewriteResult | None, _ResultCounts] = defaultdict(_ResultCounts)
        #: Result-carrying rewrites whose ``outgoing`` equals ``incoming``.
        self.identity = 0
        self.identity_examples: list[str] = []
        #: ``prevented``/``skipped`` records carrying an ``outgoing`` or a
        #: ``replaced_by``.
        self.impossible = 0
        self.impossible_examples: list[str] = []

    def add(self, record: EffectRecord) -> None:
        payload = record.payload
        if not isinstance(payload, RewritePayload):
            return
        self.rewrites += 1
        counts = self.results[payload.result]
        counts.with_outgoing += payload.outgoing is not None
        counts.with_replaced_by += bool(payload.replaced_by)
        counts.seen += 1
        if payload.result is None:
            # Pre-contract shard: its writer always produced an identity
            # ``outgoing``.
            return
        example = f"{record.record_id} ({payload.result.value})"
        if payload.outgoing is not None and (
            payload.outgoing.as_dict() == payload.incoming.as_dict()
        ):
            self.identity += 1
            keep_example(self.identity_examples, example)
        if payload.result in REWRITE_RESULTS_WITHOUT_ABILITY and (
            payload.outgoing is not None or payload.replaced_by
        ):
            self.impossible += 1
            keep_example(self.impossible_examples, example)

    def findings(self, limits: Thresholds) -> Iterator[Finding]:
        yield self._say_what_happened()
        yield self._do_not_copy_their_event()
        yield self._outcomes_agree_with_their_payload()
        yield self._name_the_substituted_ability(limits)

    def _without_a_result(self) -> int:
        """Rewrites written before the hook passed a result. Read through
        ``.get`` so a healthy window grows no ``absent`` row."""
        counts = self.results.get(None)
        return counts.seen if counts else 0

    def _by_name(self) -> list[tuple[str, _ResultCounts]]:
        """Each result's counts under its wire spelling, in spelling order."""
        return sorted(
            (
                (_NO_RESULT_LABEL if result is None else result.value, counts)
                for result, counts in self.results.items()
            ),
            key=lambda row: row[0],
        )

    def _shown(self) -> str:
        """The result histogram, for whichever rewrite finding is printing."""
        return histogram((name, counts.seen) for name, counts in self._by_name())

    def _say_what_happened(self) -> Finding:
        """Which of the five replacement outcomes the handler reported.

        Judged at zero: a missing result is a pre-contract shard in the window
        or a hook reading the wrong argument, and both are worth stopping for.
        """
        absent = self._without_a_result()
        named = self.rewrites - absent
        return Finding(
            name="rewrite records say which replacement result happened",
            ok=absent == 0,
            measured=(
                f"{named}/{self.rewrites} rewrite records carry a result "
                f"({self._shown()})"
            ),
            detail=() if absent == 0 else (
                f"{absent} carry none: either the window reaches back to shards "
                "collected before the hook passed one, or the listener is "
                "reading the wrong argument slot",
            ),
        )

    def _do_not_copy_their_event(self) -> Finding:
        """``outgoing`` copying ``incoming`` says nothing; null says "not
        rewritten in place". Counted only over records carrying a ``result``,
        since a pre-contract writer had no null to write."""
        judged = self.rewrites - self._without_a_result()
        rate = share_of(self.identity, judged)
        return Finding(
            name="a rewrite's outgoing event is not a copy of its incoming one",
            ok=self.identity == 0,
            measured=(
                f"{self.identity}/{judged} result-carrying rewrites "
                f"repeat their incoming event as outgoing ({rate:.1%})"
            ),
            detail=tuple(self.identity_examples),
        )

    def _outcomes_agree_with_their_payload(self) -> Finding:
        """``prevented`` and ``skipped`` cannot carry a payload.

        Both return before ``executeReplacementUnobserved`` runs any ability, so
        an ``outgoing`` or ``replaced_by`` on one is a listener reading a stale
        argument. ``not_replaced`` is not judged: its prevention branch writes
        ``PreventedAmount`` before returning.
        """
        ran = ", ".join(
            f"{name} {counts.with_outgoing}/{counts.seen} with outgoing, "
            f"{counts.with_replaced_by}/{counts.seen} with replaced_by"
            for name, counts in self._by_name()
        ) or "none"
        cannot = sorted(r.value for r in REWRITE_RESULTS_WITHOUT_ABILITY)
        return Finding(
            name="outgoing is null on the results that rewrote nothing",
            ok=self.impossible == 0,
            measured=(
                f"{self.impossible} {'/'.join(cannot)} records carry an "
                f"outgoing or a replaced_by, of {self.rewrites} rewrites ({ran})"
            ),
            detail=tuple(self.impossible_examples),
        )

    def _name_the_substituted_ability(self, limits: Thresholds) -> Finding:
        """Which ability ran instead, on the outcomes that ran one; a resolver
        that keys nothing reads zero here."""
        seen = named = 0
        for result in REWRITE_RESULTS_RUNNING_AN_ABILITY:
            counts = self.results.get(result)
            if counts:
                named += counts.with_replaced_by
                seen += counts.seen
        rate = share_of(named, seen)
        return floor_finding(
            name="rewrites name the ability that ran instead",
            floor=limits.min_rewrite_replaced_by_rate,
            seen=seen,
            rate=rate,
            measured=lambda limit: (
                f"{named}/{seen} replaced/updated records name a replaced_by "
                f"key ({rate:.1%}, {limit})"
            ),
            detail=(
                f"{name}: {counts.with_replaced_by}/{counts.seen} name one"
                for name, counts in self._by_name()
            ),
        )
