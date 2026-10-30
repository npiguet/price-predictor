"""Gate 2: does the prediction move the rules' way when a keyword is removed?

The perturbation is model-side and paired: each record is built twice in one
batch, as collected and with the keyword stripped from its carrier, so the two
predictions differ in the keyword alone. The count and the agreement describe
one population, decided by the table's own predicate
(:mod:`effects.domain.damage_step_keywords`).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import chain, islice
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from effects.application.model_loading import BATCH_OBSERVATIONS
from effects.domain.damage_step_keywords import (
    DAMAGE_STEP_KEYWORDS,
    MIN_DIRECTION_AGREEMENT,
    MIN_QUALIFYING_RECORDS,
    CombatParticipant,
    DamageStepKeyword,
    KeywordResolver,
    KeywordsOf,
    qualifying_observations,
    subject_id,
)
from effects.domain.effect_head_layout import FIELDS_BY_NAME, FieldSpec, FieldType
from effects.domain.evaluation_report import (
    CheckName,
    CheckResult,
    CheckStatus,
    KeywordVerdict,
)
from effects.domain.head_decoding import HeadReading
from effects.domain.records import EffectRecord

if TYPE_CHECKING:
    from effects.application.model_loading import Runnable

logger = logging.getLogger(__name__)

#: Records read eagerly before gate 2 loads its checkpoint: the loader measures
#: each slot kind's width from real records.
SLOT_WIDTH_SAMPLE = 8

PROGRESS_EVERY_RECORDS = 50_000


class Observation(NamedTuple):
    """One qualifying perturbation: a combat, the keyword row it qualifies
    for, and the carrier whose keyword the twin build strips."""

    record: EffectRecord
    row: DamageStepKeyword
    participant: CombatParticipant


class FieldAgreement(NamedTuple):
    """How many of one effect's scored observations moved the rules' way."""

    agreeing: int
    scored: int


@dataclass(frozen=True, slots=True)
class KeywordScore:
    """One keyword's population and how much of it agreed.

    ``qualifying`` counts distinct records (FR-119's 200 is a count of
    combats); ``observations`` counts (record, carrier) perturbations, since
    each carrier is its own; ``agreeing`` the observations that moved every
    field the rules' way.
    """

    qualifying: int = 0
    agreeing: int = 0
    #: ``effect label -> (agreeing, scored)``.
    per_field: dict[str, FieldAgreement] = field(default_factory=dict)
    observations: int = 0


@dataclass(slots=True)
class _Tally:
    """One keyword's running counts while the pass is still reading."""

    records: set[str] = field(default_factory=set)
    observations: int = 0
    agreeing: int = 0
    per_field: dict[str, FieldAgreement] = field(default_factory=dict)

    def add(self, record_id: str, agreed: bool, per_effect: dict[str, bool]) -> None:
        self.records.add(record_id)
        self.observations += 1
        self.agreeing += 1 if agreed else 0
        for label, ok in per_effect.items():
            agreeing, scored = self.per_field.get(label, FieldAgreement(0, 0))
            self.per_field[label] = FieldAgreement(
                agreeing + (1 if ok else 0), scored + 1,
            )

    def score(self) -> KeywordScore:
        return KeywordScore(
            len(self.records), self.agreeing, self.per_field, self.observations,
        )


def read_field(
    vector: torch.Tensor, spec: FieldSpec, category: str | None,
) -> float:
    """One field's prediction as a single scalar, gated by ``P(affected)``.

    Gated because the field heads train only where an entity was affected, so
    a move to "unaffected" (removing first strike leaves the blocker untouched
    in the first-strike step) shows in the gate alone. Any other field type
    raises: a row naming one is a table bug.
    """
    reading = HeadReading(vector)
    match spec.type:
        case FieldType.COUNT | FieldType.SIGNED_DELTA:
            return float(reading.expected(spec.name))
        case FieldType.CATEGORICAL:
            if category is None:
                raise ValueError(f"gate 2 reads a class of {spec.name}, and names none")
            return float(reading.p_class(spec.name, category))
    raise ValueError(
        f"gate 2 cannot read {spec.name}: no rule for a {spec.type} field"
    )


def score_observation(
    row: DamageStepKeyword,
    participant: CombatParticipant,
    present: dict[str, torch.Tensor],
    stripped: dict[str, torch.Tensor],
    *,
    active: frozenset[str],
    keywords_of: KeywordsOf,
) -> tuple[bool, dict[str, bool]]:
    """Whether one perturbation moved every field it should have.

    ``present`` and ``stripped`` map a subject id to its output vector in the
    two builds. An effect agrees when present minus stripped has the row's
    sign, so a prediction that did not move disagrees. An effect this combat
    cannot show (infect's creature and player fields are exclusive) or a field
    outside ``active`` is left unscored rather than counted against the keyword.
    """
    per_effect: dict[str, bool] = {}
    for effect in row.effects:
        if effect.field not in active:
            continue
        if not effect.observable(participant, keywords_of):
            continue
        subject = subject_id(effect, participant)
        before, after = present.get(subject), stripped.get(subject)
        if before is None or after is None:
            continue
        spec = FIELDS_BY_NAME[effect.field]
        change = (
            read_field(before, spec, effect.category)
            - read_field(after, spec, effect.category)
        )
        per_effect[effect.label] = (
            change > 0 if effect.direction > 0 else change < 0
        )
    return bool(per_effect) and all(per_effect.values()), per_effect


def score_keywords(
    records, rows: tuple[DamageStepKeyword, ...], runnable: Runnable,
) -> dict[str, KeywordScore]:
    """Run every qualifying perturbation and tally the eight rows.

    ``records`` is consumed as a stream; only a chunk's worth is ever held.
    """
    resolver = runnable.keyword_resolver()
    tallies = {row.keyword: _Tally() for row in rows}
    progress = _Progress()
    active = frozenset(spec.name for spec in runnable.fields)

    observations = _qualifying(records, rows, resolver, progress)
    for chunk in runnable.inference_chunks(observations, BATCH_OBSERVATIONS):
        _score_chunk(
            chunk, runnable, tallies,
            active=active, keywords_of=resolver.keywords_of,
        )
    progress.log()
    return {keyword: tally.score() for keyword, tally in tallies.items()}


@dataclass(slots=True)
class _Progress:
    """Records read and observations found, for the periodic progress line."""

    read: int = 0
    observed: int = 0
    started: float = field(default_factory=time.monotonic)

    def log(self) -> None:
        logger.info(
            "gate 2: %d records read, %d observations scored (%.0fs)",
            self.read, self.observed, time.monotonic() - self.started,
        )


def _qualifying(
    records, rows: tuple[DamageStepKeyword, ...], resolver: KeywordResolver,
    progress: _Progress,
) -> Iterator[Observation]:
    """Every qualifying observation, read lazily.

    A fork never qualifies: it re-runs a real board the population already
    holds, so it would count that combat twice toward the threshold.
    """
    for record in records:
        progress.read += 1
        if progress.read % PROGRESS_EVERY_RECORDS == 0:
            progress.log()
        if record.fork or record.interventional:
            continue
        for row, participant in qualifying_observations(record, rows, resolver):
            progress.observed += 1
            yield Observation(record, row, participant)


def _score_chunk(
    chunk: list[Observation], runnable: Runnable, tallies: dict[str, _Tally], *,
    active: frozenset[str], keywords_of: KeywordsOf,
) -> None:
    """Run one chunk of observations, paired, and fold it into ``tallies``."""
    built = [observation.record for observation in chunk] * 2
    strips = [None] * len(chunk) + [
        {observation.participant.carrier.id: frozenset({observation.row.keyword})}
        for observation in chunk
    ]
    outputs, surfaces = runnable.per_entity_outputs(built, strips)

    for offset, (record, row, participant) in enumerate(chunk):
        twin = len(chunk) + offset
        agreed, per_effect = score_observation(
            row, participant,
            _subject_vectors(surfaces[offset], outputs[offset], row, participant),
            _subject_vectors(surfaces[twin], outputs[twin], row, participant),
            active=active, keywords_of=keywords_of,
        )
        tallies[row.keyword].add(record.record_id, agreed, per_effect)


def _subject_vectors(
    surface, outputs: torch.Tensor, row: DamageStepKeyword,
    participant: CombatParticipant,
) -> dict[str, torch.Tensor]:
    """The head's output row for each subject this row reads; ``outputs`` is
    the surface's, one row per entity slot.

    Per surface, because dropping the keyword's ability token shifts every
    slot after it.
    """
    wanted = {
        subject for subject in (
            subject_id(effect, participant) for effect in row.effects
        ) if subject is not None
    }
    found: dict[str, torch.Tensor] = {}
    for column, index in enumerate(surface.entity_slots):
        key = surface.slots[index].subject_id
        if key in wanted:
            found[key] = outputs[column]
    return found


def evaluate_keyword(row: DamageStepKeyword, score: KeywordScore) -> KeywordVerdict:
    """Route one keyword: pass, or send it to a stage-three probe (FR-119).

    Under-sampled routes the same way as wrong: treating untested as passed is
    how a canary stops being one. The per-field agreement is reported but never
    routes; it says *which* field did not move, e.g. a magnitude learned
    without its consequence.
    """
    qualifying, observations = score.qualifying, score.observations
    agreement = score.agreeing / observations if observations else 0.0
    if qualifying < MIN_QUALIFYING_RECORDS:
        routed, reason = True, (
            f"under-sampled: {qualifying} qualifying records < "
            f"{MIN_QUALIFYING_RECORDS}"
        )
    elif agreement < MIN_DIRECTION_AGREEMENT:
        routed, reason = True, (
            f"direction agreement {agreement:.1%} < "
            f"{MIN_DIRECTION_AGREEMENT:.0%} on {', '.join(row.fields)}"
            f" over {observations} observations"
        )
    else:
        routed, reason = False, (
            f"direction agreement {agreement:.1%} over {observations} observations"
        )
    return KeywordVerdict(
        row.keyword, qualifying, agreement, routed_to_probe=routed,
        reason=reason + _render_per_field(score.per_field),
        observations=observations,
    )


def _render_per_field(per_field: dict[str, FieldAgreement]) -> str:
    """The per-effect tail of a keyword's report line, or nothing."""
    if not per_field:
        return ""
    parts = [
        f"{label} {agreeing / scored:.0%}"
        for label, (agreeing, scored) in per_field.items() if scored
    ]
    return f" ({', '.join(parts)})" if parts else ""


def evaluate_gate_two(verdicts: list[KeywordVerdict]) -> CheckResult:
    """The canary's routing report; it blocks nothing (FR-119)."""
    routed = [v.keyword for v in verdicts if v.routed_to_probe]
    detail = (
        "all eight keywords pass; no stage-three probe machinery is needed"
        if not routed
        else f"routed to a stage-three probe: {', '.join(routed)}"
    )
    return CheckResult(
        CheckName.GATE_2, CheckStatus.REPORTED, detail,
        {v.keyword: v.direction_agreement for v in verdicts},
    )


def gate_two_records(provenance, *, corpus_path: Path) -> Iterator:
    """The combat records of the checkpoint's **game-disjoint** games (FR-120).

    Game-disjoint rather than card-disjoint: the gate asks whether the model
    uses a keyword it has seen, and holding out the carriers would empty most
    populations to answer gate 1's question. Read from the curated corpus,
    never the raw shards, which also hold records curation refused.
    """
    from effects.domain.records import RecordKind
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.shard_io import read_records_matching

    games = frozenset(provenance.game_disjoint_games)
    if not games:
        return iter(())
    return read_records_matching(
        CorpusStore(corpus_path).game_disjoint_dir,
        game_ids=games, kinds={RecordKind.COMBAT.value},
    )


def run_gate_two(
    records: Iterable,
    load: Callable[[list], Runnable],
    rows: tuple[DamageStepKeyword, ...] = DAMAGE_STEP_KEYWORDS,
) -> list[KeywordVerdict]:
    """Score the eight keywords and route each one (FR-119 – FR-122).

    ``load`` turns a few records into the checkpoint ready to run, its batcher
    measured from them. Nothing is loaded when there is no combat record:
    every keyword is then under-sampled at n=0.
    """
    records = iter(records)
    head = list(islice(records, SLOT_WIDTH_SAMPLE))
    if not head:
        scores = {row.keyword: KeywordScore() for row in rows}
    else:
        scores = score_keywords(chain(head, records), rows, load(head))
    return [evaluate_keyword(row, scores[row.keyword]) for row in rows]
