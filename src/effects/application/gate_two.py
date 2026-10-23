"""Gate 2's numbers, measured by running a checkpoint with a keyword removed.

Gate 1 asks whether the encoder reads text; gate 2 asks whether it reads one
specific word. For each of the eight damage-step keywords the evaluator takes
the combat records where that keyword's carrier could have mattered, removes the
keyword from the carrier's **model input**, and counts how often the prediction
for the fields the rules touch moves the way the rules say.

Three choices here are the whole point of the gate.

The perturbation is **model-side and paired**: the same record is built twice in
the same batch, once as collected and once with the keyword gone, so the two
predictions differ in the keyword and in nothing else — not in the board, not in
the encoded abilities, not in a sampling decision.

The population is the **game-disjoint** validation split, not the card-disjoint
one. What is being tested is whether the model uses the keyword, not whether it
generalizes to unseen text, and holding out cards would shrink the eight
populations to nothing while answering a question gate 1 already answers.

And the count and the agreement describe **one population**, decided by the
table's own predicate (:mod:`effects.domain.damage_step_keywords`), over the
combats the games actually played: a damage-step probe or an intervention is a
fork of a real board and qualifies nothing. The under-sampled verdict and the
agreement verdict sit on the same report line, and two populations behind them
would make that line unreadable. Within it the threshold counts **records** —
FR-119's 200 is a count of combats — while the agreement averages the
**observations**, one per (record, carrier), because each carrier is its own
perturbation and a combat holding two first strikers answers twice.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from itertools import batched, chain, islice
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import torch

from effects.application.gate_one import BATCH_RECORDS
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

if TYPE_CHECKING:
    from effects.infrastructure.model_runner import Runnable

logger = logging.getLogger(__name__)

#: Records read eagerly before gate 2 loads its checkpoint. The loader measures
#: each slot kind's width from real records, and every combat record on a board
#: with a creature and a player answers that identically.
SLOT_WIDTH_SAMPLE = 8

#: Records read between two progress lines. The game-disjoint stratum streams
#: from the raw corpus and runs for many minutes with nothing else to say.
PROGRESS_EVERY_RECORDS = 50_000

#: Observations per forward pass: half of gate 1's record batch, because every
#: observation is built twice — as collected and perturbed — in one batch.
BATCH_OBSERVATIONS = BATCH_RECORDS // 2


class FieldAgreement(NamedTuple):
    """How many of one effect's scored observations moved the rules' way."""

    agreeing: int
    scored: int


@dataclass(frozen=True, slots=True)
class KeywordScore:
    """One keyword's population and how much of it agreed.

    ``qualifying`` counts distinct records, ``observations`` the (record,
    carrier) perturbations scored, and ``agreeing`` the observations that
    moved every field the rules' way.

    ``per_field`` is reported beside the total because the two halves of a row
    fail differently: a keyword whose damage field moves and whose zone field
    does not is a head that learned the magnitude and not the consequence, and
    the combined percentage alone cannot say so.
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


# ── decoding a head into one comparable number ──────────────────────────


def read_field(
    vector: torch.Tensor, spec: FieldSpec, category: str | None,
) -> float:
    """One field's prediction as a single scalar, on the field's own scale.

    Each head has a layout of its own and none of them is a bare value, so each
    type is decoded into the quantity the rules talk about — expected damage,
    expected signed life change, probability of dying — through
    :class:`HeadReading`, **gated**: ``P(affected)`` times the field's value.
    The field heads train only where an entity was affected, so they cannot
    say that an entity was not; removing first strike from an attacker leaves
    its blocker unaffected in the first-strike step, and that move is in the
    gate alone.

    Any other type raises rather than guessing: the table pins every effect to a
    real head, and a row naming a field this cannot read is a table bug.
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
    fields: tuple[FieldSpec, ...],
    keywords_of: KeywordsOf,
) -> tuple[bool, dict[str, bool]]:
    """Whether one perturbation moved every field it should have.

    ``present`` and ``stripped`` map a subject id to that subject's per-entity
    output vector in the two builds. The observed change is present minus
    stripped, and the effect agrees when its sign is the row's direction — so a
    prediction that did not move at all disagrees, which is the answer for a
    model that ignored the keyword.

    An effect this combat cannot show — one whose ``applies`` is false or whose
    subject is not on the board — is left **unscored** rather than counted
    against the keyword, through the same ``observable`` the row's ``qualifies``
    counts it by. Infect is the case that makes it necessary: its creature-side
    fields and its player-side fields are exclusive, so scoring all four on
    every combat would fail half of them by construction. A field the run never
    trained is skipped for the same reason — there is no prediction to compare.
    """
    active = {spec.name for spec in fields}
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


# ── the pass over the corpus ────────────────────────────────────────────


def score_keywords(
    records,
    rows: tuple[DamageStepKeyword, ...],
    encoder,
    model,
    batcher,
    *,
    fields: tuple[FieldSpec, ...],
) -> dict[str, KeywordScore]:
    """Run every qualifying perturbation and tally the eight rows.

    ``fields`` is the active-field tuple the run trains with, so the gate reads
    the same heads the loss shaped and reports nothing for one it never did.

    ``records`` is consumed as a stream and only a chunk's worth is ever held:
    a parsed record costs about 45 KB, and the game-disjoint split is a thousand
    games.
    """
    # The batcher's own sidecar cache, never a second one: the resolver decides
    # which entity carries a keyword and the surface builder decides which slot
    # to drop for it, and the two reading different joins is exactly how a
    # perturbation strips nothing.
    resolver = KeywordResolver(batcher.sidecars)
    tallies = {row.keyword: _Tally() for row in rows}
    progress = _Progress()

    encoder.eval()
    model.eval()
    with torch.inference_mode():
        observations = _qualifying(records, rows, resolver, progress)
        for chunk in batched(observations, BATCH_OBSERVATIONS):
            _score_chunk(
                chunk, encoder, model, batcher, tallies,
                fields=fields, keywords_of=resolver.keywords_of,
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
) -> Iterator[tuple]:
    """Every qualifying ``(record, row, participant)``, read lazily.

    A fork — a damage-step probe or an intervention — is never one: it re-runs
    a real board the population already holds, so counting it would weigh that
    combat twice and count a near-duplicate toward the threshold.
    """
    for record in records:
        progress.read += 1
        if progress.read % PROGRESS_EVERY_RECORDS == 0:
            progress.log()
        if record.fork or record.interventional:
            continue
        for row, participant in qualifying_observations(record, rows, resolver):
            progress.observed += 1
            yield record, row, participant


def _score_chunk(
    chunk: tuple, encoder, model, batcher, tallies: dict[str, _Tally], *,
    fields, keywords_of: KeywordsOf,
) -> None:
    """Run one chunk of observations, paired, and fold it into ``tallies``."""
    # Every observation twice: as collected, then with its keyword removed from
    # its carrier. One batch, so the two builds share the encoded abilities and
    # differ only in the perturbation.
    built = [record for record, _row, _participant in chunk] * 2
    strips = [None] * len(chunk) + [
        {participant.carrier.id: frozenset({row.keyword})}
        for _record, row, participant in chunk
    ]
    batch, surfaces, _encoded = batcher.build(built, encoder, strips)
    outputs = model.per_entity(model(**batch)).cpu()

    for offset, (record, row, participant) in enumerate(chunk):
        twin = len(chunk) + offset
        agreed, per_effect = score_observation(
            row, participant,
            _subject_vectors(surfaces[offset], outputs[offset], row, participant),
            _subject_vectors(surfaces[twin], outputs[twin], row, participant),
            fields=fields, keywords_of=keywords_of,
        )
        tallies[row.keyword].add(record.record_id, agreed, per_effect)


def _subject_vectors(
    surface, outputs: torch.Tensor, row: DamageStepKeyword,
    participant: CombatParticipant,
) -> dict[str, torch.Tensor]:
    """The head's output row for each subject this row reads.

    Looked up per surface rather than once, because the two builds do not share
    a slot layout: dropping the keyword's ability token shifts every slot after
    it.
    """
    wanted = {
        subject for subject in (
            subject_id(effect, participant) for effect in row.effects
        ) if subject is not None
    }
    found: dict[str, torch.Tensor] = {}
    for index in surface.entity_slots:
        slot = surface.slots[index]
        key = slot.entity_id or slot.player_id
        if key in wanted:
            found[key] = outputs[index]
    return found


# ── the verdict ─────────────────────────────────────────────────────────


def evaluate_keyword(
    row: DamageStepKeyword,
    *,
    qualifying_records: int,
    agreeing: int,
    observations: int | None = None,
    per_field: dict[str, FieldAgreement] | None = None,
) -> KeywordVerdict:
    """Route one keyword: pass, or send it to a stage-three probe (FR-119).

    Under-sampled routes the same way as wrong. A keyword with 40 qualifying
    records has not been tested, and treating "untested" as "passed" is how a
    canary stops being one.

    The threshold reads ``qualifying_records``; the agreement is ``agreeing``
    over ``observations``, which default to one per record.

    ``per_field`` is reported but never routes: a row's verdict is "did every
    field move", and the breakdown says *which* one did not — a keyword whose
    magnitude moves and whose consequence does not is a different problem from
    one the model ignores entirely, and the combined percentage cannot tell them
    apart.
    """
    if observations is None:
        observations = qualifying_records
    agreement = agreeing / observations if observations else 0.0
    breakdown = _render_per_field(per_field)
    if qualifying_records < MIN_QUALIFYING_RECORDS:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"under-sampled: {qualifying_records} qualifying records < "
                f"{MIN_QUALIFYING_RECORDS}" + breakdown
            ),
            observations=observations,
        )
    if agreement < MIN_DIRECTION_AGREEMENT:
        return KeywordVerdict(
            row.keyword, qualifying_records, agreement, routed_to_probe=True,
            reason=(
                f"direction agreement {agreement:.1%} < "
                f"{MIN_DIRECTION_AGREEMENT:.0%} on {', '.join(row.fields)}"
                f" over {observations} observations" + breakdown
            ),
            observations=observations,
        )
    return KeywordVerdict(
        row.keyword, qualifying_records, agreement, routed_to_probe=False,
        reason=(
            f"direction agreement {agreement:.1%} over {observations} "
            "observations" + breakdown
        ),
        observations=observations,
    )


def _render_per_field(per_field: dict[str, FieldAgreement] | None) -> str:
    """The per-effect tail of a keyword's report line, or nothing."""
    if not per_field:
        return ""
    parts = [
        f"{label} {agreeing / scored:.0%}"
        for label, (agreeing, scored) in per_field.items() if scored
    ]
    return f" ({', '.join(parts)})" if parts else ""


def evaluate_gate_two(verdicts: list[KeywordVerdict]) -> CheckResult:
    """The canary's routing report. **Blocks nothing** (FR-119).

    Its output is a build decision, not a ship decision: the keywords routed
    here are exactly the ones stage three needs probe machinery for, and a run
    where none are routed means that machinery is never written.
    """
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


# ── the gate ────────────────────────────────────────────────────────────


def gate_two_records(config, provenance, *, corpus_path: Path | None) -> Iterator:
    """The combat records of the checkpoint's **game-disjoint** games (FR-120).

    Game-disjoint rather than card-disjoint: the gate asks whether the model
    uses a keyword it has seen, not whether it generalizes to text it has not,
    and holding out the carriers as well would empty six of the eight
    populations to answer a question gate 1 already answers.

    From the curated corpus's ``validation/game-disjoint`` when the checkpoint
    trained on one: the raw shards also hold the records curation refused or
    repaired away. The raw corpus is read only for a checkpoint with no corpus.

    An iterator, not a list. The split is a thousand games and a parsed record
    costs about 45 KB, so the scorer reads it a chunk at a time.
    """
    from effects.domain.records import RecordKind
    from effects.infrastructure.record_io import read_records_matching

    games = frozenset(provenance.game_disjoint_games)
    if not games:
        return iter(())
    if corpus_path is not None:
        from effects.infrastructure.corpus_store import CorpusStore

        directory = CorpusStore(corpus_path).game_disjoint_dir
    else:
        directory = Path(config.records_dir)
    return read_records_matching(
        directory, game_ids=games, kinds={RecordKind.COMBAT.value},
    )


def run_gate_two(
    records: Iterable,
    load: Callable[[list], Runnable],
    rows: tuple[DamageStepKeyword, ...] = DAMAGE_STEP_KEYWORDS,
) -> list[KeywordVerdict]:
    """Score the eight keywords and route each one (FR-119 – FR-122).

    ``load`` turns a few records into the checkpoint ready to run, with a
    batcher measured from them; the evaluator passes its main checkpoint's,
    loaded once for the whole run. Nothing is loaded when ``records`` holds no
    combat record to score: every keyword is then under-sampled at n=0, which
    is the same verdict a corpus with forty records gets and for the same
    reason.
    """
    records = iter(records)
    # A handful of records up front, because the loader measures the slot
    # widths from real ones; the rest stays a stream the scorer reads in chunks.
    head = list(islice(records, SLOT_WIDTH_SAMPLE))
    if not head:
        scores = {row.keyword: KeywordScore() for row in rows}
    else:
        runnable = load(head)
        scores = score_keywords(
            chain(head, records), rows, runnable.encoder, runnable.model,
            runnable.batcher, fields=runnable.fields,
        )
    return [
        evaluate_keyword(
            row,
            qualifying_records=scores[row.keyword].qualifying,
            agreeing=scores[row.keyword].agreeing,
            observations=scores[row.keyword].observations,
            per_field=scores[row.keyword].per_field,
        )
        for row in rows
    ]
