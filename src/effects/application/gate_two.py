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
table's own predicate (:mod:`effects.domain.damage_step_keywords`). The
under-sampled verdict and the agreement verdict sit on the same report line, and
two populations behind them would make that line unreadable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import batched
from typing import NamedTuple

import torch

from effects.application.gate_one import BATCH_RECORDS
from effects.domain.damage_step_keywords import (
    CombatParticipant,
    DamageStepKeyword,
    KeywordResolver,
    KeywordsOf,
    qualifying_observations,
    subject_id,
)
from effects.domain.effect_head_layout import FIELDS_BY_NAME, ZONE_OUTCOMES, FieldSpec, FieldType
from effects.domain.head_decoding import (
    direction_balance,
    expected_magnitude,
    field_outputs,
)

logger = logging.getLogger(__name__)

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

    ``per_field`` is reported beside the total because the two halves of a row
    fail differently: a keyword whose damage field moves and whose zone field
    does not is a head that learned the magnitude and not the consequence, and
    the combined percentage alone cannot say so.
    """

    qualifying: int = 0
    agreeing: int = 0
    #: ``effect label -> (agreeing, scored)``.
    per_field: dict[str, FieldAgreement] = field(default_factory=dict)


@dataclass(slots=True)
class _Tally:
    """One keyword's running counts while the pass is still reading."""

    qualifying: int = 0
    agreeing: int = 0
    per_field: dict[str, FieldAgreement] = field(default_factory=dict)

    def add(self, agreed: bool, per_effect: dict[str, bool]) -> None:
        self.qualifying += 1
        self.agreeing += 1 if agreed else 0
        for label, ok in per_effect.items():
            agreeing, scored = self.per_field.get(label, FieldAgreement(0, 0))
            self.per_field[label] = FieldAgreement(
                agreeing + (1 if ok else 0), scored + 1,
            )

    def score(self) -> KeywordScore:
        return KeywordScore(self.qualifying, self.agreeing, self.per_field)


# ── decoding a head into one comparable number ──────────────────────────


def read_field(
    vector: torch.Tensor, spec: FieldSpec, category: str | None,
) -> float:
    """One field's prediction as a single scalar, on the field's own scale.

    Each head has a layout of its own and none of them is a bare value: a count
    predicts a log rate, a signed delta predicts three direction logits and a
    magnitude, and a categorical predicts logits over classes. The scalar has to
    be comparable between the two builds and nothing more, so each type is
    decoded into the quantity the rules talk about — expected damage, expected
    signed life change, probability of dying.

    Any other type raises rather than guessing: the table pins every effect to a
    real head, and a row naming a field this cannot read is a table bug.
    """
    slice_ = field_outputs(vector, spec)
    match spec.type:
        case FieldType.COUNT:
            return float(expected_magnitude(spec, slice_))
        case FieldType.SIGNED_DELTA:
            sign = float(direction_balance(spec, slice_))
            return sign * float(expected_magnitude(spec, slice_))
        case FieldType.CATEGORICAL:
            if spec.name != "zone_outcome" or category is None:
                raise ValueError(
                    f"gate 2 can only read a class of zone_outcome, not "
                    f"{spec.name}"
                )
            return float(torch.softmax(slice_, dim=-1)[
                ZONE_OUTCOMES.index(category)
            ])
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
    """Every qualifying ``(record, row, participant)``, read lazily."""
    for record in records:
        progress.read += 1
        if progress.read % PROGRESS_EVERY_RECORDS == 0:
            progress.log()
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
    batch, surfaces = batcher.build(built, encoder, strips)
    outputs = model.per_entity(model(**batch)).cpu()

    for offset, (_record, row, participant) in enumerate(chunk):
        twin = len(chunk) + offset
        agreed, per_effect = score_observation(
            row, participant,
            _subject_vectors(surfaces[offset], outputs[offset], row, participant),
            _subject_vectors(surfaces[twin], outputs[twin], row, participant),
            fields=fields, keywords_of=keywords_of,
        )
        tallies[row.keyword].add(agreed, per_effect)


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
