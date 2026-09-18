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

from dataclasses import dataclass, field

import torch

from effects.domain.damage_step_keywords import (
    CombatParticipant,
    DamageStepKeyword,
    KeywordsOf,
    combat_participant,
    keyword_of_line,
    subject_id,
)
from effects.domain.effect_model import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    ZONE_OUTCOMES,
    FieldSpec,
    FieldType,
)
from effects.domain.records import RecordKind

#: Observations per forward pass. Half of gate 1's record batch, because every
#: observation is built twice — as collected and perturbed — in one batch.
BATCH_OBSERVATIONS = 16


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
    per_field: dict[str, tuple[int, int]] = field(default_factory=dict)


class KeywordResolver:
    """An entity's keywords, from both channels, memoized by provenance key.

    Both channels have to be looked at. An entity's printed and
    attachment-granted keywords reach the model as ability tokens, and only a
    keyword granted until end of turn appears as a bare string in the overlay —
    so reading the overlay alone sees the rare case and misses every creature
    that printed the keyword, which is the common one.

    The memo is what makes that affordable: a corpus repeats the same few
    thousand cards across millions of combat records, and resolving each key
    once turns the walk into a dict hit.
    """

    def __init__(self, sidecars=None) -> None:
        self._sidecars = sidecars
        self._by_key: dict[object, str | None] = {}

    def _keyword_for(self, key) -> str | None:
        if key not in self._by_key:
            keyword = None
            try:
                line = self._sidecars.line_for(key)
            except (KeyError, FileNotFoundError):
                line = None
            if line is not None:
                keyword = keyword_of_line(line)
            self._by_key[key] = keyword
        return self._by_key[key]

    def keywords_of(self, entity) -> set[str]:
        # Already in this package's spelling: ``record_io`` runs Forge's
        # keyword strings through ``normalize_keyword`` at parse time.
        found = set(entity.granted_temporary.keywords)
        if self._sidecars is None:
            return found
        for key in (*entity.printed, *entity.granted_attached):
            keyword = self._keyword_for(key)
            if keyword is not None:
                found.add(keyword)
        return found


# ── who is observed ─────────────────────────────────────────────────────


def qualifying_observations(
    record, rows: tuple[DamageStepKeyword, ...], resolver: KeywordResolver,
) -> list[tuple[DamageStepKeyword, CombatParticipant]]:
    """Every ``(row, carrier)`` in one combat record the gate can observe.

    The whole condition is the row's own ``qualifies`` — at least one effect
    that applies and has a subject on the board — and it is a board fact rather
    than a model fact, so the counter and the scorer reach the same answer by
    asking the same question rather than by two rules kept in step.
    """
    if record.kind is not RecordKind.COMBAT:
        return []
    found: list[tuple[DamageStepKeyword, CombatParticipant]] = []
    wanted = {row.keyword for row in rows}
    for entity in record.state.entities:
        if entity.combat is None:
            continue
        carried = resolver.keywords_of(entity) & wanted
        if not carried:
            continue
        participant = combat_participant(record.state, entity)
        if participant is None:
            continue
        # In table order, not set-iteration order: which observations share a
        # batch would otherwise depend on the hash seed.
        for row in rows:
            if row.keyword not in carried:
                continue
            if row.qualifies(participant, resolver.keywords_of):
                found.append((row, participant))
    return found


def count_qualifying_all(
    records, rows: tuple[DamageStepKeyword, ...], sidecars=None,
) -> dict[str, int]:
    """Every row's qualifying count, in one pass over the corpus.

    One pass rather than one per keyword: each entity's keyword set is resolved
    once and checked against all eight, which is the difference between an
    evaluation that finishes and one that does not.

    The unit is a ``(record, carrier)`` observation, not a record: two creatures
    with lifelink in one combat are two independent perturbations, and the
    scorer scores them separately.
    """
    resolver = KeywordResolver(sidecars)
    counts = {row.keyword: 0 for row in rows}
    for record in records:
        for row, _participant in qualifying_observations(record, rows, resolver):
            counts[row.keyword] += 1
    return counts


def count_qualifying(records, row: DamageStepKeyword, sidecars=None) -> int:
    """One row's qualifying count.

    Without ``sidecars`` only the overlay channel is visible, and the count then
    describes a different population than the gate's table does — printed first
    strike is most first strike.
    """
    return count_qualifying_all(records, (row,), sidecars)[row.keyword]


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
    start, end = FIELD_SLICES[spec.name]
    slice_ = vector[start:end]
    match spec.type:
        case FieldType.COUNT:
            return float(torch.exp(slice_[0]))
        case FieldType.SIGNED_DELTA:
            probabilities = torch.softmax(slice_[:3], dim=-1)
            sign = float(probabilities[2] - probabilities[0])
            return sign * float(torch.exp(slice_[3]))
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
    sidecars,
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
    resolver = KeywordResolver(sidecars)
    #: keyword -> [qualifying, agreeing, {effect label: (agreeing, scored)}].
    totals: dict[str, list] = {row.keyword: [0, 0, {}] for row in rows}
    pending: list = []

    encoder.eval()
    model.eval()
    with torch.no_grad():
        for record in records:
            for row, participant in qualifying_observations(
                record, rows, resolver,
            ):
                pending.append((record, row, participant))
                if len(pending) == BATCH_OBSERVATIONS:
                    _score_chunk(
                        pending, encoder, model, batcher, totals,
                        fields=fields, keywords_of=resolver.keywords_of,
                    )
                    pending = []
        if pending:
            _score_chunk(
                pending, encoder, model, batcher, totals,
                fields=fields, keywords_of=resolver.keywords_of,
            )

    return {
        keyword: KeywordScore(
            qualifying=tally[0], agreeing=tally[1], per_field=tally[2],
        )
        for keyword, tally in totals.items()
    }


def _score_chunk(
    chunk: list, encoder, model, batcher, totals: dict[str, list], *,
    fields, keywords_of: KeywordsOf,
) -> None:
    """Run one chunk of observations, paired, and fold it into ``totals``."""
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
        tally = totals[row.keyword]
        tally[0] += 1
        tally[1] += 1 if agreed else 0
        for label, ok in per_effect.items():
            agreeing, scored = tally[2].get(label, (0, 0))
            tally[2][label] = (agreeing + (1 if ok else 0), scored + 1)


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
