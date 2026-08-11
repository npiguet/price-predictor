"""Project a draft corpus into the flat seat table the Forge workers sample from.

The seat table is the whole interface between Python and the game-evaluation
workers: Python writes it once, each worker loads it and draws its own pairings
thereafter. Keeping the projection here means the booster geometry and the
corpus schema stay in one language.

Line format, five semicolon-separated fields::

    draft_id;set_code;label;kind;cards

``kind`` is ``deck`` for a seat that plays the deck recorded in the corpus, or
``pool`` for one whose deck Forge rebuilds from its drafted pool. ``cards`` is
pipe-separated; Forge canonical card names contain neither ``;`` nor ``|``.
"""

from __future__ import annotations

import random as random_module
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from draft.domain.draft_geometry import DraftGeometry, DraftRecord

FIELD_SEPARATOR = ";"
CARD_SEPARATOR = "|"

KIND_DECK = "deck"
KIND_POOL = "pool"

#: The Forge reference label. Only these seats are eligible for diversion.
FORGE_REFERENCE_LABEL = "forge-full"
#: The label a diverted seat reports under, so the tally keeps the two apart.
FORGE_NATIVE_LABEL = "forge-native"


@dataclass(frozen=True, slots=True)
class SeatRow:
    """One seat, as the workers see it.

    Attributes:
        draft_id: groups seats into pods; opaque to the worker.
        set_code: the set the pod drafted; becomes the match row's ``set_code``.
        label: becomes ``method_A``/``method_B`` on the match row.
        kind: ``deck`` or ``pool`` — how ``cards`` is to be read.
        cards: a finished deck, or a drafted pool for a diverted seat.
    """

    draft_id: str
    set_code: str
    label: str
    kind: str
    cards: tuple[str, ...]


def record_set_code(record: DraftRecord) -> str:
    """The set every match drawn from this record is played in.

    A record's boosters all carry the same set code, so the first one speaks for
    the pod.
    """
    return record.boosters[0].set_code


def seat_rows(
    record: DraftRecord,
    *,
    native_fraction: float = 0.0,
    rng: random_module.Random | None = None,
) -> list[SeatRow]:
    """Project one draft record into one row per seat.

    A seat normally plays the deck the corpus recorded for it. With a non-zero
    ``native_fraction``, each ``forge-full`` seat is independently diverted with
    that probability: it contributes its drafted pool instead of its deck, under
    the label ``forge-native``, so Forge's own builder can be compared against
    the recorded one on the same drafted cards.

    The choice is made once here, as the seat table is written, so a seat is
    diverted in every pairing it appears in or in none.
    """
    set_code = record_set_code(record)
    geometry = None
    rows: list[SeatRow] = []

    for index, seat in enumerate(record.seats):
        divert = (
            native_fraction > 0.0
            and seat.agent == FORGE_REFERENCE_LABEL
            and (rng or random_module).random() < native_fraction
        )
        if divert:
            if geometry is None:
                geometry = DraftGeometry.from_record(record)
            cards = tuple(geometry.drafted_pool(record, index))
            label, kind = FORGE_NATIVE_LABEL, KIND_POOL
        else:
            cards = tuple(seat.deck)
            label, kind = seat.agent, KIND_DECK

        rows.append(
            SeatRow(
                draft_id=record.draft_id,
                set_code=set_code,
                label=label,
                kind=kind,
                cards=cards,
            )
        )
    return rows


def project(
    records: Iterable[DraftRecord],
    *,
    native_fraction: float = 0.0,
    rng: random_module.Random | None = None,
) -> Iterator[SeatRow]:
    """Stream every record's seat rows.

    Records are consumed one at a time and dropped, so the corpus is never held
    in memory.
    """
    for record in records:
        yield from seat_rows(record, native_fraction=native_fraction, rng=rng)


def format_seat_row(row: SeatRow) -> str:
    """Render one seat row as a seat-table line, without its newline."""
    return FIELD_SEPARATOR.join(
        (
            row.draft_id,
            row.set_code,
            row.label,
            row.kind,
            CARD_SEPARATOR.join(row.cards),
        )
    )


def parse_seat_row(line: str) -> SeatRow:
    """Parse a seat-table line. The inverse of :func:`format_seat_row`.

    Provided so the format has one definition that tests can round-trip; the
    workers parse it independently in Java.
    """
    draft_id, set_code, label, kind, cards = line.split(FIELD_SEPARATOR, 4)
    return SeatRow(
        draft_id=draft_id,
        set_code=set_code,
        label=label,
        kind=kind,
        cards=tuple(cards.split(CARD_SEPARATOR)) if cards else (),
    )


def write_seat_table(
    records: Iterable[DraftRecord],
    out,
    *,
    native_fraction: float = 0.0,
    rng: random_module.Random | None = None,
) -> tuple[int, int, int]:
    """Write every record's seats to ``out``.

    ``out`` is any text file object. Seats of one pod land on consecutive lines,
    but a reader must group by ``draft_id`` rather than rely on that.

    Returns:
        ``(seats_written, forge_reference_seats, diverted_seats)`` — the last two
        let the startup echo report how many reference seats were diverted out of
        how many were eligible.
    """
    written = 0
    reference = 0
    diverted = 0
    for record in records:
        reference += sum(1 for s in record.seats if s.agent == FORGE_REFERENCE_LABEL)
        for row in seat_rows(record, native_fraction=native_fraction, rng=rng):
            if row.label == FORGE_NATIVE_LABEL:
                diverted += 1
            out.write(format_seat_row(row))
            out.write("\n")
            written += 1
    return written, reference, diverted
