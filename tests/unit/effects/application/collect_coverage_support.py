"""Builders the collect-coverage test modules share: a converted acting
line, the gen-3 fixture shard and its held-out texts, a turn-action
record and a small converted card tree."""

from __future__ import annotations

from pathlib import Path

from effects.domain.records import RecordKind
from tests.unit.effects.builders import held_out_cards


def _convert_acting_line(cards_folder: Path) -> None:
    """A sidecar for :data:`ACTING_KEY` under ``cards_folder``, so a
    :func:`_record` acting through it is kept by the refusal rule."""
    from effects.domain.provenance import ProvenanceSidecar, SidecarLine
    from tests.unit.effects.builders import write_sidecar
    from tests.unit.effects.domain.test_coverage import ACTING_KEY

    write_sidecar(
        ProvenanceSidecar(card="shock", script_file=ACTING_KEY.script_file, lines=(
            SidecarLine(0, "spell", (ACTING_KEY,), script_text="SP$ DealDamage | NumDmg$ 2",
                prose_offset=0),
        )),
        cards_folder / "s" / "shock.provenance.json",
    )


_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


_GEN3_SHARD = _FIXTURES / "gen3-records.jsonl.gz"


_GEN3_ROOTS = {
    "cardsfolder": str(_FIXTURES / "gen3-sidecars" / "cardsfolder"),
    "tokenscripts": str(_FIXTURES / "gen3-sidecars" / "tokenscripts"),
}


def _holding(texts):
    """A holdout of exactly ``texts``."""
    return held_out_cards(texts=texts)


def _texts_by_game(shard: Path) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """``text -> games`` over the shard's resolution records, and over every
    record: each record counted through its first key with a script text."""
    from effects.domain.text_holdout import normalize_script_text
    from effects.infrastructure.shard_io import read_shard
    from effects.infrastructure.sidecar_io import SidecarCache

    cache = SidecarCache({tree: Path(root) for tree, root in _GEN3_ROOTS.items()})
    resolutions: dict[str, set[str]] = {}
    everything: dict[str, set[str]] = {}
    for record in read_shard(shard):
        for key in record.ability or ():
            line = cache.line_for(key)
            if line is None or not line.script_text:
                continue
            text = normalize_script_text(line.script_text)
            everything.setdefault(text, set()).add(record.game_id)
            if record.kind is RecordKind.RESOLUTION:
                resolutions.setdefault(text, set()).add(record.game_id)
            break
    return resolutions, everything


def _turn_action(entities, events):
    from effects.domain.records import TurnActionPayload, TurnActionSubkind
    from tests.unit.effects.builders import make_record, make_snapshot

    return make_record(
        RecordKind.TURN_ACTION, subkind=TurnActionSubkind.DRAW, generation=3,
        state=make_snapshot(entities=tuple(entities)),
        payload=TurnActionPayload(events=tuple(events)),
    )


def _two_cards(tmp_path, *names):
    cards = tmp_path / "cardsfolder"
    cards.mkdir()
    for name in names:
        (cards / f"{name}.txt").write_text(
            f"name: {name}\nmana cost: {{G}}\ntypes: creature\n", encoding="utf-8",
        )
    _convert_acting_line(cards)
    return cards
