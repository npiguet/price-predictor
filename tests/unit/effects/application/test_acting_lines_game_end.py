"""``ActingLines`` and the game-end record: the record that seeks no line is
counted apart and is not a silent empty ability.

Every record here is decoded from a Java golden: the game-end record from
``shard-game-end.jsonl``, the unkeyed mana record by editing ``record-mana.jsonl``.
"""

from __future__ import annotations

import json
from dataclasses import replace

from effects.application.corpus_validation.envelope_checks import ActingLines
from effects.application.corpus_validation.findings import Thresholds
from effects.domain.records import EffectRecord, UnresolvedReason
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import JAVA_GOLDENS

_SAYS_WHY = "every empty ability says why it is empty"


def _golden_lines(name: str) -> list[EffectRecord]:
    lines = (JAVA_GOLDENS / name).read_text(encoding="utf-8").splitlines()
    return [record_from_dict(json.loads(line)) for line in lines if line.strip()]


def _game_end_record() -> EffectRecord:
    return _golden_lines("shard-game-end.jsonl")[2]


def _says_why(*records: EffectRecord):
    acting = ActingLines()
    for record in records:
        acting.add(record)
    return next(f for f in acting.findings(Thresholds()) if f.name == _SAYS_WHY)


def test_the_game_end_record_is_not_a_silent_empty_ability():
    record = _game_end_record()
    assert record.is_game_end_record
    finding = _says_why(record)
    assert finding.ok, finding.measured
    assert "1 game-end" in finding.measured


def test_a_resolution_with_a_link_and_no_reason_is_still_silent():
    silent = replace(_game_end_record(), link_id="pair.1")
    assert not silent.is_game_end_record
    finding = _says_why(silent)
    assert not finding.ok
    assert "1 of 1 empty abilities carry no ability_unresolved reason" in finding.measured
    assert "0 game-end" in finding.measured


def test_a_game_end_record_beside_a_silent_one_leaves_the_check_failing():
    finding = _says_why(_game_end_record(), replace(_game_end_record(), link_id="pair.1"))
    assert not finding.ok
    assert "1 of 1 empty abilities carry no ability_unresolved reason" in finding.measured
    assert "1 game-end" in finding.measured


def test_an_unkeyed_mana_record_is_not_a_game_end_record():
    mana = next(r for r in _golden_lines("record-mana.jsonl") if not r.fork)
    unkeyed = replace(mana, ability=(), ability_unresolved=UnresolvedReason.NO_CARD_STATE,
                      link_id=None)
    assert unkeyed.payload.outcome is None
    assert not unkeyed.is_game_end_record
    finding = _says_why(unkeyed)
    assert finding.ok
    assert "0 game-end" in finding.measured
