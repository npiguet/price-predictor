"""validate-corpus's game boundaries: a game id names one game, the mana
reservoir's flush is not a boundary, and a finished game ends with its
markers.
"""

from __future__ import annotations

from dataclasses import replace

from effects.application.corpus_validation.findings import Thresholds
from effects.application.validate_corpus import validate_corpus
from effects.domain.event_schema import AttributionKind, Event, EventType
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    RecordKind,
    ResolutionPayload,
    TriggerPayload,
)
from tests.unit.effects.application.validate_corpus_support import (
    _activation,
    _combat,
    _healthy,
    _named,
    _resolution,
    _snapshot,
    _trigger,
    _TriggerEvent,
)


def _mana(record_id: str, turn: int, **overrides):
    """The effect half of a mana ability: what the reservoir defers.

    Its snapshot is of the turn the mana was made, and it is written when the
    game ends, so it lands after every later turn's records.
    """
    fields: dict = {
        "state": _snapshot(turn=turn),
        # A mana record resolves no stack item: its outcome is written null.
        "payload": ResolutionPayload(
            outcome=None, outcome_written=True, conditions_met=(),
            events=(
                Event(
                    type=EventType.MANA_PRODUCED,
                    subjects=("P0",),
                    params={"mana_by_color": {"G": 1}},
                    attributed_to=AttributionKind.ROOT,
                ),
            ),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _fired_triggers(records: list[EffectRecord]) -> int:
    return sum(
        1 for r in records
        if r.kind is RecordKind.TRIGGER
        and isinstance(r.payload, TriggerPayload)
        and r.payload.fired
    )


class TestGameIdNamesOneGame:
    def test_a_backward_turn_jump_inside_one_game_id_fails(self):
        records = _healthy() + [
            _resolution("run.0-L1.900", game_id="run.0-L1.0", state=_snapshot(turn=9)),
            # A second game's records landing in the first game's bucket: the
            # turn counter restarts, which no single game ever does.
            _resolution("run.0-L1.901", game_id="run.0-L1.0", state=_snapshot(turn=1)),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "1 span a backward turn jump" in finding.measured

    def test_a_deferred_half_lagging_one_turn_is_not_a_game_boundary(self):
        """Records are written when observed but snapshot what they describe."""
        records = _healthy() + [
            _resolution("run.0-L1.900", game_id="run.0-L1.0", state=_snapshot(turn=9)),
            _resolution("run.0-L1.901", game_id="run.0-L1.0", state=_snapshot(turn=8)),
        ]
        assert _named(validate_corpus(records), "one game").ok

    def test_a_second_deck_under_one_game_id_fails(self):
        """804 game ids held 31,662 games, so a bucket showed ~460 card names."""
        crowd = tuple(f"Card {index}" for index in range(200))
        records = _healthy() + [
            _resolution(
                "run.0-L1.902", game_id="run.0-L1.0",
                state=_snapshot(turn=7, names=crowd),
            ),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "distinct card names" in " ".join(finding.detail)

    def test_the_name_ceiling_is_configurable(self):
        crowd = tuple(f"Card {index}" for index in range(200))
        records = _healthy() + [
            _resolution(
                "run.0-L1.902", game_id="run.0-L1.0",
                state=_snapshot(turn=7, names=crowd),
            ),
        ]
        loose = Thresholds(max_names_per_game=500)
        assert _named(validate_corpus(records, loose), "one game").ok


class TestTheManaReservoirFlushIsNotAGameBoundary:
    """The exemption that stops the checklist from crying wolf.

    Mana activations are reservoir-sampled and flushed at ``close()`` carrying
    the snapshot of the turn the mana was made, so every healthy game ends with
    records that step backwards several turns. Un-exempted, the check failed
    168 of 177 games on a corpus with nothing wrong with it — and an invariant
    that fails on every good run is one an operator stops reading.
    """

    def test_a_flushed_mana_record_does_not_read_as_a_second_game(self):
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _mana("run.0-L1.901", turn=2),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert finding.ok, finding.measured

    def test_the_report_says_how_many_records_were_exempted(self):
        """Silently dropping records from a check is how a check starts lying."""
        records = _healthy() + [_mana(f"run.0-L1.{i}", turn=1) for i in (901, 902)]
        finding = _named(validate_corpus(records), "one game")
        exempted = _fired_triggers(_healthy()) + 2
        assert f"{exempted} records exempted as deferred reservoir flushes" in (
            finding.measured
        )

    def test_a_healthy_window_exempts_only_its_fired_triggers(self):
        """Every fired trigger is written at the game's end (connector C7)."""
        assert f"{_fired_triggers(_healthy())} records exempted" in _named(
            validate_corpus(_healthy()), "one game"
        ).measured

    def test_a_flushed_record_does_not_raise_the_bar_for_later_records(self):
        """Exempt in both directions, or the exemption invents a jump.

        A mana record whose snapshot is *ahead* of the game would otherwise set
        a high-water mark the next ordinary record steps back from.
        """
        records = _healthy() + [
            _mana("run.0-L1.900", turn=30),
            _resolution("run.0-L1.901", state=_snapshot(turn=7)),
        ]
        assert _named(validate_corpus(records), "one game").ok

    def test_a_real_game_boundary_is_still_caught_beside_a_flush(self):
        """Precise, not permissive: only the mana half is exempt."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _mana("run.0-L1.901", turn=2),
            _resolution("run.0-L1.902", state=_snapshot(turn=1)),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "1 span a backward turn jump" in finding.measured

    def test_a_resolution_that_makes_mana_and_more_is_not_exempt(self):
        """A dual-purpose line is not a reservoir record, so it is not exempt."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _resolution(
                "run.0-L1.901", state=_snapshot(turn=2),
                payload=ResolutionPayload(events=(
                    Event(type=EventType.MANA_PRODUCED, subjects=("P0",)),
                    Event(type=EventType.LIFE_CHANGE, subjects=("P0",)),
                )),
            ),
        ]
        assert not _named(validate_corpus(records), "one game").ok

    def test_a_flushed_fired_trigger_does_not_read_as_a_second_game(self):
        """Fired triggers are reservoir-sampled per line too (connector C7) and
        written when the game ends, with the board of the turn they fired."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _trigger("run.0-L1.901", state=_snapshot(turn=2)),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert finding.ok, finding.measured
        assert (
            f"{_fired_triggers(_healthy()) + 1} records exempted as deferred reservoir flushes"
            in finding.measured
        )

    def test_an_unfired_trigger_is_never_exempt(self):
        """Negatives are written as they are drawn, so a jump on one is a game
        boundary."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _trigger("run.0-L1.901", event=_TriggerEvent(fired=False), state=_snapshot(turn=2)),
        ]
        assert not _named(validate_corpus(records), "one game").ok

    def test_an_activation_half_is_never_exempt(self):
        """Nothing defers a cost half, so a jump on one is a game boundary."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _activation("run.0-L1.901", state=_snapshot(turn=2)),
        ]
        assert not _named(validate_corpus(records), "one game").ok


class TestGameEndMarkers:
    """WP7: a gen-3 game that ended with every end-of-game write done ends
    with a ``turn_action`` / ``game_end`` marker; one without was cut or is
    still running, which a window read mid-collection cannot tell apart."""

    @staticmethod
    def _record(game_id: str, index: int, **overrides):
        return _resolution(f"{game_id}.r{index}", game_id=game_id, **overrides)

    @staticmethod
    def _marker(game_id: str, withheld: int = 0):
        from tests.unit.effects.builders import game_end_marker

        return game_end_marker(TestGameEndMarkers._record(game_id, 0, state=_snapshot(turn=40)),
                               records_withheld=withheld)

    def test_games_without_a_marker_are_counted_and_watched(self):
        records = [self._record("run.0-L1.0", 1), self._marker("run.0-L1.0"),
                   self._record("run.0-L1.1", 1)]
        finding = _named(validate_corpus(records), "game-end marker")
        assert finding.ok and finding.watched
        assert "1 of 2 game(s) have no game-end marker (50.0%)" in finding.measured

    def test_no_marker_at_all_fails_once_a_later_game_closed_an_earlier_one(self):
        """A JVM plays its games one after another: an earlier game of a
        lifetime holding a later one finished or was cut, so a window none of
        whose games wrote a marker is a collection that writes none."""
        running = [self._record("run.0-L1.0", 1)]
        assert _named(validate_corpus(running), "game-end marker").ok
        closed = [*running, self._record("run.0-L1.1", 1)]
        finding = _named(validate_corpus(closed), "game-end marker")
        assert not finding.ok and not finding.watched

    def test_the_records_the_cap_withheld_are_summed(self):
        records = [self._marker("run.0-L1.0", 3), self._marker("run.0-L1.1", 4)]
        finding = _named(validate_corpus(records), "withheld")
        assert finding.watched and "7 record(s) withheld over 2 game-end marker(s)" in (
            finding.measured)

    def test_the_marker_is_no_kind_presence_misses(self):
        finding = _named(validate_corpus([self._record("run.0-L1.0", 1)]), "record kind")
        assert not any("game_end" in line for line in finding.detail)

    def test_a_gen3_game_has_ended_only_with_its_marker(self):
        """The trigger floor's finished games: a gen-3 record holding the
        outcome is no longer read as the game's end; the marker is."""
        game = "run.0-L1.0"
        negatives = [_trigger(f"{game}.t{index}", event=_TriggerEvent(fired=False), game_id=game,
                              state=_snapshot(turn=7)) for index in range(60)]
        won = Event(type=EventType.PLAYER_WON, subjects=("P0",), attributed_to="unresolved")
        outcome = _combat(f"{game}.999", game_id=game, state=_snapshot(turn=30),
                          payload=CombatPayload(events=(won,)))
        unfinished = _named(validate_corpus([*negatives, outcome]),
                            "trigger records hold positives")
        assert unfinished.watched
        finished = _named(validate_corpus([*negatives, outcome, self._marker(game)]),
                          "trigger records hold positives")
        assert not finished.ok and "0:60" in finished.measured

    def test_a_pain_land_s_mana_record_is_deferred_whatever_it_did(self):
        """Its damage and life loss are its own events (WP7): still the mana
        reservoir's record, written at the game's end with an earlier board."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        raw = next(r for r in java_golden_records() if r.get("moment") == "resolution"
                   and any(e["type"] == "damage_dealt" for e in r["payload"]["events"])
                   and any(e["type"] == "mana_produced" for e in r["payload"]["events"]))
        mana = record_from_dict(raw)
        game = mana.game_id
        later = self._record(game, 5, state=_snapshot(turn=9))
        early = replace(mana, record_id=f"{game}.r6",
                        state=replace(mana.state, global_=_snapshot(turn=2).global_))
        finding = _named(validate_corpus([later, early, self._marker(game)]), "one game")
        assert finding.ok, finding.measured
