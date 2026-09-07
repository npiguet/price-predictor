"""The card-disjoint split (T040).

Three properties, and the middle one is the whole point: the holdout is newest
cards rather than random ones, **every game holding a record that names a
held-out card is excluded**, and game-disjoint validation takes 10% of what
remains. Excluding the row and keeping the game would let a held-out card sit in
the context of a training record, which is exactly the loop that trains each
embedding from both directions — and gate 1, the shipping gate, would then score
partly on cards the encoder had already seen.
"""

from __future__ import annotations

import json

from effects.application.train_effect_model import (
    CARD_HOLDOUT_FRACTION,
    GAME_HOLDOUT_FRACTION,
    CorpusSplit,
    HeldOutCards,
    derive_split,
    load_card_files,
    load_first_printings,
    newest_first_holdout,
    record_names_held_out_card,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot


def _entity(name: str, **overrides) -> EntityState:
    defaults = {
        "id": f"E-{name}", "name": name, "zone": "battlefield",
        "controller": "P0",
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _record(game_id: str, entities=(), ability=None, **overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": RecordKind.RESOLUTION,
        "moment": Moment.RESOLUTION, "actor_player": "P0",
        "ability": ability,
        "state": StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(),
            entities=tuple(entities),
        ),
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestNewestFirstHoldout:
    def _corpus(self, n: int) -> tuple[dict[str, str], dict[str, str]]:
        files = {f"Card {i:03d}": f"cardsfolder/c/card_{i:03d}.txt" for i in range(n)}
        # Card 000 is the newest; dates descend with the index.
        printings = {
            f"Card {i:03d}": f"{2026 - i // 12:04d}-{12 - i % 12:02d}-01"
            for i in range(n)
        }
        return files, printings

    def test_the_holdout_covers_at_least_eight_percent(self):
        files, printings = self._corpus(100)
        held = newest_first_holdout(files, printings)
        assert len(held.names) / len(files) >= CARD_HOLDOUT_FRACTION

    def test_it_takes_the_newest_cards_not_random_ones(self):
        files, printings = self._corpus(100)
        held = newest_first_holdout(files, printings)
        newest = sorted(printings, key=lambda n: printings[n], reverse=True)
        assert held.names == set(newest[: len(held.names)])

    def test_the_holdout_is_deterministic(self):
        files, printings = self._corpus(100)
        assert newest_first_holdout(files, printings) == newest_first_holdout(
            files, printings
        )

    def test_a_card_with_no_printing_date_sorts_as_oldest(self):
        """An unrecognized name must not become "newest" by accident."""
        files, printings = self._corpus(50)
        files["Mystery Card"] = "cardsfolder/m/mystery_card.txt"
        held = newest_first_holdout(files, printings)
        assert "Mystery Card" not in held.names

    def test_the_holdout_records_both_names_and_script_files(self):
        files, printings = self._corpus(50)
        held = newest_first_holdout(files, printings)
        assert len(held.names) == len(held.script_files)
        assert all(f.startswith("cardsfolder/") for f in held.script_files)

    def test_token_scripts_are_not_in_the_denominator(self):
        """They are not in the printing order, so they cannot be held out."""
        files, printings = self._corpus(100)
        held = newest_first_holdout(files, printings)
        assert all("tokenscripts" not in f for f in held.script_files)

    def test_an_empty_corpus_holds_nothing_out(self):
        assert newest_first_holdout({}, {}) == HeldOutCards(frozenset(), frozenset())

    def test_the_fraction_is_configurable(self):
        files, printings = self._corpus(100)
        assert len(newest_first_holdout(files, printings, fraction=0.20).names) == 20


class TestRecordNamesHeldOutCard:
    def _held(self) -> HeldOutCards:
        return HeldOutCards(
            names=frozenset({"Serra Angel"}),
            script_files=frozenset({"cardsfolder/s/serra_angel.txt"}),
        )

    def test_a_record_whose_context_holds_the_card_names_it(self):
        record = _record("g1", entities=(_entity("Serra Angel"),))
        assert record_names_held_out_card(record, self._held())

    def test_a_record_whose_acting_ability_is_the_cards_names_it(self):
        record = _record("g1", ability=(
            ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0),
        ))
        assert record_names_held_out_card(record, self._held())

    def test_a_record_carrying_the_cards_line_on_another_entity_names_it(self):
        """An anthem's granted line names the anthem, wherever it landed."""
        recipient = _entity("Grizzly Bears", granted_attached=(
            ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "static", 0),
        ))
        assert record_names_held_out_card(
            _record("g1", entities=(recipient,)), self._held()
        )

    def test_an_unrelated_record_does_not(self):
        record = _record("g1", entities=(_entity("Grizzly Bears"),))
        assert not record_names_held_out_card(record, self._held())


class TestDeriveSplit:
    def _records(self, games: int = 100):
        records = []
        for index in range(games):
            entities = (_entity("Serra Angel"),) if index < 10 else (
                _entity("Grizzly Bears"),
            )
            records.append(_record(f"g{index}", entities=entities))
            records.append(_record(
                f"g{index}", entities=(_entity("Grizzly Bears"),),
                record_id=f"g{index}.2", kind=RecordKind.COMBAT, moment=None,
                payload=CombatPayload(),
            ))
        return records

    def _held(self) -> HeldOutCards:
        return HeldOutCards(
            names=frozenset({"Serra Angel"}),
            script_files=frozenset({"cardsfolder/s/serra_angel.txt"}),
        )

    def test_a_whole_game_is_excluded_not_just_the_naming_record(self):
        split = derive_split(self._records(), self._held())
        # Game g0's second record names no held-out card, and is still out.
        assert "g0" in split.card_disjoint_games
        assert not split.is_training_game("g0")

    def test_every_game_holding_a_naming_record_is_excluded(self):
        split = derive_split(self._records(), self._held())
        assert split.card_disjoint_games == frozenset(f"g{i}" for i in range(10))

    def test_game_disjoint_validation_takes_a_tenth_of_the_rest(self):
        split = derive_split(self._records(), self._held())
        assert len(split.game_disjoint_games) == int(90 * GAME_HOLDOUT_FRACTION)

    def test_the_two_strata_do_not_overlap(self):
        split = derive_split(self._records(), self._held())
        assert not (split.card_disjoint_games & split.game_disjoint_games)

    def test_training_games_are_what_neither_stratum_took(self):
        split = derive_split(self._records(), self._held())
        training = {
            f"g{i}" for i in range(100) if split.is_training_game(f"g{i}")
        }
        assert len(training) == 100 - 10 - 9

    def test_the_split_is_reproducible_from_the_same_corpus(self):
        records = self._records()
        assert derive_split(records, self._held()) == derive_split(
            records, self._held()
        )

    def test_a_different_seed_moves_the_game_disjoint_stratum(self):
        records = self._records()
        first = derive_split(records, self._held(), seed=1)
        second = derive_split(records, self._held(), seed=2)
        assert first.card_disjoint_games == second.card_disjoint_games
        assert first.game_disjoint_games != second.game_disjoint_games

    def test_the_held_out_card_list_is_recorded_sorted(self):
        split = derive_split(self._records(), self._held())
        assert split.held_out_cards == ("Serra Angel",)

    def test_validation_games_are_both_strata(self):
        split = derive_split(self._records(), self._held())
        assert split.validation_games == (
            split.card_disjoint_games | split.game_disjoint_games
        )

    def test_an_empty_corpus_yields_an_empty_split(self):
        split = derive_split([], self._held())
        assert split.card_disjoint_games == frozenset()
        assert split.game_disjoint_games == frozenset()

    def test_a_split_is_immutable(self):
        split = CorpusSplit(("A",), frozenset({"g1"}), frozenset({"g2"}))
        assert split.is_training_game("g3")
        assert not split.is_training_game("g1")


class TestCorpusLoaders:
    def test_card_files_are_keyed_by_the_name_line(self, tmp_path):
        (tmp_path / "s").mkdir()
        (tmp_path / "s" / "serra_angel.txt").write_text(
            "name: Serra Angel\ntypes: creature angel\n", encoding="utf-8",
        )
        assert load_card_files(tmp_path) == {
            "Serra Angel": "cardsfolder/s/serra_angel.txt",
        }

    def test_a_file_without_a_name_line_is_skipped(self, tmp_path):
        (tmp_path / "broken.txt").write_text("types: creature\n", encoding="utf-8")
        assert load_card_files(tmp_path) == {}

    def test_first_printings_take_the_earliest_release(self, tmp_path):
        path = tmp_path / "AllPrintings.json"
        path.write_text(json.dumps({"data": {
            "NEW": {"releaseDate": "2026-01-01", "cards": [{"name": "Shock"}]},
            "OLD": {"releaseDate": "1999-01-01", "cards": [{"name": "Shock"}]},
        }}), encoding="utf-8")
        assert load_first_printings(path)["Shock"] == "1999-01-01"

    def test_face_names_are_indexed_too(self, tmp_path):
        path = tmp_path / "AllPrintings.json"
        path.write_text(json.dumps({"data": {
            "SET": {"releaseDate": "2020-01-01", "cards": [
                {"name": "Front // Back", "faceName": "Front"},
            ]},
        }}), encoding="utf-8")
        printings = load_first_printings(path)
        assert printings["Front // Back"] == "2020-01-01"
        assert printings["Front"] == "2020-01-01"
