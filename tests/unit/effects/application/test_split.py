"""The card holdout and what counts as naming a held-out card (T040).

The holdout is the newest-printed cards rather than a random sample, because it
stands in for deployment to a set the model has never seen. A record names a
held-out card through its board entities or through the script files its
provenance keys point at, and either is enough to taint the game it belongs to.

Tainting the whole game rather than the naming record is the point. A held-out
card sitting in the *context* of a training record leaks through its context
role, which is exactly the loop that trains each embedding from both directions,
and gate 1 — the shipping gate — would then score partly on cards the encoder had
already seen. Which games that rule excludes, and which shards supply the
validation records, is covered in ``test_shard_streaming.py``.
"""

from __future__ import annotations

import json

from effects.application.train_effect_model import (
    HeldOutCards,
    load_card_files,
    load_first_printings,
    record_names_held_out_card,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
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


def test_functional_reprints_are_held_out_together(_ignored=None) -> None:
    """Searing Spear and Lightning Strike compile to the same script (T158).

    A name-keyed holdout puts one in training and the other in the holdout, and
    gate 1 then drops those records for having text a training card carries. The
    two must agree at every threshold, not merely at the extremes.
    """
    from effects.application.train_effect_model import text_keyed_holdout

    shared = "SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any"
    card_files = {
        "Searing Spear": "cardsfolder/s/searing_spear.txt",
        "Lightning Strike": "cardsfolder/l/lightning_strike.txt",
        "Other Thing": "cardsfolder/o/other_thing.txt",
    }
    texts_by_card = {
        "Searing Spear": [shared],
        "Lightning Strike": [shared],
        "Other Thing": ["SP$ Draw | NumCards$ 1"],
    }

    for permille in range(0, 1001, 50):
        held = text_keyed_holdout(
            card_files, texts_by_card, permille=permille, max_carriers=8,
        )
        assert ("Searing Spear" in held.names) == ("Lightning Strike" in held.names), (
            f"reprints disagreed at permille={permille}"
        )


def test_unique_text_records_counts_only_held_out_acting_text() -> None:
    """Gate 1's slice, countable from the holdout alone (T159).

    Under depletion every card carrying a held-out text is absent from the
    training pools, so "acting text on no training card" and "acting text is
    held out" are the same set — no corpus scan needed to size the stratum.
    """
    from effects.application.train_effect_model import (
        unique_text_resolution_records,
    )

    bespoke = ProvenanceKey("cardsfolder/b/bespoke.txt", 0, "spell", 0)
    common = ProvenanceKey("cardsfolder/c/common.txt", 0, "static", 0)
    texts = {bespoke: "SP$ Bespoke | Weird$ True", common: "Flying"}
    held = frozenset({"SP$ Bespoke | Weird$ True"})

    count = unique_text_resolution_records(
        [
            _record("g1", ability=(bespoke,)),
            _record("g1", ability=(common,)),
            _record("g2", ability=(bespoke,)),
            _record("g3", ability=None),
        ],
        held,
        text_of=texts.get,
    )

    assert count == 2


def test_the_holdout_carries_its_texts_for_sizing_the_stratum() -> None:
    """The guard needs the texts, not just the cards (T159)."""
    from effects.application.train_effect_model import text_keyed_holdout

    held = text_keyed_holdout(
        {"Alpha": "cardsfolder/a/alpha.txt"},
        {"Alpha": ["SP$ DealDamage | NumDmg$ 3"]},
        permille=1000, max_carriers=8,
    )

    assert held.texts == frozenset({"SP$ DealDamage | NumDmg$ 3"})


def test_a_corpus_shaped_holdout_carries_its_texts_too() -> None:
    """The ``--corpus`` path's holdout needs the same texts (task 7 fix round 1).

    A curated dataset's manifest is ``--corpus``'s only source for
    ``HeldOutCards`` — there is no sidecar scan to fall back on. Built without
    ``texts=``, ``unique_text_resolution_records`` would count against an
    empty set on every single ``--corpus`` run, so ``check_holdout`` would
    always report a zero-record card-disjoint stratum and print its "collect
    more full-strength games" warning even on a healthy dataset. Asserting the
    populated value (not just that it round-trips non-empty) is what would
    catch a manifest field that reached ``HeldOutCards`` empty.
    """
    from effects.application.train_effect_model import unique_text_resolution_records
    from tests.unit.effects.domain.test_corpus_manifest import manifest

    built = manifest(held_out_texts=("SP$ DealDamage | NumDmg$ 3",))
    held = HeldOutCards(
        names=frozenset(built.held_out_cards), script_files=frozenset(),
        texts=frozenset(built.held_out_texts),
    )

    assert held.texts == frozenset({"SP$ DealDamage | NumDmg$ 3"})

    # Closes the loop on the actual reported symptom: the stratum-sizing
    # guard now sees a non-zero count for a record whose acting text is held
    # out, rather than reading 0 on every run regardless of health.
    key = ProvenanceKey("cardsfolder/a/alpha.txt", 0, "spell", 0)
    count = unique_text_resolution_records(
        [_record("g1", ability=(key,))],
        held.texts,
        text_of={key: "SP$ DealDamage | NumDmg$ 3"}.get,
    )
    assert count == 1


class TestTheConvertedToRuntimeNameBoundary:
    """Card names change case crossing from the converted tree to Forge.

    `load_card_files` reads names out of converted card text, which is
    lowercased (`name: soul echo`). Forge's own `getName()` and every record's
    `EntityState.name` are printed case (`Soul Echo`). Matching one against the
    other silently never fires, which is how a depleted collection run came back
    holding 22% of its games tainted while reporting nothing wrong.

    Every test in this file that builds both sides from the same literal misses
    this, which is why it needs its own.
    """

    def test_a_printed_case_record_matches_a_converted_case_holdout(self) -> None:
        held = HeldOutCards(
            names=frozenset({"soul echo"}), script_files=frozenset(),
        )
        record = _record("g1", entities=(_entity("Soul Echo"),))

        assert record_names_held_out_card(record, held) is True

    def test_a_card_outside_the_holdout_still_does_not_match(self) -> None:
        held = HeldOutCards(
            names=frozenset({"soul echo"}), script_files=frozenset(),
        )
        record = _record("g1", entities=(_entity("Grizzly Bears"),))

        assert record_names_held_out_card(record, held) is False

    def test_accented_names_fold_the_same_way_java_does(self) -> None:
        """`toLowerCase(Locale.ROOT)` on the Java side; these must agree."""
        held = HeldOutCards(
            names=frozenset({"lim-dûl's vault"}), script_files=frozenset(),
        )
        record = _record("g1", entities=(_entity("Lim-Dûl's Vault"),))

        assert record_names_held_out_card(record, held) is True
