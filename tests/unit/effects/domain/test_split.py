"""The card holdout and what counts as naming a held-out card.

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

from effects.domain.provenance import ProvenanceKey
from effects.domain.split import (
    HeldOutCards,
    HoldoutRule,
    record_names_held_out_card,
)
from tests.unit.effects.record_builders import (
    held_out_cards,
    lines_of,
    make_entity,
    make_record,
    make_snapshot,
    named_entity,
    record_on_board,
)


class TestRecordNamesHeldOutCard:
    def _held(self) -> HeldOutCards:
        return held_out_cards({"Serra Angel"}, {"cardsfolder/s/serra_angel.txt"})

    def test_a_record_whose_context_holds_the_card_names_it(self):
        record = record_on_board("g1", entities=(named_entity("Serra Angel"),))
        assert record_names_held_out_card(record, self._held())

    def test_a_record_whose_acting_ability_is_the_cards_names_it(self):
        record = record_on_board("g1", ability=(
            ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0),
        ))
        assert record_names_held_out_card(record, self._held())

    def test_a_record_carrying_the_cards_line_on_another_entity_names_it(self):
        """An anthem's granted line names the anthem, wherever it landed."""
        recipient = named_entity("Grizzly Bears", lines=lines_of(granted=(
            ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "static", 0),
        )))
        assert record_names_held_out_card(
            record_on_board("g1", entities=(recipient,)), self._held()
        )

    def test_an_unrelated_record_does_not(self):
        record = record_on_board("g1", entities=(named_entity("Grizzly Bears"),))
        assert not record_names_held_out_card(record, self._held())


def test_functional_reprints_are_held_out_together(_ignored=None) -> None:
    """Searing Spear and Lightning Strike compile to the same script.

    A name-keyed holdout puts one in training and the other in the holdout, and
    gate 1 then drops those records for having text a training card carries. The
    two must agree at every threshold, not merely at the extremes.
    """
    from effects.domain.split import text_keyed_holdout

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
            card_files, texts_by_card, HoldoutRule(permille, 8),
        )
        assert ("Searing Spear" in held.names) == ("Lightning Strike" in held.names), (
            f"reprints disagreed at permille={permille}"
        )


def test_the_holdout_carries_its_texts_for_sizing_the_stratum() -> None:
    """The guard needs the texts, not just the cards."""
    from effects.domain.split import text_keyed_holdout

    held = text_keyed_holdout(
        {"Alpha": "cardsfolder/a/alpha.txt"},
        {"Alpha": ["SP$ DealDamage | NumDmg$ 3"]},
        HoldoutRule(1000, 8),
    )

    assert held.texts == frozenset({"SP$ DealDamage | NumDmg$ 3"})


class TestTheConvertedToRuntimeNameBoundary:
    """Card names change case crossing from the converted tree to Forge.

    `ConvertedTrees.card_files` reads names out of converted card text, which is
    lowercased (`name: soul echo`). Forge's own `getName()` and every record's
    `EntityState.name` are printed case (`Soul Echo`). Matching one against the
    other silently never fires, which is how a depleted collection run came back
    holding 22% of its games tainted while reporting nothing wrong.

    Every test in this file that builds both sides from the same literal misses
    this, which is why it needs its own.
    """

    def test_a_printed_case_record_matches_a_converted_case_holdout(self) -> None:
        held = held_out_cards({"soul echo"})
        record = record_on_board("g1", entities=(named_entity("Soul Echo"),))

        assert record_names_held_out_card(record, held) is True

    def test_a_card_outside_the_holdout_still_does_not_match(self) -> None:
        held = held_out_cards({"soul echo"})
        record = record_on_board("g1", entities=(named_entity("Grizzly Bears"),))

        assert record_names_held_out_card(record, held) is False

    def test_accented_names_fold_the_same_way_java_does(self) -> None:
        """`toLowerCase(Locale.ROOT)` on the Java side; these must agree."""
        held = held_out_cards({"lim-dûl's vault"})
        record = record_on_board("g1", entities=(named_entity("Lim-Dûl's Vault"),))

        assert record_names_held_out_card(record, held) is True


class TestTokenScriptsCarryHeldOutTexts:
    """A token script is a carrier like a card.

    Counted out, a token printing a held-out text trains on it: no holdout
    test ever names its script, so its records and every game it appears in
    reach the training stratum.
    """

    _TEXT = "SP$ DealDamage | NumDmg$ 3"

    def test_a_token_carrying_a_held_out_text_is_held_out_by_script_file(self):
        from effects.domain.split import text_keyed_holdout

        held = text_keyed_holdout(
            {"Alpha": "cardsfolder/a/alpha.txt"},
            {"Alpha": [self._TEXT]},
            HoldoutRule(1000, 8),
            token_texts={"tokenscripts/r_1_1_goblin_shock.txt": [self._TEXT]},
        )
        assert "tokenscripts/r_1_1_goblin_shock.txt" in held.script_files
        assert held.names == frozenset({"alpha"})

    def test_a_token_counts_toward_the_carrier_ceiling(self):
        from effects.domain.split import text_keyed_holdout

        held = text_keyed_holdout(
            {"Alpha": "cardsfolder/a/alpha.txt"},
            {"Alpha": [self._TEXT]},
            HoldoutRule(1000, 1),
            token_texts={"tokenscripts/goblin.txt": [self._TEXT]},
        )
        assert not held.texts

    def test_a_text_only_a_token_carries_can_be_held_out(self):
        from effects.domain.split import text_keyed_holdout

        held = text_keyed_holdout(
            {"Alpha": "cardsfolder/a/alpha.txt"}, {"Alpha": ["SP$ Draw"]},
            HoldoutRule(1000, 8),
            token_texts={"tokenscripts/goblin.txt": [self._TEXT]},
        )
        assert self._TEXT in held.texts
        assert held.script_files == frozenset({
            "cardsfolder/a/alpha.txt", "tokenscripts/goblin.txt",
        })


class TestHeldOutTextMatching:
    """Whether a text is held out, under the unit the holdout was chosen by."""

    def test_a_template_unit_holds_out_every_text_sharing_the_template(self):
        from effects.domain.split import text_keyed_holdout

        held = text_keyed_holdout(
            {"Alpha": "cardsfolder/a/alpha.txt"},
            {"Alpha": ["SP$ DealDamage | NumDmg$ 3"]},
            HoldoutRule(1000, 8),
        )
        assert held.holds_text("SP$ DealDamage | NumDmg$ 7")
        assert not held.holds_text("SP$ Draw | NumCards$ 1")


def test_a_held_out_token_on_the_board_names_its_game_held_out():
    from effects.domain.provenance import ProvenanceKey

    token = ProvenanceKey("tokenscripts/goblin.txt", 0, "spell", 1)
    record = make_record(state=make_snapshot(entities=(
        make_entity("E1", "Goblin Token", lines=lines_of(printed=(token,))),
    )))
    held = held_out_cards(script_files={"tokenscripts/goblin.txt"})
    assert record_names_held_out_card(record, held)


class TestVariantNamingHeldOutText:
    """A variant script is named by its keys, an entity's lines included."""

    _VARIANT = ProvenanceKey("variant-scripts/bolt_variant_ab12.txt", 0, "spell", 0)

    def _on_board(self) -> object:
        entity = named_entity("Bolt Variant ab12", lines=lines_of(printed=(self._VARIANT,)))
        return record_on_board("g1", entities=(entity,))

    def test_a_variant_printing_a_held_out_text_on_the_board_names_the_game(self):
        prints = {self._VARIANT.script_file}.__contains__
        assert record_names_held_out_card(self._on_board(), held_out_cards(), prints)

    def test_a_variant_printing_no_held_out_text_does_not(self):
        assert not record_names_held_out_card(
            self._on_board(), held_out_cards(), lambda script_file: False)

    def test_without_the_variant_check_the_record_names_nothing(self):
        assert not record_names_held_out_card(self._on_board(), held_out_cards())

    def test_only_variant_tree_keys_reach_the_check(self):
        asked: list[str] = []

        def prints(script_file: str) -> bool:
            asked.append(script_file)
            return False

        record = record_on_board("g1", ability=(
            ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0), self._VARIANT))
        record_names_held_out_card(record, held_out_cards(), prints)
        assert asked == [self._VARIANT.script_file]


def test_a_game_holding_no_variant_is_placed_as_before_over_the_gen3_fixture():
    """The fixture holds no variant key, so the variant check is never asked and
    every record answers as it did without it."""
    from effects.domain.records import key_sites
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    held = held_out_cards({"Grizzly Bears", "Lightning Bolt"})
    records = list(read_shard(GEN3_RECORDS))
    assert not [k for r in records for k in key_sites(r) if k.tree == "variant-scripts"]

    def never(_script_file: str) -> bool:
        raise AssertionError("no variant to ask about")

    for record in records:
        assert record_names_held_out_card(record, held, never) == (
            record_names_held_out_card(record, held))
