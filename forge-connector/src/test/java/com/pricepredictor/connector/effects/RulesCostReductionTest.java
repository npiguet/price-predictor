package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The reductions a spell's keywords allow (convoke, improvise, waterbend,
 * delve, emerge) are the rules', not the AI's choice of what to tap or
 * sacrifice (root spec § 8.8, findings 8.1 and 8.2): {@code affordable} asks
 * whether the cost can be paid at all, choosing the payment freely, and
 * {@code cost_after_adjustment} reads the most the board lets the keywords
 * pay. Reading a candidate changes nothing on the game.
 */
@ExtendWith(ForgeExtension.class)
class RulesCostReductionTest {

    private final Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    private final Player you = game.getPlayers().get(0);

    private void inPlay(Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            TestBoard.inPlay(game, owner, name);
        }
    }

    private static DecisionCandidate read(SpellAbility candidate, Player activator) {
        return DecisionCandidate.read(candidate, activator).orElseThrow();
    }

    private SpellAbility inHand(String name) {
        return TestBoard.inHand(game, you, name).getFirstSpellAbility();
    }

    private void assertReads(String mana, boolean affordable, SpellAbility candidate, String why) {
        DecisionCandidate read = read(candidate, you);
        assertEquals(mana, read.manaCost(), why);
        assertEquals(affordable, read.verdict().affordable(), why);
        assertEquals(!affordable, read.verdict().failed(PlayabilityCheck.COST), why);
    }

    // ── convoke ─────────────────────────────────────────────────────────

    /** A's Stoke the Flames ({2}{R}{R}, convoke) over a Mountain, a Raging Goblin and two Grizzly Bears in {@code phase}. */
    private static DecisionCandidate stokeOverThreeCreatures(PhaseType phase) {
        Game board = TestBoard.startedTwoPlayerGame(phase);
        Player a = board.getPlayers().get(0);
        TestBoard.inPlay(board, a, "Mountain");
        TestBoard.inPlay(board, a, "Raging Goblin");
        TestBoard.inPlay(board, a, "Grizzly Bears");
        TestBoard.inPlay(board, a, "Grizzly Bears");
        return read(TestBoard.inHand(board, a, "Stoke the Flames").getFirstSpellAbility(), a);
    }

    /**
     * The Goblin pays an {R}, the Bears the {2} and the Mountain the other
     * {R}, before combat as after it: the AI convokes nothing before its own
     * attack, and the rules do not care when it would.
     */
    @Test
    void convokeReadsTheSameBeforeAndAfterCombat() {
        for (PhaseType phase : List.of(PhaseType.MAIN1, PhaseType.MAIN2)) {
            DecisionCandidate read = stokeOverThreeCreatures(phase);
            assertEquals("{R}", read.manaCost(), phase.toString());
            assertTrue(read.verdict().affordable(), phase.toString());
            assertFalse(read.verdict().failed(PlayabilityCheck.COST), phase.toString());
        }
    }

    /** A green creature pays generic mana or green, never the red: three Bears and a Mountain leave an {R} unpaid. */
    @Test
    void aConvokingCreaturePaysOnlyItsOwnColour() {
        inPlay(you, "Mountain", 1);
        inPlay(you, "Grizzly Bears", 3);
        assertReads("{R}{R}", false, inHand("Stoke the Flames"), "three green creatures and one Mountain");
    }

    /**
     * Llanowar Elves either convokes or taps for mana, never both: with two
     * Mountains it is the third of four mana. The cost still reads the most
     * convoke could pay, the Elves' {1}.
     */
    @Test
    void aManaCreatureEitherConvokesOrMakesMana() {
        inPlay(you, "Mountain", 2);
        inPlay(you, "Llanowar Elves", 1);
        SpellAbility stoke = inHand("Stoke the Flames");
        assertReads("{1}{R}{R}", false, stoke, "two Mountains and the Elves make three of four");

        inPlay(you, "Grizzly Bears", 1);
        assertReads("{R}{R}", true, stoke, "the Bears and the Elves pay the {2}");
    }

    // ── improvise ───────────────────────────────────────────────────────

    /** Reverse Engineer ({3}{U}{U}): three Ornithopters pay the {3}, two Islands the {U}{U}, and two Forests do not. */
    @Test
    void improvisePaysTheGenericAndTheLandsTheColour() {
        inPlay(you, "Ornithopter", 3);
        inPlay(you, "Forest", 2);
        SpellAbility engineer = inHand("Reverse Engineer");
        assertReads("{U}{U}", false, engineer, "two Forests cannot pay {U}{U}");

        inPlay(you, "Island", 2);
        assertReads("{U}{U}", true, engineer, "two Islands can");
    }

    // ── waterbend ───────────────────────────────────────────────────────

    /**
     * Benevolent River Spirit ({U}{U}, waterbend {5}): each untapped artifact
     * or creature pays {1} of the five, and no more than five, so under
     * Sphere of Resistance the sixth generic is the Islands' to pay.
     */
    @Test
    void waterbendPaysAtMostItsOwnAmount() {
        inPlay(you, "Island", 2);
        inPlay(you, "Ornithopter", 6);
        SpellAbility spirit = inHand("Benevolent River Spirit");
        assertReads("{U}{U}", true, spirit, "six Ornithopters cover the waterbend {5}");

        inPlay(you, "Sphere of Resistance", 1);
        assertReads("{1}{U}{U}", false, spirit, "seven artifacts pay five; two Islands do not pay three");

        inPlay(you, "Island", 1);
        assertReads("{1}{U}{U}", true, spirit, "a third Island pays the {1}");
    }

    // ── delve ───────────────────────────────────────────────────────────

    /** Treasure Cruise ({7}{U}) delves the whole graveyard: three cards, and five Islands pay the {4}{U}. */
    @Test
    void delveCountsTheWholeGraveyard() {
        for (String name : List.of("Grizzly Bears", "Shock", "Forest")) {
            TestBoard.inGraveyard(game, you, name);
        }
        inPlay(you, "Island", 4);
        SpellAbility cruise = inHand("Treasure Cruise");
        assertReads("{4}{U}", false, cruise, "four Islands");

        inPlay(you, "Island", 1);
        assertReads("{4}{U}", true, cruise, "five Islands");
        assertEquals(3, you.getCardsIn(ZoneType.Graveyard).size(), "reading delved nothing");
    }

    // ── emerge ──────────────────────────────────────────────────────────

    private static SpellAbility emergeOf(Card card) {
        return card.getSpells().stream().filter(SpellAbility::isEmerge).findFirst().orElseThrow();
    }

    /**
     * Wretched Gryff (emerge {5}{U}) beside a Prodigal Sorcerer (mana value 3)
     * and a Grizzly Bears: the cost reads reduced by the larger, and reading
     * it sacrifices nothing: no creature is left marked as paying a cost, and
     * the Sorcerer's own ability still reads playable.
     */
    @Test
    void emergeTakesTheLargestSacrificeAndMarksNothing() {
        Card sorcerer = TestBoard.inPlay(game, you, "Prodigal Sorcerer");
        TestBoard.inPlay(game, you, "Grizzly Bears");
        inPlay(you, "Island", 3);
        SpellAbility emerge = emergeOf(TestBoard.inHand(game, you, "Wretched Gryff"));

        assertReads("{2}{U}", true, emerge, "the Sorcerer takes {3} off {5}{U}");
        read(emerge, you);

        for (Card card : you.getCardsIn(ZoneType.Battlefield)) {
            assertFalse(card.isUsedToPay(), card + " is left marked as paying a cost");
        }
        assertNull(emerge.getSacrificedAsEmerge(), "the emerge ability names a sacrifice");
        SpellAbility ping = TestBoard.ability(sorcerer, ApiType.DealDamage);
        assertTrue(PlayabilityVerdict.of(ping, you).canPlay(), "the Sorcerer's ability reads unplayable");
    }
}
