package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlyReplaced;
import static com.pricepredictor.connector.effects.directed.rules.RuleTwinChecks.assertOnlySwapped;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of Auras: a type-setting Aura (205.1a),
 * Aura swap (701.12e, 702.65a, 702.65b) and a bestowed Aura whose target
 * is gone (702.103e). Each rule's scenario base binds, a board missing it
 * binds nothing, and the fork differs from its twin by the variant's one
 * fact, as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesAurasTest {

    // ── 205.1a, a type-setting Aura: counters ───────────────────────────

    /** A's three Forests and Song of the Dryads in hand, B's {@code creature}, as A's first main phase begins. */
    private static RuleBoard songBoard(String creature, int forests) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < forests; i++) {
            board.card(board.you, "Forest");
        }
        board.subject = board.card(board.opponent, creature);
        board.other = TestBoard.inHand(board.game, board.you, "Song of the Dryads");
        return board;
    }

    @Test
    void typeSettingAuraBindsTheAuraAndTheCreature() {
        RuleBoard board = songBoard("Grizzly Bears", 3);
        Binding binding = TypeSettingAura.bind(board.game).orElseThrow();
        assertEquals(board.you, binding.player("caster"));
        assertEquals(board.other, binding.card("aura"));
        assertEquals(board.subject, binding.card("creature"));
    }

    @Test
    void typeSettingAuraBindsNothingWithoutTheLands() {
        assertTrue(TypeSettingAura.bind(songBoard("Grizzly Bears", 2).game).isEmpty());
    }

    /** A creature that already has a +1/+1 counter is no board the variant's one counter describes. */
    @Test
    void typeSettingAuraBindsNothingOnACreatureWithACounter() {
        RuleBoard board = songBoard("Grizzly Bears", 3);
        board.subject.setCounters(CounterEnumType.P1P1, 1);
        assertTrue(TypeSettingAura.bind(board.game).isEmpty());
    }

    /** The battlefield's one creature has hexproof: Song of the Dryads, cast by its opponent, cannot target it. */
    @Test
    void typeSettingAuraBindsNothingOnACreatureTheAuraCannotTarget() {
        assertTrue(TypeSettingAura.bind(songBoard("Gladecover Scout", 3).game).isEmpty());
    }

    /** Holy Strength adds to its creature's power and toughness and sets no card type. */
    @Test
    void typeSettingAuraBindsNothingForAnAuraThatSetsNoType() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        board.card(board.you, "Plains");
        board.card(board.opponent, "Grizzly Bears");
        TestBoard.inHand(board.game, board.you, Cards.HOLY_STRENGTH);
        assertTrue(TypeSettingAura.bind(board.game).isEmpty());
    }

    @Test
    void typeSettingAuraForksTheCreatureWithACounter() {
        RuleBoard board = songBoard("Grizzly Bears", 3);
        RuleTwin twin = board.twin(TypeSettingAura.rule());
        for (String change : twin.changes()) {
            assertTrue(change.startsWith(id(board.subject) + "."), "the edit changed " + change);
        }
        assertEquals(1, twin.fork().card("creature").getCounters(CounterEnumType.P1P1));
    }

    // ── 701.12e, Aura swap: move_card ───────────────────────────────────

    private static RuleBoard auraSwapBoard(String... hand) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        Card bears = board.card(board.you, "Grizzly Bears");
        board.other = board.card(board.you, "Arcanum Wings");
        board.other.attachToEntity(bears, null);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        for (String name : hand) {
            board.subject = TestBoard.inHand(board.game, board.you, name);
        }
        return board;
    }

    @Test
    void auraSwapBindsTheHandsCardAndTheAttachedAura() {
        RuleBoard board = auraSwapBoard("Giant Growth");
        Binding binding = AuraSwap.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapBindsNothingWithAnAuraInHand() {
        assertTrue(AuraSwap.bind(auraSwapBoard("Giant Growth", "Wild Growth").game).isEmpty());
    }

    @Test
    void auraSwapBindsNothingWithAnEmptyHand() {
        assertTrue(AuraSwap.bind(auraSwapBoard().game).isEmpty());
    }

    /** Holy Strength is white: Black Knight, with protection from white, could not take it. */
    @Test
    void auraSwapBindsNothingOnAHostWithProtection() {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        Card knight = board.card(board.you, "Black Knight");
        board.card(board.you, "Arcanum Wings").attachToEntity(knight, null);
        for (int i = 0; i < 3; i++) {
            board.card(board.you, "Island");
        }
        TestBoard.inHand(board.game, board.you, "Giant Growth");
        assertTrue(AuraSwap.bind(board.game).isEmpty());
    }

    @Test
    void auraSwapForksAHolyStrengthInTheCardsPlace() {
        RuleBoard board = auraSwapBoard("Giant Growth");
        RuleTwin twin = board.twin(AuraSwap.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 702.65a, Aura swap's exchange, registered with 701.12e: move_card ─

    /** 702.65a-1's base, Grizzly Bears in hand, is 701.12e's registration's board too. */
    @Test
    void auraSwapBindsTheExchangeScenariosHandCard() {
        RuleBoard board = auraSwapBoard("Grizzly Bears");
        Binding binding = AuraSwap.rule().precondition(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapBindsNothingWithTheAuraTheExchangeWaitsFor() {
        assertTrue(AuraSwap.rule().precondition(auraSwapBoard(Cards.HOLY_STRENGTH).game).isEmpty());
    }

    @Test
    void auraSwapForksAHolyStrengthInTheExchangeScenariosCardsPlace() {
        RuleBoard board = auraSwapBoard("Grizzly Bears");
        RuleTwin twin = board.twin(AuraSwap.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 702.65b, an Aura swap whose Aura cannot enchant: move_card ──────

    @Test
    void auraSwapCantEnchantBindsAnAuraInHandThatCannotEnchantTheHost() {
        RuleBoard board = auraSwapBoard("Wild Growth");
        Binding binding = AuraSwapCantEnchant.bind(board.game).orElseThrow();
        assertEquals(board.subject, binding.card("card"));
        assertEquals(board.other, binding.card("aura"));
    }

    @Test
    void auraSwapCantEnchantBindsNothingForAnAuraThatCanEnchant() {
        assertTrue(AuraSwapCantEnchant.bind(auraSwapBoard(Cards.HOLY_STRENGTH).game).isEmpty());
    }

    @Test
    void auraSwapCantEnchantForksHolyStrengthInTheAurasPlace() {
        RuleBoard board = auraSwapBoard("Wild Growth");
        RuleTwin twin = board.twin(AuraSwapCantEnchant.rule());
        List<Card> hand = List.copyOf(twin.forkGame().getPlayers().get(0).getCardsIn(ZoneType.Hand));
        assertEquals(List.of(Cards.HOLY_STRENGTH), hand.stream().map(Card::getName).toList());
        assertOnlySwapped(twin, board.subject, hand.get(0));
    }

    // ── 702.103e, a bestowed Aura whose target is gone: replace_card ────

    /** A's Hopeful Eidolon in hand over four Plains and its creature {@code creature}, the battlefield's one. */
    private static RuleBoard eidolonBoard(String creature) {
        RuleBoard board = new RuleBoard(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            board.card(board.you, "Plains");
        }
        board.subject = board.card(board.you, creature);
        TestBoard.inHand(board.game, board.you, "Hopeful Eidolon");
        return board;
    }

    @Test
    void bestowIllegalTargetBindsTheOneCreatureTheAuraCouldEnchant() {
        RuleBoard board = eidolonBoard("Grizzly Bears");
        assertEquals(board.subject, BestowIllegalTarget.bind(board.game).orElseThrow().card("creature"));
    }

    @Test
    void bestowIllegalTargetBindsNothingWithASecondCreatureToTarget() {
        RuleBoard board = eidolonBoard("Grizzly Bears");
        board.card(board.opponent, "Hill Giant");
        assertTrue(BestowIllegalTarget.bind(board.game).isEmpty());
    }

    /** One damage on the Bears would kill the 2/1 Ghost in their place at the fork's first check. */
    @Test
    void bestowIllegalTargetBindsNothingWhenTheGhostWouldNotSurvive() {
        RuleBoard board = eidolonBoard("Grizzly Bears");
        board.subject.setDamage(1);
        assertTrue(BestowIllegalTarget.bind(board.game).isEmpty());
    }

    @Test
    void bestowIllegalTargetForksASkulkingGhostInTheCreaturesPlace() {
        RuleBoard board = eidolonBoard("Grizzly Bears");
        assertOnlyReplaced(board.twin(BestowIllegalTarget.rule()), board.subject, BestowIllegalTarget.GHOST);
    }
}
