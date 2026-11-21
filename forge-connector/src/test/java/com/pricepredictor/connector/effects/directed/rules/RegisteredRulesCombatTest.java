package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.ForgeExtension;
import com.pricepredictor.connector.effects.TestBoard;
import com.pricepredictor.connector.effects.directed.Binding;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.combat.Combat;
import forge.game.combat.CombatUtil;
import forge.game.keyword.Keyword;
import forge.game.phase.PhaseType;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static com.pricepredictor.connector.effects.directed.rules.CardName.named;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The registered directed rules of combat: simultaneous combat damage
 * (510.2), trample over planeswalkers and a planeswalker gone from combat
 * (702.19c, 702.19f), unleash's can't block (702.98a) and skulk
 * (702.118b). Each rule's scenario base binds, a board missing it binds
 * nothing, and the fork differs from its twin by the variant's one fact,
 * as {@link RegisteredRulesTest} sets out.
 */
@ExtendWith(ForgeExtension.class)
class RegisteredRulesCombatTest {

    // ── 510.2 (703.4m), simultaneous combat damage: replace_card ────────

    /** A's Grizzly Bears attacks, B's {@code blocker} blocks it; the first-strike damage step begins. */
    static RuleBoard blockedBoard(CardName blocker) {
        RuleBoard board = new RuleBoard(PhaseType.COMBAT_FIRST_STRIKE_DAMAGE);
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = board.card(board.opponent, blocker.name());
        TestBoard.blockedCombat(new TestBoard.CombatSides(board.game, board.you, board.opponent), board.other, board.subject);
        return board;
    }

    @Test
    void simultaneousDamageBindsTheBlockedAttackerAndItsBlocker() {
        RuleBoard board = blockedBoard(named("Grizzly Bears"));
        Binding binding = SimultaneousDamage.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void simultaneousDamageBindsNothingWhenTheBlockerStrikesFirst() {
        assertTrue(SimultaneousDamage.bind(blockedBoard(named(SimultaneousDamage.KNIGHT)).game).isEmpty());
    }

    @Test
    void simultaneousDamageForksAFirstStrikeBlockerInTheBlockersPlace() {
        RuleBoard board = blockedBoard(named("Grizzly Bears"));
        RuleTwin twin = board.twin(SimultaneousDamage.rule());
        Game fork = twin.forkGame();
        Card knight = fork.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(SimultaneousDamage.KNIGHT)).findFirst().orElseThrow();
        // The live board with the knight's id wherever the blocker's stood:
        // what is left differs only in what makes the card that card, its
        // first strike included (granted_temporary lists every keyword).
        Set<String> left = changes(renamed(twin.real(), id(board.subject), id(knight)), twin.forked());
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
        for (String change : left) {
            assertTrue(change.startsWith(id(knight) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        assertTrue(fork.getCombat().isBlocking(knight, twin.fork().card("attacker")));
        assertTrue(knight.hasFirstStrike());
    }

    // ── 702.98a, unleash can't block: counters ──────────────────────────

    /** Grizzly Bears attacks B, who controls {@code blocker} with {@code counters} +1/+1 counters; no blocker declared yet. */
    static RuleBoard unleashBoard(CardName blocker, int counters) {
        RuleBoard board = new RuleBoard(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Grizzly Bears");
        board.subject = board.card(board.opponent, blocker.name());
        board.subject.setCounters(CounterEnumType.P1P1, counters);
        board.attacking(board.other);
        return board;
    }

    @Test
    void unleashBindsTheAttackerAndAnUnleashedCreatureWithACounter() {
        RuleBoard board = unleashBoard(named("Rakdos Cackler"), 1);
        Binding binding = UnleashCantBlock.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void unleashBindsNothingWithoutTheCounterOrTheKeyword() {
        assertTrue(UnleashCantBlock.bind(unleashBoard(named("Rakdos Cackler"), 0).game).isEmpty());
        assertTrue(UnleashCantBlock.bind(unleashBoard(named("Runeclaw Bear"), 1).game).isEmpty());
    }

    @Test
    void unleashForksTheCreatureWithoutItsCounter() {
        RuleBoard board = unleashBoard(named("Rakdos Cackler"), 1);
        RuleTwin twin = board.twin(UnleashCantBlock.rule());
        String cackler = id(board.subject);
        assertTrue(twin.changes().contains(cackler + ".counters"), twin.changes().toString());
        assertTrue(Set.of(cackler + ".counters", cackler + ".pt_boosts").containsAll(twin.changes()),
                twin.changes().toString());
        Card forked = twin.fork().card("blocker");
        assertEquals(0, forked.getCounters(CounterEnumType.P1P1));
        assertFalse(CombatUtil.canBlock(board.subject), "the live creature may not block");
        assertTrue(CombatUtil.canBlock(forked), "with no counter it may");
    }

    // ── 702.19c, trample over planeswalkers: replace_card ───────────────

    /** Your {@code attacker} attacks B's Jace Beleren of {@code loyalty} unblocked; the damage step begins. */
    private static RuleBoard planeswalkerBoard(CardName attacker, int loyalty) {
        return planeswalkerBoard(attacker, loyalty, PhaseType.COMBAT_DAMAGE);
    }

    /** {@link #planeswalkerBoard(CardName, int)} as {@code step} begins. */
    private static RuleBoard planeswalkerBoard(CardName attacker, int loyalty, PhaseType step) {
        RuleBoard board = new RuleBoard(step);
        board.subject = board.card(board.you, attacker.name());
        board.other = board.card(board.opponent, "Jace Beleren");
        board.other.setCounters(CounterEnumType.LOYALTY, loyalty);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.subject, board.other);
        combat.setBlocked(board.subject, false);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void trampleOverPlaneswalkersBindsAnUnblockedAttackerWithoutTheKeyword() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        assertEquals(board.subject, TrampleOverPlaneswalkers.bind(board.game).orElseThrow().card("attacker"));
    }

    /** Thrasta's own attack is the side the counts hold already. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWithTheKeyword() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard(named(Cards.THRASTA), 3).game).isEmpty());
    }

    /** A granted trample over planeswalkers is the keyword already, and stays on the Thrasta in its place. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAGrantedKeyword() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        board.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(Boards.tramplesOverPlaneswalkers(board.subject));
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** Thrasta's 7 does not top a loyalty of 7: the fork's attack would have no excess. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenThrastaWouldHaveNoExcess() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard(named("Colossal Dreadmaw"), 7).game).isEmpty());
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard(named("Colossal Dreadmaw"), 6).game).isPresent());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingOnABlockedAttacker() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        Card bears = board.card(board.opponent, "Grizzly Bears");
        board.game.getCombat().addBlocker(board.subject, bears);
        board.game.getCombat().setBlocked(board.subject, true);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersBindsNothingAtAnotherBoundary() {
        assertTrue(TrampleOverPlaneswalkers.bind(planeswalkerBoard(named("Colossal Dreadmaw"), 3,
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    /** The Dreadmaw's 7 marked damage would kill a 7/7 Thrasta. */
    @Test
    void trampleOverPlaneswalkersBindsNothingWhenThrastaWouldDie() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        board.subject.setDamage(5);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isPresent());
        board.subject.setDamage(7);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    /** A first striker deals its damage in the first-strike step, not in the regular one the rule forks at. */
    @Test
    void trampleOverPlaneswalkersBindsNothingForAFirstStriker() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        board.subject.addChangedCardKeywords(List.of("First Strike"), null, false,
                board.game.getNextTimestamp(), null);
        assertTrue(TrampleOverPlaneswalkers.bind(board.game).isEmpty());
    }

    @Test
    void trampleOverPlaneswalkersForksThrastaAttackingTheSamePlaneswalker() {
        RuleBoard board = planeswalkerBoard(named("Colossal Dreadmaw"), 3);
        RuleTwin twin = board.twin(TrampleOverPlaneswalkers.rule());
        Card thrasta = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(Cards.THRASTA)).findFirst().orElseThrow();
        for (String change : changes(renamed(twin.real(), id(board.subject), id(thrasta)), twin.forked())) {
            assertTrue(change.startsWith(id(thrasta) + ".") || change.contains(id(thrasta)),
                    "replacing changed " + change);
        }
        assertTrue(Boards.tramplesOverPlaneswalkers(thrasta));
        Combat combat = twin.forkGame().getCombat();
        assertFalse(combat.isBlocked(thrasta));
        assertEquals(combat.getDefenderByAttacker(thrasta).getId(), board.other.getId());
    }

    // ── 702.19f, a planeswalker gone from combat: replace_card ─────────

    /**
     * Your {@code attacker} attacks B's Jace Beleren unblocked, and Jace has
     * left the battlefield, so the attacker attacks Forge's placeholder
     * (506.4c), while your Grizzly Bears attacks B; {@code step} begins.
     */
    private static RuleBoard goneWalkerBoard(CardName attacker, PhaseType step) {
        RuleBoard board = planeswalkerBoard(attacker, 3, step);
        Card bears = board.card(board.you, "Grizzly Bears");
        board.game.getCombat().addAttacker(bears, board.opponent);
        board.game.getCombat().setBlocked(bears, false);
        board.game.getAction().moveToGraveyard(board.other, null);
        assertTrue(Combat.isRemovedDefender(board.game.getCombat().getDefenderByAttacker(board.subject)));
        return board;
    }

    private static RuleBoard goneWalkerBoard(CardName attacker) {
        return goneWalkerBoard(attacker, PhaseType.COMBAT_DAMAGE);
    }

    @Test
    void aGoneWalkersAttackerWithoutTheKeywordBindsTheNoDamageRule() {
        RuleBoard board = goneWalkerBoard(named("Craw Wurm"));
        assertEquals(board.subject, NoDamageToAGonePlaneswalker.bind(board.game).orElseThrow().card("attacker"));
    }

    /** Thrasta's own attack is the side 702.19f's counts hold already. */
    @Test
    void aGoneWalkersTramplerBindsNothing() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(goneWalkerBoard(named(Cards.THRASTA)).game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingWhileThePlaneswalkerStays() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(planeswalkerBoard(named("Craw Wurm"), 3).game).isEmpty());
    }

    /** Alone in the step, the attacker would deal no damage in the live game, which then writes no combat record. */
    @Test
    void theGoneWalkerRuleBindsNothingWithoutAnotherCombatantsDamage() {
        RuleBoard board = goneWalkerBoard(named("Craw Wurm"));
        Card bears = board.game.getCombat().getAttackers().stream()
                .filter(card -> card.getName().equals("Grizzly Bears")).findFirst().orElseThrow();
        board.game.getCombat().removeFromCombat(bears);
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingAtAnotherBoundary() {
        assertTrue(NoDamageToAGonePlaneswalker.bind(goneWalkerBoard(named("Craw Wurm"),
                PhaseType.COMBAT_DECLARE_BLOCKERS).game).isEmpty());
    }

    @Test
    void theGoneWalkerRuleBindsNothingForAGrantedKeyword() {
        RuleBoard granted = goneWalkerBoard(named("Craw Wurm"));
        granted.subject.addChangedCardKeywords(List.of("Trample:Planeswalker"), null, false,
                granted.game.getNextTimestamp(), null);
        assertTrue(NoDamageToAGonePlaneswalker.bind(granted.game).isEmpty());
    }

    /** Ghalta's 8 marked damage would kill a 7/7 Thrasta. */
    @Test
    void aGoneWalkersAttackerBindsNothingWhenThrastaWouldDie() {
        RuleBoard board = goneWalkerBoard(named("Ghalta, Primal Hunger"));
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isPresent());
        board.subject.setDamage(8);
        assertTrue(NoDamageToAGonePlaneswalker.bind(board.game).isEmpty());
    }

    /** The fork's {@code name} attacks the copy's own placeholder; nothing else changed but the new card. */
    private static void assertReplacedAttackingTheGoneWalker(RuleBoard board, RuleTwin twin, CardName name) {
        Card replacement = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(name.name())).findFirst().orElseThrow();
        Combat combat = twin.forkGame().getCombat();
        assertTrue(combat.isAttacking(replacement));
        assertTrue(Combat.isRemovedDefender(combat.getDefenderByAttacker(replacement)));
        assertEquals(board.opponent.getId(), combat.getDefenderPlayerByAttacker(replacement).getId());
        for (String change : changes(renamed(twin.real(), id(board.subject), id(replacement)), twin.forked())) {
            assertTrue(change.startsWith(id(replacement) + ".") || change.contains(id(replacement)),
                    "replacing changed " + change);
        }
    }

    @Test
    void aGoneWalkersAttackerForksThrasta() {
        RuleBoard board = goneWalkerBoard(named("Craw Wurm"));
        assertReplacedAttackingTheGoneWalker(board, board.twin(NoDamageToAGonePlaneswalker.rule()),
                named(Cards.THRASTA));
    }

    // ── 702.118b, skulk: replace_card ───────────────────────────────────

    /** Furtive Homunculus attacks B, who controls {@code blocker}; no blocker declared yet. */
    private static RuleBoard skulkBoard(CardName blocker) {
        return skulkBoard(blocker, PhaseType.COMBAT_DECLARE_BLOCKERS);
    }

    /** {@link #skulkBoard(CardName)} as {@code step} begins. */
    private static RuleBoard skulkBoard(CardName blocker, PhaseType step) {
        RuleBoard board = new RuleBoard(step);
        board.other = board.card(board.you, "Furtive Homunculus");
        board.subject = board.card(board.opponent, blocker.name());
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        return board;
    }

    @Test
    void skulkBlockerBindsTheAttackerAndTheCreatureWithGreaterPower() {
        RuleBoard board = skulkBoard(named("Centaur Courser"));
        Binding binding = SkulkBlocker.bind(board.game).orElseThrow();
        assertEquals(board.other, binding.card("attacker"));
        assertEquals(board.subject, binding.card("blocker"));
    }

    @Test
    void skulkBlockerBindsNothingWhenTheCreatureMayBlock() {
        assertTrue(SkulkBlocker.bind(skulkBoard(named("Grizzly Bears")).game).isEmpty());
    }

    /** A +1/+1 counter the Elves would carry leaves them at the attacker's power 2, not below it. */
    @Test
    void skulkBlockerBindsNothingWhenTheElvesWouldNotStayBelowTheAttacker() {
        RuleBoard board = skulkBoard(named("Centaur Courser"));
        board.subject.setCounters(CounterEnumType.P1P1, 1);
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    @Test
    void skulkBlockerBindsNothingAtAnotherBoundary() {
        assertTrue(SkulkBlocker.bind(skulkBoard(named("Centaur Courser"), PhaseType.COMBAT_DAMAGE).game).isEmpty());
    }

    /** Phantom Warrior cannot be blocked at all, the Elves no more than the Courser: no skulk to fork away. */
    @Test
    void skulkBlockerBindsNothingWithoutSkulk() {
        RuleBoard board = new RuleBoard(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Phantom Warrior");
        board.subject = board.card(board.opponent, "Centaur Courser");
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, board.opponent);
        board.game.getPhaseHandler().setCombat(combat);
        assertFalse(CombatUtil.canBlock(board.other, board.subject));
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    /** An evasion besides skulk (flying), or any protection (from green, from creatures), would keep the Elves out too. */
    @Test
    void skulkBlockerBindsNothingForAnotherEvasionOrAProtection() {
        for (String keyword : List.of("Flying", "Protection from green", "Protection:Creature:creatures",
                "Protection from blue")) {
            RuleBoard board = skulkBoard(named("Centaur Courser"));
            board.other.addChangedCardKeywords(List.of(keyword), null, false, board.game.getNextTimestamp(), null);
            assertTrue(SkulkBlocker.bind(board.game).isEmpty(), keyword);
        }
    }

    /** Protection from green keeps Llanowar Elves out of the block as well, the case the list is for. */
    @Test
    void protectionFromGreenWouldKeepTheElvesOut() {
        RuleBoard board = skulkBoard(named("Centaur Courser"));
        board.other.addChangedCardKeywords(List.of("Protection from green"), null, false,
                board.game.getNextTimestamp(), null);
        Card elves = board.card(board.opponent, SkulkBlocker.ELVES);
        assertFalse(CombatUtil.canBlock(board.other, elves));
        assertTrue(SkulkBlocker.otherEvasions().contains(Keyword.PROTECTION));
    }

    @Test
    void skulkBlockerBindsNothingForATappedCreatureOrOneThatCannotBlock() {
        RuleBoard tapped = skulkBoard(named("Centaur Courser"));
        tapped.subject.setTapped(true);
        assertTrue(SkulkBlocker.bind(tapped.game).isEmpty(), "tapped");
        assertTrue(SkulkBlocker.bind(skulkBoard(named("Craven Giant")).game).isEmpty(), "can't block");
    }

    /** One marked damage would kill the 1/1 Elves. */
    @Test
    void skulkBlockerBindsNothingWhenTheElvesWouldDie() {
        RuleBoard board = skulkBoard(named("Centaur Courser"));
        board.subject.setDamage(1);
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    /** Attacking a battle no player protects, the attacker has no defending player to block it. */
    @Test
    void skulkBlockerBindsNothingWithoutADefendingPlayer() {
        RuleBoard board = new RuleBoard(PhaseType.COMBAT_DECLARE_BLOCKERS);
        board.other = board.card(board.you, "Furtive Homunculus");
        board.subject = board.card(board.opponent, "Centaur Courser");
        Card battle = board.card(board.opponent, "Invasion of Amonkhet");
        battle.setProtectingPlayer(board.opponent);
        Combat combat = new Combat(board.you);
        combat.addAttacker(board.other, battle);
        board.game.getPhaseHandler().setCombat(combat);
        battle.setProtectingPlayer(null);
        assertNull(combat.getDefenderPlayerByAttacker(board.other));
        assertTrue(SkulkBlocker.bind(board.game).isEmpty());
    }

    @Test
    void skulkBlockerForksLlanowarElvesAmongTheLegalBlockers() {
        RuleBoard board = skulkBoard(named("Centaur Courser"));
        RuleTwin twin = board.twin(SkulkBlocker.rule());
        Card elves = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals(SkulkBlocker.ELVES)).findFirst().orElseThrow();
        // The Elves' mana ability adds to what B can make untapped.
        Set<String> identity = Set.of("name", "colors", "subtypes", "mana_value", "mana_cost",
                "base_pt", "lines", "granted_temporary");
        for (String change : changes(renamed(twin.real(), id(board.subject), id(elves)), twin.forked())) {
            assertTrue(change.equals("P1.untapped_production") || change.startsWith(id(elves) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change);
        }
        assertTrue(CombatUtil.canBlock(twin.fork().card("attacker"), elves));
        assertFalse(CombatUtil.canBlock(board.other, board.subject), "the live Courser still cannot block");
    }
}
