package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.simulation.GameCopier;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The fork fidelity audit: a copy compared with the game it came from on
 * everything a record's state says, before anything in the copy is changed.
 */
@ExtendWith(ForgeExtension.class)
class ForkFidelityTest {

    private static Game board() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        TestBoard.inPlay(live, you, "Grizzly Bears").addCounterInternal(
                CounterEnumType.P1P1, 2, you, false, null, java.util.Map.of());
        TestBoard.inPlay(live, you, "Forest").tap(true, null, null);
        TestBoard.inPlay(live, opponent, "Serra Angel");
        TestBoard.inHand(live, you, "Lightning Bolt");
        TestBoard.inGraveyard(live, opponent, "Giant Growth");
        live.getAction().checkStaticAbilities();
        return live;
    }

    @Test
    void aFaithfulCopyDiffersInNothing() {
        Game live = board();
        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * After combat: a creature attacked, another blocked it, damage was
     * dealt and a creature died. The copier copied none of the turn's combat
     * or zone history, so the copy's creatures had not attacked or blocked,
     * no damage linked them and no creature had died this turn.
     */
    @Test
    void aCopyTakenAfterCombatKeepsTheTurnsHistory() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        Card attacker = TestBoard.inPlay(live, you, "Hill Giant");
        Card blocker = TestBoard.inPlay(live, opponent, "Grizzly Bears");
        Card chump = TestBoard.inPlay(live, opponent, "Llanowar Elves");
        attacker.getDamageHistory().setCreatureAttackedThisCombat(opponent, 0);
        blocker.addBlockedThisTurn(attacker);
        attacker.addBlockedByThisTurn(blocker);
        attacker.getDamageHistory().registerDamage(1, true, attacker, blocker, new java.util.HashMap<>());
        live.getAction().moveTo(forge.game.zone.ZoneType.Graveyard, chump, null, null);
        assertEquals(1, live.getLeftBattlefieldThisTurn().size(), "the chump blocker did not leave");

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * The monarch is a command-zone effect card holding the designation's
     * triggers; the copy's effect card came out with no lines and no colour.
     */
    @Test
    void aCopyKeepsTheMonarchsEffectCard() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        live.getAction().becomeMonarch(live.getPlayers().get(0), null);

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * An effect card a resolved spell left in the command zone names that
     * spell's line through the ability that made it. The copier set an
     * effect's source only on battlefield cards, so the copy's effect card
     * named no line and nothing it was created by.
     */
    @Test
    void aCopyKeepsWhatMadeAnEffectCard() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN2);
        Player you = live.getPlayers().get(0);
        forge.game.spellability.SpellAbility spell =
                TestBoard.inGraveyard(live, you, "Giant Growth").getFirstSpellAbility();
        spell.setActivatingPlayer(you);
        Card effect = forge.game.ability.SpellAbilityEffect.createEffect(
                spell, you, "Giant Growth's Effect", null);
        live.getAction().moveTo(forge.game.zone.ZoneType.Command, effect, spell, null);

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * Sewer Plague's perpetual -2/-2 and upkeep trigger on a creature since
     * returned to its owner's hand: the copier rebuilt the card in hand from
     * its rules, so the copy's pt_boosts and lines lacked them.
     */
    @Test
    void aCopyKeepsAPerpetualChangeOffTheBattlefield() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        Card bears = TestBoard.inPlay(live, opponent, "Grizzly Bears");
        TestBoard.cast(live, TestBoard.inHand(live, you, "Sewer Plague"), you, bears);
        live.getStack().resolveStack();
        live.getAction().moveTo(forge.game.zone.ZoneType.Hand, live.getCardState(bears), null, null);

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * A player at max speed: the speed effect card shows its back face, Max
     * Speed!, its trigger on the front. The copy showed its front, with no
     * trigger.
     */
    @Test
    void aCopyKeepsTheSpeedEffectCardsFace() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        for (int i = 0; i < 4; i++) {
            live.getPlayers().get(0).increaseSpeed();
        }

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /** A player's keyword from a static (Leyline of Sanctity's hexproof) and its Keyword Effects card. */
    @Test
    void aCopyKeepsAPlayersKeywordFromAStatic() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(live, live.getPlayers().get(0), "Leyline of Sanctity");
        live.getAction().checkStateEffects(true);

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /**
     * A player's keyword from a resolved spell (Veil of Summer's hexproof
     * from blue and black, until end of turn) and its Keyword Effects card.
     */
    @Test
    void aCopyKeepsAPlayersKeywordFromAResolvedSpell() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Card veil = TestBoard.inHand(live, you, "Veil of Summer");
        Card onStack = live.getAction().moveToStack(veil, null);
        TestBoard.castAndResolve(live, TestBoard.spellOf(onStack, you));
        assertTrue(you.getKeywords().toString().contains("Hexproof"), you.getKeywords().toString());

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
        assertEquals(you.getKeywords().toString(), fork.getPlayer(you.getId()).getKeywords().toString());
    }

    /**
     * Spell Queller exiled a spell from the stack and remembers it as the
     * stack's card, which names nothing that exiled it: the live snapshot
     * wrote the one link twice, keyed to Spell Queller's line and plain, the
     * copy's, holding the exiled card, once.
     */
    @Test
    void aCopyKeepsWhatSpellQuellerExiled() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = live.getPlayers().get(0);
        Player opponent = live.getPlayers().get(1);
        forge.game.spellability.SpellAbility shock =
                TestBoard.cast(live, TestBoard.inHand(live, opponent, "Shock"), opponent, you);
        Card queller = TestBoard.inPlay(live, you, "Spell Queller");
        forge.game.spellability.SpellAbility exile = queller.getTriggers().iterator().next().ensureAbility();
        exile.setActivatingPlayer(you);
        exile.resetTargets();
        exile.getTargets().add(shock);
        live.getStack().add(exile);
        live.getStack().resolveStack();
        live.getAction().checkStateEffects(true);
        assertEquals(1, live.getCardsIn(forge.game.zone.ZoneType.Exile).size(), "Shock was not exiled");

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    /** "Target creature blocks this turn if able": the requirement survives the copy. */
    @Test
    void aCopyKeepsACreaturesRequirementToBlock() {
        Game live = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Card attacker = TestBoard.inPlay(live, live.getPlayers().get(0), "Hill Giant");
        Card blocker = TestBoard.inPlay(live, live.getPlayers().get(1), "Grizzly Bears");
        blocker.addMustBlockCard(live.getNextTimestamp(), attacker);

        Game fork = new GameCopier(live).makeCopy();

        assertEquals(List.of(), ForkFidelity.differences(live, fork));
    }

    @Test
    void aCopyThatDiffersIsNamedByEntityAndField() {
        Game live = board();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Card angel = live.getCardsIn(forge.game.zone.ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Serra Angel")).findFirst().orElseThrow();
        copier.find(angel).tap(true, null, null);

        List<String> differences = ForkFidelity.differences(live, fork);

        assertTrue(differences.stream().anyMatch(line -> line.startsWith(
                "entity " + Refs.of(angel) + " (Serra Angel).tapped: live=false fork=true")),
                differences.toString());
    }

    @Test
    void aPlayerThatDiffersIsNamedByField() {
        Game live = board();
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();
        Player you = live.getPlayers().get(0);
        ((Player) copier.find(you)).setLife(13, null);

        List<String> differences = ForkFidelity.differences(live, fork);

        assertTrue(differences.contains("player " + Refs.of(you) + ".life: live=20 fork=13"),
                differences.toString());
    }

    // ── the copier's own report and mapper ──────────────────────────────

    /**
     * The fidelity line names the kinds of reference the copier could not
     * map, with their counts, and only those: a faithful copy that dropped
     * nothing reports nothing.
     */
    @Test
    void theFidelityLineNamesWhatTheCopierDropped() {
        Game live = board();
        java.util.Map<GameCopier.NoCounterpart, Integer> tally =
                new java.util.EnumMap<>(GameCopier.NoCounterpart.class);
        tally.put(GameCopier.NoCounterpart.LINK, 2);
        tally.put(GameCopier.NoCounterpart.COMMAND, 0);

        String line = ForkFidelity.line(live, 0, tally);

        assertTrue(line != null && line.endsWith(", no counterpart {LINK=2}"), line);
        assertEquals(null, ForkFidelity.line(live, 0, java.util.Map.of()));
    }

    /** What the copier left out is listed after the differences, one line per example, kind first. */
    @Test
    void theReportListsExamplesOfWhatTheCopierLeftOut() {
        java.util.Map<GameCopier.NoCounterpart, List<String>> examples =
                new java.util.EnumMap<>(GameCopier.NoCounterpart.class);
        examples.put(GameCopier.NoCounterpart.COMMAND, List.of("PumpEffect$1: no counterpart for Token (7)"));
        examples.put(GameCopier.NoCounterpart.ACTIVATION, List.of("Bike (3) crew", "Bike (3) saddle"));
        examples.put(GameCopier.NoCounterpart.LINK, List.of());

        assertEquals(List.of(
                "no counterpart ACTIVATION: Bike (3) crew",
                "no counterpart ACTIVATION: Bike (3) saddle",
                "no counterpart COMMAND: PumpEffect$1: no counterpart for Token (7)"),
                ForkFidelity.exampleLines(examples));
        assertEquals(List.of(), ForkFidelity.exampleLines(null));
    }

    /** A live ability maps into the copy through the copier's own mapper, onto the copy's card. */
    @Test
    void aLiveAbilityMapsOntoItsCopysCard() {
        Game live = board();
        Card bolt = live.getPlayers().get(0).getCardsIn(forge.game.zone.ZoneType.Hand).get(0);
        GameCopier copier = new GameCopier(live);
        Game fork = copier.makeCopy();

        forge.game.spellability.SpellAbility mapped =
                Forks.mappedAbility(copier, bolt.getFirstSpellAbility());

        assertTrue(mapped != null, "the Bolt's spell has a counterpart");
        assertTrue(mapped.getHostCard().getGame() == fork, "on the copy's card");
        assertEquals(bolt.getId(), mapped.getHostCard().getId());
        assertEquals(bolt.getFirstSpellAbility().getDescription(), mapped.getDescription());
    }
}
