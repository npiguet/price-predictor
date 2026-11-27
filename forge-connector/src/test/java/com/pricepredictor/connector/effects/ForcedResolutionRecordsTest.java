package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static com.pricepredictor.connector.effects.ForkTestCaps.forkCaps;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static com.pricepredictor.connector.effects.RecordText.payloadOf;
import static com.pricepredictor.connector.effects.RecordText.stateOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a forced resolution's record holds: the board it acted on, read where
 * an observed resolution's is, and what live attributes to the forced line.
 */
@ExtendWith(ForgeExtension.class)
class ForcedResolutionRecordsTest {

    @TempDir
    Path tempDir;

    // ── a forced resolution's state is the board it acted on ────────────

    /**
     * A forced targeted spell's record names the target it resolved at, as
     * an observed resolution record does.
     *
     * <p>The fork draws its targets itself, so a state rendered before that
     * draw holds an ability with no target: Disfigure resolving at the one
     * creature on the board, recorded as aimed at nothing.
     */
    @Test
    void aForcedTargetedSpellsRecordNamesItsTarget() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        // In hand, where the fork's copier can find it.
        Card disfigure = TestBoard.inHand(game, caster, "Disfigure");
        SpellAbility spell = disfigure.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        ShardRun.Fork fork = ShardRun.forked(tempDir, new ShardRun.ForkRun(game, forkCaps().interventions(1).probes(0).build(), 1L, spell));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");

        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String target = "\"E" + bears.getId() + "\"";
        assertTrue(records.get(0).contains("\"refs\":{\"targets\":[" + target + "]"),
                records.get(0));
    }

    /**
     * A forced X spell's record carries the X it resolved with.
     *
     * <p>The fork announces X itself, from its own seeded source, between one
     * and what the caster's lands could pay, so the seed fixes the value:
     * Blaze with five Mountains, which could pay X up to four, and a seed that
     * draws neither bound.
     */
    @Test
    void aForcedXSpellsRecordCarriesItsX() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        for (int i = 0; i < 5; i++) {
            TestBoard.inPlay(game, caster, "Mountain");
        }
        Card blaze = TestBoard.inHand(game, caster, "Blaze");
        SpellAbility spell = blaze.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        ShardRun.Fork fork = ShardRun.forked(tempDir, new ShardRun.ForkRun(game, forkCaps().interventions(1).probes(0).build(), 9L, spell));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");

        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        assertTrue(records.get(0).contains("\"modes\":[],\"x\":2,"), records.get(0));
        assertTrue(records.get(0).contains("\"x_paid\":2"), records.get(0));
    }

    // ── a forced resolution's state is read where an observed one's is ──

    /**
     * A forced resolution's state is the board an observed resolution of
     * the same spell records: the spell on the stack, its card in the stack
     * zone rather than in hand, and priority with its caster. The two boards
     * are built alike and one casts Pyroclasm while the other forks and
     * forces it. Pyroclasm has no target, and the observed cast here pays
     * no cost, so nothing separates the two states.
     */
    @Test
    void aForcedResolutionsStateIsTheObservedResolutionsState() throws Throwable {
        Game observedGame = pyroclasmBoard();
        SpellAbility cast = firstSpellInHand(observedGame);
        List<String> observed = ShardRun.observed(tempDir.resolve("observed"), observedGame,
                bracket -> {
                    observedGame.getStack().addAndUnfreeze(cast);
                    observedGame.getStack().resolveStack();
                    observedGame.getAction().checkStateEffects(false);
                    TestBoard.nextStep(observedGame);
                });
        List<String> observedHalves = records(observed, "resolution");
        assertFalse(observedHalves.isEmpty(), observed.toString());
        String observedHalf = observedHalves.get(0);

        Game forkedGame = pyroclasmBoard();
        ShardRun.Fork fork = ShardRun.forked(tempDir.resolve("forked"), new ShardRun.ForkRun(forkedGame, forkCaps().interventions(1).probes(0).build(), 1L, firstSpellInHand(forkedGame)));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> forced = fork.records();
        assertEquals(1, forced.size(), forced.toString());

        assertTrue(stateOf(forced.get(0)).contains("\"stack_size\":1,"), forced.get(0));
        assertEquals(stateOf(observedHalf), stateOf(forced.get(0)));
    }

    // ── a forced record holds what live attributes to the forced line ───

    /**
     * A trigger the forced spell's resolution causes is not the spell's: live
     * gives it a bracket and a record of its own. Attended Knight's enters
     * trigger makes a Soldier, and the forced record holds the Knight's own
     * arrival and nothing of the token.
     */
    @Test
    void aForcedSpellsRecordLeavesOutTheTriggerItCaused() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card knight = TestBoard.inHand(game, caster, "Attended Knight");
        SpellAbility spell = knight.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String forced = forcedRecord(game, spell, 1L);
        String payload = payloadOf(forced);

        assertEquals(List.of(EventType.ZONE_CHANGE.wireName()), eventTypes(forced), payload);
        assertTrue(payload.contains("\"subjects\":[\"E" + knight.getId() + "\"],"
                + "\"params\":{\"from_zone\":\"stack\",\"to_zone\":\"battlefield\""), payload);
    }

    /**
     * A cast trigger resolves above the forced spell, before it, and is not
     * the spell's either: Monastery Swiftspear's prowess pump is not in the
     * forced Pyroclasm's record, though the damage it took is.
     */
    @Test
    void aForcedSpellsRecordLeavesOutItsCastTrigger() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card swiftspear = TestBoard.inPlay(game, caster, "Monastery Swiftspear");
        SpellAbility spell = TestBoard.inHand(game, caster, "Pyroclasm").getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String payload = payloadOf(forcedRecord(game, spell, 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"E"
                + swiftspear.getId() + "\"]"), payload);
        assertFalse(payload.contains(EventType.PT_CHANGE.wireName()), payload);
        assertFalse(payload.contains(EventType.KEYWORD_CHANGE.wireName()), payload);
    }

    /**
     * After the forced spell has resolved only the deaths of the state-based
     * check that follows join its record, as live holds them. Pyroclasm goes
     * to the graveyard once it has resolved, and Tarmogoyf grows for the
     * sorcery there: that stat change is Tarmogoyf's own static at work, not
     * Pyroclasm's effect. The deaths of the damaged Tarmogoyf and bears in
     * the check that follows are Pyroclasm's.
     */
    @Test
    void afterTheResolutionOnlyTheFollowingChecksDeathsJoin() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card goyf = TestBoard.inPlay(game, game.getPlayers().get(1), "Tarmogoyf");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility spell = TestBoard.inHand(game, caster, "Pyroclasm").getFirstSpellAbility();
        spell.setActivatingPlayer(caster);

        String payload = payloadOf(forcedRecord(game, spell, 1L));

        assertFalse(payload.contains(EventType.PT_CHANGE.wireName()), payload);
        for (Card died : List.of(goyf, bears)) {
            assertTrue(payload.contains("\"subjects\":[\"E" + died.getId() + "\"],\"params\":"
                    + "{\"from_zone\":\"battlefield\",\"to_zone\":\"graveyard\""), payload);
        }
    }

    /**
     * A death in the state-based check between the cast and the resolution
     * is nobody's, forced as live (FR-029): no resolution is held and no
     * damage step open, so live counts it and files it nowhere, the fork
     * refuses it, and the two records agree. Maro counts its caster's hand,
     * and casting Pyroclasm, the hand's one card, leaves it 0/0 before
     * Pyroclasm resolves; its shrink precedes the effect half's board, read
     * as the resolution starts (FR-042b), and is in neither record.
     */
    @Test
    void aDeathInTheCheckBeforeTheResolutionIsNobodys() throws Throwable {
        Game observedGame = maroBoard();
        String observed = payloadOf(
                observedEffectHalf(observedGame, firstSpellInHand(observedGame)));

        Game forkedGame = maroBoard();
        Card maro = forkedGame.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> card.getName().equals("Maro")).findFirst().orElseThrow();
        String forced = payloadOf(forcedRecord(forkedGame, firstSpellInHand(forkedGame), 1L));

        assertFalse(forced.contains("{\"type\":\"pt_change\",\"subjects\":[\"" + Refs.of(maro)
                + "\"]"), forced);
        assertFalse(forced.contains("\"subjects\":[\"" + Refs.of(maro)
                + "\"],\"params\":{\"from_zone\":\"battlefield\""), forced);
        assertEquals(observed, forced);
    }

    /**
     * A cast trigger above the forced spell owns what it does, and the
     * deaths of the check after it: Earthshaker deals 2 to each creature
     * without flying when an Arcane spell is cast, and the opposing bears it
     * kills are nowhere in the forced Lava Spike's record.
     */
    @Test
    void aCastTriggersDamageAndDeathsAreNotTheForcedSpells() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        TestBoard.inPlay(game, caster, "Earthshaker");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, caster, "Lava Spike");

        String payload = payloadOf(forcedRecord(game, firstSpellInHand(game), 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"P"), payload);
        assertFalse(payload.contains("\"E" + bears.getId() + "\""), payload);
    }

    /**
     * Nothing in the aftermath of an item above the forced spell is the
     * spell's, though the spell is the top of the stack again: it has not
     * begun to resolve, and live sends that aftermath to no record.
     * Managorger Hydra's cast trigger puts a +1/+1 counter on a Hydra that
     * carries a -1/-1 counter, above a forced Lava Spike, and the state-based
     * check after the trigger removes the pair: an outcome of that check
     * that is no death, and not Lava Spike's.
     */
    @Test
    void theAftermathOfACastTriggerIsNotTheForcedSpells() throws IOException {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        // The anthem keeps the 1/1 Hydra alive under its -1/-1 counter.
        TestBoard.inPlay(game, caster, "Glorious Anthem");
        Card hydra = TestBoard.inPlay(game, caster, "Managorger Hydra");
        hydra.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
        TestBoard.inHand(game, caster, "Lava Spike");

        String payload = payloadOf(forcedRecord(game, firstSpellInHand(game), 1L));

        assertTrue(payload.contains("\"type\":\"damage_dealt\",\"subjects\":[\"P"), payload);
        assertFalse(payload.contains("\"E" + hydra.getId() + "\""), payload);
    }

    /**
     * A death the forced Kavu's enters trigger causes is the trigger's, as
     * live files it: the Kavu's held window is closed by the time it
     * happens. The forced Kavu's record equals the observed Kavu's.
     */
    @Test
    void aDeathTheForcedKavusTriggerCausesIsNotTheKavus() throws Throwable {
        Game observedGame = kavuBoard();
        String observed = payloadOf(
                observedEffectHalf(observedGame, firstSpellInHand(observedGame)));

        Game forkedGame = kavuBoard();
        String forced = forcedRecord(forkedGame, firstSpellInHand(forkedGame), 1L);

        assertEquals(List.of(EventType.ZONE_CHANGE.wireName()), eventTypes(forced), forced);
        assertEquals(observed, payloadOf(forced));
    }

    /** Maro for the caster, whose hand is Pyroclasm alone, and the other seat's bears. */
    private static Game maroBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Maro");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Pyroclasm");
        return game;
    }

    /** Two creatures for the other seat and Flametongue Kavu in the caster's hand. */
    private static Game kavuBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Llanowar Elves");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Flametongue Kavu");
        return game;
    }

    private static SpellAbility firstSpellInHand(Game game) {
        Player caster = game.getPlayers().get(0);
        SpellAbility spell = caster.getCardsIn(ZoneType.Hand).get(0).getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        return spell;
    }

    /**
     * The first effect half an observed cast of {@code spell} writes, played
     * the way a game plays it: the check before priority, the resolution,
     * the check after it, and the triggers until the stack is empty.
     */
    private String observedEffectHalf(Game game, SpellAbility spell) throws Throwable {
        List<String> records = ShardRun.observed(tempDir.resolve("observed"), game, bracket -> {
            // A live collector has watched the board form, as the fork's sink
            // primes it: the stat change the cast's move to the stack makes
            // has a reading to differ from on both sides.
            bracket.rebaselineStats();
            game.getStack().addAndUnfreeze(spell);
            game.getAction().checkStateEffects(false);
            while (!game.getStack().isEmpty()) {
                game.getStack().resolveStack();
                game.getAction().checkStateEffects(false);
                game.getStack().addAllTriggeredAbilitiesToStack();
            }
            TestBoard.nextStep(game);
        });
        List<String> halves = records(records, "resolution");
        assertFalse(halves.isEmpty(), records.toString());
        return halves.get(0);
    }

    /** The one forced record. */
    private String forcedRecord(Game game, SpellAbility spell, long seed) throws IOException {
        ShardRun.Fork fork = ShardRun.forked(
                tempDir.resolve("fork-" + seed), new ShardRun.ForkRun(game, forkCaps().interventions(1).probes(0).build(), seed, spell));
        assertTrue(fork.wrote(), "the forced resolution observed nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        return records.get(0);
    }

    /** A Mountain for its caster, Grizzly Bears for the other seat, and Pyroclasm in hand. */
    private static Game pyroclasmBoard() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inPlay(game, game.getPlayers().get(0), "Mountain");
        TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        TestBoard.inHand(game, game.getPlayers().get(0), "Pyroclasm");
        return game;
    }
}
