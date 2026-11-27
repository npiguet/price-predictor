package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventGameFinished;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordPicks.withEventOfType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The outcomes a resolution or combat record reads off the bus besides the
 * moves themselves: a permanent destroyed, a permanent
 * sacrificed, a card turning face up, a player losing the game. Each was
 * written only on a trigger's or a replacement's record, from its run
 * parameters, and never on the record of what did it.
 */
@ExtendWith(ForgeExtension.class)
class BusOutcomeEventsTest {

    private static final String WHY = "for the resolution and state-based-loop hooks";

    @TempDir
    Path tempDir;

    /** Murder destroys Grizzly Bears: {@code destroyed}, then the bears' move. */
    @Test
    void aDestroySpellsHalfSaysThePermanentWasDestroyed() throws Throwable {
        assertBearsEventsOfSpell(List.of("destroyed", "zone_change"),
                new BearsSpell("Murder", "cardsfolder/m/murder.txt", Target.BEARS));
    }

    /**
     * A Lightning Bolt's lethal damage: the state-based check destroys the
     * bears, and the {@code destroyed} joins the record the death joins,
     * the Bolt's held half.
     */
    @Test
    void aStateBasedDestructionJoinsTheRecordItsDeathJoins() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card bolt = TestBoard.inHand(game, caster, "Lightning Bolt");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, bolt, caster, bears);
            game.getStack().resolveStack();
            for (int step = 0; step < 10 && game.getCardsIn(forge.game.zone.ZoneType.Battlefield)
                    .stream().anyMatch(c -> c.getId() == bears.getId()); step++) {
                game.getPhaseHandler().mainLoopStep();
            }
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        assertEquals(List.of("damage_dealt", "destroyed", "zone_change"),
                typesNaming(half, Refs.of(bears)), half);
    }

    /** Cruel Edict: the opponent sacrifices its creature, {@code sacrificed} then the move. */
    @Test
    void aSacrificeSpellsHalfSaysThePermanentWasSacrificed() throws Throwable {
        assertBearsEventsOfSpell(List.of("sacrificed", "zone_change"),
                new BearsSpell("Cruel Edict", "cardsfolder/c/cruel_edict.txt", Target.OPPONENT));
    }

    /**
     * Break Open turns the opponent's face-down Battering Craghorn face up:
     * a {@code face_change} to {@code front}, in FaceState's spelling.
     */
    @Test
    void aCardTurnedFaceUpIsAFaceChangeToItsFront() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card craghorn = TestBoard.inPlay(game, game.getPlayers().get(1), "Battering Craghorn");
        craghorn.turnFaceDown(true);
        Card breakOpen = TestBoard.inHand(game, caster, "Break Open");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, breakOpen, caster, craghorn);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        assertTrue(!craghorn.isFaceDown(), "Break Open turned the Craghorn face up");
        String half = only(records, "resolution", "cardsfolder/b/break_open.txt");
        List<Map<String, Object>> turned = events(half).stream()
                .filter(event -> "face_change".equals(event.get("type"))).toList();
        assertEquals(1, turned.size(), half);
        assertEquals(List.of(Refs.of(craghorn)), turned.get(0).get("subjects"), half);
        assertEquals("front", params(turned.get(0)).get("to_state"), half);
    }

    /**
     * A Lightning Bolt takes the opponent from 3 life to 0: the game's
     * outcome names the loser, with Forge's reason, before the winner.
     */
    @Test
    void theGamesOutcomeNamesItsLoser() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        opponent.setLife(3, null);
        SpellAbility bolt = TestBoard.spellOf(TestBoard.inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(opponent);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            TestBoard.castAndResolve(game, bolt);
            // The game's end publishes its outcome on the bus.
            assertTrue(game.isGameOver(), "the opponent lost");
            collector.finishGame(Set.of());
        });

        List<String> outcome = withEventOfType(records, "player_lost");
        assertEquals(1, outcome.size(), records.toString());
        List<Map<String, Object>> told = events(outcome.get(0)).stream()
                .filter(event -> event.get("type").toString().startsWith("player_")).toList();
        assertEquals(List.of("player_lost", "player_won"),
                told.stream().map(event -> event.get("type")).toList(), outcome.get(0));
        assertEquals(List.of(Refs.of(opponent)), told.get(0).get("subjects"), outcome.get(0));
        assertEquals("LifeReachedZero", params(told.get(0)).get("reason"), outcome.get(0));
        // The winner, though the bolt's own bracket had closed before the
        // state-based check found the game over.
        assertEquals(List.of(Refs.of(caster)), told.get(1).get("subjects"), outcome.get(0));
    }

    /**
     * Door to Nothingness makes its target lose: Forge ends the game while
     * the ability resolves, and the ability's effect half holds the loss and
     * the win it made, written once in the game. The half is written at the
     * game's end, which a match publishes when its game loop returns.
     */
    @Test
    void aLoseTheGameAbilitysHalfHoldsTheLossAndTheWin() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        SpellAbility loses = abilityOfPermanent(game, "Door to Nothingness",
                forge.game.ability.ApiType.LosesGame);
        loses.resetTargets();
        loses.getTargets().add(opponent);

        assertGameEndInResolutionHalf(resolveToTheGamesEnd(game, loses, "the opponent lost"),
                "cardsfolder/d/door_to_nothingness.txt", new GameEnd(opponent, "SpellEffect", caster));
    }

    /**
     * Halo Fountain's "you win the game": the ability's effect half holds
     * the win and the opponent's loss, written once in the game.
     */
    @Test
    void aWinTheGameAbilitysHalfHoldsTheWinAndTheLoss() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        SpellAbility wins = abilityOfPermanent(game, "Halo Fountain",
                forge.game.ability.ApiType.WinsGame);

        List<String> resolved = resolveToTheGamesEnd(game, wins, "the caster won");

        GameEnd end = new GameEnd(opponent, "OpponentWon", caster);
        assertGameEndInResolutionHalf(resolved, "cardsfolder/h/halo_fountain.txt", end);
    }

    /**
     * Both players at 1 life lose 1 at once: a draw. Each gets a {@code
     * player_lost}, in the record the game's outcome is written in, and no
     * {@code player_won} is written.
     */
    @Test
    void aSimultaneousLossNamesBothLosersAndNoWinner() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        caster.setLife(1, null);
        opponent.setLife(1, null);
        SpellAbility drain = forge.game.ability.AbilityFactory.getAbility(
                "SP$ LoseLife | Defined$ Player | LifeAmount$ 1",
                TestBoard.inHand(game, caster, "Lightning Bolt"));
        drain.setActivatingPlayer(caster);

        List<String> records = ShardRun.observed(tempDir, game, collector -> {
            TestBoard.castAndResolve(game, drain);
            assertTrue(game.isGameOver(), "both players lost");
            collector.finishGame(Set.of());
        });

        List<String> outcome = withEventOfType(records, "player_lost");
        assertEquals(1, outcome.size(), records.toString());
        List<Object> losers = events(outcome.get(0)).stream()
                .filter(event -> "player_lost".equals(event.get("type")))
                .<Object>map(event -> ((List<?>) event.get("subjects")).get(0)).toList();
        assertEquals(Set.of(Refs.of(caster), Refs.of(opponent)), Set.copyOf(losers), outcome.get(0));
        assertEquals(2, losers.size(), outcome.get(0));
        assertEquals(List.of(), withEventOfType(records, "player_won"), records.toString());
    }

    /**
     * The game's end is in {@code half}: one {@code player_lost} naming the
     * loser with {@code reason}, then one {@code player_won} naming the
     * winner, and no other record of the game writes either.
     */
    private static void assertGameEndWrittenOnceIn(List<String> records, String half, GameEnd end) {
        List<Map<String, Object>> told = events(half).stream()
                .filter(event -> event.get("type").toString().startsWith("player_")).toList();
        assertEquals(List.of("player_lost", "player_won"),
                told.stream().map(event -> event.get("type")).toList(), half);
        assertEquals(List.of(Refs.of(end.loser())), told.get(0).get("subjects"), half);
        assertEquals(end.reason(), params(told.get(0)).get("reason"), half);
        assertEquals(List.of(Refs.of(end.winner())), told.get(1).get("subjects"), half);
        assertEquals(List.of(half), withEventOfType(records, "player_lost"), records.toString());
        assertEquals(List.of(half), withEventOfType(records, "player_won"), records.toString());
    }

    /** What a spell at the opponent's Grizzly Bears aims at. */
    private enum Target { BEARS, OPPONENT }

    /** A spell cast at the opponent's Grizzly Bears or at the opponent, and the script of its resolution record. */
    private record BearsSpell(String name, String script, Target target) {
    }

    /** Who lost, Forge's reason, and who won. */
    private record GameEnd(Player loser, String reason, Player winner) {
    }

    /** The events of the spell's resolution record naming the bears are of {@code types}, in order. */
    private void assertBearsEventsOfSpell(List<String> types, BearsSpell spell) throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card card = TestBoard.inHand(game, caster, spell.name());

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, card, caster, spell.target() == Target.BEARS ? bears : opponent);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", spell.script());
        assertEquals(types, typesNaming(half, Refs.of(bears)), half);
    }

    /** The {@code api} ability of the named permanent the first player controls, that player activating it. */
    private static SpellAbility abilityOfPermanent(Game game, String name, forge.game.ability.ApiType api) {
        Player controller = game.getPlayers().get(0);
        SpellAbility ability = TestBoard.ability(TestBoard.inPlay(game, controller, name), api);
        ability.setActivatingPlayer(controller);
        return ability;
    }

    /** The game's end is written once, in the resolution record of {@code script}. */
    private static void assertGameEndInResolutionHalf(List<String> records, String script, GameEnd end) {
        assertGameEndWrittenOnceIn(records, only(records, "resolution", script), end);
    }

    /** Resolves {@code ability} on the stack, asserts the game is over ({@code why}) and publishes its end. */
    private List<String> resolveToTheGamesEnd(Game game, SpellAbility ability, String why) throws Throwable {
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().add(ability);
            game.getStack().resolveStack();
            assertTrue(game.isGameOver(), why);
            game.fireEvent(new GameEventGameFinished());
        });
    }

    /** The types of the record's events naming {@code ref}, in order. */
    private static List<Object> typesNaming(String record, String ref) {
        return events(record).stream()
                .filter(event -> ((List<?>) event.get("subjects")).contains(ref))
                .map(event -> event.get("type")).toList();
    }
}
