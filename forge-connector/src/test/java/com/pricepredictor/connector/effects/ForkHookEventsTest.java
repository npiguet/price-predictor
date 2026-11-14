package com.pricepredictor.connector.effects;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.payloadOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

/**
 * What a fork hears of the hooks: a forced resolution and a damage-step
 * probe each record the events the live record takes from the hooks rather
 * than the bus, on a game recorded as a worker records it.
 */
@ExtendWith(ForgeExtension.class)
class ForkHookEventsTest {

    /** Boundaries published: at one in twenty, the gate passes at about ten. */
    private static final int BOUNDARIES = 200;

    @TempDir
    Path tempDir;

    private static void boundaries(Game game, int count) {
        for (int i = 0; i < count; i++) {
            game.fireEvent(new GameEventTurnPhase(
                    game.getPhaseHandler().getPlayerTurn(), game.getPhaseHandler().getPhase(), ""));
        }
    }

    /** Every record the game's recording wrote, under {@code caps}, while {@code body} ran. */
    private List<String> recorded(Game game, CollectionCaps caps, String why, ShardRun.Played body)
            throws Throwable {
        RecordShardWriter writer = ShardRun.recordedWithClauseHook(tempDir, game, caps, why, body);
        return Files.exists(writer.path()) ? ShardRun.readShard(writer.path()) : List.of();
    }

    /** The forced resolutions' halves among {@code records}. */
    private static List<String> forced(List<String> records) {
        return records.stream()
                .filter(line -> Boolean.TRUE.equals(parse(line).get("interventional"))
                        && "resolution".equals(parse(line).get("kind"))
                        && parse(line).get("link_id") == null)
                .toList();
    }

    /**
     * A forced token spell's record holds the {@code token_created} the
     * outcome hook reports, as an observed one does: before the forks heard
     * the hooks, its tokens' moves were all the record held.
     */
    @Test
    void aForcedTokenSpellRecordsTheTokenTheOutcomeHookReports() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        TestBoard.inHand(game, game.getPlayers().get(0), "Raise the Alarm");
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        List<String> forced = forced(recorded(game, caps, "to hear the outcome hook",
                () -> boundaries(game, BOUNDARIES)));

        assertEquals(1, forced.size(), forced.toString());
        assertTrue(eventTypes(forced.get(0)).contains(EventType.TOKEN_CREATED.wireName()),
                payloadOf(forced.get(0)));
    }

    /**
     * A forced spell dividing X among its targets announces X before it
     * draws them (601.2b-c): drawn the other way round the amount was zero,
     * no target was taken and the resolution did nothing.
     */
    @Test
    void aForcedDividedXSpellDealsTheXItAnnounced() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        TestBoard.inHand(game, you, "Rolling Thunder");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        List<String> forced = forced(recorded(game, caps, "to force a line as a worker does",
                () -> boundaries(game, BOUNDARIES)));

        assertEquals(1, forced.size(), forced.toString());
        int dealt = events(forced.get(0)).stream()
                .filter(event -> EventType.DAMAGE_DEALT.wireName().equals(event.get("type")))
                .mapToInt(event -> ((Number) ((Map<?, ?>) event.get("params")).get("amount")).intValue())
                .sum();
        Object x = ((Map<?, ?>) ((Map<?, ?>) parse(forced.get(0)).get("state")).get("refs")).get("x");
        assertTrue(dealt >= 1, payloadOf(forced.get(0)));
        assertEquals(((Number) x).intValue(), dealt, "the damage dealt is the X announced");
    }

    /**
     * A forced bestow whose target left before it resolved (Skulking Ghost
     * sacrifices itself on becoming a target) resolves as a creature spell,
     * as an observed one does: resolved, its card entering the battlefield,
     * not fizzled with nothing.
     */
    @Test
    void aForcedBestowWhoseTargetLeftResolvesAsACreature() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Skulking Ghost");
        game.getTriggerHandler().resetActiveTriggers();
        Card naiad = TestBoard.inHand(game, you, "Nimbus Naiad");
        SpellAbility bestow = naiad.getSpellAbilities().stream()
                .filter(SpellAbility::isBestow).findFirst().orElseThrow();
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 1L, bestow, "to read the resolution hook's fizzle");

        assertTrue(fork.wrote(), "the forced bestow wrote nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        Map<String, Object> outcome = RecordLines.payload(records.get(0));
        assertEquals("resolved", outcome.get("outcome"), payloadOf(records.get(0)));
        assertTrue(events(records.get(0)).stream().anyMatch(event ->
                        EventType.ZONE_CHANGE.wireName().equals(event.get("type"))
                                && "battlefield".equals(((Map<?, ?>) event.get("params")).get("to_zone"))),
                payloadOf(records.get(0)));
    }

    /**
     * A forced spell whose one target left before it resolved fizzles as an
     * observed one does (608.2b): its effect half holds its own card's move
     * off the stack and nothing else, and says it fizzled.
     */
    @Test
    void aForcedSpellWhoseTargetLeftFizzlesWithItsMoveOffTheStack() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player you = game.getPlayers().get(0);
        Card ghost = TestBoard.inPlay(game, game.getPlayers().get(1), "Skulking Ghost");
        game.getTriggerHandler().resetActiveTriggers();
        Card blade = TestBoard.inHand(game, you, "Murder");
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 1L, blade.getFirstSpellAbility(), "to read the resolution hook's fizzle");

        assertTrue(fork.wrote(), "the fizzled forced spell wrote nothing");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        assertEquals("fizzled", RecordLines.payload(records.get(0)).get("outcome"), payloadOf(records.get(0)));
        List<Map<String, Object>> held = events(records.get(0));
        assertEquals(1, held.size(), payloadOf(records.get(0)));
        assertEquals(EventType.ZONE_CHANGE.wireName(), held.get(0).get("type"));
        assertEquals(List.of(Refs.of(blade)), held.get(0).get("subjects"), payloadOf(records.get(0)));
        assertTrue(ghost.getZone() != null && ghost.getZone().is(forge.game.zone.ZoneType.Battlefield),
                "the live ghost was sacrificed by the fork");
    }

    /**
     * A probe's branch records the damage a protection prevented, which the
     * outcome hook reports, as the observed combat record it mirrors does:
     * otherwise gate 2's difference holds a prevention the stripped keyword
     * did not cause.
     */
    @Test
    void aProbeRecordsTheDamageAProtectionPrevented() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DAMAGE);
        Player you = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card dreadmaw = TestBoard.inPlay(game, you, "Colossal Dreadmaw");
        Card watcher = TestBoard.inPlay(game, opponent, "Coast Watcher");
        Combat combat = TestBoard.blockedCombat(game, you, opponent, dreadmaw, watcher);
        CollectionCaps caps = TestCaps.caps().everyDecision().manaCap(2000).interventions(0).probes(1)
                .probeKeywords("trample").build();

        List<String> records = recorded(game, caps, "to hear the outcome hook", () -> {
            TestBoard.enterStep(game, PhaseType.COMBAT_DAMAGE);
            assertTrue(combat.assignCombatDamage(false), "the step assigned no damage");
            combat.dealAssignedDamage();
            game.getAction().checkStateEffects(false);
            TestBoard.enterStep(game, PhaseType.COMBAT_END);
        });

        List<String> combats = records.stream()
                .filter(line -> "combat".equals(parse(line).get("kind"))).toList();
        List<String> observed = combats.stream()
                .filter(line -> Boolean.FALSE.equals(parse(line).get("fork"))).toList();
        List<String> probes = combats.stream()
                .filter(line -> Boolean.TRUE.equals(parse(line).get("fork"))).toList();
        assertEquals(1, observed.size(), combats.toString());
        assertTrue(eventTypes(observed.get(0)).contains(EventType.DAMAGE_PREVENTED.wireName()),
                "the observed record, the control: " + payloadOf(observed.get(0)));
        assertEquals(1, probes.size(), combats.toString());
        assertTrue(eventTypes(probes.get(0)).contains(EventType.DAMAGE_PREVENTED.wireName()),
                payloadOf(probes.get(0)));
        assertFalse(probes.get(0).contains("\"copy_dropped\""), probes.get(0));
    }
}
