package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.GameStage;
import forge.game.card.Card;
import forge.game.card.CardView;
import forge.game.card.CounterType;
import forge.game.combat.Combat;
import forge.game.event.GameEventCardDamaged;
import forge.game.phase.PhaseHandler;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordEvents.payload;
import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A {@code spell_cast}'s {@code from_zone} and {@code without_paying}, and a
 * {@code damage_dealt}'s {@code excess}, on boards the engine really plays:
 * a spell cast from the hand, flashed back from the graveyard, evoked and
 * cascaded into; damage beyond lethal to a creature, from a deathtouch
 * source, to a planeswalker beyond its loyalty, and through a damage
 * replacement (task B50).
 */
@ExtendWith(ForgeExtension.class)
class CastZoneAndExcessTest {

    private static final String WHY = "for trigger and rewrite records";
    private static final String CONTEMPLATION = "cardsfolder/c/contemplation.txt";

    @TempDir
    Path tempDir;

    // ── spell_cast ──────────────────────────────────────────────────────

    /** Think Twice cast from the hand: its zone, and no free cast. */
    @Test
    void aSpellCastFromTheHandNamesTheHand() throws Throwable {
        assertPaidCastFrom(new Cast(TestBoard::inHand, "Think Twice", 2, sa -> !sa.isFlashback()), "hand");
    }

    /** Think Twice flashed back: cast from the graveyard, its flashback cost paid. */
    @Test
    void aFlashbackNamesTheGraveyardAndIsPaid() throws Throwable {
        assertPaidCastFrom(new Cast(TestBoard::inGraveyard, "Think Twice", 3, SpellAbility::isFlashback),
                "graveyard");
    }

    /** Mulldrifter evoked: an alternative cost, paid, so no free cast. */
    @Test
    void anEvokedSpellIsPaid() throws Throwable {
        assertPaidCastFrom(new Cast(TestBoard::inHand, "Mulldrifter", 3, SpellAbility::isEvoke), "hand");
    }

    /**
     * Bloodbraid Elf's cascade exiles a Mountain and a Lightning Bolt and
     * casts the Bolt from exile without paying its mana cost.
     */
    @Test
    void aCascadedSpellIsCastFromExileWithoutPaying() throws Throwable {
        Game game = game();
        Player you = you(game);
        TestBoard.inPlay(game, you, "Contemplation");
        Card elf = TestBoard.inPlay(game, you, "Bloodbraid Elf");
        Card bolt = game.getAction().moveToLibrary(TestBoard.card(game, you, "Lightning Bolt"), 0, null);
        game.getAction().moveToLibrary(TestBoard.card(game, you, "Mountain"), 0, null);
        WrappedAbility cascade = triggerOf(elf, you);

        List<String> records = played(game, () -> {
            game.getStack().add(cascade);
            game.getStack().resolveStack();
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                    "the cascade cast the Bolt");
            game.getTriggerHandler().runWaitingTriggers();
        });

        Map<String, Object> cast = only(fired(records, CONTEMPLATION));
        assertEquals("exile", params(cast).get("from_zone"), cast.toString());
        assertEquals(true, params(cast).get("without_paying"), cast.toString());
    }

    /** Where a card is put before it is cast. */
    @FunctionalInterface
    private interface Placement {
        Card put(Game game, Player owner, String name);
    }

    /** A card put somewhere, the Islands that pay for it, and which of its spell abilities is cast. */
    private record Cast(Placement placement, String name, int islands, Predicate<SpellAbility> which) {
    }

    /** {@code cast} played as the AI plays it: Contemplation sees it cast from {@code zone}, paid. */
    private void assertPaidCastFrom(Cast cast, String zone) throws Throwable {
        Game game = game();
        Player you = you(game);
        lands(game, you, "Island", cast.islands());
        Card card = cast.placement().put(game, you, cast.name());

        Map<String, Object> event = castThrough(game, you, card, cast.which());

        assertEquals(zone, params(event).get("from_zone"), event.toString());
        assertFalse(params(event).containsKey("without_paying"), event.toString());
    }

    /** Cast the card's spell ability {@code which} picks, as the AI casts it, and Contemplation's one cast event. */
    private Map<String, Object> castThrough(
            Game game, Player caster, Card card, Predicate<SpellAbility> which) throws Throwable {
        TestBoard.inPlay(game, caster, "Contemplation");
        SpellAbility spell = spellAbility(card, caster, which);
        List<String> records = played(game, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, spell, null), "the AI cast " + spell);
            game.getTriggerHandler().runWaitingTriggers();
        });
        return only(fired(records, CONTEMPLATION));
    }

    private static SpellAbility spellAbility(Card card, Player caster, Predicate<SpellAbility> which) {
        for (SpellAbility sa : card.getAllPossibleAbilities(caster, true)) {
            if (sa.isSpell() && which.test(sa)) {
                sa.setActivatingPlayer(caster);
                return sa;
            }
        }
        throw new AssertionError("no such spell on " + card);
    }

    /** The card's first trigger, wrapped as the trigger handler puts it on the stack. */
    private static WrappedAbility triggerOf(Card card, Player controller) {
        Trigger trigger = card.getTriggers().iterator().next();
        SpellAbility execute = trigger.ensureAbility();
        execute.setActivatingPlayer(controller);
        WrappedAbility wrapped = new WrappedAbility(trigger, execute, controller);
        wrapped.setActivatingPlayer(controller);
        return wrapped;
    }

    // ── damage_dealt's excess ───────────────────────────────────────────

    /**
     * A 5/5 blocked by a 2/2: the 2/2 is dealt 5, 3 beyond its lethal 2; the
     * 5/5 is dealt 2, not lethal, and carries 0.
     */
    @Test
    void damageBeyondLethalToACreatureCarriesTheExcess() throws Throwable {
        Map<String, Long> excess = combatExcess("Durkwood Baloth", "Grizzly Bears");
        assertEquals(Map.of("Grizzly Bears", 3L, "Durkwood Baloth", 0L), excess);
    }

    /**
     * Gifted Aetherborn (2/3, deathtouch) blocked by Hill Giant (3/3): for a
     * deathtouch source lethal damage is 1 (702.2c), so the giant's 2 is 1
     * beyond it; the giant's 3 to the Aetherborn is exactly lethal, 0 beyond.
     */
    @Test
    void deathtouchDamageIsLethalAtOne() throws Throwable {
        Map<String, Long> excess = combatExcess("Gifted Aetherborn", "Hill Giant");
        assertEquals(Map.of("Hill Giant", 1L, "Gifted Aetherborn", 0L), excess);
    }

    /**
     * Gifted Aetherborn attacks Jace Beleren at 1 loyalty: its 2 damage is 1
     * beyond the loyalty, which Hooded Blightfang's damage trigger sees. The
     * bus tells no planeswalker's damage, only its loyalty lost.
     */
    @Test
    void damageBeyondAPlaneswalkersLoyaltyCarriesTheExcess() throws Throwable {
        Game game = combatGame();
        Player attacking = game.getPlayers().get(0);
        Player defending = game.getPlayers().get(1);
        Card aetherborn = TestBoard.inPlay(game, attacking, "Gifted Aetherborn");
        TestBoard.inPlay(game, attacking, "Hooded Blightfang");
        Card jace = TestBoard.inPlay(game, defending, "Jace Beleren");
        jace.setCounters(CounterType.getType("LOYALTY"), 1);
        Combat combat = new Combat(attacking);
        combat.addAttacker(aetherborn, jace);
        combat.orderAttackersForDamageAssignment();
        game.getPhaseHandler().setCombat(combat);

        List<String> records = played(game, () -> throughCombat(game));

        Map<String, Object> damage = only(fired(records, "cardsfolder/h/hooded_blightfang.txt"));
        assertEquals(List.of(Refs.of(jace)), damage.get("subjects"), damage.toString());
        assertEquals(2L, params(damage).get("amount"), damage.toString());
        assertEquals(1L, params(damage).get("excess"), damage.toString());
    }

    /**
     * Lightning Bolt at Grizzly Bears under Furnace of Rath: the damage as it
     * arrives is 3, 1 beyond lethal; as the replacement doubles it, 6, 4
     * beyond. Both are measured on the board before the damage is dealt.
     */
    @Test
    void aDamageReplacementMeasuresBothSidesBeforeTheDamage() throws Throwable {
        Game game = game();
        Player you = you(game);
        TestBoard.inPlay(game, you, "Furnace of Rath");
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card bolt = TestBoard.inHand(game, you, "Lightning Bolt");

        List<String> records = played(game, () -> {
            TestBoard.cast(game, bolt, you, bears);
            game.getStack().resolveStack();
        });

        Map<String, Object> rewrite = only(rewrites(records, "cardsfolder/f/furnace_of_rath.txt"));
        Map<String, Object> incoming = params(asMap(rewrite.get("incoming")));
        Map<String, Object> outgoing = params(asMap(rewrite.get("outgoing")));
        assertEquals(List.of(3L, 1L), Arrays.asList(incoming.get("amount"), incoming.get("excess")), rewrite.toString());
        assertEquals(List.of(6L, 4L), Arrays.asList(outgoing.get("amount"), outgoing.get("excess")), rewrite.toString());
    }

    /** A damage event the engine measured no excess for (built by hand) carries none, rather than 0. */
    @Test
    void anUnmeasuredDamageCarriesNoExcess() {
        Card bears = TestCards.build("Grizzly Bears");
        EffectEvent event = BusEvents.cardDamaged(new GameEventCardDamaged(
                CardView.get(bears), CardView.get(bears), 2, GameEventCardDamaged.DamageType.Normal), false);
        assertFalse(event.hasParam(EventParam.EXCESS), event.toString());
    }

    /** {@code attacker} attacks, {@code blocker} blocks it: each damaged creature's excess, by name. */
    private Map<String, Long> combatExcess(String attacker, String blocker) throws Throwable {
        Game game = combatGame();
        Player attacking = game.getPlayers().get(0);
        Card attacks = TestBoard.inPlay(game, attacking, attacker);
        Card blocks = TestBoard.inPlay(game, game.getPlayers().get(1), blocker);
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacks, game.getPlayers().get(1));
        combat.addBlocker(attacks, blocks);
        combat.setBlocked(attacks, true);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        game.getPhaseHandler().setCombat(combat);

        List<String> records = ShardRun.observed(tempDir, game, collector -> throughCombat(game));

        Map<String, String> names = Map.of(Refs.of(attacks), attacker, Refs.of(blocks), blocker);
        Map<String, Long> excess = new HashMap<>();
        for (String record : records) {
            for (Map<String, Object> event : RecordEvents.events(record)) {
                if ("damage_dealt".equals(event.get("type"))) {
                    String subject = ((List<?>) event.get("subjects")).get(0).toString();
                    excess.put(names.get(subject), (Long) params(event).get("excess"));
                }
            }
        }
        return excess;
    }

    private static void throughCombat(Game game) {
        PhaseHandler phases = game.getPhaseHandler();
        for (int step = 0; step < 20 && !phases.is(PhaseType.COMBAT_END); step++) {
            phases.mainLoopStep();
        }
        assertTrue(phases.is(PhaseType.COMBAT_END), "the game reached the end of combat");
    }

    private static Game combatGame() {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
        game.setAge(GameStage.Play);
        return game;
    }

    // ── the records ─────────────────────────────────────────────────────

    /** Play {@code body} with every hook installed, the triggers of cards put on the board by hand made active. */
    private List<String> played(Game game, ShardRun.Played body) throws Throwable {
        game.getTriggerHandler().resetActiveTriggers();
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, body);
    }

    /** The events of every fired trigger record acting through {@code script}. */
    private static List<Map<String, Object>> fired(List<String> records, String script) {
        List<Map<String, Object>> events = new ArrayList<>();
        for (String record : ofKind(records, "trigger", script)) {
            if (Boolean.TRUE.equals(payload(record).get("fired"))) {
                events.add(asMap(payload(record).get("event")));
            }
        }
        return events;
    }

    /** The payloads of every rewrite record acting through {@code script}. */
    private static List<Map<String, Object>> rewrites(List<String> records, String script) {
        return ofKind(records, "rewrite", script).stream().map(RecordEvents::payload).toList();
    }

    private static List<String> ofKind(List<String> records, String kind, String script) {
        return records.stream()
                .filter(record -> kind.equals(parse(record).get("kind")) && script.equals(scriptOf(record)))
                .toList();
    }

    private static <T> T only(List<T> found) {
        assertEquals(1, found.size(), found.toString());
        return found.get(0);
    }

    private static void lands(Game game, Player owner, String name, int count) {
        for (int i = 0; i < count; i++) {
            TestBoard.inPlay(game, owner, name);
        }
    }

    private static Game game() {
        return TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
    }

    private static Player you(Game game) {
        return game.getPlayers().get(0);
    }
}
