package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.WrappedAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.optionOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.payload;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.RecordLines.scriptOf;
import static com.pricepredictor.connector.effects.TestBoard.inHand;
import static com.pricepredictor.connector.effects.TestBoard.inPlay;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.priorityWithTriggers;
import static com.pricepredictor.connector.effects.TestBoard.spellOf;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The record moments (root spec § 8.14, FR-042a–FR-042f), on real casts and
 * real resolutions recorded as a worker records them: every hook installed,
 * the engine choosing, paying and resolving.
 *
 * <p>Each record describes one moment, its board read just before it: a cost
 * half before its card moved or its costs were paid, an effect half when its
 * resolution started, a mana record before its source was tapped, a rewrite
 * record before the replacement ran.
 */
@ExtendWith(ForgeExtension.class)
class RecordMomentsTest {

    private static final String WHY = "to record a cast's moments";
    private static final String BOLT = "cardsfolder/l/lightning_bolt.txt";

    @TempDir
    Path tempDir;

    // ── reading records ─────────────────────────────────────────────────

    /** The entity {@code ref} in a record's state. */
    private static Map<String, Object> entity(String record, String ref) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object entity : (List<?>) state.get("entities")) {
            Map<String, Object> map = asMap(entity);
            if (ref.equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + ref + " in " + record);
    }

    /** The player {@code ref} in a record's state. */
    private static Map<String, Object> player(String record, String ref) {
        Map<String, Object> state = asMap(parse(record).get("state"));
        for (Object player : (List<?>) state.get("players")) {
            Map<String, Object> map = asMap(player);
            if (ref.equals(map.get("id"))) {
                return map;
            }
        }
        throw new AssertionError("no " + ref + " in " + record);
    }

    private static String ref(Card card) {
        return "E" + card.getId();
    }

    /** The records acting through {@code script} at {@code moment}. */
    private static List<String> through(List<String> all, String moment, String script) {
        return records(all, moment).stream().filter(r -> script.equals(scriptOf(r))).toList();
    }

    private static Map<String, Object> zoneChange(String subject, String from, String to) {
        return Map.of("subjects", List.of(subject), "from_zone", from, "to_zone", to);
    }

    /** Whether {@code event} carries every entry of {@code expected}. */
    private static boolean carries(Map<String, Object> event, Map<String, Object> expected) {
        Map<String, Object> params = RecordLines.params(event);
        for (Map.Entry<String, Object> entry : expected.entrySet()) {
            Object actual = event.containsKey(entry.getKey())
                    ? event.get(entry.getKey()) : params.get(entry.getKey());
            if (!entry.getValue().equals(actual)) {
                return false;
            }
        }
        return true;
    }

    // ── FR-042a: the cost half ──────────────────────────────────────────

    /**
     * A Lightning Bolt the AI casts and pays for with a Mountain: the cost
     * half's board has the Bolt still in hand and the Mountain untapped; its
     * events are the Bolt's move to the stack, its costs the red mana, and it
     * carries no outcome. The effect half's board has the Bolt on the stack,
     * it says resolved, and the Bolt's move to the graveyard follows its
     * damage, before the Giant's death the half holds. The Mountain's mana record is read before it tapped: it
     * untapped, the pool empty, and its events the tap, then the mana.
     */
    @Test
    void aSpellIsCastFromItsHandAndResolvesOffTheStack() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card giant = inPlay(game, game.getPlayers().get(1), "Hill Giant");
        Card mountain = inPlay(game, caster, "Mountain");
        Card bolt = inHand(game, caster, "Lightning Bolt");
        SpellAbility spell = spellOf(bolt, caster);
        spell.resetTargets();
        spell.getTargets().add(giant);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, spell, null),
                    "the AI cast the Bolt");
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            nextStep(game);
        });

        String cost = only(all, "activation", BOLT);
        assertFalse(payload(cost).containsKey("outcome"), cost);
        assertEquals("hand", entity(cost, ref(bolt)).get("zone"), cost);
        assertEquals(false, entity(cost, ref(mountain)).get("tapped"), cost);
        List<Map<String, Object>> castEvents = events(cost);
        assertEquals(1, castEvents.size(), cost);
        assertTrue(carries(castEvents.get(0), zoneChange(ref(bolt), "hand", "stack")), cost);
        assertEquals(Map.of("R", 1L),
                asMap(payload(cost).get("costs")).get("mana_by_color"), cost);

        String effect = only(all, "resolution", BOLT);
        assertEquals("resolved", payload(effect).get("outcome"), effect);
        assertEquals(linkOf(cost), linkOf(effect));
        assertEquals("stack", entity(effect, ref(bolt)).get("zone"), effect);
        // The Bolt leaves the stack after its own events, and the Giant's
        // death, the state-based consequence the half holds, comes after.
        assertEquals(List.of("damage_dealt", "zone_change", "zone_change"), eventTypes(effect),
                effect);
        List<Map<String, Object>> resolved = events(effect);
        assertTrue(carries(resolved.get(1), zoneChange(ref(bolt), "stack", "graveyard")), effect);
        assertTrue(carries(resolved.get(2), zoneChange(ref(giant), "battlefield", "graveyard")),
                effect);

        List<String> manaRecords = records(all, "resolution").stream()
                .filter(r -> linkOf(r) == null && eventTypes(r).contains("mana_produced"))
                .toList();
        assertEquals(1, manaRecords.size(), all.toString());
        String mana = manaRecords.get(0);
        assertNull(linkOf(mana), mana);
        assertEquals(false, entity(mana, ref(mountain)).get("tapped"), mana);
        assertEquals(0L, asMap(player(mana, "P" + caster.getId()).get("floating_mana")).get("R"),
                mana);
        assertEquals(List.of("tapped", "mana_produced"), eventTypes(mana), mana);
        assertEquals(List.of(ref(mountain)), events(mana).get(0).get("subjects"), mana);
        assertNull(payload(mana).get("outcome"), mana);
    }

    /**
     * Prodigal Pyromancer's tap ability: its cost half is read before its
     * cost is paid, the Pyromancer untapped, and holds no event, an
     * activated ability moving no card.
     */
    @Test
    void anActivatedAbilitysCostHalfIsReadBeforeItsCostsArePaid() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card pyromancer = inPlay(game, caster, "Prodigal Pyromancer");
        SpellAbility ping = TestBoard.ability(pyromancer, forge.game.ability.ApiType.DealDamage);
        ping.setActivatingPlayer(caster);
        ping.resetTargets();
        ping.getTargets().add(opponent);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, ping, null),
                    "the AI activated the Pyromancer");
            game.getStack().resolveStack();
            nextStep(game);
        });

        String pyro = "cardsfolder/p/prodigal_pyromancer.txt";
        String cost = only(all, "activation", pyro);
        assertEquals(false, entity(cost, ref(pyromancer)).get("tapped"), cost);
        assertEquals(List.of(), events(cost), cost);
        assertEquals(List.of(ref(pyromancer)),
                asMap(payload(cost).get("costs")).get("tapped"), cost);
        String effect = only(all, "resolution", pyro);
        assertEquals("resolved", payload(effect).get("outcome"), effect);
        assertEquals(true, entity(effect, ref(pyromancer)).get("tapped"), effect);
        assertEquals(19, opponent.getLife());
    }

    // ── FR-042b: the effect half ────────────────────────────────────────

    /**
     * A Bolt at a Hill Giant, answered by a Giant Growth that resolves first:
     * the Bolt's effect half is read as it starts to resolve, so it shows the
     * Giant already 6/6, which the cast-time board did not.
     */
    @Test
    void theEffectHalfIsReadWhenItsResolutionStarts() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card giant = inPlay(game, opponent, "Hill Giant");
        SpellAbility bolt = spellOf(inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(giant);
        SpellAbility growth = spellOf(inHand(game, opponent, "Giant Growth"), opponent);
        growth.resetTargets();
        growth.getTargets().add(giant);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(bolt);
            game.getStack().addAndUnfreeze(growth);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            nextStep(game);
        });

        assertEquals(ZoneType.Battlefield, game.getZoneOf(giant).getZoneType(), "the Giant lived");
        String cost = only(all, "activation", BOLT);
        String effect = only(all, "resolution", BOLT);
        assertEquals(List.of(0L, 0L),
                asMap(entity(cost, ref(giant)).get("pt_boosts")).get("temporary"), cost);
        assertEquals(List.of(3L, 3L),
                asMap(entity(effect, ref(giant)).get("pt_boosts")).get("temporary"), effect);
    }

    /**
     * A Bolt whose target died before it resolved: its effect half says
     * fizzled and holds no event (FR-042b), under the cost half's link.
     */
    @Test
    void aFizzledSpellWritesAnEmptyEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility bolt = spellOf(inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(bears);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(bolt);
            game.getAction().moveToGraveyard(bears, null);
            game.getStack().resolveStack();
            nextStep(game);
        });

        String cost = only(all, "activation", BOLT);
        String effect = only(all, "resolution", BOLT);
        assertEquals("fizzled", payload(effect).get("outcome"), effect);
        assertEquals(List.of(), events(effect), effect);
        assertEquals(linkOf(cost), linkOf(effect));
    }

    /**
     * A Counterspell on a Bolt: the Bolt writes no effect half, and the
     * Counterspell's holds the Bolt's {@code spell_countered} and the Bolt's
     * card leaving the stack (FR-042b).
     */
    @Test
    void aCounteredSpellsFateIsTheCounterspellsEffectHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        SpellAbility bolt = spellOf(inHand(game, caster, "Lightning Bolt"), caster);
        bolt.resetTargets();
        bolt.getTargets().add(opponent);
        SpellAbility counter = spellOf(inHand(game, opponent, "Counterspell"), opponent);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(bolt);
            Card onStack = bolt.getHostCard();
            counter.resetTargets();
            counter.getTargets().add(game.getStack().peekAbility());
            game.getStack().addAndUnfreeze(counter);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            assertEquals(ZoneType.Graveyard,
                    game.getZoneOf(game.getCardState(onStack)).getZoneType(), "countered");
            nextStep(game);
        });

        assertEquals(20, opponent.getLife());
        assertEquals(List.of(), through(all, "resolution", BOLT), all.toString());
        only(all, "activation", BOLT);
        String countering = only(all, "resolution", "cardsfolder/c/counterspell.txt");
        String boltRef = ref(bolt.getHostCard());
        assertTrue(eventTypes(countering).contains("spell_countered"), countering);
        assertTrue(events(countering).stream().anyMatch(
                event -> carries(event, zoneChange(boltRef, "stack", "graveyard"))), countering);
    }

    // ── C3: mana a resolving spell makes ────────────────────────────────

    /**
     * Dark Ritual's mana is its effect half's {@code mana_produced}; no mana
     * record is written for it (FR-042c).
     */
    @Test
    void aSpellsManaIsItsEffectHalfsAndNoManaRecord() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility ritual = spellOf(inHand(game, caster, "Dark Ritual"), caster);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(ritual);
            game.getStack().resolveStack();
            nextStep(game);
        });

        String effect = only(all, "resolution", "cardsfolder/d/dark_ritual.txt");
        assertTrue(linkOf(effect) != null, "the ritual's half is linked, no mana record: " + all);
        List<Map<String, Object>> made = events(effect).stream()
                .filter(event -> "mana_produced".equals(event.get("type"))).toList();
        assertEquals(1, made.size(), effect);
        assertEquals(Map.of("B", 3L), RecordLines.params(made.get(0)).get("mana_by_color"),
                effect);
    }

    // ── FR-042d, FR-042e: entering tapped; library cards ────────────────

    /**
     * Rampant Growth puts a Forest from the library onto the battlefield
     * tapped: its effect half has a {@code tapped} event right after the
     * Forest's move, and renders the Forest as it was in the library,
     * untapped and not entered this turn.
     */
    @Test
    void aLandPutOntoTheBattlefieldTappedIsTappedInTheRecordThatMovedIt() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card forest = game.getAction().moveToLibrary(
                TestBoard.card(game, caster, "Forest"), null, null);
        SpellAbility growth = spellOf(inHand(game, caster, "Rampant Growth"), caster);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(growth);
            game.getStack().resolveStack();
            nextStep(game);
        });

        assertTrue(game.getCardState(forest).isTapped(), "the Forest entered tapped");
        String effect = only(all, "resolution", "cardsfolder/r/rampant_growth.txt");
        List<Map<String, Object>> resolved = events(effect);
        int moved = -1;
        for (int i = 0; i < resolved.size(); i++) {
            if (carries(resolved.get(i), zoneChange(ref(forest), "library", "battlefield"))) {
                moved = i;
            }
        }
        assertTrue(moved >= 0, effect);
        assertEquals("tapped", resolved.get(moved + 1).get("type"), effect);
        assertEquals(List.of(ref(forest)), resolved.get(moved + 1).get("subjects"), effect);
        Map<String, Object> asInLibrary = entity(effect, ref(forest));
        assertEquals("library", asInLibrary.get("zone"), effect);
        assertEquals(false, asInLibrary.get("tapped"), effect);
        assertEquals(false, asMap(asInLibrary.get("history")).get("entered_this_turn"), effect);
    }

    // ── H1-2: the rewrite record's board ────────────────────────────────

    /**
     * Jungle Hollow enters tapped through its own replacement: the rewrite
     * record's board is read before the replacement ran, the land untapped.
     */
    @Test
    void aGainLandsRewriteRecordShowsItUntapped() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card hollow = inHand(game, caster, "Jungle Hollow");

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getAction().moveToPlay(hollow, caster, null, null);
            nextStep(game);
        });

        assertTrue(game.getCardState(hollow).isTapped(), "the Hollow entered tapped");
        List<String> rewrites = all.stream()
                .filter(r -> r.contains("\"kind\":\"rewrite\""))
                .filter(r -> "cardsfolder/j/jungle_hollow.txt".equals(scriptOf(r)))
                .toList();
        assertEquals(1, rewrites.size(), all.toString());
        assertEquals(false, entity(rewrites.get(0), ref(hollow)).get("tapped"), rewrites.get(0));
    }

    // ── A1: a charm's modes, each in its own half ───────────────────────

    /**
     * Academic Probation with both modes chosen, each ending in an Effect:
     * the first mode's {@code continuous_effect_created} is in its own
     * mode's half, not the last mode's, and each half holds its own Effect
     * alone (A1: a clause's events are filed when its own effect is over,
     * before the next mode begins).
     */
    @Test
    void aCharmsFirstModeKeepsItsOwnEvents() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        SpellAbility charm = TestBoard.charmOf(inHand(game, caster, "Academic Probation"), caster);
        for (AbilitySub clause : TestBoard.chain(charm, 0, 1)) {
            clause.setActivatingPlayer(caster);
            if (clause.usesTargeting()) {
                clause.getTargets().add(bears);
            }
        }

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(charm);
            game.getStack().resolveStack();
            nextStep(game);
        });

        List<String> halves = through(all, "resolution", "cardsfolder/a/academic_probation.txt");
        assertEquals(2, halves.size(), all.toString());
        String first = halves.stream().filter(h -> Long.valueOf(0).equals(optionOf(h)))
                .findFirst().orElseThrow();
        String second = halves.stream().filter(h -> Long.valueOf(1).equals(optionOf(h)))
                .findFirst().orElseThrow();
        assertEquals(List.of("CantCast"), layersCreated(first), first);
        assertEquals(List.of("DBCantStatic"), layersCreated(second), second);
    }

    /** The {@code layers} of each {@code continuous_effect_created} in a record. */
    private static List<Object> layersCreated(String record) {
        return events(record).stream()
                .filter(event -> "continuous_effect_created".equals(event.get("type")))
                .map(event -> RecordLines.params(event).get("layers"))
                .toList();
    }

    // ── A3: a clause that did not run emits nothing ─────────────────────

    /**
     * Vigorous Charge cast unkicked: its {@code Condition$ Kicked} Effect
     * clause did not run, so no {@code continuous_effect_created}, and its
     * condition reads unmet, as Forge decided it.
     */
    @Test
    void anUnkickedClauseCreatesNoEffect() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = inPlay(game, caster, "Grizzly Bears");
        SpellAbility charge = spellOf(inHand(game, caster, "Vigorous Charge"), caster);
        charge.resetTargets();
        charge.getTargets().add(bears);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(charge);
            game.getStack().resolveStack();
            nextStep(game);
        });

        String effect = only(all, "resolution", "cardsfolder/v/vigorous_charge.txt");
        assertFalse(eventTypes(effect).contains("continuous_effect_created"), effect);
        assertEquals(List.of(Map.of("segment", "0", "met", false)),
                payload(effect).get("conditions_met"), effect);
    }

    // ── A2: a trigger's root clause, once; triggers have no cost half ───

    /** Sea Gate Stormcaller in play, its enters trigger on the stack. */
    private static void stormcallerEnters(Game game, Player caster) {
        SpellAbility stormcaller = spellOf(inHand(game, caster, "Sea Gate Stormcaller"), caster);
        game.getStack().addAndUnfreeze(stormcaller);
        game.getStack().resolveStack();
        priorityWithTriggers(game);
    }

    private static final String STORMCALLER = "cardsfolder/s/sea_gate_stormcaller.txt";

    /**
     * Sea Gate Stormcaller's enters trigger, whose root is a DelayedTrigger:
     * exactly one {@code delayed_trigger_created}, and the trigger, nobody's
     * cast, has no cost half and no link.
     */
    @Test
    void aTriggersDelayedTriggerRootIsWrittenOnce() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            stormcallerEnters(game, caster);
            game.getStack().resolveStack();
            nextStep(game);
        });

        String trigger = only(all, "resolution", STORMCALLER, "trigger");
        assertEquals(List.of("delayed_trigger_created"), eventTypes(trigger), trigger);
        assertEquals("resolved", payload(trigger).get("outcome"), trigger);
        assertNull(linkOf(trigger), trigger);
        assertEquals(1, through(all, "activation", STORMCALLER).size(),
                "the creature spell's cost half alone: " + all);
    }

    /** A controller that declines every optional trigger, telling the confirm hook as the AI does. */
    private static final class Declining extends PlayerControllerAi {
        Declining(Game game, Player player) {
            super(game, player, player.getLobbyPlayer());
        }

        @Override
        public boolean confirmTrigger(WrappedAbility wrapper) {
            try {
                Method note = PlayerControllerAi.class.getDeclaredMethod(
                        "noteEffectRecordConfirm", SpellAbility.class, boolean.class);
                note.setAccessible(true);
                return (Boolean) note.invoke(null, wrapper.getWrappedAbility(), false);
            } catch (ReflectiveOperationException e) {
                throw new AssertionError(e);
            }
        }
    }

    /**
     * The same trigger made optional and declined: its effect half says
     * declined and holds no event, the wrapper's clause no longer reported.
     */
    @Test
    void aDeclinedTriggerCreatesNothingAndSaysDeclined() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        caster.dangerouslySetController(new Declining(game, caster));

        SpellAbility stormcaller = spellOf(inHand(game, caster, "Sea Gate Stormcaller"), caster);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().addAndUnfreeze(stormcaller);
            // On the stack the card is a new object, which keeps its
            // triggers onto the battlefield: the enters trigger made a "you may".
            for (Trigger trigger : stormcaller.getHostCard().getTriggers()) {
                trigger.putParam("OptionalDecider", "You");
            }
            game.getStack().resolveStack();
            priorityWithTriggers(game);
            game.getStack().resolveStack();
            nextStep(game);
        });

        String trigger = only(all, "resolution", STORMCALLER, "trigger");
        assertEquals("declined", payload(trigger).get("outcome"), trigger);
        assertEquals(List.of(), events(trigger), trigger);
    }
}
