package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventPlayerPriority;
import forge.game.event.GameEventTurnPhase;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.player.PlayerView;
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
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A spell cast while another ability is resolving.
 *
 * <p>Forge freezes the stack while an ability resolves, and a spell a
 * {@code Play} effect casts is the one thing that goes onto it anyway: its
 * {@code GameEventSpellAbilityCast} arrives in the middle of the resolution,
 * the resolution then finishes and publishes {@code GameEventSpellResolved},
 * and the cast spell resolves later, after a priority. The resolution keeps
 * its bracket until it ends; the cast's bracket opens at that priority.
 *
 * <p>Driven through the real engine — the stack, the resolution, the
 * {@code Play} effect and the AI's cast — with the collector on the game's
 * own bus. The priority is the one event fired by hand, as {@code
 * PhaseHandler.mainLoopStep} fires it before each state-based check.
 */
@ExtendWith(ForgeExtension.class)
class MidResolutionCastTest {

    @TempDir
    Path tempDir;

    /** Tetsuo's {@code Choices$} position of "you may cast an instant or sorcery". */
    private static final int TETSUO_CAST_MODE = 1;

    private static Method resolving() {
        return HookShapes.method(HookShapes.Clause.class, "onClauseResolving");
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

    /** What {@code PhaseHandler.mainLoopStep} does before the next player acts. */
    private static void priority(Game game) {
        PlayerView turn = PlayerView.get(game.getPhaseHandler().getPlayerTurn());
        game.fireEvent(new GameEventPlayerPriority(turn, PhaseType.MAIN1, turn));
        game.getAction().checkStateEffects(false);
    }

    /** The next real action: a step boundary, which writes the held halves. */
    private static void nextStep(Game game) {
        game.fireEvent(new GameEventTurnPhase(
                game.getPhaseHandler().getPlayerTurn(), PhaseType.MAIN2, ""));
    }

    /** Tetsuo on the battlefield, equipped with a mana value 1 Equipment. */
    private static Card equippedTetsuo(Game game, Player owner) {
        Card tetsuo = TestBoard.inPlay(game, owner, "Tetsuo, Imperial Champion");
        Card bonesplitter = TestBoard.inPlay(game, owner, "Bonesplitter");
        bonesplitter.attachToEntity(tetsuo, null);
        return tetsuo;
    }

    private static List<String> records(List<String> all, String moment) {
        List<String> found = new ArrayList<>();
        for (String record : all) {
            if (record.contains("\"moment\":\"" + moment + "\"")) {
                found.add(record);
            }
        }
        return found;
    }

    private static String scriptOf(String record) {
        int at = record.indexOf("\"script_file\":\"");
        assertTrue(at >= 0, record);
        int from = at + "\"script_file\":\"".length();
        return record.substring(from, record.indexOf('"', from));
    }

    private static String linkOf(String record) {
        int at = record.indexOf("\"link_id\":\"");
        if (at < 0) {
            return null;
        }
        int from = at + "\"link_id\":\"".length();
        return record.substring(from, record.indexOf('"', from));
    }

    /** The one record of {@code moment} acting through {@code script}. */
    private static String only(List<String> all, String moment, String script) {
        List<String> found = new ArrayList<>();
        for (String record : records(all, moment)) {
            if (scriptOf(record).equals(script)) {
                found.add(record);
            }
        }
        assertEquals(1, found.size(), moment + " through " + script + ": " + all);
        return found.get(0);
    }

    private static String zoneChange(Card card, String from, String to) {
        return "{\"type\":\"zone_change\",\"subjects\":[\"E" + card.getId()
                + "\"],\"params\":{\"from_zone\":\"" + from + "\",\"to_zone\":\"" + to + "\"";
    }

    private static final String TETSUO = "cardsfolder/t/tetsuo_imperial_champion.txt";
    private static final String BOLT = "cardsfolder/l/lightning_bolt.txt";
    private static final String BLOODBRAID = "cardsfolder/b/bloodbraid_elf.txt";

    // ── a charm whose mode casts a spell ────────────────────────────────

    /**
     * Tetsuo's trigger, cast mode chosen, casts the Lightning Bolt in its
     * controller's hand. The charm's half acts through the charm's own option
     * line under the charm's link; the Bolt's half acts through the Bolt's
     * line, with no option and none of the charm's events.
     */
    @Test
    void aCharmThatCastsASpellWritesItsOwnHalfAndTheSpellItsOwn() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card tetsuo = equippedTetsuo(game, caster);
        Card bolt = TestBoard.inHand(game, caster, "Lightning Bolt");
        WrappedAbility trigger = triggerOf(tetsuo, caster);
        TestBoard.chain(trigger.getWrappedAbility(), TETSUO_CAST_MODE);

        List<String> all = ShardRun.recordsWithClauseHook(
                tempDir, game, "to split the charm's mode", () -> {
                    game.getStack().add(trigger);
                    game.getStack().resolveStack();
                    assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                            "the cast mode put the Bolt on the stack");
                    priority(game);
                    game.getStack().resolveStack();
                    priority(game);
                    nextStep(game);
                });

        String charmCost = only(all, "activation", TETSUO);
        String charmHalf = only(all, "resolution", TETSUO);
        String boltCost = only(all, "activation", BOLT);
        String boltHalf = only(all, "resolution", BOLT);

        assertNotNull(linkOf(charmCost), "the charm's cost half links its effect half: " + charmCost);
        assertEquals(linkOf(charmCost), linkOf(charmHalf));
        assertTrue(charmHalf.contains("\"option\":" + TETSUO_CAST_MODE + "}"),
                "the charm's half acts through its cast mode: " + charmHalf);
        assertFalse(charmHalf.contains("damage_dealt"),
                "the Bolt's damage is not the charm's: " + charmHalf);

        assertNotNull(linkOf(boltCost), boltCost);
        assertEquals(linkOf(boltCost), linkOf(boltHalf));
        assertFalse(boltHalf.contains("\"option\""),
                "the Bolt inherits no option of the charm's: " + boltHalf);
        assertTrue(boltHalf.contains("damage_dealt"), boltHalf);
    }

    /**
     * Between the cast and the end of the resolution the charm is still the
     * one resolving: its events after the cast are its own, and a refusal in
     * the cast spell's resolution, which itself did nothing, reads as
     * {@code declined}. Driven by hand, because nothing in a real game both
     * declines the cast spell's only clause and lets the test see when.
     */
    @Test
    void aSpellCastMidResolutionStartsWithNothingOfTheCharms() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card tetsuo = equippedTetsuo(game, caster);
        WrappedAbility trigger = triggerOf(tetsuo, caster);
        List<AbilitySub> clones = TestBoard.chain(trigger.getWrappedAbility(), TETSUO_CAST_MODE);
        SpellAbility bolt = TestBoard.inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            game.subscribeToEvents(bracket);
            bracket.beginBracket(trigger);
            // MagicStack.resolveStack: resolving from here until finishResolving.
            game.getStack().setResolving(true);
            collectors.clauses().clauseHandler().invoke(null, resolving(), new Object[]{trigger});
            collectors.clauses().clauseHandler().invoke(null, resolving(), new Object[]{clones.get(0)});
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.KEYWORD_CHANGE));
            bracket.beginBracket(bolt);
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.COUNTER_CHANGE));
            bracket.endBracket(trigger.getId(), false);
            game.getStack().setResolving(false);
            priority(game);
            bracket.noteDeclined();
            bracket.endBracket(bolt.getId(), false);
            bracket.onNextAction();
        });

        String charmHalf = only(all, "resolution", TETSUO);
        assertTrue(charmHalf.contains("keyword_change") && charmHalf.contains("counter_change"),
                "both of the charm's events, before and after the cast, are the charm's: "
                        + charmHalf);
        String boltCost = only(all, "activation", BOLT);
        assertTrue(boltCost.contains("\"outcome\":\"declined\""),
                "the Bolt's refusal is judged on the Bolt's own events: " + boltCost);
        assertEquals(1, records(all, "resolution").size(), "a refusal writes no half: " + all);
    }

    /**
     * A spell cast mid-resolution that fizzles takes nothing of the charm's
     * with it: the charm's half has been written at its own resolution.
     */
    @Test
    void aFizzledSpellCastMidResolutionLeavesTheCharmsHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.onBattlefield(game, game.getPlayers().get(1), "Grizzly Bears");
        Card tetsuo = equippedTetsuo(game, caster);
        TestBoard.inHand(game, caster, "Lightning Bolt");
        WrappedAbility trigger = triggerOf(tetsuo, caster);
        TestBoard.chain(trigger.getWrappedAbility(), TETSUO_CAST_MODE);

        List<String> all = ShardRun.recordsWithClauseHook(
                tempDir, game, "to split the charm's mode", () -> {
                    game.getStack().add(trigger);
                    game.getStack().resolveStack();
                    // Wherever the AI aimed it, the Bolt now points at the
                    // bears, which leave before it resolves.
                    SpellAbility cast = game.getStack().peekAbility();
                    cast.resetTargets();
                    cast.getTargets().add(bears);
                    priority(game);
                    game.getAction().exile(bears, null, null);
                    game.getStack().resolveStack();
                    priority(game);
                    nextStep(game);
                });

        String boltCost = only(all, "activation", BOLT);
        assertTrue(boltCost.contains("\"outcome\":\"fizzled\""), boltCost);
        String charmCost = only(all, "activation", TETSUO);
        String charmHalf = only(all, "resolution", TETSUO);
        assertEquals(linkOf(charmCost), linkOf(charmHalf));
        assertTrue(charmHalf.contains("\"option\":" + TETSUO_CAST_MODE + "}"), charmHalf);
    }

    // ── cascade ─────────────────────────────────────────────────────────

    /**
     * Bloodbraid Elf's cascade exiles a Mountain and a Lightning Bolt, casts
     * the Bolt and puts the Mountain on the bottom. The trigger writes its own
     * effect half with all three moves; the Bolt's half holds its damage and
     * not the Mountain's return, which happened before the Bolt resolved.
     */
    @Test
    void aCascadeTriggerKeepsItsHalfAndTheCascadedSpellGetsItsOwn() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elf = TestBoard.inPlay(game, caster, "Bloodbraid Elf");
        Card bolt = game.getAction().moveToLibrary(
                TestBoard.card(game, caster, "Lightning Bolt"), 0, null);
        Card mountain = game.getAction().moveToLibrary(
                TestBoard.card(game, caster, "Mountain"), 0, null);
        assertEquals(mountain, caster.getCardsIn(ZoneType.Library).get(0), "the Mountain is on top");
        WrappedAbility cascade = triggerOf(elf, caster);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(cascade);
            game.getStack().resolveStack();
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                    "the cascade cast the Bolt");
            priority(game);
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String cascadeCost = only(all, "activation", BLOODBRAID);
        String cascadeHalf = only(all, "resolution", BLOODBRAID);
        assertNotNull(linkOf(cascadeCost), "the trigger issues its link: " + cascadeCost);
        assertEquals(linkOf(cascadeCost), linkOf(cascadeHalf));
        assertTrue(cascadeHalf.contains(zoneChange(mountain, "library", "exile")), cascadeHalf);
        assertTrue(cascadeHalf.contains(zoneChange(bolt, "library", "exile")), cascadeHalf);
        assertTrue(cascadeHalf.contains(zoneChange(mountain, "exile", "library")),
                "the Mountain's return is the trigger's: " + cascadeHalf);

        String boltHalf = only(all, "resolution", BOLT);
        assertTrue(boltHalf.contains("damage_dealt"), boltHalf);
        assertFalse(boltHalf.contains("\"subjects\":[\"E" + mountain.getId() + "\"]"),
                "nothing of the Mountain's is the Bolt's: " + boltHalf);
    }
}
