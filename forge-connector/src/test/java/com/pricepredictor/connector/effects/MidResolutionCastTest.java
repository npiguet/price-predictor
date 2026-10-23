package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CounterEnumType;
import forge.game.cost.CostPayment;
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

import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static com.pricepredictor.connector.effects.RecordLines.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.linkOf;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static com.pricepredictor.connector.effects.RecordLines.withEventOfType;
import static com.pricepredictor.connector.effects.TestBoard.nextStep;
import static com.pricepredictor.connector.effects.TestBoard.priority;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A spell cast while another ability is resolving.
 *
 * <p>Forge freezes the stack while an ability resolves, and a spell a
 * {@code Play} effect casts is the one thing that goes onto it anyway: its
 * {@code GameEventSpellAbilityCast} arrives in the middle of the resolution,
 * the resolution then finishes and publishes {@code GameEventSpellResolved},
 * and the cast spell resolves later, after a priority. The resolution keeps
 * its bracket until it ends; the cast's bracket opens once the cast spell
 * starts resolving, and the payment of its costs is in neither half.
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

    /** The card's first trigger, wrapped as the trigger handler puts it on the stack. */
    private static WrappedAbility triggerOf(Card card, Player controller) {
        Trigger trigger = card.getTriggers().iterator().next();
        SpellAbility execute = trigger.ensureAbility();
        execute.setActivatingPlayer(controller);
        WrappedAbility wrapped = new WrappedAbility(trigger, execute, controller);
        wrapped.setActivatingPlayer(controller);
        return wrapped;
    }

    /** Tetsuo on the battlefield, equipped with a mana value 1 Equipment. */
    private static Card equippedTetsuo(Game game, Player owner) {
        Card tetsuo = TestBoard.inPlay(game, owner, "Tetsuo, Imperial Champion");
        Card bonesplitter = TestBoard.inPlay(game, owner, "Bonesplitter");
        bonesplitter.attachToEntity(tetsuo, null);
        return tetsuo;
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
        // Casting the Bolt moved it to the stack, which no event records,
        // and its free cost paid nothing: the mode did nothing observable.
        assertEquals(List.of(), eventTypes(charmHalf), charmHalf);

        assertNotNull(linkOf(boltCost), boltCost);
        assertEquals(linkOf(boltCost), linkOf(boltHalf));
        assertFalse(boltHalf.contains("\"option\""),
                "the Bolt inherits no option of the charm's: " + boltHalf);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(boltHalf), boltHalf);
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
            collectors.clauses().clauseHandler().invoke(null, HookShapes.clauseResolving(), new Object[]{trigger});
            collectors.clauses().clauseHandler().invoke(null, HookShapes.clauseResolving(), new Object[]{clones.get(0)});
            bracket.recordClauseEvent(new EffectEvent(EventType.KEYWORD_CHANGE));
            bracket.beginBracket(bolt);
            bracket.recordClauseEvent(new EffectEvent(EventType.COUNTER_CHANGE));
            bracket.endBracket(trigger.getId(), false);
            game.getStack().setResolving(false);
            priority(game);
            // The Bolt's own resolution, which refuses its only clause.
            game.getStack().setResolving(true);
            bracket.noteDeclined();
            bracket.endBracket(bolt.getId(), false);
            game.getStack().setResolving(false);
            bracket.onNextAction();
        });

        String charmHalf = only(all, "resolution", TETSUO);
        assertEquals(List.of("keyword_change", "counter_change"), eventTypes(charmHalf),
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
        assertEquals(List.of(), eventTypes(charmHalf), charmHalf);
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
        assertEquals(List.of("zone_change", "zone_change", "zone_change"),
                eventTypes(cascadeHalf), "the three moves and nothing of the Bolt's: " + cascadeHalf);

        String boltHalf = only(all, "resolution", BOLT);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(boltHalf),
                "nothing of the Mountain's is the Bolt's: " + boltHalf);
    }

    /**
     * Cascade's Bolt resolves after a state-based check that annihilates a
     * +1/+1 and a -1/-1 counter on the Elf. That check is the aftermath of
     * the cascade's resolution, not the Bolt's: the Bolt's bracket opens when
     * the Bolt starts resolving, so the counters are not in the Bolt's half.
     */
    @Test
    void aStateBasedCheckBeforeTheCastSpellResolvesIsNotTheCastSpells() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elf = TestBoard.inPlay(game, caster, "Bloodbraid Elf");
        Card bolt = game.getAction().moveToLibrary(
                TestBoard.card(game, caster, "Lightning Bolt"), 0, null);
        WrappedAbility cascade = triggerOf(elf, caster);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(cascade);
            game.getStack().resolveStack();
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(bolt)).getZoneType(),
                    "the cascade cast the Bolt");
            elf.addCounterInternal(CounterEnumType.P1P1, 1, caster, false, null, null);
            elf.addCounterInternal(CounterEnumType.M1M1, 1, caster, false, null, null);
            priority(game);
            assertEquals(0, elf.getCounters(CounterEnumType.P1P1),
                    "the check annihilated the counters");
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String boltHalf = only(all, "resolution", BOLT);
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(boltHalf), boltHalf);
    }

    /**
     * A resolution that wins the game with a cast still waiting: the win
     * arrives at the state-based check after the resolution, with nothing
     * resolving, and is written as the orphaned event it is. Driven by hand,
     * as the game ends before the waiting spell could ever resolve.
     */
    @Test
    void aWinAfterAResolutionThatCastASpellIsStillWritten() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player victim = game.getPlayers().get(1);
        Card elf = TestBoard.inPlay(game, caster, "Bloodbraid Elf");
        WrappedAbility cascade = triggerOf(elf, caster);
        SpellAbility bolt = TestBoard.inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            game.subscribeToEvents(bracket);
            bracket.beginBracket(cascade);
            game.getStack().setResolving(true);
            bracket.beginBracket(bolt);
            victim.setLife(0, null);
            bracket.endBracket(cascade.getId(), false);
            game.getStack().setResolving(false);
            priority(game);
            assertTrue(game.isGameOver(), "the state-based check ended the game");
            bracket.finishGame(Set.of(bolt.getId()));
        });

        List<String> won = withEventOfType(all, "player_won");
        assertEquals(1, won.size(), "the win is written: " + all);
        assertNull(linkOf(won.get(0)), "an orphan, nobody's half: " + won.get(0));
    }

    /**
     * A cast spell whose resolution shows nothing — no event, no mode, no
     * refusal — still writes its linked half when it resolves, with no
     * events. Driven by hand, because a spell the engine resolves always
     * publishes something the bracket hears.
     */
    @Test
    void aCastSpellThatShowsNothingWritesAnEmptyLinkedHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elf = TestBoard.inPlay(game, caster, "Bloodbraid Elf");
        WrappedAbility cascade = triggerOf(elf, caster);
        SpellAbility bolt = TestBoard.inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            game.subscribeToEvents(bracket);
            bracket.beginBracket(cascade);
            game.getStack().setResolving(true);
            bracket.beginBracket(bolt);
            bracket.endBracket(cascade.getId(), false);
            game.getStack().setResolving(false);
            priority(game);
            game.getStack().setResolving(true);
            bracket.endBracket(bolt.getId(), false);
            game.getStack().setResolving(false);
            bracket.onNextAction();
        });

        String boltCost = only(all, "activation", BOLT);
        String boltHalf = only(all, "resolution", BOLT);
        assertNotNull(linkOf(boltCost), boltCost);
        assertEquals(linkOf(boltCost), linkOf(boltHalf));
        assertEquals(List.of(), eventTypes(boltHalf), boltHalf);
    }

    /**
     * A refusal while a player pays for the spell the resolving ability casts
     * — an optional cost part of it turned down — is the cast spell's, not
     * the resolving ability's: the resolving ability, which did nothing
     * observable, still reads {@code resolved}.
     */
    @Test
    void aRefusalDuringTheCastSpellsPaymentIsNotTheResolvingAbilitys() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card elf = TestBoard.inPlay(game, caster, "Bloodbraid Elf");
        WrappedAbility cascade = triggerOf(elf, caster);
        SpellAbility bolt = TestBoard.inHand(game, caster, "Lightning Bolt").getFirstSpellAbility();
        bolt.setActivatingPlayer(caster);
        bolt.setCastFromPlayEffect(true);

        List<String> all = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(cascade);
            game.getStack().setResolving(true);
            // CostPayment.payCost, paying the Bolt that the cascade casts.
            game.costPaymentStack.push(
                    bolt.getPayCosts().getCostParts().get(0),
                    new CostPayment(bolt.getPayCosts(), bolt));
            bracket.noteDeclined();
            game.costPaymentStack.pop();
            bracket.endBracket(cascade.getId(), false);
            game.getStack().setResolving(false);
            bracket.onNextAction();
        });

        String cascadeCost = only(all, "activation", BLOODBRAID);
        assertEquals("resolved", RecordLines.outcomeOf(cascadeCost), cascadeCost);
        assertEquals(linkOf(cascadeCost), linkOf(only(all, "resolution", BLOODBRAID)));
    }

    // ── the cast spell's own costs ──────────────────────────────────────

    private static final String TEMPER = "cardsfolder/f/fiery_temper.txt";

    /**
     * A forced Kari Zev's Expertise casts the Bone Splinters in its caster's
     * hand, whose additional cost sacrifices a creature. The sacrifice pays
     * the Splinters' cost, and the fork leaves it out of the Expertise's
     * record as the live bracket does.
     */
    @Test
    void aForcedResolutionLeavesOutThePaymentOfASpellItCasts() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.inPlay(game, caster, "Llanowar Elves");
        TestBoard.inPlay(game, opponent, "Serra Angel");
        TestBoard.inHand(game, caster, "Bone Splinters");
        SpellAbility expertise = TestBoard.inHand(game, caster, "Kari Zev's Expertise")
                .getFirstSpellAbility();
        expertise.setActivatingPlayer(caster);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).probes(0).build();

        ShardRun.Fork fork = ShardRun.forked(tempDir, game, caps, 5L, expertise);

        assertTrue(fork.wrote(), "the forced resolution observed something");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String record = records.get(0);
        assertTrue(record.contains("\"interventional\":true"), record);
        assertFalse(record.contains("\"to_zone\":\"graveyard\""),
                "the sacrifice that paid for the Splinters is not the Expertise's: " + record);
    }

    /**
     * Fiery Temper's madness trigger casts it for {R}, paid with a Mountain.
     * The tap pays the Temper's cost and is in the Temper's cost half; it is
     * not the trigger's effect, whose half is empty.
     */
    @Test
    void aMadnessTriggerDoesNotCarryThePaymentOfTheSpellItCasts() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card mountain = TestBoard.inPlay(game, caster, "Mountain");
        Card temper = game.getAction().exile(
                TestBoard.inHand(game, caster, "Fiery Temper"), null, null);
        WrappedAbility madness = triggerOf(temper, caster);

        List<String> all = ShardRun.observed(tempDir, game, bracket -> {
            game.getStack().add(madness);
            game.getStack().resolveStack();
            assertTrue(mountain.isTapped(), "the Mountain paid for the Temper");
            assertEquals(ZoneType.Stack, game.getZoneOf(game.getCardState(temper)).getZoneType(),
                    "the madness trigger cast the Temper");
            priority(game);
            game.getStack().resolveStack();
            priority(game);
            nextStep(game);
        });

        String madnessHalf = only(all, "resolution", TEMPER, "keyword");
        assertEquals(List.of(), eventTypes(madnessHalf), madnessHalf);
        String temperCost = only(all, "activation", TEMPER, "spell");
        assertTrue(temperCost.contains("\"mana_by_color\":{\"R\":1}"), temperCost);
        String temperHalf = only(all, "resolution", TEMPER, "spell");
        assertEquals(List.of("damage_dealt", "life_change"), eventTypes(temperHalf), temperHalf);
    }

    /**
     * Tetsuo's cast mode casts Bone Splinters, whose additional cost
     * sacrifices a creature. The sacrifice pays the Splinters' cost; the
     * charm's half is empty.
     */
    @Test
    void aSpellsAdditionalCostIsNotTheEffectOfTheCharmThatCastIt() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card tetsuo = equippedTetsuo(game, caster);
        Card elves = TestBoard.inPlay(game, caster, "Llanowar Elves");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Serra Angel");
        Card splinters = TestBoard.inHand(game, caster, "Bone Splinters");
        WrappedAbility trigger = triggerOf(tetsuo, caster);
        TestBoard.chain(trigger.getWrappedAbility(), TETSUO_CAST_MODE);

        List<String> all = ShardRun.recordsWithClauseHook(
                tempDir, game, "to split the charm's mode", () -> {
                    game.getStack().add(trigger);
                    game.getStack().resolveStack();
                    assertEquals(ZoneType.Stack,
                            game.getZoneOf(game.getCardState(splinters)).getZoneType(),
                            "the cast mode put Bone Splinters on the stack");
                    assertEquals(ZoneType.Graveyard,
                            game.getZoneOf(game.getCardState(elves)).getZoneType(),
                            "the Elves paid the additional cost");
                    priority(game);
                    game.getStack().resolveStack();
                    priority(game);
                    nextStep(game);
                });

        String charmHalf = only(all, "resolution", TETSUO);
        assertEquals(List.of(), eventTypes(charmHalf), charmHalf);
    }
}
