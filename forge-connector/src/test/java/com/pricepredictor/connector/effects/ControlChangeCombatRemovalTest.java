package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.TriggerType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A creature leaves combat when its control changes (506.4), also when the
 * change is made outside any clause: an exchange of control and a static
 * gaining or losing control only set the new controller, and Forge's zone
 * correction in the static check after the clause takes the creature out of
 * combat. Each is one {@code removed_from_combat} naming the creature, in
 * the half of the resolution whose static check made it, as the {@code
 * GainControl} clause's own removal ({@link CombatRemovalTest}) is.
 */
@ExtendWith(ForgeExtension.class)
class ControlChangeCombatRemovalTest {

    private static final String WHY = "for the resolution hook";

    @TempDir
    Path tempDir;

    /** A game in the declare-blockers step of the first player's turn, and its two seats. */
    private record Seats(Game game, Player you, Player opponent) {

        static Seats inCombat() {
            Game game = TestBoard.startedTwoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
            return new Seats(game, game.getPlayers().get(0), game.getPlayers().get(1));
        }

        Card yours(String name) {
            return TestBoard.inPlay(game, you, name);
        }

        Card theirs(String name) {
            return TestBoard.inPlay(game, opponent, name);
        }

        /** {@code attacker} attacking the player it does not belong to, unblocked. */
        Combat attacking(Card attacker) {
            Combat combat = new Combat(you);
            combat.addAttacker(attacker, opponent);
            combat.setBlocked(attacker, false);
            game.getPhaseHandler().setCombat(combat);
            return combat;
        }

        /** {@code name} from {@code caster}'s hand, through the stack zone, cast at {@code targets}. */
        SpellAbility castAt(Player caster, String name, Card... targets) {
            Card onStack = game.getAction().moveToStack(TestBoard.inHand(game, caster, name), null);
            SpellAbility spell = TestBoard.spellOf(onStack, caster);
            spell.resetTargets();
            for (Card target : targets) {
                spell.getTargets().add(target);
            }
            return spell;
        }
    }

    /** The records of {@code spell} put on the stack and resolved, and the next step entered. */
    private List<String> resolving(Seats seats, SpellAbility spell) throws Throwable {
        Game game = seats.game();
        return ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().add(spell);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });
    }

    /** The record's {@code removed_from_combat} events. */
    private static List<Map<String, Object>> removals(String record) {
        return events(record).stream()
                .filter(event -> "removed_from_combat".equals(event.get("type"))).toList();
    }

    /** One {@code removed_from_combat} in {@code half}, naming {@code creature}, the root clause's. */
    private static void removedOnce(String half, Card creature) {
        List<Map<String, Object>> removals = removals(half);
        assertEquals(1, removals.size(), half);
        assertEquals(List.of(Refs.of(creature)), removals.get(0).get("subjects"), half);
        assertEquals("root", removals.get(0).get("attributed_to"), half);
    }

    /**
     * Switcheroo exchanging your attacking Grizzly Bears for the defender's
     * Hill Giant: the bears change controller and leave combat, once; the
     * giant, in no combat, leaves none.
     */
    @Test
    void anAttackerExchangedAwayIsRemovedFromCombat() throws Throwable {
        Seats seats = Seats.inCombat();
        Card giant = seats.theirs("Hill Giant");
        yourAttackingBearsAreTaken(seats, bears -> seats.castAt(seats.you(), "Switcheroo", bears, giant),
                "cardsfolder/s/switcheroo.txt");
        assertEquals(seats.you(), giant.getController(), "the premise: the exchange took place");
    }

    /**
     * Control Magic entering on the defender's Hill Giant, which blocks your
     * Grizzly Bears: its static takes the giant, which leaves combat, once.
     */
    @Test
    void aBlockerControlMagicTakesIsRemovedFromCombat() throws Throwable {
        Seats seats = Seats.inCombat();
        Card bears = seats.yours("Grizzly Bears");
        Card giant = seats.theirs("Hill Giant");
        Combat combat = TestBoard.blockedCombat(new TestBoard.CombatSides(seats.game(), seats.you(), seats.opponent()), bears, giant);

        List<String> records = resolving(seats, seats.castAt(seats.you(), "Control Magic", giant));

        assertEquals(seats.you(), giant.getController(), "the premise: the static took the giant");
        assertFalse(combat.isBlocking(giant), "the premise: the change removed it");
        assertTrue(combat.isAttacking(bears), "the premise: the bears still attack");
        removedOnce(only(records, "resolution", "cardsfolder/c/control_magic.txt"), giant);
    }

    /**
     * The opponent's Disenchant destroying the Control Magic on the Hill
     * Giant that attacks them for you: the steal ends, and the giant goes
     * back to them and leaves combat, once, in Disenchant's half.
     */
    @Test
    void anAttackerWhoseStealEndsIsRemovedFromCombat() throws Throwable {
        Seats seats = Seats.inCombat();
        Card giant = seats.theirs("Hill Giant");
        Card magic = seats.yours("Control Magic");
        magic.attachToEntity(giant, null);
        seats.game().getAction().checkStateEffects(false);
        assertEquals(seats.you(), giant.getController(), "the premise: you control the giant");
        Combat combat = seats.attacking(giant);

        List<String> records = resolving(seats, seats.castAt(seats.opponent(), "Disenchant", magic));

        assertEquals(seats.opponent(), giant.getController(), "the premise: the steal ended");
        assertFalse(combat.isAttacking(giant), "the premise: the change removed it");
        removedOnce(only(records, "resolution", "cardsfolder/d/disenchant.txt"), giant);
    }

    /**
     * Act of Treason on your attacking Grizzly Bears: its {@code GainControl}
     * clause corrects the zone itself, and its clause rule tells the removal,
     * so the collector tells it once.
     */
    @Test
    void aGainControlClausesRemovalIsToldOnce() throws Throwable {
        Seats seats = Seats.inCombat();
        yourAttackingBearsAreTaken(seats, bears -> seats.castAt(seats.opponent(), "Act of Treason", bears),
                "cardsfolder/a/act_of_treason.txt");
    }

    /**
     * The opponent's tapped Tahngarth, First Mate, its "whenever an opponent
     * attacks" trigger put on the stack by the engine when your Grizzly Bears
     * attack, and accepted: its one {@code GainControl} clause gives you
     * control of Tahngarth until end of combat. The clause sets the controller
     * and corrects the zone and publishes no stats change, so its {@code
     * control_change} is read where the zone is corrected: the resolution's
     * half holds it, the clause's, {@code root}.
     */
    @Test
    void aRealTriggersControlChangeIsInItsRecord() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS,
                new forge.ai.LobbyPlayerAi("a", null), DecliningSeat.accepting("b"));
        game.setAge(forge.game.GameStage.Play);
        Seats seats = new Seats(game, game.getPlayers().get(0), game.getPlayers().get(1));
        Card bears = seats.yours("Grizzly Bears");
        Card tahngarth = seats.theirs("Tahngarth, First Mate");
        tahngarth.tap(true, null, null);
        Combat combat = seats.attacking(bears);
        game.getTriggerHandler().resetActiveTriggers();

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Map<AbilityKey, Object> params = AbilityKey.newMap();
            params.put(AbilityKey.Attackers, combat.getAttackers());
            params.put(AbilityKey.AttackingPlayer, seats.you());
            params.put(AbilityKey.AttackedTarget, List.of(seats.opponent()));
            game.getTriggerHandler().runTrigger(TriggerType.AttackersDeclared, params, false);
            game.getStack().addAllTriggeredAbilitiesToStack();
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        assertEquals(seats.you(), tahngarth.getController(), "the premise: the trigger took Tahngarth");
        String half = only(records, "resolution", "cardsfolder/t/tahngarth_first_mate.txt");
        List<Map<String, Object>> changes = events(half).stream()
                .filter(event -> "control_change".equals(event.get("type"))).toList();
        assertEquals(1, changes.size(), half);
        assertEquals(List.of(Refs.of(tahngarth)), changes.get(0).get("subjects"), half);
        assertEquals("root", changes.get(0).get("attributed_to"), half);
    }

    /**
     * Your Grizzly Bears, attacking, taken by the opponent through {@code
     * play}: they change controller and leave combat, once, in the half of
     * {@code script}.
     */
    private void yourAttackingBearsAreTaken(
            Seats seats, Function<Card, SpellAbility> play, String script) throws Throwable {
        Card bears = seats.yours("Grizzly Bears");
        Combat combat = seats.attacking(bears);

        List<String> records = resolving(seats, play.apply(bears));

        assertEquals(seats.opponent(), bears.getController(), "the premise: control changed");
        assertFalse(combat.isAttacking(bears), "the premise: the change removed it");
        removedOnce(only(records, "resolution", script), bears);
    }
}
