package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The {@link CombatRemovalApiRules}: a combat phase ended, a creature
 * removed from combat or reselected to attack another defender.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsCombatRemovalTest {

    // The combat tests below build their cards in a game of their own, not
    // TestCards.game(): combat state (PhaseHandler.setCombat) is per-Game and
    // the shared game is used by every other ApiEvents test, so mutating
    // combat on it would leak across tests that never asked for it.

    // ── removed_from_combat: RemoveFromCombat ───────────────────────────────

    /** A combat on {@code game} in which every one of {@code attackers} attacks {@code defender}. */
    private static Combat combatOf(Game game, int attackerControllerId, Player defender, Card... attackers) {
        Combat combat = new Combat(new Player("attacker", game, attackerControllerId));
        game.getPhaseHandler().setCombat(combat);
        for (Card attacker : attackers) {
            combat.addAttacker(attacker, defender);
        }
        return combat;
    }

    /** {@code script} built on {@code host}, with exactly {@code targets} chosen. */
    private static SpellAbility targeting(String script, Card host, Card... targets) {
        SpellAbility sa = AbilityFactory.getAbility(script, host);
        sa.resetTargets();
        for (Card target : targets) {
            sa.getTargets().add(target);
        }
        return sa;
    }

    /**
     * A {@code RemoveFromCombat} clause hosted by {@code c1}, targeting both
     * {@code c1} and {@code c2}, attackers in {@code combat}.
     */
    private record RemovalClause(Card c1, Card c2, Combat combat, SpellAbility sa) {
    }

    /**
     * Grizzly Bears and Runeclaw Bear attacking in a game of their own, and a
     * {@code RemoveFromCombat} clause targeting both.
     */
    private static RemovalClause removalOfTwoAttackers(int attackerControllerId, int defenderId) {
        Game game = TestBoard.anotherGame();
        Card c1 = TestBoard.cardIn(game, "Grizzly Bears");
        Card c2 = TestBoard.cardIn(game, "Runeclaw Bear");
        Combat combat = combatOf(game, attackerControllerId, new Player("defender", game, defenderId), c1, c2);
        SpellAbility sa = targeting("DB$ RemoveFromCombat | ValidTgts$ Creature", c1, c1, c2);
        return new RemovalClause(c1, c2, combat, sa);
    }

    @Test
    void endingTheCombatPhaseNamesTheAttackingPlayer() {
        Game game = TestBoard.anotherGame();
        Card attacker = TestBoard.cardIn(game, "Grizzly Bears");
        Combat combat = combatOf(game, 93205, new Player("defender", game, 93206), attacker);
        Player attacking = combat.getAttackingPlayer();
        SpellAbility sa = AbilityFactory.getAbility("SP$ EndCombatPhase", attacker);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        game.getPhaseHandler().setCombat(null);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(EventType.COMBAT_ENDED, event.type());
        assertEquals(List.of(Refs.of(attacking)), event.subjects());
    }

    @Test
    void endingTheCombatPhaseOutsideCombatEmitsNothing() {
        Game game = TestBoard.anotherGame();
        Card host = TestBoard.cardIn(game, "Grizzly Bears");
        game.getPhaseHandler().setCombat(null);
        SpellAbility sa = AbilityFactory.getAbility("SP$ EndCombatPhase", host);

        assertNull(TestCards.onlyEvent(observe(sa)));
    }

    /**
     * Two real attacking creatures, both removed -- not a single-entry
     * fixture where "first only" and "both" would agree.
     */
    @Test
    void removeFromCombatNamesEveryCreatureActuallyRemoved() {
        RemovalClause clause = removalOfTwoAttackers(93201, 93202);
        Card c1 = clause.c1();
        Card c2 = clause.c2();

        ApiEvents.ClauseObservation<?> observed = observe(clause.sa());
        clause.combat().removeFromCombat(c1);
        clause.combat().removeFromCombat(c2);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(EventType.REMOVED_FROM_COMBAT, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
                event.subjects());
    }

    /**
     * Only one of the two targets is genuinely removed -- {@code
     * RemoveFromCombatEffect}'s own per-target guards (stale LKI, not in
     * play) leave the other untouched, and this must report exactly the one
     * that actually left, not both and not neither.
     */
    @Test
    void removeFromCombatNamesOnlyTheCreatureThatWasActuallyRemoved() {
        RemovalClause clause = removalOfTwoAttackers(93203, 93204);
        Card c1 = clause.c1();

        ApiEvents.ClauseObservation<?> observed = observe(clause.sa());
        clause.combat().removeFromCombat(c1);
        // c2 stays attacking -- its own removal was skipped by a guard this
        // rule does not replicate.
        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(List.of(Refs.of(c1)), event.subjects());
    }

    // ── removed_from_combat: ChangeCombatants ───────────────────────────────

    /**
     * {@code portal_manipulator.txt} is {@code DB$ ChangeCombatants |
     * ValidTgts$ Creature.attacking | TargetMin$ 0 | TargetMax$ MaxTargets} --
     * any number of attackers, not just one, so two are reselected here: a
     * single-entry fixture would not tell "every attacker" from "the first".
     * The plausible wrong implementation this falsifies is a {@code
     * reselectedAttackers} that returns on the first matching entry instead of
     * continuing the loop.
     *
     * <p>{@code addToCombat}'s reselection branch calls {@code
     * combat.removeFromCombat(c)} immediately before re-adding the same
     * creature against the new defender --
     * simulated here with the same two public {@code Combat} calls, not a
     * hand-rolled approximation.
     *
     * <p>{@code ValidTgts$ Creature.attacking} on the ability text, and a
     * host distinct from the attackers, are load-bearing, not decoration:
     * {@code SpellAbility.usesTargeting()} is exactly {@code
     * targetRestrictions != null}, which {@code AbilityFactory} only sets when
     * the script declares {@code ValidTgts$}. Without it, {@code
     * affectedCards} falls through to its {@code Defined$} branch (default
     * {@code "Self"} -- the host), so a host that is itself an attacker would
     * make the host-fallback and the manually-added targets name the same card
     * and the test pass for a reason that had nothing to do with targeting.
     */
    @Test
    void changeCombatantsNamesEveryReselectedAttackerNotJustTheFirst() {
        Game game = TestBoard.anotherGame();
        Player oldDefender = new Player("old-defender", game, 93307);
        Player newDefender = new Player("new-defender", game, 93308);
        Card host = TestBoard.cardIn(game, "Mountain");
        Card c1 = TestBoard.cardIn(game, "Grizzly Bears");
        Card c2 = TestBoard.cardIn(game, "Runeclaw Bear");
        Combat combat = combatOf(game, 93306, oldDefender, c1, c2);
        SpellAbility sa = targeting(
                "DB$ ChangeCombatants | ValidTgts$ Creature.attacking | Attacking$ True", host, c1, c2);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        combat.removeFromCombat(c1);
        combat.addAttacker(c1, newDefender);
        combat.removeFromCombat(c2);
        combat.addAttacker(c2, newDefender);
        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(EventType.REMOVED_FROM_COMBAT, event.type());
        assertEquals(
                List.of(Refs.of(c1), Refs.of(c2)),
                event.subjects());
    }

    /**
     * The player declines the optional reselection (or reselects the same
     * defender): {@code addToCombat} never calls {@code removeFromCombat} at
     * all in that case, so nothing should be reported. The plausible wrong
     * implementation this falsifies is a rule that reports every clause
     * targeting an attacker, whether or not a reselection actually happened.
     */
    @Test
    void changeCombatantsNamesNoOneWhenTheDefenderDidNotChange() {
        Game game = TestBoard.anotherGame();
        Card host = TestBoard.cardIn(game, "Mountain");
        Card c = TestBoard.cardIn(game, "Grizzly Bears");
        combatOf(game, 93304, new Player("defender", game, 93305), c);
        SpellAbility sa = targeting(
                "DB$ ChangeCombatants | ValidTgts$ Creature.attacking | Attacking$ True", host, c);

        ApiEvents.ClauseObservation<?> observed = observe(sa);
        // Nothing changes: declined, or reselected the same defender.
        EffectEvent event = TestCards.onlyEvent(observed);

        assertNull(event, "the defender never changed, so no removal happened");
    }
}
