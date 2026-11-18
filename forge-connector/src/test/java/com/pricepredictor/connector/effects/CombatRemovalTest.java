package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * A creature leaves combat as part of what a clause does (506.4): it
 * regenerates (701.15), changes controller, or phases out (702.26). Each is
 * a {@code removed_from_combat} naming the creature, as an explicit removal
 * is, read on a real board after the clause's own effect ran.
 */
@ExtendWith(ForgeExtension.class)
class CombatRemovalTest {

    /** The clause's rule observed, its own effect run, its events built: what the clause hooks bracket. */
    private static List<EffectEvent> resolved(SpellAbility clause) {
        ApiEvents.ClauseObservation<?> observed = ApiEvents.observe(clause, ApiEvents::ruleFor);
        clause.getApi().getSpellEffect().resolve(clause);
        return observed.finish();
    }

    /** {@code attacker} of yours attacking the opponent, unblocked, in the declare-blockers step. */
    private static Combat attacking(CharacterizationBoard board, Card attacker) {
        Combat combat = new Combat(board.you);
        combat.addAttacker(attacker, board.opponent);
        combat.setBlocked(attacker, false);
        board.game.getPhaseHandler().setCombat(combat);
        return combat;
    }

    private static void assertRemoved(Card creature, List<EffectEvent> events) {
        assertEquals(1, events.size(), events.toString());
        assertEquals(EventType.REMOVED_FROM_COMBAT, events.get(0).type());
        assertEquals(List.of(Refs.of(creature)), events.get(0).subjects());
    }

    /**
     * Drudge Skeletons' shield: its {@code DB$ Regeneration | Defined$
     * ReplacedCard}, the ability the shield's replacement runs in place of
     * the destruction.
     */
    @Test
    void aRegeneratedAttackerIsRemovedFromCombat() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card skeletons = TestBoard.inPlay(board.game, board.you, "Drudge Skeletons");
        Combat combat = attacking(board, skeletons);
        board.resolve(board.ability(skeletons, ApiType.Regenerate), board.you);
        Card shield = board.game.getCardsIn(ZoneType.Command).get(0);
        SpellAbility regeneration = shield.getReplacementEffects().get(0).getOverridingAbility();
        regeneration.setActivatingPlayer(board.you);
        regeneration.setReplacingObject(AbilityKey.Card, skeletons);

        List<EffectEvent> events = resolved(regeneration);

        assertFalse(combat.isAttacking(skeletons), "the premise: regeneration removed it");
        assertRemoved(skeletons, events);
    }

    /** Act of Treason: {@code SP$ GainControl | ValidTgts$ Creature | ...}, cast on an attacker. */
    @Test
    void anAttackerWhoseControlChangedIsRemovedFromCombat() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card bears = TestBoard.inPlay(board.game, board.you, "Grizzly Bears");
        Combat combat = attacking(board, bears);
        SpellAbility treason = board.spell(board.hand("Act of Treason", board.opponent), board.opponent);
        treason.resetTargets();
        treason.getTargets().add(bears);

        List<EffectEvent> events = resolved(treason);

        assertEquals(board.opponent, bears.getController(), "the premise: control changed");
        assertFalse(combat.isAttacking(bears), "the premise: the change removed it");
        assertRemoved(bears, events);
    }

    /** A control change of a creature not in combat removes nobody. */
    @Test
    void aControlChangeOutsideCombatRemovesNobody() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card attacker = TestBoard.inPlay(board.game, board.you, "Hill Giant");
        attacking(board, attacker);
        Card bears = TestBoard.inPlay(board.game, board.you, "Grizzly Bears");
        SpellAbility treason = board.spell(board.hand("Act of Treason", board.opponent), board.opponent);
        treason.resetTargets();
        treason.getTargets().add(bears);

        assertEquals(List.of(), resolved(treason));
    }

    /** Blink Dog: {@code AB$ Phases | Defined$ Self}, activated while it attacks. */
    @Test
    void anAttackerThatPhasedOutIsRemovedFromCombat() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card dog = TestBoard.inPlay(board.game, board.you, "Blink Dog");
        Combat combat = attacking(board, dog);
        SpellAbility phase = board.ability(dog, ApiType.Phases);
        phase.setActivatingPlayer(board.you);

        List<EffectEvent> events = resolved(phase);

        assertFalse(combat.isAttacking(dog), "the premise: phasing out removed it");
        assertRemoved(dog, events);
    }
}
