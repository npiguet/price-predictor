package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.event.EventValueChangeType;
import forge.game.event.GameEventZone;

import java.util.Optional;
import java.util.stream.Stream;

/**
 * A creature a change of control takes out of combat (506.4), seen where
 * Forge corrects its zone.
 *
 * <p>An exchange of control and a static gaining or losing control (Control
 * Magic entering or leaving) only set the new controller. {@code
 * GameAction.controllerChangeZoneCorrection}, in the static check after the
 * clause, moves the permanent to its new controller's battlefield and then
 * calls {@code Combat.removeFromCombat}; neither publishes anything of its
 * own, but the move publishes the battlefield's {@code GameEventZone}. A
 * combatant added to a battlefield is that move, since every other move onto
 * the battlefield takes the card out of combat before it leaves its zone
 * ({@code GameAction.changeZone}), so its removal is filed there, on the
 * condition that the creature is out of that combat by the time its record
 * is written.
 *
 * <p>A clause whose own rule tells every combatant leaving combat while it
 * runs ({@link CombatRemovalApiRules#tellsEveryCombatRemoval}: a {@code GainControl},
 * which corrects the zone itself, a {@code Regeneration}, a {@code Phases})
 * keeps that telling, and nothing is filed here meanwhile.
 */
final class ControlCorrection {

    private ControlCorrection() {
    }

    /** The {@code removed_from_combat} a battlefield's zone event tells, or empty. */
    static Optional<EffectEvent> removalAt(Game game, GameEventZone event) {
        if (event.mode() != EventValueChangeType.Added || event.card() == null) {
            return Optional.empty();
        }
        Combat combat = game.getCombat();
        Card combatant = combatantOf(combat, event.card().getId());
        if (combatant == null || CombatRemovalApiRules.tellsEveryCombatRemoval(PatchHooks.currentSubAbility())) {
            return Optional.empty();
        }
        return Optional.of(new EffectEvent(EventType.REMOVED_FROM_COMBAT)
                .subject(Refs.of(combatant))
                .onlyIf(() -> !CombatShape.isCombatant(combat, combatant)));
    }

    /** The attacker or blocker of {@code combat} whose id is {@code id}, or null. */
    private static Card combatantOf(Combat combat, int id) {
        if (combat == null) {
            return null;
        }
        return Stream.concat(combat.getAttackers().stream(), combat.getAllBlockers().stream())
                .filter(card -> card.getId() == id)
                .findFirst().orElse(null);
    }
}
