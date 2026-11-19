package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.TargetingClauses;
import forge.game.GameObject;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;

import java.util.List;

/**
 * The random seat does not offer an attach line whose only legal target is
 * what its host is already attached to.
 *
 * <p>Attaching to what it is attached to does nothing (701.3b), and the seat
 * never passes while a non-mana ability is playable, so an Equip with one
 * creature on the board would be drawn again and again, each a record with no
 * event. An Equip that can move the Equipment to another creature is not
 * touched, nor is any other activation.
 */
final class RepeatAttachGate {

    private RepeatAttachGate() {
    }

    /** Whether {@code sa} is an attach whose every legal target is its host's current attachment. */
    static boolean attachesOnlyToWhatItIsOn(SpellAbility sa) {
        if (sa.getApi() != ApiType.Attach || !sa.isActivatedAbility()) {
            return false;
        }
        Card host = sa.getHostCard();
        return host != null && host.isAttachedToEntity() && onlyTargets(sa, host.getEntityAttachedTo());
    }

    private static boolean onlyTargets(SpellAbility sa, GameObject attachedTo) {
        List<GameObject> targets = TargetingClauses.candidatesOf(sa);
        return !targets.isEmpty() && targets.stream().allMatch(attachedTo::equals);
    }
}
