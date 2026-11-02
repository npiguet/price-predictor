package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.trigger.TriggerHandler;
import forge.game.trigger.TriggerType;

import java.util.List;

/**
 * {@code remove_permanent}: take a permanent off the battlefield into no zone
 * (FR-011a).
 *
 * <p>It leaves combat, what is attached to it is unattached, it is unattached
 * from what it is attached to, its soulbond pairing ends, and it is taken out
 * of the battlefield's card list. It goes to no zone: no leaves-the-battlefield
 * or dies trigger fires (a Blood Artist or a Doomed Dissenter fires nothing),
 * no graveyard or exile gains a card, and no per-turn count moves. An Aura or
 * Equipment it held stays on the battlefield unattached; an Aura attached to
 * nothing then goes to its owner's graveyard by the state-based check once the
 * fork plays on, the game's consequence of the board rather than the edit's.
 *
 * <p>The unattaching goes through {@code Card.unattachFromEntity}, which also
 * ends a bestowed Aura's creature form and runs the until-unattached
 * commands, with the becomes-unattached trigger (Grafted Exoskeleton)
 * suppressed for the span. What is linked to the permanent otherwise — a card
 * exiled until it leaves (Banishing Light), what it remembers — stays as it
 * is: in no zone, the permanent never left the battlefield as the game counts
 * it.
 *
 * @param permanent the fork's permanent
 */
public record RemovePermanentOperation(Card permanent) implements BoardEdit {

    @Override
    public ForkOperation operation() {
        return ForkOperation.REMOVE_PERMANENT;
    }

    @Override
    public void apply(Game fork) {
        takeOff(fork, BoardEdit.onBattlefield(permanent));
        BoardEdit.settle(fork);
    }

    /**
     * The removing step, shared with {@link ReplaceCardOperation}: out of
     * combat, unattached both ways, unpaired, out of the battlefield and into
     * no zone.
     */
    static void takeOff(Game fork, Card permanent) {
        Combat combat = fork.getCombat();
        if (combat != null) {
            combat.removeFromCombat(permanent);
        }
        unattachSilently(fork, permanent);
        if (permanent.isPaired()) {
            Card partner = permanent.getPairedWith();
            partner.setPairedWith(null);
            permanent.setPairedWith(null);
        }
        permanent.getZone().remove(permanent);
        permanent.setZone(null);
    }

    /** Unattach what is attached to {@code permanent}, and it from its host, firing no trigger. */
    private static void unattachSilently(Game fork, Card permanent) {
        TriggerHandler triggers = fork.getTriggerHandler();
        boolean wasSuppressed = triggers.isTriggerSuppressed(TriggerType.Unattached);
        if (!wasSuppressed) {
            triggers.suppressMode(TriggerType.Unattached);
        }
        try {
            for (Card attached : List.copyOf(permanent.getAttachedCards())) {
                attached.unattachFromEntity(permanent);
            }
            GameEntity host = permanent.getEntityAttachedTo();
            if (host != null) {
                permanent.unattachFromEntity(host);
            }
        } finally {
            if (!wasSuppressed) {
                triggers.clearSuppression(TriggerType.Unattached);
            }
        }
    }
}
