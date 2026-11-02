package com.pricepredictor.connector.effects.directed.operations;

import com.pricepredictor.connector.effects.directed.ForkOperation;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.zone.ZoneType;

/**
 * One scenario variant's change, applied to a fork's copy of a live game
 * (FR-011): the objects it names are the fork's, already mapped from the live
 * game.
 *
 * <p>Every edit is <b>silent</b> (FR-011a, D2). A variant describes a
 * different board, not an event on the board, so an edit raises none of the
 * events the same change would raise in play: no trigger, no replacement, no
 * zone-change event and no per-turn count. None goes through
 * {@code GameAction.moveTo}, which would fire the zone-change triggers and
 * replacements and add to the counts (cards drawn, permanents put into a
 * graveyard, lands played) a variant does not change. What the changed board
 * then does — an Aura left unattached going to its owner's graveyard, a
 * creature at 0 toughness dying — happens when the fork plays on, as the
 * game's own consequence of the board.
 *
 * <p>An edit that cannot describe its variant on this board — a card no longer
 * where the precondition found it, a keyword the card does not carry, a name
 * the card database lacks — throws, and {@code Forks.ForkRun} discards the
 * fork rather than record a board the variant does not name.
 */
public interface BoardEdit {

    /** The operation this edit is an instance of, for the record's {@code directed} tag. */
    ForkOperation operation();

    /** Apply the change to {@code fork}, ending with the engine's static-ability check. */
    void apply(Game fork);

    /**
     * What every edit ends with: the engine's static-ability check, so the
     * board's continuous effects account for the change before anything reads
     * it.
     *
     * <p>The active-trigger list is rebuilt first. Forge rebuilds it at its
     * own checkpoints (the state-based check, a resolution, a phase), and the
     * static check runs the state triggers from it: a permanent the edit
     * took off the battlefield would otherwise still be asked about, and one
     * it put there would not yet be. Rebuilding it fires nothing.
     */
    static void settle(Game fork) {
        fork.getTriggerHandler().resetActiveTriggers(false, null);
        fork.getAction().checkStaticAbilities();
    }

    /** {@code card}, failing when it is not on the battlefield. */
    static Card onBattlefield(Card card) {
        if (card == null || card.getZone() == null || !card.getZone().is(ZoneType.Battlefield)) {
            throw new IllegalStateException("not on the battlefield: " + card);
        }
        return card;
    }
}
