package com.pricepredictor.connector.effects;

import forge.game.Game;

import java.util.List;

/**
 * A fork's premise: the state-based check its edit makes true, run as the
 * fork's first act and recorded as a {@code turn_action} record of subkind
 * {@code fork_edit}.
 *
 * <p>A protection fork's grant or a directed fork's board edit is applied
 * before the fork's collectors listen, so the edit is no event of the fork.
 * What it makes true is: an Aura or Equipment the protection makes illegal
 * falls off, a creature the edit leaves at 0 toughness dies. Left to the
 * fork's first priority, that check runs with no resolution held and no
 * turn-based action's window open, so its moves are dropped and its
 * {@code unattached} goes to the orphans. Run here instead, with the fork's
 * collectors listening, everything the bracket would route while it runs is
 * this record's, its board read after the edit and before the check, its
 * actor the active player. A check that changes nothing writes no record.
 *
 * <p>A tap of the fork's bracket ({@link BusBracketCollector#addTap}),
 * taking nothing outside {@link #settle}. The triggers the check puts on the
 * stack (a state trigger the edit makes true, a "when this is put into a
 * graveyard") are the fork's to record as it plays on.
 */
final class ForkPremise implements EventRouter.EventTap {

    private final Game fork;
    private final CollectorContext context;
    /** The events the check published so far, or null outside {@link #settle}. */
    private EventBuffer events;

    ForkPremise(Game fork, CollectorContext context) {
        this.fork = fork;
        this.context = context;
    }

    @Override
    public boolean take(EffectEvent event, String namedCause) {
        if (events == null) {
            return false;
        }
        events.file(EventAttribution.stampAt(event, null, null, namedCause));
        return true;
    }

    /** Run the state-based check the edit makes, and write what it did. */
    void settle() {
        Snapshot state = new SnapshotBuilder(fork).ofBoard();
        String actor = Refs.of(fork.getPhaseHandler().getPlayerTurn());
        events = new EventBuffer();
        List<EffectEvent> taken;
        try {
            StateBasedCheck.run(fork);
        } finally {
            taken = events.drain();
            events = null;
        }
        if (!taken.isEmpty()) {
            context.emit(context.newTurnAction(TurnActionSubkind.FORK_EDIT)
                    .actor(actor)
                    .state(state.withEventCards(taken).state())
                    .payload(RecordPayloads.eventsPayload(taken)));
        }
    }
}
