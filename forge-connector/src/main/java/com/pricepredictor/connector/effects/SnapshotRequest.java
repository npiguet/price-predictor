package com.pricepredictor.connector.effects;

import forge.game.card.Card;

/**
 * What one snapshot is of, in the shape every {@link SnapshotBuilder} entry
 * point builds: which cards it points at, the board it is read from, who is
 * acting and what is about to be handled.
 *
 * <p>A request starts from {@link #of} (no suppressed static, no source, no
 * pending event) and each {@code with} method sets the one thing a shape
 * adds, so no caller passes the combination of nulls that picks a shape.
 *
 * @param referenced the cards the record points at, carried whatever zone
 *                   they sit in (tier 1)
 * @param board      the board as it stands, or without one static: a
 *                   {@code continuous} record's, which carries no
 *                   {@code affects} from a static, whose answer it is, and
 *                   names each card's controller on it
 * @param acting     the acting spell ability and the targets its links name
 * @param source     the host of an acting line that is not a spell ability,
 *                   or null
 * @param pending    the event a trait is about to handle, or null
 */
record SnapshotRequest(
        Iterable<Card> referenced, StaticSuppression board, SnapshotLinks.Acting acting,
        Card source, EffectEvent pending) {

    /** The board as it stands, around {@code referenced} and {@code acting}. */
    static SnapshotRequest of(Iterable<Card> referenced, SnapshotLinks.Acting acting) {
        return new SnapshotRequest(referenced, StaticSuppression.NONE, acting, null, null);
    }

    /** This request, read off {@code suppression}'s board. */
    SnapshotRequest withBoard(StaticSuppression suppression) {
        return new SnapshotRequest(referenced, suppression, acting, source, pending);
    }

    /** This request, its refs naming {@code host} as the source of the acting line. */
    SnapshotRequest withSource(Card host) {
        return new SnapshotRequest(referenced, board, acting, host, pending);
    }

    /** This request, for a trait about to handle {@code event}. */
    SnapshotRequest withPending(EffectEvent event) {
        return new SnapshotRequest(referenced, board, acting, source, event);
    }
}
