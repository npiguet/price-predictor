package com.pricepredictor.connector;

/**
 * Where every booster of a draft pod sits and which way it travels: the
 * {@code drafts.jsonl} geometry contract.
 *
 * <p>A transcript records no seat per pick. The Python reader
 * ({@code src/draft/domain/draft_geometry.py}) recovers it from the booster
 * ordering alone, so this class and that module are twins: booster {@code k}
 * belongs to pack {@code k / POD_SIZE + 1} and was opened by seat
 * {@code k mod POD_SIZE}, and the pick at offset {@code j} of it was made by
 * seat {@code (opening + j * dir) mod POD_SIZE}, with {@code dir} {@code +1}
 * on odd packs (pass left) and {@code -1} on even ones (pass right). A change
 * here that the reader does not share attributes every pick to the wrong
 * seat.
 *
 * <p>Seats are 0-based; packs and picks 1-based, as the transcript and the
 * pick protocol number them.
 */
final class PodGeometry {

    static final int POD_SIZE = 8;
    static final int PACKS = 3;
    /** Every booster the pod opens: one per seat per pack. */
    static final int BOOSTERS = PACKS * POD_SIZE;

    private PodGeometry() {
    }

    /** The index in {@code boosters} of the booster {@code openingSeat} opens in {@code pack}. */
    static int boosterIndex(int pack, int openingSeat) {
        return (pack - 1) * POD_SIZE + openingSeat;
    }

    /** {@code +1} (left) for odd packs, {@code -1} (right) for even ones. */
    static int passDirection(int pack) {
        return pack % 2 == 1 ? 1 : -1;
    }

    /** The seat {@code seat} passes its pack to in {@code pack}. */
    static int nextSeat(int seat, int pack) {
        return Math.floorMod(seat + passDirection(pack), POD_SIZE);
    }
}
