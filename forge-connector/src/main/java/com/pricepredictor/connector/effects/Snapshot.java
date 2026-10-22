package com.pricepredictor.connector.effects;

/**
 * One record's state snapshot, held as its rendered blocks until it is written.
 *
 * <p>Kept in blocks rather than as one string because the refs block is the one
 * part of a snapshot that is not read at the same moment as the rest. The board
 * must be captured before the ability resolves, but the modes a charm chose,
 * the X it announced and the card, colour or type its controller named are set
 * by the engine <b>during</b> resolution — so a snapshot taken at the cast
 * carries an empty refs block for exactly the abilities whose choice is the
 * interesting part of the effect. Re-rendering the whole snapshot at resolution
 * is not the alternative: that would describe the board the ability had
 * already changed, which is the one thing a pre-event snapshot must not do.
 *
 * @param global       the {@code global} block
 * @param players      the {@code players} array
 * @param entities     the {@code entities} array, every inclusion tier
 * @param refs         the {@code refs} block for the acting line
 * @param pendingEvent the {@code pending_event} a trait is about to handle, or
 *                     {@code null} rendered
 */
record Snapshot(
        String global, String players, String entities, String refs, String pendingEvent) {

    /**
     * The {@code state.tiers} every snapshot carries: all four, always.
     *
     * <p>Still written although it never varies, because the record schema is
     * frozen and older corpora carry shallower vectors, which readers keep
     * treating as uncollected rather than empty.
     */
    static final String TIERS_JSON = "[1,2,3,4]";

    /** The same board, with its refs block replaced by a later reading. */
    Snapshot withRefs(String laterRefs) {
        return new Snapshot(global, players, entities, laterRefs, pendingEvent);
    }

    /** The state as a record writes it, its blocks in the order the schema fixes. */
    String render() {
        return Json.object()
                .raw("global", global)
                .raw("players", players)
                .raw("entities", entities)
                .raw("refs", refs)
                .raw("pending_event", pendingEvent)
                .raw("tiers", TIERS_JSON)
                .render();
    }
}
