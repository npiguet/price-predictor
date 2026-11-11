package com.pricepredictor.connector.effects;

import forge.game.spellability.SpellAbility;

/**
 * One stack item's bracket in the live collector, from its cast until it
 * resolves or leaves the stack without resolving.
 *
 * <p>Built at the cast. A cast whose stack top holds no ability builds none,
 * and nothing is filed for it.
 */
final class OpenResolution {
    final SpellAbility ability;
    final ProvenanceKey.Resolved keys;
    final String actor;
    /**
     * The board read when the item was cast, the effect half's board only
     * where its resolution's start was never told ({@link #resolutionState}):
     * a worker without the resolution hook, or a bracket a test drives by
     * hand.
     */
    final Snapshot state;
    /**
     * The link its cost half carries and its effect halves claim, or null
     * for an item that has no cost half: a triggered ability or a copy, which
     * nobody cast (root § 8.14).
     */
    final String linkId;
    /** What its cast paid, or null for an item with no cost half. */
    final PaidCosts costs;
    /** How many targets its root line had at the cast, for a partial fizzle. */
    final int targetsAtCast;
    /**
     * What the bus filed for this item, split by chosen mode as it resolves;
     * its own and no other's. From its cast to its resolution's start it
     * gathers the cast's consequences, which reach no record once the
     * resolution starts (they precede its board); from there, what it does
     * while it resolves.
     */
    final ModeSplit modes = new ModeSplit();

    /**
     * The board when its resolution started (FR-042b), or null until then.
     */
    private Snapshot resolutionState;

    /**
     * The X it announced ({@link RefsJson#announcedX}), read once: when it
     * was put on the stack, its costs paid, for an item somebody cast; when
     * its resolution started otherwise. Every refs block of the item carries
     * this one reading, never one taken after the resolution (B3).
     */
    private Integer x;
    private boolean xRead;

    /**
     * Whether the controller turned something down while this bracket was open.
     *
     * <p>Scoped to the bracket rather than matched against the declined
     * ability's id, because Forge resolves a copy of the ability it was handed
     * and an id comparison would silently never match — the same reason
     * {@link EventAttribution} counts a clause upward from the pointer instead
     * of downward from the root. What keeps the flag from over-claiming is the
     * second half of the test in {@link #wasDeclined}: a resolution that
     * produced any outcome at all is {@code resolved} however many optional
     * clauses inside it were declined.
     */
    private boolean declined;

    OpenResolution(
            SpellAbility ability, ProvenanceKey.Resolved keys, String actor, Snapshot state) {
        this(ability, keys, actor, state, null, null);
    }

    OpenResolution(
            SpellAbility ability, ProvenanceKey.Resolved keys, String actor, Snapshot state,
            String linkId, PaidCosts costs) {
        this.ability = ability;
        this.keys = keys;
        this.actor = actor;
        this.state = state;
        this.linkId = linkId;
        this.costs = costs;
        this.targetsAtCast = targetCount(ability);
        if (linkId != null) {
            readX();
        }
    }

    /** Its resolution has started, on this board (FR-042b). */
    void resolutionStarted(Snapshot board) {
        resolutionState = board;
        readX();
    }

    private void readX() {
        if (!xRead) {
            x = RefsJson.announcedX(ability);
            xRead = true;
        }
    }

    /** The X it announced, read once; null when it has none or none was read yet. */
    Integer x() {
        return x;
    }

    /**
     * The board a root effect half is read against: the one taken when the
     * resolution started, or, where that was never told, the cast's.
     */
    Snapshot effectState() {
        return resolutionState != null ? resolutionState : state;
    }

    /** The controller turned something down while this bracket was open. */
    void markDeclined() {
        declined = true;
    }

    /**
     * Whether this resolution is a refusal rather than a resolution.
     *
     * <p>Both halves are required. A refusal alone is not enough: a line whose
     * mandatory clause drew a card and whose optional clause was declined did
     * something, and {@code resolved} is the honest word for it. An empty event
     * list alone is not enough either — plenty of resolutions produce nothing
     * this collector can observe. Together they are the shape the schema
     * describes: offered, turned down, and nothing happened.
     */
    boolean wasDeclined() {
        return declined && modes.isEmpty();
    }

    /**
     * What became of the item, now that it has finished resolving: its
     * effect half's {@code outcome} (FR-042b).
     *
     * @param fizzled whether Forge removed it for having no legal target left
     */
    ActivationOutcome outcome(boolean fizzled) {
        if (fizzled) {
            return ActivationOutcome.FIZZLED;
        }
        if (wasDeclined()) {
            return ActivationOutcome.DECLINED;
        }
        return outcomeOf(targetsAtCast, ability);
    }

    /**
     * Whether a resolution lost some of its targets on the way.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * from the ability before resolving it, and only then does the ability
     * resolve — so comparing the count with the one read at the cast is the
     * engine's own answer to "did part of this spell do nothing". Nothing else
     * in the record distinguishes a Lightning Helix that hit from one whose
     * creature had already died.
     *
     * <p>Only the root line's targets are compared. A partial fizzle confined to
     * a sub-ability reads as {@code resolved}, which understates rather than
     * invents. Losing <b>every</b> target without the engine calling it a fizzle
     * is the {@code CantFizzle} case (Gilded Drake), where the spell still does
     * what it does: that is {@code resolved} too.
     */
    static ActivationOutcome outcomeOf(int targetsAtCast, SpellAbility resolved) {
        if (resolved == null) {
            return ActivationOutcome.RESOLVED;
        }
        int surviving = targetCount(resolved);
        return surviving > 0 && surviving < targetsAtCast
                ? ActivationOutcome.PARTIALLY_FIZZLED
                : ActivationOutcome.RESOLVED;
    }

    static int targetCount(SpellAbility ability) {
        return ability == null || ability.getTargets() == null
                ? 0 : ability.getTargets().size();
    }

    /** Whether this is the bracket of the ability the engine names by id. */
    boolean is(int abilityId) {
        return ability.getId() == abilityId;
    }

    /**
     * Whether a clause is on this resolution's card; see
     * {@link BusBracketCollector#beginMode}.
     */
    boolean hosts(SpellAbility clause) {
        return clause != null && clause.getHostCard() == ability.getHostCard();
    }
}
