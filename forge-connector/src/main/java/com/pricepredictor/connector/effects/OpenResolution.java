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
    /**
     * How many targets its chain had at the cast, every link's, for a
     * partial fizzle where the resolution hook never told the targets as
     * cast ({@link #castTargets}).
     */
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
     * Its chain's targets when its resolution started, after Forge's check
     * for illegal targets removed the illegal ones; null until then.
     */
    private ChainTargets resolvingTargets;

    /**
     * Its chain's targets as the resolution hook told them, before that
     * check (608.2b); null until then, or where the hook could not copy them.
     */
    private ChainTargets castTargets;

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

    /**
     * Its resolution has started, on this board (FR-042b). A refusal made
     * since its cast was the cast's consequence and precedes this board, as
     * the events gathered meanwhile do: it is not the resolution's.
     */
    void resolutionStarted(Snapshot board) {
        resolutionStarted(board, null, null);
    }

    /**
     * Its resolution has started, on this board, holding {@code resolving}
     * as its targets, and {@code atCast} before Forge's check for illegal
     * targets (null where the hook could not tell them).
     */
    void resolutionStarted(Snapshot board, ChainTargets resolving, ChainTargets atCast) {
        resolutionState = board;
        resolvingTargets = resolving;
        castTargets = atCast;
        declined = false;
        readX();
    }

    /**
     * One effect half's {@code targets}: what its links held as its
     * resolution started, or, where that was never told, what they hold now.
     */
    java.util.List<String> targetsOfHalf(int half, int halves, Integer option) {
        ChainTargets targets = resolvingTargets != null
                ? resolvingTargets : ChainTargets.now(ability);
        return targets.ofHalf(half, halves, option);
    }

    /**
     * Its targets before Forge's check for illegal targets, which its effect
     * half's boards carry whatever zone they are in now, or as they last
     * existed (the card that made it fizzle is among them), with the links
     * narrowed to the mode beginning now, the next of {@link #modes}
     * ({@link ChainTargets#beginningMode}): what that mode's half's board
     * links. Null where they are unknown.
     */
    ChainTargets castTargetsOfMode(Integer option) {
        return castTargets == null ? null : castTargets.beginningMode(modes.begun(), option);
    }

    /**
     * Of a half's {@code targets} and {@code targets_at_cast}, the refs that
     * name a spell or ability on the stack, in the order written.
     */
    java.util.List<String> onStackOf(java.util.List<String> targets, java.util.List<String> atCast) {
        java.util.List<String> written = new java.util.ArrayList<>(targets);
        if (atCast != null) {
            written.addAll(atCast);
        }
        ChainTargets now = resolvingTargets != null ? resolvingTargets : ChainTargets.now(ability);
        return ChainTargets.onStackOf(written, now, castTargets);
    }

    /** One effect half's {@code targets_at_cast}, or null where they are unknown. */
    java.util.List<String> targetsAtCastOfHalf(int half, int halves, Integer option) {
        return castTargets == null ? null : castTargets.ofHalf(half, halves, option);
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
        if (castTargets != null && resolvingTargets != null) {
            return outcomeOf(castTargets.count(), resolvingTargets.count());
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
     * <p>Every link's targets are counted, a charm's modes' and a
     * {@code SubAbility$}'s with the root's: a charm whose one mode lost its
     * target while another kept its own partially fizzled. Losing <b>every</b>
     * target without the engine calling it a fizzle
     * is the {@code CantFizzle} case (Gilded Drake), where the spell still does
     * what it does: that is {@code resolved} too.
     */
    static ActivationOutcome outcomeOf(int targetsAtCast, SpellAbility resolved) {
        if (resolved == null) {
            return ActivationOutcome.RESOLVED;
        }
        return outcomeOf(targetsAtCast, targetCount(resolved));
    }

    /** The outcome from the chain's target counts before and after the illegal-target check. */
    static ActivationOutcome outcomeOf(int targetsAtCast, int surviving) {
        return surviving > 0 && surviving < targetsAtCast
                ? ActivationOutcome.PARTIALLY_FIZZLED
                : ActivationOutcome.RESOLVED;
    }

    /** How many targets the ability's chain holds, every link's. */
    static int targetCount(SpellAbility ability) {
        return ability == null ? 0 : ChainTargets.now(ability).count();
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

    /**
     * Whether a clause is on this resolution's own line: its chain's root is
     * the item the stack resolves, or, for a triggered ability, the ability
     * its stack wrapper resolves. A clause another line of the same card
     * resolves meanwhile — an enters-the-battlefield replacement's (Eddymurk
     * Crab's "enters tapped if it's not your turn"), a static trigger's — is
     * on the card ({@link #hosts}) and not on this line.
     */
    boolean onItsLine(SpellAbility clause) {
        if (!hosts(clause)) {
            return false;
        }
        SpellAbility root = clause.getRootAbility();
        return root == ability
                || ability instanceof forge.game.trigger.WrappedAbility wrapper
                        && root == wrapper.getWrappedAbility();
    }
}
