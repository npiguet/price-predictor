package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.List;

/**
 * The cast tracking of a {@link BusBracketCollector}: the board read as a
 * cast or an activation started (FR-042a), kept until its cast event, and
 * the cost half that event writes on it.
 *
 * <p>It holds the started cast of its own; the context, the snapshot
 * builder, the payment ledger and the casts under way are the collector's,
 * shared.
 */
final class CastTracker {

    private final CollectorContext context;
    private final SnapshotBuilder snapshots;
    /** The counters and life each payment under way has moved, until its cast event reads them. */
    private final PaymentLedger payments;
    /** The cast on its way onto the stack, where the bus said its card came from. */
    private final CastsUnderWay castsUnderWay;

    /**
     * The board read as a cast or an activation started (FR-042a), until its
     * cast event writes its cost half; null when no cast is under way.
     */
    private PendingCast pendingCast;

    /**
     * A cast's board, read before its card moved or its costs were paid, and
     * the targets its chain held then, each ref once.
     */
    record PendingCast(
            SpellAbility ability, int hostId, Snapshot board, List<String> targets) {

        /** Whether the item a cast event put on the stack is this cast's. */
        boolean isOf(SpellAbility pushed) {
            return pushed == ability || pushed.getOriginalAbility() == ability
                    || (pushed.getHostCard() != null && pushed.getHostCard().getId() == hostId
                            && pushed.isSpell() == ability.isSpell());
        }
    }

    CastTracker(CollectorContext context, SnapshotBuilder snapshots,
            PaymentLedger payments, CastsUnderWay castsUnderWay) {
        this.context = context;
        this.snapshots = snapshots;
        this.payments = payments;
        this.castsUnderWay = castsUnderWay;
    }

    /** A cast has started: read the board its cost half describes, with its targets now. */
    void started(SpellAbility cast) {
        Card host = cast.getHostCard();
        pendingCast = new PendingCast(cast, host == null ? -1 : host.getId(),
                snapshots.ofResolving(cast), ChainTargets.now(cast).all());
    }

    /** No cast is under way any more: one started and never pushed was given up. */
    void forgetStarted() {
        pendingCast = null;
    }

    /**
     * The cast whose start was told for the item a cast event put on the
     * stack, no longer pending; null when none was told (a copy, a worker
     * without the cast hook). A started cast the pushed item is not, other
     * than for a trigger, was given up.
     */
    PendingCast startedCast(SpellAbility pushed) {
        PendingCast started = pendingCast;
        if (started == null || pushed == null) {
            return null;
        }
        if (started.isOf(pushed)) {
            pendingCast = null;
            return started;
        }
        if (!pushed.isTrigger()) {
            pendingCast = null;
        }
        return null;
    }

    /**
     * The bracket of an item a cast event put on the stack. An item somebody
     * cast or activated — not a triggered ability, not a copy — has a cost
     * half: its link and what it paid, read now.
     *
     * @param started its cast's start, or null
     */
    OpenResolution openResolution(SpellAbility ability, PendingCast started) {
        // The board a degraded worker's effect half is read against, the
        // resolution's start being told only by the hook.
        Snapshot atCast = started != null ? started.board() : snapshots.ofResolving(ability);
        if (!hasCostHalf(ability)) {
            return new OpenResolution(
                    ability, ProvenanceResolver.resolve(ability), actorOf(ability), atCast);
        }
        return new OpenResolution(
                ability, ProvenanceResolver.resolve(ability), actorOf(ability), atCast,
                linkOf(context.gameId(), ability), PaidCosts.of(ability, payments.take(ability)));
    }

    /** The link a cast or activated item's halves carry in game {@code gameId}: its cost half's. */
    static String linkOf(String gameId, SpellAbility item) {
        return gameId + ".link." + item.getId();
    }

    /**
     * Whether a stack item has a cost half: something cast or activated.
     * A triggered ability is not (root § 8.14), nor a copy of a spell or an
     * ability, which is put on the stack without being cast.
     */
    static boolean hasCostHalf(SpellAbility ability) {
        if (ability.isTrigger() || ability.isCopied()) {
            return false;
        }
        Card host = ability.getHostCard();
        return !(ability.isSpell() && host != null && host.isCopiedSpell());
    }

    /**
     * Write a cast's cost half (FR-042a): the board read as its cast started
     * — or at the cast, where its start was not told — with the refs block
     * read now, its targets, modes and X chosen, and the {@code targets}
     * links those refs name; its events, a spell card's {@code zone_change}
     * to the stack; and its costs.
     */
    void writeCostHalf(OpenResolution cast, PendingCast started) {
        Snapshot board = started != null
                ? withTargetsChosen(started, cast.ability)
                        .withRefs(RefsJson.renderWithX(cast.ability, cast.x()))
                : cast.state;
        context.emit(context.newResolution(Moment.ACTIVATION)
                .actor(cast.actor)
                .ability(cast.keys)
                .linkId(cast.linkId)
                .state(board.render())
                .payload(RecordPayloads.costPayload(castMove(cast.ability), cast.costs)));
    }

    /**
     * A cast's board as it started, its {@code targets} links read again
     * where its modes or targets were chosen after the read: Forge moves a
     * spell's card to the stack, where the cast hook reads the board, before
     * it chains a charm's modes, and a play that defers its targets (the
     * random seat's charm) chooses them after that too.
     */
    private Snapshot withTargetsChosen(PendingCast started, SpellAbility pushed) {
        return ChainTargets.now(pushed).all().equals(started.targets())
                ? started.board()
                : snapshots.withTargetsChosen(started.board(), pushed);
    }

    /**
     * A spell's own move to the stack, from the zone the bus said its card
     * came from (601.2a); nothing for an activated ability, which moves no
     * card, or a card whose arrival was not seen.
     */
    private List<EffectEvent> castMove(SpellAbility ability) {
        Card host = ability.getHostCard();
        if (!ability.isSpell() || host == null) {
            return List.of();
        }
        String from = castsUnderWay.arrivedFrom(host.getId());
        if (from == null) {
            return List.of();
        }
        EffectEvent moved = new EffectEvent(EventType.ZONE_CHANGE)
                .subject(Refs.of(host))
                .param("from_zone", from)
                .param("to_zone", CardSpelling.zoneName(ZoneType.Stack));
        return List.of(EventAttribution.stampAt(moved, ability, null, null));
    }

    private static String actorOf(SpellAbility ability) {
        return ability == null ? null : Refs.ofNullable(ability.getActivatingPlayer());
    }
}
