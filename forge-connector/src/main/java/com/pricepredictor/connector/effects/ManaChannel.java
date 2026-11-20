package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * One resolution record per mana ability that produced mana (FR-042c).
 *
 * <p>A mana ability never uses the stack, so the bracket collector sees no
 * cast and no resolution for it and the corpus has no effect half at all.
 * That is what leaves the role-polarity probe unrunnable: paying {@code R} is
 * observable from stage one, producing it is not.
 *
 * <p>The record describes the activation's moment (root § 8.14): its board is
 * read before the mana ability's costs are paid ({@link #paymentHandler}, the
 * source untapped and the pool without the new mana), and its events are what
 * the activation changed — a {@code tapped} event for each permanent its cost
 * tapped, then {@code mana_produced}. So "tapping a source that says add {G}
 * produces {G}" reads off the record, and the board never already holds the
 * label.
 *
 * <p>Mana a resolving spell or trigger makes — Dark Ritual, a "whenever you
 * cast" mana trigger — is not a mana ability's: its {@code mana_produced}
 * goes into that resolution's effect half, and no mana record is written
 * (C3). A triggered mana ability (Wild Growth's "whenever enchanted land is
 * tapped for mana") whose payment the hook reports is an activation of its
 * own: its mana is a mana record keyed to its own line (the Aura's), read
 * before its own payment. One whose payment the hook does not report joins
 * the record of the activation that set it off, while that record is still
 * being built; with no such record, its mana reaches none.
 *
 * <p>A mana ability's own other effects are its record's too, taken from the
 * bracket ({@link EventRouter.EventTap}) rather than dropped as a
 * payment's: what its cost did beyond the taps (a creature sacrificed for
 * mana: its {@code sacrificed} and {@code zone_change}), before
 * {@code mana_produced}, and what its sub-abilities did (a pain land's
 * damage to its controller), after it.
 *
 * <p>Capped at {@code --mana-cap} records per unique mana-ability text, by a
 * reservoir that holds the sampled records until the game ends.
 */
final class ManaChannel implements EventRouter.EventTap {

    private final LiveGame live;
    private final CollectorContext context;
    /** The sampled mana records, held until the game ends. */
    private final RecordReservoir reservoir;
    /** Where mana a resolving spell or trigger makes is filed, and whose stat differ re-reads. */
    private final BusBracketCollector bracket;

    /**
     * The board read before each mana ability's costs, by ability identity,
     * until its mana arrives: activations nest (a filter land's own {1} is
     * paid by another land's mana ability), and a mana ability's mana is
     * produced after its payment is over.
     */
    private final Map<SpellAbility, Activation> pending = new IdentityHashMap<>();

    /**
     * The record of the mana ability whose mana arrived last, still taking
     * the mana of the triggered mana abilities its activation set off; null
     * between activations.
     */
    private Building building;

    /** A mana ability's activation, from before its costs until its mana. */
    private static final class Activation {
        final RecordState state;
        /** The permanents its cost tapped, read once the payment succeeded. */
        List<String> tapped = List.of();
        /** What its payment did besides the taps: a sacrifice's events. */
        final List<EffectEvent> paid = new ArrayList<>();

        Activation(RecordState state) {
            this.state = state;
        }
    }

    /** A mana record being built, keyed for the reservoir by the activation's own mana. */
    private record Building(
            SpellAbility source, String key, String actor, ProvenanceKey.Resolved ability,
            RecordState state, List<EffectEvent> events) {
    }

    ManaChannel(
            LiveGame live, CollectorContext context, RecordReservoir reservoir,
            BusBracketCollector bracket) {
        this.live = live;
        this.context = context;
        this.reservoir = reservoir;
        this.bracket = bracket;
    }

    /**
     * The payment hook's listener: before a mana ability's costs are paid,
     * its board is read; once they are, the permanents they tapped. A payment
     * that failed produces no mana and keeps nothing.
     *
     * <p>Only a mana ability's own payment is read: a spell's or an
     * activated ability's costs belong to its cost half, and an effect's
     * request is the resolving ability's. Package-private for the reason
     * {@link #handler} is.
     */
    PatchHooks.Listener paymentHandler() {
        return PatchHooks.PAYMENT_LISTENER.listener((callback, args) -> {
            SpellAbility ability = args.get(0, SpellAbility.class);
            if (ability != null && !ability.isManaAbility() && live.owns(ability)
                    && callback == PatchHooks.Callback.ON_COSTS_PAID) {
                // The payment the mana went to is over: no triggered mana of
                // the last activation comes after it (V3 #15).
                finishBuilding();
            }
            if (ability == null || !ability.isManaAbility() || !live.owns(ability)
                    || !reservoir.keepsAny()) {
                return;
            }
            switch (callback) {
                case ON_PAYING_COSTS -> payingCosts(ability);
                case ON_COSTS_PAID -> costsPaid(ability, Boolean.TRUE.equals(args.get(2)));
                default -> { }
            }
        });
    }

    private void payingCosts(SpellAbility ability) {
        finishBuilding();
        // No stat re-read here: a mana record holds no stat change, and a
        // re-read would take the enclosing cast's unpublished changes out of
        // its cost half whether or not this activation is kept (V3 #15).
        pending.put(ability, new Activation(live.snapshots().ofBoard().state()));
    }

    /**
     * An event published while one of this game's mana abilities is paid
     * for or resolves is that activation's: its payment's, but for the taps
     * its {@code tapped} events already say; its own clauses', read off the
     * resolving clause's root.
     */
    @Override
    public boolean take(EffectEvent event, String namedCause) {
        if (pending.isEmpty() && building == null) {
            return false;
        }
        Object clause = PatchHooks.currentSubAbility();
        if (clause instanceof SpellAbility resolving && resolving.getRootAbility() != null
                && resolving.getRootAbility().isManaAbility()) {
            SpellAbility root = resolving.getRootAbility();
            List<EffectEvent> into = building != null && building.source() == root
                    ? building.events()
                    : pending.containsKey(root) ? pending.get(root).paid : null;
            if (into == null) {
                return false;
            }
            into.add(EventAttribution.stamp(event, root, namedCause));
            return true;
        }
        Activation paying = payingFor();
        if (paying == null) {
            return false;
        }
        if (event.type() != EventType.TAPPED) {
            paying.paid.add(EventAttribution.stampAt(event, null, null, namedCause));
        }
        return true;
    }

    /** The activation whose costs are being paid now, from the top of the payment stack. */
    private Activation payingFor() {
        var game = live.game();
        var top = game == null ? null : game.costPaymentStack.peek();
        SpellAbility paid = top == null || top.payment() == null
                ? null : top.payment().getAbility();
        return paid == null ? null : pending.get(paid);
    }

    private void costsPaid(SpellAbility ability, boolean paid) {
        if (!paid) {
            pending.remove(ability);
            return;
        }
        Activation activation = pending.get(ability);
        if (activation != null) {
            activation.tapped = PaidCosts.of(ability).tapped();
        }
    }

    /**
     * The mana hook's listener.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    PatchHooks.Listener handler() {
        return PatchHooks.MANA_LISTENER.listener((callback, args) -> {
            SpellAbility ability = args.get(0, SpellAbility.class);
            if (ability == null) {
                return;
            }
            String produced = String.valueOf(args.get(2));
            String player = live.playerIdOrActive(args.get(1, Player.class));
            made(ability, player, produced, manaEvent(player, produced));
        });
    }

    /**
     * The {@code mana_produced} event for mana the engine reports, or null
     * when it reports none: Forge passes {@code "0"} when a mana effect
     * computes zero (Songs of the Damned with no creature in the graveyard),
     * and zero mana is not an event.
     */
    private static EffectEvent manaEvent(String player, String produced) {
        Map<String, Integer> byColor = ProducedMana.byColor(produced);
        if (byColor.values().stream().noneMatch(count -> count > 0)) {
            return null;
        }
        return new EffectEvent(EventType.MANA_PRODUCED)
                .subject(player)
                .param(EventParam.MANA_BY_COLOR, byColor);
    }

    /**
     * A clause's own effect is over. The clause that carries the mana part
     * ending with its activation still pending is one whose effect never
     * reported its mana (Forge sends a root's notice before its
     * sub-abilities run, so a root that only prepares the mana, Nykthos's
     * colour choice, does not end the activation; the sub-ability that makes
     * the mana does, the activation looked up by its root):
     * Forge skips the mana hook when the mana string is empty (a noted type
     * nothing noted) and when a {@code ProduceMana} replacement prevents the
     * mana. The activation is over all the same, and writes its taps and
     * payments as the zero-mana case does.
     */
    void clauseResolved(SpellAbility clause) {
        SpellAbility root = clause.getManaPart() == null ? null : clause.getRootAbility();
        if (root != null && pending.containsKey(root)) {
            produced(root, live.playerIdOrActive(root.getActivatingPlayer()), "", null);
        }
    }

    /** Route the mana the engine reported; {@code event} is null for zero mana. */
    private void made(SpellAbility ability, String player, String produced, EffectEvent event) {
        // A fork's own mana abilities can resolve through the inline path
        // too (a forced ability, or costs it pays along the way), and this
        // hook is a JVM static like the others: the fork's are its
        // fork-bound recording's.
        if (!live.owns(ability)) {
            return;
        }
        if (ability.isManaAbility()) {
            produced(ability, player, produced, event);
        } else if (event != null && bracket != null) {
            // A resolving spell's or trigger's mana: its effect half's (C3).
            bracket.recordClauseEvent(event);
        }
    }

    /**
     * A mana ability's mana, or its producing nothing ({@code event} null):
     * either way its activation is over, and its record holds what the
     * activation did. A mana ability's mana is a root clause's, at an instant.
     */
    private void produced(
            SpellAbility ability, String player, String produced, EffectEvent event) {
        Activation activation = pending.remove(ability);
        if (activation == null) {
            joinBuilding(ability, event);
            return;
        }
        finishBuilding();
        List<EffectEvent> events = new ArrayList<>();
        for (String tapped : activation.tapped) {
            events.add(EventAttribution.stampAt(
                    new EffectEvent(EventType.TAPPED).subject(tapped), null, null, null));
        }
        events.addAll(activation.paid);
        if (event != null) {
            events.add(rootAtInstant(event));
        }
        if (events.isEmpty()) {
            return;
        }
        building = new Building(
                ability, capKey(ability, produced), player, ProvenanceResolver.resolve(ability),
                activation.state, events);
    }

    /**
     * A triggered mana ability the building activation set off, or the same
     * ability adding mana a second time.
     */
    private void joinBuilding(SpellAbility ability, EffectEvent event) {
        if (event != null && joins(ability)) {
            building.events().add(rootAtInstant(event));
        }
    }

    private boolean joins(SpellAbility ability) {
        return building != null && (ability.isTrigger() || building.source() == ability);
    }

    private static EffectEvent rootAtInstant(EffectEvent event) {
        return event.attributedTo(EventAttribution.ROOT).duration(EventDuration.INSTANT);
    }

    /**
     * The record being built is complete: offered to the reservoir, which
     * keeps it or not.
     */
    private void finishBuilding() {
        Building done = building;
        building = null;
        if (done == null) {
            return;
        }
        int slot = reservoir.slot(done.key());
        if (slot < 0) {
            return;
        }
        // Held rather than written: which activation survives is not known
        // until the game ends.
        EffectRecord record = context.newResolution(Moment.RESOLUTION)
                .actor(done.actor())
                .ability(done.ability())
                .state(done.state())
                // A mana ability resolves in place, at the player's action,
                // where no clause is filed: no condition is read, and it is
                // no stack item, so it has no outcome.
                .payload(RecordPayloads.resolutionPayload(done.events(), List.of(), null));
        reservoir.hold(done.key(), slot, record);
    }

    /**
     * The cap's key: the ability's {@link AbilityLines#key}, and what it
     * produced, because a filter land producing {@code W} and the same land
     * producing {@code U} are different observations of the same line.
     */
    private static String capKey(SpellAbility ability, String produced) {
        return AbilityLines.key(ability) + "|" + produced;
    }

    /**
     * Write the sampled mana records, once the game can produce no more.
     *
     * <p>They are held until here because reservoir sampling does not know
     * which activation won until the last one has happened. A game whose JVM
     * dies loses its reservoir — at most {@code --mana-cap} records per mana
     * ability, against a crash that already costs the rest of the game.
     */
    void flush() {
        finishBuilding();
        pending.clear();
        for (EffectRecord record : reservoir.drain()) {
            context.emitAtGameEnd(record);
        }
    }
}
