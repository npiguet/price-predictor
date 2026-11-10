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
 * tapped for mana") is never paid for, so it has no board of its own: its
 * mana joins the record of the mana ability whose activation set it off,
 * which is still being built when it arrives, so the land's record reads
 * "tapping it made {G}{G}"; with no such record, its mana reaches none.
 *
 * <p>Capped at {@code --mana-cap} records per unique mana-ability text, by a
 * reservoir that holds the sampled records until the game ends.
 */
final class ManaChannel {

    private final LiveGame live;
    private final CollectorContext context;
    /** The sampled mana records, held until the game ends. */
    private final ManaReservoir reservoir;
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
        final String state;
        /** The permanents its cost tapped, read once the payment succeeded. */
        List<String> tapped = List.of();

        Activation(String state) {
            this.state = state;
        }
    }

    /** A mana record being built, keyed for the reservoir by the activation's own mana. */
    private record Building(
            String key, String actor, ProvenanceKey.Resolved ability, String state,
            List<EffectEvent> events) {
    }

    ManaChannel(
            LiveGame live, CollectorContext context, ManaReservoir reservoir,
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
        // A record's stat changes are measured from its own board (FR-042f).
        if (bracket != null) {
            bracket.rebaselineAtSnapshot();
        }
        pending.put(ability, new Activation(live.snapshots().ofBoard().render()));
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
            String produced = String.valueOf(args.get(2));
            if (ability == null || produced.isEmpty()) {
                return;
            }
            String player = live.playerIdOrActive(args.get(1, Player.class));
            EffectEvent event = new EffectEvent(EventType.MANA_PRODUCED)
                    .subject(player)
                    .param("mana_by_color", RunParamNormalizer.manaByColor(produced));
            // A fork's own mana abilities can resolve through the inline path
            // too (forceResolution's forced ability, or costs it pays along
            // the way), and this hook is a JVM static like the others.
            if (!live.owns(ability)) {
                if (!ability.isManaAbility()) {
                    Forks.forkHooks().manaProduced(ability, event);
                }
                return;
            }
            if (!ability.isManaAbility()) {
                // A resolving spell's or trigger's mana: its effect half's (C3).
                if (bracket != null) {
                    bracket.recordClauseEvent(event);
                }
                return;
            }
            produced(ability, player, produced, event);
        });
    }

    private void produced(
            SpellAbility ability, String player, String produced, EffectEvent event) {
        Activation activation = pending.remove(ability);
        if (activation == null) {
            if (ability.isTrigger() && building != null) {
                // A triggered mana ability the building activation set off.
                building.events().add(event);
            }
            return;
        }
        finishBuilding();
        List<EffectEvent> events = new ArrayList<>();
        for (String tapped : activation.tapped) {
            events.add(new EffectEvent(EventType.TAPPED).subject(tapped));
        }
        events.add(event);
        building = new Building(
                capKey(ability, produced), player, ProvenanceResolver.resolve(ability),
                activation.state, events);
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
            context.emit(record);
        }
    }
}
