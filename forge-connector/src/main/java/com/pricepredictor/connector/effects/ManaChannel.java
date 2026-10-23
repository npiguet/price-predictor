package com.pricepredictor.connector.effects;

import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.lang.reflect.InvocationHandler;
import java.util.List;

/**
 * One resolution record per mana ability that produced mana.
 *
 * <p>A mana ability never uses the stack, so the bracket collector sees no
 * cast and no resolution for it and the corpus has no effect half at all.
 * That is what leaves the role-polarity probe unrunnable: paying {@code R} is
 * observable from stage one, producing it is not.
 *
 * <p>Capped at {@code --mana-cap} records per unique mana-ability text, by a
 * reservoir that holds the sampled records until the game ends.
 */
final class ManaChannel {

    private final LiveGame live;
    private final CollectorContext context;
    /** The sampled mana records, held until the game ends. */
    private final ManaReservoir reservoir;

    ManaChannel(LiveGame live, CollectorContext context, ManaReservoir reservoir) {
        this.live = live;
        this.context = context;
        this.reservoir = reservoir;
    }

    /**
     * The mana hook's listener.
     *
     * <p>Package-private: the argument order has no compiler behind it, so
     * this hook's own game-identity test calls it with a synthesized argument
     * array.
     */
    InvocationHandler handler() {
        return PatchHooks.MANA_LISTENER.listener((callback, args) -> {
            SpellAbility ability = args.get(0, SpellAbility.class);
            String produced = String.valueOf(args.get(2));
            // A fork's own mana abilities can resolve through the inline path
            // too (forceResolution's forced ability,
            // or costs it pays along the way), and this hook is a JVM static
            // like the others.
            if (ability == null || produced.isEmpty() || !live.owns(ability)) {
                return;
            }
            String key = capKey(ability, produced);
            int slot = reservoir.slot(key);
            if (slot < 0) {
                return;
            }
            String player = live.playerIdOrActive(args.get(1, Player.class));
            EffectEvent event = new EffectEvent(EffectEvent.MANA_PRODUCED)
                    .subject(player)
                    .param("mana_by_color", RunParamNormalizer.manaByColor(produced));
            // Held rather than written: which activation survives is not known
            // until the game ends. The state is snapshotted here, at the moment
            // the mana was made, so a record that does survive describes the
            // board it was actually produced on.
            EffectRecord record = context.newResolution(Moment.RESOLUTION)
                    .actor(player)
                    .ability(ProvenanceResolver.resolve(ability))
                    .state(live.snapshots().ofBoard().render())
                    .payload(EffectRecord.eventsPayload(List.of(event)));
            reservoir.hold(key, slot, record);
        });
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
        for (EffectRecord record : reservoir.drain()) {
            context.emit(record);
        }
    }
}
