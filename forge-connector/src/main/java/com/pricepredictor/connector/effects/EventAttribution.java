package com.pricepredictor.connector.effects;

import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

import java.util.Locale;
import java.util.Set;

/**
 * The two fields an event carries about where it came from and how long it lasts.
 *
 * <p>Both were declared on the event shape and never written. {@code duration}
 * is why {@code pt_duration} and {@code type_color_duration} were the only two
 * head fields no target path fed: an effect that pumps a creature and one that
 * pumps it permanently produced identical records, so there was nothing to
 * learn the difference from.
 */
final class EventAttribution {

    private EventAttribution() {
    }

    /** The head's three buckets; the full vocabulary is DURATIONS in Python. */
    static final String INSTANT = "instant";
    static final String END_OF_TURN = "end_of_turn";
    static final String PERMANENT = "permanent";

    /**
     * Script durations that outlast the turn.
     *
     * <p>Forge uses a dozen spellings and the head has three buckets, so the
     * mapping is by exception: these last, anything else beginning
     * {@code Until} or {@code EOT} ends with the turn or sooner.
     * {@code UntilHostLeavesPlay} is here because an aura's grant lasts as long
     * as the aura does, which is not a turn boundary.
     */
    private static final Set<String> LASTING = Set.of(
            "permanent", "perpetual", "aslongascontrol", "untilhostleavesplay");

    /**
     * Which sub-ability produced this event, as its index down the chain.
     *
     * <p>The sidecar addresses a sub-ability by a flat chain index, so this
     * walks the acting ability's chain looking for the one the engine says is
     * resolving. Null means the root line, which is also the fallback when the
     * pointer names an ability that is not on this chain — attribution never
     * drops an event.
     */
    static String attributedTo(SpellAbility root) {
        if (!(PatchHooks.currentSubAbility() instanceof AbilitySub resolving)) {
            return null;
        }
        // Counted upward from the resolving clause rather than downward from
        // the acting one. Forge resolves a copy of the ability it was handed,
        // so the bracket's root and the pointer's chain need not be the same
        // objects, and a walk down from the root matched nothing at all.
        int index = 0;
        SpellAbility parent = resolving.getParent();
        while (parent instanceof AbilitySub above) {
            index++;
            parent = above.getParent();
        }
        return parent == null ? null : String.valueOf(index);
    }

    /**
     * How long the acting script says its effect lasts.
     *
     * <p>Read from the resolving clause's own {@code Duration$}, falling back to
     * the root line's — a sub-ability that declares none inherits the line's.
     * The source is the script rather than the engine's cleanup registration
     * because the script is where the answer is written down; an effect whose
     * duration lives only in Java reads as {@code instant}, which is also the
     * right answer for the many events that do not last at all.
     */
    static String duration(SpellAbility root) {
        String declared = declaredDuration(root);
        if (declared == null) {
            return INSTANT;
        }
        String key = declared.toLowerCase(Locale.ROOT);
        if (LASTING.contains(key)) {
            return PERMANENT;
        }
        return END_OF_TURN;
    }

    private static String declaredDuration(SpellAbility root) {
        Object current = PatchHooks.currentSubAbility();
        if (current instanceof SpellAbility resolving) {
            String own = resolving.getParam("Duration");
            if (own != null && !own.isEmpty()) {
                return own;
            }
        }
        if (root == null) {
            return null;
        }
        String own = root.getParam("Duration");
        return own == null || own.isEmpty() ? null : own;
    }

    /**
     * Stamp both onto an event, unless it already carries them.
     *
     * <p>Applied where an event is filed rather than where it is built, because
     * only the collector knows which ability's bracket it landed in.
     */
    static EffectEvent stamp(EffectEvent event, SpellAbility root) {
        if (event == null) {
            return null;
        }
        return event
                .duration(duration(root))
                .attributedTo(attributedTo(root));
    }
}
