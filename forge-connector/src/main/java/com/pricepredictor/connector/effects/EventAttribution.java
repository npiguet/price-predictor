package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;

import java.util.Locale;
import java.util.Set;

/**
 * The three fields an event carries about where it came from and how long it lasts.
 *
 * <p>All three were declared on the event shape and never written by the bus
 * paths. {@code duration} is why {@code pt_duration} and
 * {@code type_color_duration} were the only two head fields no target path fed:
 * an effect that pumps a creature and one that pumps it permanently produced
 * identical records, so there was nothing to learn the difference from.
 *
 * <p>{@code cause} joined them here rather than living in each shaper because
 * it answers the same question about the same instant as the other two, and
 * because it has a default the shapers cannot know: an event the engine names
 * no causer for was still produced by whatever line was resolving. Stamping all
 * three at one door is what stopped the fork paths writing nulls while the
 * observed path wrote answers.
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
     * The acting line's own top-level clause produced this event.
     *
     * <p>A literal rather than {@code null}, because "the root did it" and "no
     * pointer was available" are different facts and the corpus could not tell
     * them apart. A single-clause ability legitimately has no sub-ability and
     * the root is the right answer; the pointer failing to land looks exactly
     * the same from the record. Reported apart, the sub-ability share becomes a
     * number that can be read; conflated, it was 3.2% of nothing interpretable.
     */
    static final String ROOT = "root";

    /**
     * No clause could be named for this event.
     *
     * <p>Three ways to arrive here, all of them honest and none of them the
     * root: no clause was resolving at all (a combat damage step, a
     * state-based-action death after the bracket's ability finished), the hook
     * is absent because this is an unpatched checkout, or the pointer named a
     * chain whose walk reached no top-level line.
     */
    static final String UNRESOLVED = "unresolved";

    /**
     * Which sub-ability produced this event, as its index down the chain.
     *
     * <p>The sidecar addresses a sub-ability by a flat chain index, so this
     * walks the acting ability's chain looking for the one the engine says is
     * resolving. {@link #ROOT} and {@link #UNRESOLVED} are the other two
     * answers; attribution never drops an event and never returns null.
     */
    static String attributedTo(SpellAbility root) {
        return attributedTo(root, PatchHooks.currentSubAbility());
    }

    /**
     * The same answer, from a pointer the caller already read.
     *
     * <p>The hook is read once where an event is stamped and the arithmetic
     * over it is what can be wrong, so the two are separated: this half needs
     * no patched checkout and no resolving thread to exercise.
     *
     * <p>A pointer that is a {@link SpellAbility} but not an {@link AbilitySub}
     * is a top-level clause resolving, which is {@link #ROOT}. That reading is
     * as precise as the pointer allows and no more: {@code resolveApiAbility}
     * sets it for every clause of every line, so a top-level clause of some
     * <em>other</em> line resolving inside this bracket also reads as the root.
     * Identity against {@code root} cannot narrow it — Forge resolves a copy of
     * the ability it was handed, so the bracket's root and the pointer are
     * routinely different objects for the same line.
     */
    static String attributedTo(SpellAbility root, Object pointer) {
        if (!(pointer instanceof SpellAbility resolvingLine)) {
            // Null, or something that is not an ability at all: the hook is
            // absent, or nothing was resolving when the event was published.
            return UNRESOLVED;
        }
        if (!(resolvingLine instanceof AbilitySub resolving)) {
            return ROOT;
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
        // A chain that ends in nothing is a detached sub-ability: the index
        // counts clauses of a line this collector never saw, so it names
        // nothing a reader could join to.
        return parent == null ? UNRESOLVED : String.valueOf(index);
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
        return duration(root, PatchHooks.currentSubAbility());
    }

    static String duration(SpellAbility root, Object pointer) {
        String declared = declaredDuration(root, pointer);
        if (declared == null) {
            return INSTANT;
        }
        String key = declared.toLowerCase(Locale.ROOT);
        if (LASTING.contains(key)) {
            return PERMANENT;
        }
        return END_OF_TURN;
    }

    /**
     * The card whose line produced this outcome, as an entity ref.
     *
     * <p>The bracket's own answer to "what caused this", for the many bus
     * events that name no source of their own — a life change, a counter, a
     * tap, a card moving. It is the host card of the clause that was resolving,
     * falling back to the acting line's, read in the same order and from the
     * same pointer as {@link #duration}: a clause that resolves out of an
     * effect card names that card rather than the line the bracket opened on.
     *
     * <p>Deliberately the same value the trait-derived side produces for a
     * {@code SpellAbility} cause, so a {@code zone_change} written from a
     * replacement's parameter map and one written from the bus name the same
     * card in the same spelling.
     *
     * <p>Null when nothing was resolving — a combat damage step, a turn-based
     * action — which writes no {@code cause} at all rather than a placeholder.
     * A caller that has a better answer than the bracket, such as the source
     * the bus itself named on a damage event, passes it to
     * {@link #stamp(EffectEvent, SpellAbility, Object, String)} and it wins.
     */
    static String cause(SpellAbility root, Object pointer) {
        Card host = hostOf(pointer instanceof SpellAbility resolving ? resolving : null);
        if (host == null) {
            host = hostOf(root);
        }
        return host == null ? null : SnapshotBuilder.entityId(host);
    }

    private static Card hostOf(SpellAbility ability) {
        return ability == null ? null : ability.getHostCard();
    }

    private static String declaredDuration(SpellAbility root, Object pointer) {
        if (pointer instanceof SpellAbility resolving) {
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
     * Stamp all three onto an event, from the bracket alone.
     *
     * <p>Applied where an event is filed rather than where it is built, because
     * only the collector knows which ability's bracket it landed in.
     */
    static EffectEvent stamp(EffectEvent event, SpellAbility root) {
        return stamp(event, root, PatchHooks.currentSubAbility(), null);
    }

    /**
     * The same, with a cause the caller read off the event itself.
     *
     * <p>{@code namedCause} wins over the bracket's, and null defers to it.
     * The precedence is the caller's to state rather than something read back
     * off the half-built event: the engine's own naming — the creature a
     * {@code GameEventCardDamaged} says dealt the damage — is more precise than
     * "whatever line was resolving", and in a combat damage step it is the only
     * answer there is.
     */
    static EffectEvent stamp(EffectEvent event, SpellAbility root, String namedCause) {
        // Read once: all three fields answer the same question about the same
        // instant, and reading the hook twice could straddle a clause boundary
        // in a nested resolution.
        return stamp(event, root, PatchHooks.currentSubAbility(), namedCause);
    }

    static EffectEvent stamp(EffectEvent event, SpellAbility root, Object pointer) {
        return stamp(event, root, pointer, null);
    }

    static EffectEvent stamp(
            EffectEvent event, SpellAbility root, Object pointer, String namedCause) {
        if (event == null) {
            return null;
        }
        return event
                .duration(duration(root, pointer))
                .attributedTo(attributedTo(root, pointer))
                .cause(namedCause != null ? namedCause : cause(root, pointer));
    }
}
