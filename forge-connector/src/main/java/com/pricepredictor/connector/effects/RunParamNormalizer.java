package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.RunParamValues.counterMapTotals;
import static com.pricepredictor.connector.effects.RunParamValues.subjectOf;

/**
 * A patched hook's run-parameter map, read into an event.
 *
 * <p>A replacement's and a trigger's parameter maps are the only place those
 * records say what they are about. Stateless: the reading depends on the trait's
 * side (trigger or replacement), its mode and the map alone, never on the game
 * it came from — save the {@code RemoveCounter} replacement's removal, which
 * reads the affected card's count, and a face change, which reads the face the
 * card shows.
 *
 * <p>Each mode is read by its own row ({@link RunParamRows}), which names its
 * event type, its subject and every parameter it reads: Forge spells the same
 * quantity differently per mode, and a key one mode fills with an amount
 * another fills with an ordinal or a running total.
 */
final class RunParamNormalizer {

    private RunParamNormalizer() {
    }

    /** What a trait whose mode could not be read is counted under. */
    static final String UNKNOWN_MODE = "?";

    /**
     * A patched hook's parameter map as an event.
     *
     * <p>The map is Forge's own {@code AbilityKey} enum, which this side does
     * not link against, so it is read by key <b>name</b> — but by the name of
     * the key, never by the list of them, and what the record carries is the
     * <b>values</b>.
     *
     * <p>Read off the key list instead — a subject per map entry, a
     * {@code cause} of the comma-joined key names — both halves of a rewrite
     * record would be byte-identical by construction, and {@code cause} a
     * near-constant string per event type. Every value that matters already
     * has a slot in the event vocabulary, so filling those slots needs no
     * change to the wire format.
     */
    static EffectEvent describeParams(Object trait, Object params) {
        return describeParams(trait, params, null);
    }

    /**
     * The same, a counter map read for one counter type alone: its {@code
     * counter_type} that type and its {@code delta} that type's count, summed
     * over the map's players (0 where the map holds none of it). An {@code
     * AddCounter} replacement's record over several types is written once per
     * type it changed ({@link #changedCounterTypes}).
     *
     * @param counterType the type, as {@link CardSpelling#counterName} spells
     *                    it, or null to read the whole map
     */
    static EffectEvent describeParams(Object trait, Object params, String counterType) {
        String mode = modeOf(trait);
        Map<String, Object> byName = paramsByName(params);
        ModeRow row = RunParamRows.rowOf(trait, mode);
        EffectEvent event = new EffectEvent(row.type());
        // The engine's own name for what happened, kept whether or not the
        // vocabulary has a member for it. Forge has some two hundred trigger
        // modes and the vocabulary is a closed set of outcomes, so mapping
        // every one would be a table nobody could keep true; carrying the mode
        // loses nothing and lets a later stage map more of them.
        if (mode != null) {
            event.param(EventParam.MODE, mode);
        }
        if (byName.isEmpty()) {
            return event;
        }
        row.subjects().of(byName).forEach(event::subject);
        ModeRow.RunParams read = new ModeRow.RunParams(byName, counterType);
        for (ModeRow.Slot slot : row.slots()) {
            slot.read(event, read);
        }
        event.cause(causeOf(byName));
        if (byName.get("CardLKI") instanceof Card lastKnown) {
            event.lastKnown(lastKnown);
        }
        return event;
    }

    /**
     * Whether this is a prevention that prevented nothing (FR-035), which is
     * written nowhere: a shield whose amount is used up still answers the
     * damage, and a {@code NoPreventDamage} event leaves a prevented amount
     * of 0 in the map. Only a {@code damage_prevented} with a known amount
     * of 0 or less is; an unread amount says nothing either way.
     */
    static boolean preventedNothing(EffectEvent event) {
        return event != null
                && event.type() == EventType.DAMAGE_PREVENTED
                && event.param(EventParam.AMOUNT) instanceof Number amount
                && amount.intValue() <= 0;
    }

    /**
     * The run parameters keyed by the name Forge spells them with.
     *
     * <p>{@code AbilityKey.toString()} is the key's own name, so this is a
     * rename rather than a reading: what it buys is that everything below can
     * ask for {@code "Origin"} without linking against the enum.
     */
    static Map<String, Object> paramsByName(Object params) {
        if (!(params instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> byName = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            byName.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return byName;
    }

    /**
     * Keys that name the object a run-parameter map blames, most specific first.
     *
     * <p>Forge spells the causing object six ways. {@code Cause} answers for
     * {@code ChangesZone}, {@code Sacrificed} and {@code Exiled};
     * {@code SpellAbility} answers for {@code SpellCast} and {@code LifeLost};
     * the other four are where the rest keep it — {@code Source} on
     * {@code CounterAdded} ({@code Card.addCounterInternal}) and on
     * {@code LifeGained} ({@code Player.gainLife}), {@code SourceSA} on
     * {@code BecomesTarget} ({@code SpellAbilityStackInstance.updateTarget}),
     * {@code Causer} on {@code Destroyed} ({@code GameAction.destroy}),
     * {@code Activator} on {@code TapsForMana}
     * ({@code AbilityManaPart.tapsForMana}).
     *
     * <p>Ordered rather than merged because a map often holds several: an
     * explicit {@code Cause} is the engine's own answer and outranks a source
     * card, which outranks the player who happened to be acting.
     *
     * <p>{@code DamageSource} is deliberately absent. The damage types
     * normalize it into their own {@code source} slot, and repeating it here
     * would make the cause channel a copy of a field the reader already has
     * rather than a second fact about the event.
     */
    private static final List<String> CAUSE_KEYS = List.of(
            "Cause", "SpellAbility", "SourceSA", "Source", "Causer", "Activator");

    /**
     * The object that caused this event, as a ref, or null when none is named.
     *
     * <p>{@code cause} is documented as an entity or player ref, never a list
     * of run-parameter key names, which nothing could learn from.
     * A cause that is a spell or ability is named by the card it is on, which
     * is the identity the snapshot carries.
     *
     * <p>Null is a real answer and stays one. {@code Phase} carries only the
     * turn's player ({@code PhaseHandler.onPhaseBegin}, with the {@code Phase}
     * key commented out), {@code Untaps} carries only the card
     * ({@code Card.untap}), and {@code Attacks} and {@code Blocks} carry
     * combat's shape and no actor: nothing in the game caused them in the sense
     * this field means, and inventing one would be worse than the gap.
     */
    private static String causeOf(Map<String, Object> byName) {
        for (String key : CAUSE_KEYS) {
            String ref = subjectOf(byName.get(key));
            if (ref != null) {
                return ref;
            }
        }
        return null;
    }

    /**
     * The counter types an {@code AddCounter} replacement changed the count
     * of, when its map names several, in the order of their names; empty
     * otherwise: for any other trait, a map of one type, a call that changed
     * no type's count, and run parameters Forge could not copy ({@code
     * before} null), where nothing can be compared.
     *
     * <p>A type's count is summed over the map's players, before and after;
     * a type on one side only counts 0 on the other. Each type named is one
     * record's: the result and the ability that ran are the call's, but a
     * replacement reading one type ({@code ValidCounterType$}) leaves the
     * others' counts as they came, and a record of the whole map would say
     * {@code updated} beside the counters it never touched.
     *
     * <p>Ordered by name, not as the map yields them: Forge's counter
     * multisets hash an enum's identity, which differs from one JVM to the
     * next.
     */
    static List<String> changedCounterTypes(Object trait, Object before, Object after) {
        if (before == null || !(trait instanceof ReplacementEffect replacement)) {
            return List.of();
        }
        if (!isAddCounter(replacement)) {
            return List.of();
        }
        Map<String, Integer> was = counterMapTotals(paramsByName(before).get("CounterMap"));
        Map<String, Integer> now = counterMapTotals(paramsByName(after).get("CounterMap"));
        Set<String> types = new TreeSet<>(was.keySet());
        types.addAll(now.keySet());
        if (types.size() < 2) {
            return List.of();
        }
        List<String> changed = new ArrayList<>();
        for (String type : types) {
            if (was.getOrDefault(type, 0).intValue() != now.getOrDefault(type, 0).intValue()) {
                changed.add(type);
            }
        }
        return changed;
    }

    /** The replacement mode whose record is written once per counter type it changed. */
    static final String ADD_COUNTER = "AddCounter";

    private static boolean isAddCounter(ReplacementEffect replacement) {
        return ADD_COUNTER.equals(modeOf(replacement));
    }

    /**
     * A trigger's or replacement's mode.
     *
     * <p>Null for anything else. The trigger and rewrite listeners declare
     * these two types for the trait, so anything else is a signature that
     * drifted, and its record carries no mode rather than a guessed one.
     */
    static String modeOf(Object trait) {
        Object mode = trait instanceof Trigger trigger ? trigger.getMode()
                : trait instanceof ReplacementEffect replacement ? replacement.getMode()
                : null;
        return mode == null ? null : String.valueOf(mode);
    }
}
