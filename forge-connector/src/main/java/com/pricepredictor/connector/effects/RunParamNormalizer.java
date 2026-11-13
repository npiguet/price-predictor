package com.pricepredictor.connector.effects;

import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.card.CounterType;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
 * <p>Each mode is read by its own row ({@link #TRIGGER_MODES},
 * {@link #REPLACEMENT_MODES}): Forge spells the same quantity differently per
 * mode, and a key one mode fills with an amount another fills with an
 * ordinal or a running total.
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
        String mode = modeOf(trait);
        Map<String, Object> byName = paramsByName(params);
        Reading reading = readingOf(trait, mode);
        EffectEvent event = new EffectEvent(reading.type());
        // The engine's own name for what happened, kept whether or not the
        // vocabulary has a member for it. Forge has some two hundred trigger
        // modes and the vocabulary is a closed set of outcomes, so mapping
        // every one would be a table nobody could keep true; carrying the mode
        // loses nothing and lets a later stage map more of them.
        if (mode != null) {
            event.param("mode", mode);
        }
        if (byName.isEmpty()) {
            return event;
        }
        for (String subject : subjectsOf(byName)) {
            event.subject(subject);
        }
        normalizeParams(event, mode, reading, byName);
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
                && event.params().get("amount") instanceof Number amount
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
     * How this trait's mode is read: its own row of the trigger table or of
     * the replacement table, else the generic type with nothing read.
     *
     * <p>Checked for null before the lookup: a trait whose {@code getMode()}
     * is absent — a hook whose signature drifted, so the argument is not the
     * trait we expect — must not take an NPE inside a dynamic proxy, which the
     * engine would surface as an {@code UndeclaredThrowableException} in the
     * middle of {@code canRunTrigger}.
     */
    private static Reading readingOf(Object trait, String mode) {
        Map<String, Reading> table = trait instanceof ReplacementEffect
                ? REPLACEMENT_MODES : TRIGGER_MODES;
        Reading reading = mode == null ? null : table.get(mode);
        return reading != null ? reading : GENERIC;
    }

    /**
     * Keys that name what an event is <b>about</b>, most specific first.
     *
     * <p>A parameter map names one card several times over: {@code Card},
     * {@code CardLKI} and {@code Affected} routinely all point at it. The
     * others in the map — {@code Cause}, {@code DamageSource},
     * {@code LastState*} — are causes and context, not subjects, and reading
     * every entry as one would make the subject list meaningless.
     */
    private static final List<String> SUBJECT_KEYS = List.of(
            "Affected", "Card", "Player", "DamageTarget", "Attacker",
            // Combat's own names, in the order that keeps each event about the
            // creature it is about: an AttackerBlocked map holds Attacker and
            // Blockers and is about the attacker, a Blocks map holds Blocker
            // and Attackers and is about the blocker, and an AttackersDeclared
            // map holds only the collection. Without these three, every one of
            // them would name no subject at all.
            "Blocker", "Attackers", "Attached",
            // The Aura or Equipment an Attached or Unattached trigger is
            // about; AttachTarget and Object name what it hangs or hung on.
            "AttachSource",
            // Last, and only as a last resort. CardLKI is a copy of the card as
            // it was, so wherever the live object is also in the map one of the
            // keys above names it and this is never reached. Where it is not --
            // MagicStack puts CardLKI, Activator and SpellAbility and no Card at
            // all -- it is the only name the map has for the spell that was
            // cast, and without it every SpellCast event would name no
            // subject whatever.
            "CardLKI");

    /** How many subjects one event may name before the list is capped. */
    private static final int MAX_SUBJECTS = 8;

    /**
     * What this event is about: the first subject key the map carries.
     *
     * <p>A collection value contributes each of its members — a damage-all
     * event is about all of them — deduplicated, because the same card can
     * appear twice in one collection after a state change.
     */
    private static Set<String> subjectsOf(Map<String, Object> byName) {
        for (String key : SUBJECT_KEYS) {
            Set<String> subjects = new LinkedHashSet<>();
            refsIn(byName.get(key), subjects);
            if (!subjects.isEmpty()) {
                return subjects;
            }
        }
        return Set.of();
    }

    /**
     * Add what one parameter value names to {@code into}: the value itself, or
     * each member of a collection, until {@code into} holds
     * {@value #MAX_SUBJECTS}.
     */
    private static void refsIn(Object value, Set<String> into) {
        Iterable<?> members = value instanceof Iterable<?> many
                ? many : Collections.singletonList(value);
        for (Object member : members) {
            String ref = subjectOf(member);
            if (ref != null && into.size() < MAX_SUBJECTS) {
                into.add(ref);
            }
        }
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
     * Fill the slots this event type declares, from the values the map holds,
     * as the mode's row reads them.
     *
     * <p>Per type and no wider: the vocabulary normalizes an outcome so that the
     * same thing from two different script APIs lands in the same slots, and a
     * key written outside the type's own set is one no reader will look for. A
     * type this has no reading for carries its mode and its cause and nothing
     * else, claiming no more than it read.
     */
    private static void normalizeParams(
            EffectEvent event, String mode, Reading reading, Map<String, Object> byName) {
        switch (event.type()) {
            case ZONE_CHANGE -> {
                event.param("from_zone", CardSpelling.zoneName(byName.get("Origin")));
                event.param("to_zone", CardSpelling.zoneName(byName.get("Destination")));
            }
            case DAMAGE_DEALT -> {
                event.param("amount", intOf(byName.get("DamageAmount")));
                Boolean combat = boolOf(byName.get("IsCombatDamage"));
                event.param("combat", combat != null
                        ? combat : boolOf(byName.get("IsCombat")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            // DamagePreventedOnce (CardDamageTable): the damage prevented, as
            // DamageAmount.
            case DAMAGE_PREVENTED -> {
                event.param("amount", intOf(byName.get("DamageAmount")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            case DESTROYED ->
                    event.param("regenerable", boolOf(byName.get("Regeneration")));
            case COUNTER_CHANGE -> counterParams(event, reading, byName);
            // LifeAmount on the triggers, LifeGained on the GainLife
            // replacement, Amount on LifeReduced and PayLife's.
            case LIFE_CHANGE -> event.param("delta", signed(
                    reading, firstInt(byName, "LifeAmount", "LifeGained", "Amount")));
            case CARD_DRAWN, CARD_MILLED, CARD_DISCARDED, TOKEN_CREATED, SPELL_COPIED ->
                    event.param("count", reading.amount().read(byName));
            // Scry and Surveil each spell their own count, and neither is
            // Number: read under the general names alone this type carried no
            // count at all.
            case CARD_LOOKED_AT -> {
                event.param("count", firstInt(
                        byName, "ScryNum", "SurveilNum", "Number", "Num"));
                event.param("from_zone", CardSpelling.zoneName(byName.get("Origin")));
            }
            case MANA_PRODUCED, MANA_LOST ->
                    event.param("mana_by_color", manaOf(byName));
            case PHASED ->
                    // Two modes, one type, and the direction is the whole of
                    // the difference between them.
                    event.param("out", !"PhaseIn".equals(mode));
            case ATTACHED ->
                    event.param("attached_to", firstRef(byName, "AttachTarget"));
            // Card.unattachFromEntity: the permanent or player it left, as Object.
            case UNATTACHED ->
                    event.param("detached_from", firstRef(byName, "Object"));
            case ATTACKERS_DECLARED ->
                    // Attacked first: CombatUtil.checkDeclaredAttacker spells
                    // the entity this creature attacked that way and no other.
                    event.param("defender", firstRef(
                            byName, "Attacked", "Defender", "AttackedTarget",
                            "DefendingPlayer", "Defenders"));
            case BLOCKERS_DECLARED ->
                    event.param("blocked", firstRef(
                            byName, "Attackers", "Attacker"));
            case BECAME_BLOCKED ->
                    event.param("blockers", refList(byName, "Blockers", "Blocker"));
            case FACE_CHANGE ->
                    event.param("to_state", faceState(byName));
            case PLAYER_LOST ->
                    event.param("reason", textOf(byName.get("LoseReason")));
            default -> {
                // Tapped, untapped, spell_cast and the generic outcome all
                // carry their subject and their mode, which is everything
                // the vocabulary declares for them.
            }
        }
    }

    /**
     * A magnitude read off the map, signed the way its mode moves it: a loss
     * mode's amount negative, a gain mode's positive, and a mode of sign 0
     * the amount as read.
     */
    private static Integer signed(Reading reading, Integer amount) {
        if (amount == null || reading.sign() == 0) {
            return amount;
        }
        int size = Math.abs(amount);
        return reading.sign() < 0 ? -size : size;
    }

    /**
     * Which counter moved and by how much, signed: positive for counters put
     * on, negative for counters removed.
     *
     * <p>Forge says it three ways. {@code CounterAdded} and {@code
     * CounterRemoved} fire once per counter, their {@code CounterAmount} and
     * {@code NewCounterAmount} the count after that one counter moved, so each
     * is one counter. {@code CounterAddedOnce} and {@code CounterRemovedOnce}
     * carry the number moved as {@code CounterAmount}. The {@code AddCounter}
     * replacement hands over {@code CounterMap}, a per-player multiset that it
     * rewrites <b>in place</b> — which is also why the engine-side copy of the
     * parameter map has to reach inside it, or both halves of the record read
     * the same however carefully this reads them. The {@code RemoveCounter}
     * replacement carries the count the removal would leave ({@code Result}),
     * so what it removes is that count less the card's count now
     * ({@link #removalOf}).
     *
     * <p>A map naming several counter types at once has one delta and no single
     * type: the total is the honest answer and a joined type name would be a
     * value no reader could match.
     */
    private static void counterParams(
            EffectEvent event, Reading reading, Map<String, Object> byName) {
        Object type = byName.get("CounterType");
        if (type != null) {
            event.param("counter_type", CardSpelling.counterName(type));
        }
        Integer amount = reading.amount().read(byName);
        if (amount != null) {
            event.param("delta", signed(reading, amount));
            return;
        }
        Map<String, Integer> total = counterMapTotals(byName.get("CounterMap"));
        if (total.isEmpty()) {
            return;
        }
        if (total.size() == 1 && type == null) {
            event.param("counter_type", total.keySet().iterator().next());
        }
        int sum = 0;
        for (Integer count : total.values()) {
            sum += count;
        }
        event.param("delta", signed(reading, sum));
    }

    /**
     * What a {@code RemoveCounter} replacement's removal comes to, as a
     * signed change: the count it would leave ({@code Result}, never below 0)
     * less the affected object's count of that type now, the removal not
     * having happened yet; null where either is unreadable.
     */
    private static Integer removalOf(Map<String, Object> byName) {
        Integer result = intOf(byName.get("Result"));
        if (result == null
                || !(byName.get("CounterType") instanceof CounterType counterType)
                || !(byName.get("Affected") instanceof GameEntity affected)) {
            return null;
        }
        return Math.max(result, 0) - affected.getCounters(counterType);
    }

    /**
     * A {@code CounterMap}'s counts per counter type, summed over its players,
     * in the order the map yields them; empty for anything that is not one.
     */
    private static Map<String, Integer> counterMapTotals(Object counterMap) {
        Map<String, Integer> total = new LinkedHashMap<>();
        if (!(counterMap instanceof Map<?, ?> counters)) {
            return total;
        }
        for (Object value : counters.values()) {
            if (!(value instanceof com.google.common.collect.Multiset<?> multiset)) {
                continue;
            }
            for (com.google.common.collect.Multiset.Entry<?> entry
                    : multiset.entrySet()) {
                total.merge(
                        CardSpelling.counterName(entry.getElement()), entry.getCount(), Integer::sum);
            }
        }
        return total;
    }

    /**
     * The mana a mana event moved, in the slot the vocabulary declares.
     *
     * <p>Forge hands produced mana over as its own short text -- {@code "G"},
     * {@code "G G"}, {@code "1"} -- so the reading is the one
     * {@link #manaByColor(String)} already does for the activation channel,
     * which is what keeps a tapped land and a replaced mana production
     * comparable. A value that is not text yields nothing rather than an empty
     * object: absent is the honest answer for a mana structure this side does
     * not link against, and {@code {}} would claim no mana was produced.
     */
    private static Map<String, Integer> manaOf(Map<String, Object> byName) {
        Object produced = byName.get("Produced");
        if (!(produced instanceof String)) {
            produced = byName.get("Mana");
        }
        if (!(produced instanceof String text)) {
            return null;
        }
        Map<String, Integer> mana = manaByColor(text.trim());
        return mana.isEmpty() ? null : mana;
    }

    /**
     * Which face a card turned to: the face it shows now, in an entity's
     * {@code face_state} spelling ({@code front}, {@code back},
     * {@code flipped}, {@code melded}, {@code face_down}).
     *
     * <p>Read off the card rather than the mode: Forge runs the {@code
     * Transform} and {@code TurnFaceUp} replacements and the {@code
     * Transformed} and {@code TurnFaceUp} triggers once the card has turned
     * (a transform lands on either face), and no map carries a usable state
     * name: {@code AbilityKey.CardState} is a {@code CardState} object only
     * door unlocking puts there ({@code Card.unlockRoom}).
     */
    private static String faceState(Map<String, Object> byName) {
        Object card = byName.get("Card");
        if (!(card instanceof Card)) {
            card = byName.get("Affected");
        }
        return card instanceof Card turned ? CardFacts.faceState(turned) : null;
    }

    /** The first of these keys that names an entity, single or collection. */
    private static String firstRef(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            Set<String> refs = new LinkedHashSet<>();
            refsIn(byName.get(key), refs);
            if (!refs.isEmpty()) {
                return refs.iterator().next();
            }
        }
        return null;
    }

    /** Every entity these keys name, deduplicated, or null for none. */
    private static List<String> refList(Map<String, Object> byName, String... keys) {
        Set<String> refs = new LinkedHashSet<>();
        for (String key : keys) {
            refsIn(byName.get(key), refs);
        }
        return refs.isEmpty() ? null : new ArrayList<>(refs);
    }

    /** A parameter's own text, or null where it holds none. */
    private static String textOf(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    private static Integer firstInt(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            Integer value = intOf(byName.get(key));
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static Integer intOf(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return value == null ? null : Integer.valueOf(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            // A script variable Forge has not evaluated yet. Absent is the
            // right answer; zero would be a claim.
            return null;
        }
    }

    private static Boolean boolOf(Object value) {
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value);
        return "true".equalsIgnoreCase(text) || "false".equalsIgnoreCase(text)
                ? Boolean.valueOf(text.equalsIgnoreCase("true")) : null;
    }

    /**
     * The entity a run-parameter value names, or null when it names none.
     *
     * <p>A replacement's and a trigger's parameter maps are the only place
     * those records say what they are about: unread, every rewrite and trigger
     * event would name no subjects at all, leaving the pending-event overlay
     * with nothing to attach to.
     */
    private static String subjectOf(Object value) {
        if (value instanceof SpellAbility ability && ability.getHostCard() != null) {
            return Refs.of(ability.getHostCard());
        }
        return Refs.ofAny(value).orElse(null);
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

    /** How a mode's amount is read off its map; null where the map holds none. */
    @FunctionalInterface
    interface Amount {
        Integer read(Map<String, Object> byName);

        /** Nothing read here: the event type's own reading applies. */
        Amount NONE = byName -> null;
        /** One: a mode Forge fires once per card or per counter. */
        Amount ONE = byName -> 1;

        /** The first of these keys holding a number. */
        static Amount key(String... keys) {
            return byName -> firstInt(byName, keys);
        }

        /** How many objects the collection under {@code key} holds. */
        static Amount sizeOf(String key) {
            return byName -> byName.get(key) instanceof Collection<?> many ? many.size() : null;
        }
    }

    /**
     * One mode's reading: its event type, how its amount is read, and the
     * direction it moves a life total or a counter count: {@code 1} a gain
     * (written positive), {@code -1} a loss (written negative), {@code 0}
     * the amount as read, already signed.
     */
    record Reading(EventType type, Amount amount, int sign) {

        static Reading of(EventType type) {
            return new Reading(type, Amount.NONE, 1);
        }

        static Reading of(EventType type, Amount amount) {
            return new Reading(type, amount, 1);
        }

        static Reading loss(EventType type, Amount amount) {
            return new Reading(type, amount, -1);
        }
    }

    /** A mode with no row: the generic type, carrying its mode and nothing read. */
    private static final Reading GENERIC = Reading.of(EventType.STATE_FLAG_CHANGE);

    /**
     * Forge's trigger modes ({@code TriggerType} names) that have a
     * counterpart in the event vocabulary, each read as its own run
     * parameters hold it.
     *
     * <p>Deliberately partial. The vocabulary names observable outcomes and
     * Forge names engine hooks, so only some pairs are the same thing; the rest
     * carry their mode as a parameter and take the generic type.
     */
    static final Map<String, Reading> TRIGGER_MODES = Map.ofEntries(
            Map.entry("ChangesZone", Reading.of(EventType.ZONE_CHANGE)),
            Map.entry("Destroyed", Reading.of(EventType.DESTROYED)),
            Map.entry("Sacrificed", Reading.of(EventType.SACRIFICED)),
            Map.entry("DamageDone", Reading.of(EventType.DAMAGE_DEALT)),
            Map.entry("DamageDealtOnce", Reading.of(EventType.DAMAGE_DEALT)),
            Map.entry("DamageAll", Reading.of(EventType.DAMAGE_DEALT)),
            Map.entry("DamagePreventedOnce", Reading.of(EventType.DAMAGE_PREVENTED)),
            // Drawn, Discarded, Milled and TokenCreated fire once per card;
            // Drawn's Number and TokenCreated's Num are the turn's ordinals,
            // not counts.
            Map.entry("Drawn", Reading.of(EventType.CARD_DRAWN, Amount.ONE)),
            Map.entry("Discarded", Reading.of(EventType.CARD_DISCARDED, Amount.ONE)),
            Map.entry("Milled", Reading.of(EventType.CARD_MILLED, Amount.ONE)),
            Map.entry("LifeGained", Reading.of(EventType.LIFE_CHANGE)),
            Map.entry("LifeLost", Reading.loss(EventType.LIFE_CHANGE, Amount.NONE)),
            Map.entry("PayLife", Reading.loss(EventType.LIFE_CHANGE, Amount.NONE)),
            Map.entry("CounterAdded", Reading.of(EventType.COUNTER_CHANGE, Amount.ONE)),
            Map.entry("CounterAddedOnce",
                    Reading.of(EventType.COUNTER_CHANGE, Amount.key("CounterAmount"))),
            Map.entry("CounterRemoved", Reading.loss(EventType.COUNTER_CHANGE, Amount.ONE)),
            Map.entry("CounterRemovedOnce",
                    Reading.loss(EventType.COUNTER_CHANGE, Amount.key("CounterAmount"))),
            Map.entry("Taps", Reading.of(EventType.TAPPED)),
            Map.entry("TapsForMana", Reading.of(EventType.MANA_PRODUCED)),
            Map.entry("Untaps", Reading.of(EventType.UNTAPPED)),
            Map.entry("Attached", Reading.of(EventType.ATTACHED)),
            Map.entry("Unattached", Reading.of(EventType.UNATTACHED)),
            Map.entry("SpellCast", Reading.of(EventType.SPELL_CAST)),
            Map.entry("SpellAbilityCast", Reading.of(EventType.SPELL_CAST)),
            Map.entry("Countered", Reading.of(EventType.SPELL_COUNTERED)),
            Map.entry("TokenCreated", Reading.of(EventType.TOKEN_CREATED, Amount.ONE)),
            Map.entry("TokenCreatedOnce",
                    Reading.of(EventType.TOKEN_CREATED, Amount.sizeOf("Cards"))),
            Map.entry("AttackersDeclared", Reading.of(EventType.ATTACKERS_DECLARED)),
            // Attacks is one creature being declared as an attacker, which is
            // exactly what attackers_declared names, and one of the heaviest
            // trait-derived modes. Under the generic type it would carry
            // nothing but its own name; as attackers_declared it carries the
            // defender.
            Map.entry("Attacks", Reading.of(EventType.ATTACKERS_DECLARED)),
            Map.entry("Blocks", Reading.of(EventType.BLOCKERS_DECLARED)),
            // Retargeted: the subject of an AttackerBlocked event is the
            // attacker, not the blocker, so blockers_declared would put the wrong
            // creature in the slot named "blocked". became_blocked is the type
            // whose subject is the attacker and whose param is the blockers.
            Map.entry("AttackerBlocked", Reading.of(EventType.BECAME_BLOCKED)),
            Map.entry("AttackerBlockedByCreature", Reading.of(EventType.BECAME_BLOCKED)),
            Map.entry("TurnFaceUp", Reading.of(EventType.FACE_CHANGE)),
            Map.entry("Transformed", Reading.of(EventType.FACE_CHANGE)),
            Map.entry("Shuffled", Reading.of(EventType.LIBRARY_SHUFFLED)),
            Map.entry("Scry", Reading.of(EventType.CARD_LOOKED_AT)),
            Map.entry("Surveil", Reading.of(EventType.CARD_LOOKED_AT)),
            Map.entry("PhaseOut", Reading.of(EventType.PHASED)),
            Map.entry("PhaseIn", Reading.of(EventType.PHASED)));

    /**
     * Forge's replacement modes ({@code ReplacementType} names) that have a
     * counterpart in the event vocabulary, read the same way.
     *
     * <p>A replacement mode with no row takes the generic type and carries
     * nothing but its mode -- and an event with no value in it cannot differ
     * from itself, so the record reads as a rewrite of nothing however much
     * the replacement changed: a rewrite that halved a draw would be dropped
     * as an identity pair.
     */
    static final Map<String, Reading> REPLACEMENT_MODES = Map.ofEntries(
            Map.entry("Moved", Reading.of(EventType.ZONE_CHANGE)),
            Map.entry("Destroy", Reading.of(EventType.DESTROYED)),
            Map.entry("DamageDone", Reading.of(EventType.DAMAGE_DEALT)),
            Map.entry("DealtDamage", Reading.of(EventType.DAMAGE_DEALT)),
            Map.entry("Draw", Reading.of(EventType.CARD_DRAWN, Amount.ONE)),
            Map.entry("DrawCards", Reading.of(EventType.CARD_DRAWN, Amount.key("Number"))),
            Map.entry("Mill", Reading.of(EventType.CARD_MILLED, Amount.key("Number"))),
            Map.entry("GainLife", Reading.of(EventType.LIFE_CHANGE)),
            Map.entry("LifeReduced", Reading.loss(EventType.LIFE_CHANGE, Amount.NONE)),
            Map.entry("PayLife", Reading.loss(EventType.LIFE_CHANGE, Amount.NONE)),
            Map.entry("AddCounter", Reading.of(EventType.COUNTER_CHANGE)),
            Map.entry("RemoveCounter", new Reading(
                    EventType.COUNTER_CHANGE, RunParamNormalizer::removalOf, 0)),
            Map.entry("CreateToken", Reading.of(EventType.TOKEN_CREATED)),
            Map.entry("Untap", Reading.of(EventType.UNTAPPED)),
            Map.entry("Tap", Reading.of(EventType.TAPPED)),
            Map.entry("Counter", Reading.of(EventType.SPELL_COUNTERED)),
            Map.entry("CopySpell", Reading.of(EventType.SPELL_COPIED, Amount.key("Amount"))),
            Map.entry("ProduceMana", Reading.of(EventType.MANA_PRODUCED)),
            Map.entry("LoseMana", Reading.of(EventType.MANA_LOST)),
            Map.entry("Attached", Reading.of(EventType.ATTACHED)),
            Map.entry("Transform", Reading.of(EventType.FACE_CHANGE)),
            Map.entry("TurnFaceUp", Reading.of(EventType.FACE_CHANGE)),
            Map.entry("DeclareBlocker", Reading.of(EventType.BLOCKERS_DECLARED)),
            Map.entry("RollDice", Reading.of(EventType.DICE_ROLLED)),
            Map.entry("GameWin", Reading.of(EventType.PLAYER_WON)),
            Map.entry("GameLoss", Reading.of(EventType.PLAYER_LOST)));

    /**
     * Forge's produced-mana string as counts per colour letter.
     *
     * <p>Returned as a map rather than rendered here, because
     * {@link EffectEvent#param} renders a map as a JSON object and a string as
     * a JSON string — and the reader expects {@code {"B":1}}, not {@code "{...}"}.
     */
    static Map<String, Integer> manaByColor(String produced) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String atom : produced.split(" ")) {
            if (atom.isEmpty()) {
                continue;
            }
            // A bare number is that many generic; anything else is one mana of
            // the colour it names, "C" included.
            if (atom.chars().allMatch(Character::isDigit)) {
                counts.merge("C", Integer.parseInt(atom), Integer::sum);
            } else {
                counts.merge(atom.toUpperCase(java.util.Locale.ROOT), 1, Integer::sum);
            }
        }
        return counts;
    }
}
