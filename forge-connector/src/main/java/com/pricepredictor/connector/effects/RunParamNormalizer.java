package com.pricepredictor.connector.effects;

import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
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
 * mode and the map alone, never on the game it came from.
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
        EffectEvent event = new EffectEvent(typeOf(mode, byName));
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
        normalizeParams(event, mode, byName);
        event.cause(causeOf(byName));
        return event;
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
     * Which outcome this is, before any parameter is read into it.
     *
     * <p>Almost always the mode's own mapping. The exception is prevention:
     * Forge reports it through the damage keys, and an event that says
     * {@code damage_dealt} with no amount is not what happened.
     */
    private static String typeOf(String mode, Map<String, Object> byName) {
        // Checked before the lookup rather than left to getOrDefault:
        // EVENT_BY_MODE is a Map.of, which throws on a null key instead of
        // answering the default. A trait whose getMode() is absent -- a hook
        // whose signature drifted, so the argument is not the trait we expect --
        // would take that NPE inside a dynamic proxy, and the engine would
        // surface it as an UndeclaredThrowableException in the middle of
        // canRunTrigger.
        String type = mode == null
                ? EffectEvent.STATE_FLAG_CHANGE
                : EVENT_BY_MODE.getOrDefault(mode, EffectEvent.STATE_FLAG_CHANGE);
        if (EffectEvent.DAMAGE_DEALT.equals(type)
                && !byName.containsKey("DamageAmount")
                && byName.containsKey("PreventedAmount")) {
            return EffectEvent.DAMAGE_PREVENTED;
        }
        return type;
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
     * {@code CounterAdded} ({@code Card.java:1790}) and on {@code LifeGained}
     * ({@code Player.java:487}), {@code SourceSA} on {@code BecomesTarget}
     * ({@code SpellAbilityStackInstance.java:170}), {@code Causer} on
     * {@code Destroyed} ({@code GameAction.java:2161}), {@code Activator} on
     * {@code TapsForMana} ({@code AbilityManaPart.java:251}).
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
     * turn's player ({@code PhaseHandler.java:438}, with the {@code Phase} key
     * commented out), {@code Untaps} carries only the card
     * ({@code Card.java:4864}), and {@code Attacks} and {@code Blocks} carry
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
     * Fill the slots this event type declares, from the values the map holds.
     *
     * <p>Per type and no wider: the vocabulary normalizes an outcome so that the
     * same thing from two different script APIs lands in the same slots, and a
     * key written outside the type's own set is one no reader will look for. A
     * type this has no reading for carries its mode and its cause and nothing
     * else, claiming no more than it read.
     */
    private static void normalizeParams(
            EffectEvent event, String mode, Map<String, Object> byName) {
        switch (event.type()) {
            case EffectEvent.ZONE_CHANGE -> {
                event.param("from_zone", zoneName(byName.get("Origin")));
                event.param("to_zone", zoneName(byName.get("Destination")));
            }
            case EffectEvent.DAMAGE_DEALT -> {
                event.param("amount", intOf(byName.get("DamageAmount")));
                Boolean combat = boolOf(byName.get("IsCombatDamage"));
                event.param("combat", combat != null
                        ? combat : boolOf(byName.get("IsCombat")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            case EffectEvent.DAMAGE_PREVENTED -> {
                event.param("amount", intOf(byName.get("PreventedAmount")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            case EffectEvent.DESTROYED ->
                    event.param("regenerable", boolOf(byName.get("Regeneration")));
            case EffectEvent.COUNTER_CHANGE -> counterParams(event, byName);
            // "Amount" last: it is how the replacement side spells the same
            // quantity the trigger side calls LifeAmount, and without it a
            // life-total replacement carries no number to differ in.
            case EffectEvent.LIFE_CHANGE -> event.param(
                    "delta",
                    firstInt(byName, "LifeAmount", "LifeGained", "Amount"));
            case EffectEvent.CARD_DRAWN, EffectEvent.CARD_MILLED,
                    EffectEvent.CARD_DISCARDED ->
                    event.param("count", firstInt(byName, "Number", "Num"));
            // Scry and Surveil each spell their own count, and neither is
            // Number: read under the general names alone this type carried no
            // count at all.
            case EffectEvent.CARD_LOOKED_AT -> {
                event.param("count", firstInt(
                        byName, "ScryNum", "SurveilNum", "Number", "Num"));
                event.param("from_zone", zoneName(byName.get("Origin")));
            }
            case EffectEvent.POISON_CHANGE -> event.param(
                    "delta", firstInt(byName, "Amount", "Num", "Number"));
            case EffectEvent.MANA_PRODUCED, EffectEvent.MANA_LOST ->
                    event.param("mana_by_color", manaOf(byName));
            case EffectEvent.PHASED ->
                    // Two modes, one type, and the direction is the whole of
                    // the difference between them.
                    event.param("out", !"PhaseIn".equals(mode));
            case EffectEvent.ATTACHED ->
                    event.param("attached_to", firstRef(
                            byName, "AttachTarget", "Attached", "Card"));
            case EffectEvent.UNATTACHED ->
                    event.param("detached_from", firstRef(
                            byName, "AttachTarget", "Attached", "Card"));
            case EffectEvent.ATTACKERS_DECLARED ->
                    // Attacked first: CombatUtil.checkDeclaredAttacker spells
                    // the entity this creature attacked that way and no other.
                    event.param("defender", firstRef(
                            byName, "Attacked", "Defender", "AttackedTarget",
                            "DefendingPlayer", "Defenders"));
            case EffectEvent.BLOCKERS_DECLARED ->
                    event.param("blocked", firstRef(
                            byName, "Attackers", "Attacker"));
            case EffectEvent.BECAME_BLOCKED ->
                    event.param("blockers", refList(byName, "Blockers", "Blocker"));
            case EffectEvent.FACE_CHANGE ->
                    event.param("to_state", faceState(mode));
            case EffectEvent.PLAYER_LOST ->
                    event.param("reason", textOf(byName.get("LoseReason")));
            case EffectEvent.SPELL_COPIED ->
                    event.param("count", firstInt(byName, "Number", "Num"));
            case EffectEvent.TOKEN_CREATED -> event.param(
                    "count", firstInt(byName, "TokenNum", "Number", "Num"));
            default -> {
                // Tapped, untapped, phased, spell_cast and the generic outcome
                // all carry their subject and their mode, which is everything
                // the vocabulary declares for them.
            }
        }
    }

    /**
     * Which counter moved and by how much.
     *
     * <p>Forge says it two ways. The plain one names the type and the number.
     * The other hands over {@code CounterMap}, a per-player multiset that a
     * counter-replacement rewrites <b>in place</b> — which is also why the
     * engine-side copy of the parameter map has to reach inside it, or both
     * halves of the record read the same however carefully this reads them.
     *
     * <p>A map naming several counter types at once has one delta and no single
     * type: the total is the honest answer and a joined type name would be a
     * value no reader could match.
     */
    private static void counterParams(EffectEvent event, Map<String, Object> byName) {
        Object type = byName.get("CounterType");
        Integer delta = firstInt(
                byName, "CounterNum", "CounterAmount", "NewCounterAmount");
        if (type != null) {
            event.param("counter_type", counterName(type));
        }
        if (delta != null) {
            event.param("delta", delta);
            return;
        }
        if (!(byName.get("CounterMap") instanceof Map<?, ?> counters)) {
            return;
        }
        Map<String, Integer> total = new LinkedHashMap<>();
        for (Object value : counters.values()) {
            if (!(value instanceof com.google.common.collect.Multiset<?> multiset)) {
                continue;
            }
            for (com.google.common.collect.Multiset.Entry<?> entry
                    : multiset.entrySet()) {
                total.merge(
                        counterName(entry.getElement()), entry.getCount(), Integer::sum);
            }
        }
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
        event.param("delta", sum);
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
     * Which face a card turned to.
     *
     * <p>Read from the mode rather than from a parameter, because the two modes
     * that reach this type each mean exactly one direction and neither map
     * carries a usable state name: {@code AbilityKey.CardState} exists but is a
     * {@code CardState} object put there by door unlocking alone
     * ({@code Card.java:8146}), and rendering it would fill the slot with an
     * object's {@code toString} instead of a face.
     */
    private static String faceState(String mode) {
        return "TurnFaceUp".equals(mode) ? "face_up" : "transformed";
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

    private static String counterName(Object type) {
        return type instanceof forge.game.card.CounterType counter
                ? counter.getName().toUpperCase(java.util.Locale.ROOT)
                : String.valueOf(type).toUpperCase(java.util.Locale.ROOT);
    }

    /** A zone, however the parameter spells it. */
    private static String zoneName(Object value) {
        if (value instanceof ZoneType zone) {
            return zone.name().toLowerCase(java.util.Locale.ROOT);
        }
        return value == null
                ? null : String.valueOf(value).toLowerCase(java.util.Locale.ROOT);
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
            return SnapshotBuilder.entityId(ability.getHostCard());
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

    /**
     * Forge modes that have a counterpart in the event vocabulary.
     *
     * <p>Deliberately partial. The vocabulary names observable outcomes and
     * Forge names engine hooks, so only some pairs are the same thing; the rest
     * carry their mode as a parameter and take the generic type.
     */
    private static final Map<String, String> EVENT_BY_MODE = Map.ofEntries(
            Map.entry("ChangesZone", EffectEvent.ZONE_CHANGE),
            Map.entry("Moved", EffectEvent.ZONE_CHANGE),
            Map.entry("Destroy", EffectEvent.DESTROYED),
            Map.entry("Destroyed", EffectEvent.DESTROYED),
            Map.entry("Sacrificed", EffectEvent.SACRIFICED),
            Map.entry("DamageDone", EffectEvent.DAMAGE_DEALT),
            Map.entry("DamageDealtOnce", EffectEvent.DAMAGE_DEALT),
            Map.entry("DamageAll", EffectEvent.DAMAGE_DEALT),
            Map.entry("Drawn", EffectEvent.CARD_DRAWN),
            Map.entry("Draw", EffectEvent.CARD_DRAWN),
            // The replacement side's spelling of the same event. This table was
            // written from the trigger vocabulary, so 31 of Forge's 42
            // ReplacementTypes fell through to the generic outcome carrying
            // nothing but their mode -- and an event with no value in it cannot
            // differ from itself, which is how a rewrite that halved a draw got
            // dropped as an identity pair. Only the five whose quantity a
            // reading above already picks up are added here; the rest are what
            // the identity tally is instrumented to answer.
            Map.entry("DrawCards", EffectEvent.CARD_DRAWN),
            Map.entry("Discarded", EffectEvent.CARD_DISCARDED),
            Map.entry("Milled", EffectEvent.CARD_MILLED),
            Map.entry("Mill", EffectEvent.CARD_MILLED),
            Map.entry("LifeGained", EffectEvent.LIFE_CHANGE),
            Map.entry("LifeLost", EffectEvent.LIFE_CHANGE),
            Map.entry("GainLife", EffectEvent.LIFE_CHANGE),
            Map.entry("LoseLife", EffectEvent.LIFE_CHANGE),
            Map.entry("LifeReduced", EffectEvent.LIFE_CHANGE),
            Map.entry("PayLife", EffectEvent.LIFE_CHANGE),
            Map.entry("Poisoned", EffectEvent.POISON_CHANGE),
            Map.entry("CounterAdded", EffectEvent.COUNTER_CHANGE),
            Map.entry("CounterAddedOnce", EffectEvent.COUNTER_CHANGE),
            Map.entry("CounterRemoved", EffectEvent.COUNTER_CHANGE),
            Map.entry("AddCounter", EffectEvent.COUNTER_CHANGE),
            Map.entry("RemoveCounter", EffectEvent.COUNTER_CHANGE),
            Map.entry("Taps", EffectEvent.TAPPED),
            Map.entry("TapsForMana", EffectEvent.MANA_PRODUCED),
            Map.entry("Untaps", EffectEvent.UNTAPPED),
            Map.entry("Attached", EffectEvent.ATTACHED),
            Map.entry("Unattach", EffectEvent.UNATTACHED),
            Map.entry("SpellCast", EffectEvent.SPELL_CAST),
            Map.entry("SpellAbilityCast", EffectEvent.SPELL_CAST),
            Map.entry("Countered", EffectEvent.SPELL_COUNTERED),
            Map.entry("TokenCreated", EffectEvent.TOKEN_CREATED),
            Map.entry("CreateToken", EffectEvent.TOKEN_CREATED),
            Map.entry("TokenCreatedOnce", EffectEvent.TOKEN_CREATED),
            Map.entry("Attackers", EffectEvent.ATTACKERS_DECLARED),
            Map.entry("AttackersDeclared", EffectEvent.ATTACKERS_DECLARED),
            // Attacks is one creature being declared as an attacker, which is
            // exactly what attackers_declared names, and one of the heaviest
            // trait-derived modes. Under the generic type it would carry
            // nothing but its own name; as attackers_declared it carries the
            // defender.
            Map.entry("Attacks", EffectEvent.ATTACKERS_DECLARED),
            Map.entry("Blocks", EffectEvent.BLOCKERS_DECLARED),
            // Retargeted: the subject of an AttackerBlocked event is the
            // attacker, not the blocker, so blockers_declared would put the wrong
            // creature in the slot named "blocked". became_blocked is the type
            // whose subject is the attacker and whose param is the blockers.
            Map.entry("AttackerBlocked", EffectEvent.BECAME_BLOCKED),
            Map.entry("AttackerBlockedByCreature", EffectEvent.BECAME_BLOCKED),
            Map.entry("TurnFaceUp", EffectEvent.FACE_CHANGE),
            Map.entry("Shuffled", EffectEvent.LIBRARY_SHUFFLED),
            Map.entry("Scry", EffectEvent.CARD_LOOKED_AT),
            Map.entry("Surveil", EffectEvent.CARD_LOOKED_AT),
            Map.entry("PhaseOut", EffectEvent.PHASED),
            Map.entry("PhaseIn", EffectEvent.PHASED),
            // ── the replacement side's own spellings ────────────────────
            // ReplacementType names its modes differently from TriggerType,
            // and the entries above follow the trigger vocabulary. A
            // replacement mode with no entry here takes the generic type and
            // carries nothing but its mode -- and an event with no value in it
            // cannot differ from itself, so the record reads as a rewrite of
            // nothing however much the replacement changed.
            Map.entry("Untap", EffectEvent.UNTAPPED),
            Map.entry("Tap", EffectEvent.TAPPED),
            Map.entry("DealtDamage", EffectEvent.DAMAGE_DEALT),
            Map.entry("Counter", EffectEvent.SPELL_COUNTERED),
            Map.entry("CopySpell", EffectEvent.SPELL_COPIED),
            Map.entry("ProduceMana", EffectEvent.MANA_PRODUCED),
            Map.entry("LoseMana", EffectEvent.MANA_LOST),
            Map.entry("Transform", EffectEvent.FACE_CHANGE),
            Map.entry("DeclareBlocker", EffectEvent.BLOCKERS_DECLARED),
            Map.entry("RollDice", EffectEvent.DICE_ROLLED),
            Map.entry("GameWin", EffectEvent.PLAYER_WON),
            Map.entry("GameLoss", EffectEvent.PLAYER_LOST));

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
