package com.pricepredictor.connector.effects;

import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.pricepredictor.connector.json.Json;
import forge.game.Game;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.replacement.ReplacementEffect;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;
import java.util.WeakHashMap;

/**
 * Effects from resolved spells that wait or last, as command-zone objects
 * (FR-024, Decision C4): {@code floating_effects}, {@code delayed_triggers}
 * and {@code dungeons} of a snapshot's global block.
 *
 * <p>Two kinds of floating effect, found two ways:
 *
 * <ul>
 *   <li><b>An effect card</b> ({@code Foo's Effect}) already sits in the
 *       command zone and carries everything: the ability that made it, its
 *       statics' affected cards, what it remembers, its {@code Duration$}, and
 *       a counting-down shield's amount in the SVar its replacement spends.
 *       It is an entity of the snapshot, and its floating effect carries its
 *       entity id.</li>
 *   <li><b>A timestamped change</b> — Giant Growth's +3/+3, "gains flying
 *       until end of turn" — lives only as rows a resolving ability wrote
 *       into a card's layer tables under static id 0, which name no line.
 *       The one join is the timestamp: this class notes, at each live
 *       clause's {@code ON_CLAUSE_RESOLVING}, the game's timestamp, the line
 *       resolving and its {@code Duration$}, and closes the note at
 *       {@code ON_CLAUSE_RESOLVED}; a row's timestamp falls inside the
 *       innermost note open when Forge drew it. A snapshot builds one object
 *       per timestamp still present on some card, id {@code F<timestamp>}. A
 *       row no note covers (one a fork drew, or one written before the
 *       collector listened) is not listed: it names no line.</li>
 * </ul>
 *
 * <p>Pending delayed triggers come from Forge's own lists (the patched
 * {@code TriggerHandler.getPendingDelayedTriggers}), id {@code D<trigger id>},
 * keyed to the ability that spawned them; none are listed on an unpatched
 * Forge.
 *
 * <p>The notes are kept per game in a weak registry, because the clause hook
 * and the several snapshot builders over one game are built apart.
 */
final class FloatingEffects {

    private static final Map<Game, FloatingEffects> BY_GAME =
            Collections.synchronizedMap(new WeakHashMap<>());

    /** The notes of {@code game}, made on first use. */
    static FloatingEffects of(Game game) {
        return BY_GAME.computeIfAbsent(game, g -> new FloatingEffects());
    }

    /** One live clause's resolution: the timestamps Forge drew while it ran. */
    private static final class Note {
        final long start;
        final ProvenanceKey line;
        final String duration;
        /**
         * The clause while it runs, to close the right note; dropped once
         * closed, because a clause holds its game and the registry holds
         * games only weakly.
         */
        SpellAbility clause;
        /** The game's timestamp when the clause finished; open until then. */
        long end = Long.MAX_VALUE;

        Note(long start, ProvenanceKey line, String duration, SpellAbility clause) {
            this.start = start;
            this.line = line;
            this.duration = duration;
            this.clause = clause;
        }

        boolean covers(long timestamp) {
            return timestamp > start && timestamp <= end;
        }
    }

    /** By (start, order noted): clauses nest, and nested ones can share a start. */
    private final NavigableMap<long[], Note> notes = new TreeMap<>((a, b) ->
            a[0] != b[0] ? Long.compare(a[0], b[0]) : Long.compare(a[1], b[1]));
    private final Deque<Note> open = new ArrayDeque<>();
    private long noted;

    private FloatingEffects() {
    }

    // ── bookkeeping at the clause hook ───────────────────────────────

    /** A live clause begins resolving: note the timestamp, its line and its duration. */
    synchronized void clauseResolving(Game game, SpellAbility clause) {
        Note note = new Note(game.getTimestamp(), ProvenanceResolver.of(clause),
                declaredDuration(clause), clause);
        notes.put(new long[] {note.start, noted++}, note);
        open.push(note);
    }

    /** A live clause finished, or threw: close its note. */
    synchronized void clauseResolved(Game game, SpellAbility clause) {
        for (var it = open.iterator(); it.hasNext(); ) {
            Note note = it.next();
            if (note.clause == clause) {
                note.end = game.getTimestamp();
                note.clause = null;
                it.remove();
                return;
            }
        }
    }

    /** The innermost note that covers a timestamp, or null. */
    private synchronized Note noteAt(long timestamp) {
        for (Note note : notes.headMap(new long[] {timestamp, Long.MAX_VALUE}, false)
                .descendingMap().values()) {
            if (note.covers(timestamp)) {
                return note;
            }
        }
        return null;
    }

    /**
     * The clause's own {@code Duration$}, else its root line's, as
     * {@link EventAttribution} reads it.
     */
    static String declaredDuration(SpellAbility clause) {
        if (clause == null) {
            return null;
        }
        String own = clause.getParam("Duration");
        if (own != null && !own.isEmpty()) {
            return own;
        }
        SpellAbility root = clause.getRootAbility();
        String rooted = root == null ? null : root.getParam("Duration");
        return rooted == null || rooted.isEmpty() ? null : rooted;
    }

    /**
     * A {@code Duration$} as one of three values. {@code permanent} for
     * {@code Permanent}; {@code until_leaves} for a duration that ends when
     * an object changes (it leaves play, its control is lost, it untaps or is
     * unattached or turned face down, "as long as"); {@code end_of_turn} for
     * every other, which ends at a step of some turn — absent, which Forge
     * reads as until end of turn, until end of combat, until your next turn,
     * or a host-leaves duration that also ends at end of turn or combat.
     */
    static String bucket(String duration) {
        if (duration == null || duration.isEmpty()) {
            return "end_of_turn";
        }
        if (duration.equals("Permanent")) {
            return "permanent";
        }
        if (duration.endsWith("OrEOT") || duration.endsWith("OrEndOfCombat")) {
            return "end_of_turn";
        }
        if (duration.startsWith("UntilHostLeavesPlay") || duration.startsWith("AsLongAs")
                || duration.equals("UntilLoseControlOfHost") || duration.endsWith("Untaps")
                || duration.equals("UntilUnattached") || duration.equals("UntilFacedown")) {
            return "until_leaves";
        }
        return "end_of_turn";
    }

    // ── the snapshot's three lists ───────────────────────────────────

    /**
     * {@code floating_effects}: one object per effect card in the command
     * zone, then one per timestamp a noted clause wrote on a card in
     * {@code cards}, oldest first.
     */
    String floatingEffectsJson(Game game, Collection<Card> cards) {
        List<String> objects = new ArrayList<>();
        for (Card effect : game.getCardsIn(ZoneType.Command)) {
            if (isEffectCard(effect)) {
                objects.add(effectCardJson(game, effect));
            }
        }
        Map<Long, Set<String>> affectedByTimestamp = new TreeMap<>();
        for (Card card : cards) {
            for (long timestamp : timestampsOfResolvedRows(card)) {
                affectedByTimestamp.computeIfAbsent(timestamp, t -> new LinkedHashSet<>())
                        .add(Refs.of(card));
            }
        }
        for (Map.Entry<Long, Set<String>> row : affectedByTimestamp.entrySet()) {
            Note note = noteAt(row.getKey());
            if (note == null) {
                continue;
            }
            objects.add(objectJson("F" + row.getKey(), keysOf(note.line),
                    List.copyOf(row.getValue()), List.of(), bucket(note.duration), null));
        }
        return "[" + String.join(",", objects) + "]";
    }

    /** An effect card: an immutable command-zone card that is no emblem and no dungeon. */
    static boolean isEffectCard(Card card) {
        return card.isImmutable() && !card.isEmblem() && !card.getType().isDungeon();
    }

    private static String effectCardJson(Game game, Card effect) {
        SpellAbility source = effect.getEffectSourceAbility();
        Set<String> affected = new LinkedHashSet<>();
        for (StaticAbility stat : effect.getStaticAbilities()) {
            StaticEffect applied = ForgeFields.staticEffectOf(game, stat);
            if (applied != null) {
                for (Card each : applied.getAffectedCards()) {
                    affected.add(Refs.of(each));
                }
            }
        }
        List<String> remembered = new ArrayList<>();
        for (Object each : effect.getRemembered()) {
            Refs.ofAny(each).ifPresent(remembered::add);
        }
        return objectJson(Refs.of(effect),
                keysOf(source == null ? null : ProvenanceResolver.of(source)),
                List.copyOf(affected), remembered,
                // An effect no ability made is a designation's (the monarch,
                // the initiative, the city's blessing): it lasts until the
                // game takes it away.
                source == null ? "permanent" : bucket(declaredDuration(source)),
                shieldLeft(effect));
    }

    /**
     * The amount left on a shield that counts down: the {@code Number$} SVar
     * an effect card's replacement spends ({@code Amount$ ShieldAmount} on a
     * "prevent the next N damage" shield, which {@code ReplaceDamage} lowers
     * as damage is prevented). Null for any other effect.
     */
    static Integer shieldLeft(Card effect) {
        for (ReplacementEffect replacement : effect.getReplacementEffects()) {
            SpellAbility spends = replacement.getOverridingAbility();
            String amount = spends == null ? null : spends.getParam("Amount");
            if (amount == null || !effect.hasSVar(amount)) {
                continue;
            }
            String value = effect.getSVar(amount);
            if (value.startsWith("Number$")) {
                try {
                    return Integer.parseInt(value.substring("Number$".length()).trim());
                } catch (NumberFormatException e) {
                    return null;
                }
            }
        }
        return null;
    }

    /**
     * The timestamps of the rows a resolving ability wrote on a card — static
     * id 0 — in every layer table that keys by timestamp: P/T set and boost,
     * keywords, granted traits, types, colours and names.
     */
    static Set<Long> timestampsOfResolvedRows(Card card) {
        Set<Long> timestamps = new LinkedHashSet<>();
        addResolvedRows(timestamps, card.getSetPTTable());
        addResolvedRows(timestamps, card.getPTBoostTable());
        addResolvedRows(timestamps, card.getChangedCardKeywords());
        addResolvedRows(timestamps, card.getChangedCardTraits());
        addResolvedRows(timestamps, ForgeFields.changedCardTypes(card));
        addResolvedRows(timestamps, ForgeFields.changedCardColors(card));
        if (PatchHooks.CHANGED_NAMES.read(card) instanceof Table<?, ?, ?> names) {
            addResolvedRows(timestamps, names);
        }
        return timestamps;
    }

    private static void addResolvedRows(Set<Long> into, Table<?, ?, ?> table) {
        if (table == null) {
            return;
        }
        for (Table.Cell<?, ?, ?> cell : table.cellSet()) {
            if (cell.getColumnKey() instanceof Long staticId && staticId == 0L
                    && cell.getRowKey() instanceof Long timestamp) {
                into.add(timestamp);
            }
        }
    }

    /**
     * {@code delayed_triggers}: each pending delayed trigger once, keyed to
     * the ability that spawned it, with what it remembers. Duration
     * {@code end_of_turn} for one that lapses at the turn's end, else
     * {@code permanent}: it waits until it fires.
     */
    static String delayedTriggersJson(Game game) {
        Object pending = PatchHooks.PENDING_DELAYED_TRIGGERS.read(game.getTriggerHandler());
        if (pending == null) {
            return "[]";
        }
        Set<Trigger> thisTurn = identitySet();
        thisTurn.addAll(triggersIn(PatchHooks.read(pending, "thisTurn")));
        // Forge's own order, each trigger once: thisTurn is a subset of
        // delayed, and a player-defined trigger can join delayed at its turn.
        List<Trigger> all = triggersIn(PatchHooks.read(pending, "delayed"));
        if (PatchHooks.read(pending, "playerDefined") instanceof Multimap<?, ?> byPlayer) {
            all.addAll(triggersIn(byPlayer.values()));
        }
        Set<Trigger> listed = identitySet();
        List<String> objects = new ArrayList<>();
        for (Trigger trigger : all) {
            if (!listed.add(trigger)) {
                continue;
            }
            List<String> remembered = new ArrayList<>();
            for (Object each : trigger.getTriggerRemembered()) {
                Refs.ofAny(each).ifPresent(remembered::add);
            }
            objects.add(objectJson("D" + trigger.getId(),
                    keysOf(ProvenanceResolver.of(trigger)), List.of(), remembered,
                    thisTurn.contains(trigger) ? "end_of_turn" : "permanent", null));
        }
        return "[" + String.join(",", objects) + "]";
    }

    /** {@code dungeons}: each dungeon in a command zone, its player and its current room. */
    static String dungeonsJson(Game game) {
        List<String> objects = new ArrayList<>();
        for (Card card : game.getCardsIn(ZoneType.Command)) {
            if (card.getType().isDungeon()) {
                objects.add(Json.object()
                        .value("player", CardFacts.ownerOf(card))
                        .value("dungeon", Refs.of(card))
                        .value("room", card.getCurrentRoom())
                        .render());
            }
        }
        return "[" + String.join(",", objects) + "]";
    }

    private static String objectJson(
            String id, List<ProvenanceKey> sourceLines, List<String> affected,
            List<String> remembered, String duration, Integer shieldLeft) {
        return Json.object()
                .value("id", id)
                .raw("source_lines", ProvenanceKey.toJsonArray(sourceLines))
                .raw("affected", Json.stringArray(affected))
                .raw("remembered", Json.stringArray(remembered))
                .value("duration", duration)
                .value("shield_left", shieldLeft)
                .render();
    }

    private static List<ProvenanceKey> keysOf(ProvenanceKey key) {
        return key == null ? List.of() : List.of(key);
    }

    private static List<Trigger> triggersIn(Object list) {
        List<Trigger> triggers = new ArrayList<>();
        if (list instanceof Iterable<?> items) {
            for (Object item : items) {
                if (item instanceof Trigger trigger) {
                    triggers.add(trigger);
                }
            }
        }
        return triggers;
    }

    private static Set<Trigger> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }
}
