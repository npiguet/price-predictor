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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

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
 *       statics' affected cards and players, what it remembers, its {@code Duration$}, and
 *       a counting-down shield's amount in the SVar its replacement spends.
 *       It is an entity of the snapshot, and its floating effect carries its
 *       entity id.</li>
 *   <li><b>A timestamped change</b> — Giant Growth's +3/+3, "gains flying
 *       until end of turn" — lives only as rows a resolving ability wrote
 *       into a card's layer tables under static id 0, which name no line.
 *       The one join is the timestamp: {@link ClauseNotes} notes, at each live
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
 */
final class FloatingEffects {

    private FloatingEffects() {
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
     * {@code Permanent} and {@code Perpetual} (a change that follows the card
     * into every zone for the rest of the game); {@code until_leaves} for a
     * duration that ends when
     * an object changes (it leaves play, its control is lost, it untaps or is
     * unattached, turned face down or face up, "as long as"); {@code end_of_turn} for
     * every other, which ends at a step of some turn — absent, which Forge
     * reads as until end of turn, until end of combat, until your next turn,
     * or a host-leaves duration that also ends at end of turn or combat.
     */
    static FloatingDuration bucket(String duration) {
        if (duration == null || duration.isEmpty()) {
            return FloatingDuration.END_OF_TURN;
        }
        if (lastsForGood(duration)) {
            return FloatingDuration.PERMANENT;
        }
        if (endsAtTurnOrCombatEnd(duration)) {
            return FloatingDuration.END_OF_TURN;
        }
        return endsOnObjectChange(duration) ? FloatingDuration.UNTIL_LEAVES : FloatingDuration.END_OF_TURN;
    }

    private static boolean lastsForGood(String duration) {
        return duration.equals("Permanent") || duration.equals("Perpetual");
    }

    /** A duration that also ends at the end of the turn or of combat. */
    private static boolean endsAtTurnOrCombatEnd(String duration) {
        return duration.endsWith("OrEOT") || duration.endsWith("OrEndOfCombat");
    }

    private static boolean endsOnObjectChange(String duration) {
        return duration.startsWith("UntilHostLeavesPlay") || duration.startsWith("AsLongAs")
                || duration.equals("UntilLoseControlOfHost") || duration.endsWith("Untaps")
                || duration.equals("UntilUnattached") || duration.equals("UntilFacedown")
                || duration.equals("UntilAnimatedFaceup");
    }

    // ── the snapshot's three lists ───────────────────────────────────

    /**
     * {@code floating_effects}: one object per effect card in the command
     * zone, then one per timestamp a noted clause wrote on a card in
     * {@code cards}, oldest first.
     *
     * @param board the board as it stands, or a {@code continuous} record's
     *              board without its static, on which an effect card's
     *              statics name nothing they affect: which permanents a
     *              static touches is that record's answer
     * @param notes the game's clause notes, which name a timestamp's line
     */
    static String floatingEffectsJson(
            Game game, Collection<Card> cards, StaticSuppression board, ClauseNotes notes) {
        List<String> objects = new ArrayList<>();
        for (Card effect : game.getCardsIn(ZoneType.Command)) {
            if (isEffectCard(effect)) {
                objects.add(effectCardJson(game, effect, board));
            }
        }
        for (Timestamped row : timestamped(notes, cards)) {
            objects.add(new ListedObject(row.id(), keysOf(row.line()),
                    row.affected(), List.of(), bucket(row.duration()), null).json());
        }
        return "[" + String.join(",", objects) + "]";
    }

    /**
     * A timestamped floating effect, as {@code floating_effects} lists it and
     * the snapshot's links read it.
     *
     * @param id       {@code F<timestamp>}
     * @param host     the ref of the card whose line drew it, or null
     * @param line     the key of that line, or null
     * @param affected the entities carrying a row of the timestamp
     * @param duration the clause's {@code Duration$}, unbucketed
     */
    record Timestamped(
            String id, String host, ProvenanceKey line, List<String> affected, String duration) {
    }

    /**
     * One per timestamp a noted clause wrote on a card in {@code cards},
     * oldest first; a timestamp no note covers is left out.
     */
    static List<Timestamped> timestamped(ClauseNotes notes, Collection<Card> cards) {
        List<Timestamped> rows = new ArrayList<>();
        for (Map.Entry<Long, Set<String>> row : affectedByTimestamp(cards).entrySet()) {
            ClauseNotes.Note note = notes.noteAt(row.getKey());
            if (note != null) {
                rows.add(new Timestamped("F" + row.getKey(), note.host(), note.line(),
                        List.copyOf(row.getValue()), note.duration()));
            }
        }
        return rows;
    }

    /** The refs of the cards carrying a resolved row, by the row's timestamp, oldest first. */
    private static Map<Long, Set<String>> affectedByTimestamp(Collection<Card> cards) {
        Map<Long, Set<String>> affectedByTimestamp = new TreeMap<>();
        for (Card card : cards) {
            for (long timestamp : timestampsOfResolvedRows(card)) {
                affectedByTimestamp.computeIfAbsent(timestamp, t -> new LinkedHashSet<>())
                        .add(Refs.of(card));
            }
        }
        return affectedByTimestamp;
    }

    /** An effect card: an immutable command-zone card that is no emblem and no dungeon. */
    static boolean isEffectCard(Card card) {
        return card.isImmutable() && !card.isEmblem() && !card.getType().isDungeon();
    }

    private static String effectCardJson(Game game, Card effect, StaticSuppression board) {
        SpellAbility source = effect.getEffectSourceAbility();
        List<String> affected = affectedBy(game, effect, board);
        List<String> remembered = new ArrayList<>();
        for (Object each : effect.getRemembered()) {
            Refs.ofAny(each).ifPresent(remembered::add);
        }
        return new ListedObject(Refs.of(effect),
                keysOf(source == null ? null : ProvenanceResolver.of(source)),
                affected, remembered, effectCardDuration(source), shieldLeft(effect)).json();
    }

    /**
     * Every static's cards, then every static's players: Explore's effect
     * card affects its caster, whose land plays it raises.
     */
    private static List<String> affectedBy(Game game, Card effect, StaticSuppression board) {
        Set<String> cards = new LinkedHashSet<>();
        Set<String> players = new LinkedHashSet<>();
        for (StaticAbility stat : board.showsAffected() ? effect.getStaticAbilities() : List.<StaticAbility>of()) {
            StaticEffect applied = ForgeFields.staticEffectOf(game, stat);
            for (String each : applied == null ? List.<String>of() : Refs.affectedBy(applied)) {
                (Refs.isEntity(each) ? cards : players).add(each);
            }
        }
        List<String> affected = new ArrayList<>(cards);
        affected.addAll(players);
        return affected;
    }

    /**
     * An effect card's duration: its ability's, bucketed. An effect no
     * ability made is a designation's (the monarch, the initiative, the
     * city's blessing): it lasts until the game takes it away.
     */
    private static FloatingDuration effectCardDuration(SpellAbility source) {
        return source == null ? FloatingDuration.PERMANENT : bucket(declaredDuration(source));
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
                return numberOf(value);
            }
        }
        return null;
    }

    /** A {@code Number$} SVar's amount, or null when it is no number. */
    private static Integer numberOf(String value) {
        try {
            return Integer.parseInt(value.substring("Number$".length()).trim());
        } catch (NumberFormatException e) {
            return null;
        }
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
        // "Target creature can't block this turn": a KW$ HIDDEN grant is in
        // no keyword table but its own; an Animate's CantHaveKeyword$ has
        // one too.
        addResolvedRows(timestamps, ForgeFields.hiddenExtrinsicKeywords(card));
        addResolvedRows(timestamps, ForgeFields.cantHaveKeywords(card));
        addResolvedControl(timestamps, card);
        return timestamps;
    }

    /**
     * A resolved GainControl (Blatant Thievery, Act of Treason) files its
     * change under its timestamp alone, with no static id: a static's is
     * filed under the static's timestamp, which no clause note covers.
     */
    private static void addResolvedControl(Set<Long> into, Card card) {
        Map<?, ?> controllers = ForgeFields.tempControllers(card);
        if (controllers == null) {
            return;
        }
        Set<Long> statics = ForgeFields.staticEffectTimestamps(card.getGame());
        for (Object timestamp : controllers.keySet()) {
            if (timestamp instanceof Long resolved && !statics.contains(resolved)) {
                into.add(resolved);
            }
        }
    }

    private static <R, C, V> void addResolvedRows(Set<Long> into, Table<R, C, V> table) {
        if (table == null) {
            return;
        }
        for (Table.Cell<R, C, V> cell : table.cellSet()) {
            Long timestamp = resolvedRowTimestamp(cell);
            if (timestamp != null) {
                into.add(timestamp);
            }
        }
    }

    /** A row's timestamp when a resolving ability wrote it (static id 0), else null. */
    private static Long resolvedRowTimestamp(Table.Cell<?, ?, ?> cell) {
        if (!(cell.getColumnKey() instanceof Long staticId) || staticId != 0L) {
            return null;
        }
        return cell.getRowKey() instanceof Long timestamp ? timestamp : null;
    }

    /**
     * {@code delayed_triggers}: each pending delayed trigger once, keyed to
     * the ability that spawned it, with what it remembers. Duration
     * {@code end_of_turn} for one that lapses at the turn's end, else
     * {@code permanent}: it waits until it fires.
     */
    static String delayedTriggersJson(Game game) {
        List<String> objects = new ArrayList<>();
        for (Pending pending : pendingDelayedTriggers(game)) {
            objects.add(new ListedObject(pending.id(),
                    keysOf(ProvenanceResolver.of(pending.trigger())), List.of(), pending.remembered(),
                    pending.thisTurn() ? FloatingDuration.END_OF_TURN : FloatingDuration.PERMANENT,
                    null).json());
        }
        return "[" + String.join(",", objects) + "]";
    }

    /**
     * A pending delayed trigger, as {@code delayed_triggers} lists it and the
     * snapshot's links read it.
     *
     * @param thisTurn whether it lapses at the turn's end
     */
    record Pending(Trigger trigger, boolean thisTurn) {

        /** {@code D<trigger id>}. */
        String id() {
            return "D" + trigger.getId();
        }

        /** The refs of the cards and players it holds. */
        List<String> remembered() {
            List<String> remembered = new ArrayList<>();
            for (Object each : trigger.getTriggerRemembered()) {
                Refs.ofAny(each).ifPresent(remembered::add);
            }
            return remembered;
        }
    }

    /**
     * Every pending delayed trigger once, in Forge's own order; none on an
     * unpatched Forge.
     */
    static List<Pending> pendingDelayedTriggers(Game game) {
        Object pending = PatchHooks.PENDING_DELAYED_TRIGGERS.read(game.getTriggerHandler());
        if (pending == null) {
            return List.of();
        }
        Set<Trigger> thisTurn = identitySet();
        thisTurn.addAll(triggersIn(PatchHooks.PENDING_THIS_TURN.read(pending)));
        // Forge's own order, each trigger once: thisTurn is a subset of
        // delayed, and a player-defined trigger can join delayed at its turn.
        List<Trigger> all = triggersIn(PatchHooks.PENDING_DELAYED.read(pending));
        if (PatchHooks.PENDING_PLAYER_DEFINED.read(pending) instanceof Multimap<?, ?> byPlayer) {
            all.addAll(triggersIn(byPlayer.values()));
        }
        Set<Trigger> listed = identitySet();
        List<Pending> triggers = new ArrayList<>();
        for (Trigger trigger : all) {
            if (listed.add(trigger)) {
                triggers.add(new Pending(trigger, thisTurn.contains(trigger)));
            }
        }
        return triggers;
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

    /**
     * One object of a command-zone list, as {@code floating_effects} and
     * {@code delayed_triggers} render it.
     *
     * @param shieldLeft the amount left on a counting-down shield, or null
     */
    private record ListedObject(
            String id, List<ProvenanceKey> sourceLines, List<String> affected,
            List<String> remembered, FloatingDuration duration, Integer shieldLeft) {

        String json() {
            return Json.object()
                    .value("id", id)
                    .raw("source_lines", ProvenanceKey.toJsonArray(sourceLines))
                    .raw("affected", Json.stringArray(affected))
                    .raw("remembered", Json.stringArray(remembered))
                    .value("duration", duration)
                    .value("shield_left", shieldLeft)
                    .render();
        }
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
