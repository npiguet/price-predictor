package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;

import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.pricepredictor.connector.effects.RunParamValues.refsIn;

/**
 * What a mode's event is about, read off its run-parameter map: the entities
 * under the keys its row names, or the members of the batch or table it
 * hands over. Each is capped at {@value RunParamValues#MAX_SUBJECTS} and
 * deduplicated, because the same card can appear twice in one collection
 * after a state change.
 */
final class RunParamSubjects {

    private RunParamSubjects() {
    }

    /**
     * The entities under the first of these keys that names any. A map names
     * one card several times over ({@code Card}, {@code CardLKI} and {@code
     * Affected} routinely all point at it), and the others in it ({@code
     * Cause}, {@code DamageSource}, {@code LastState*}) are causes and context,
     * not subjects: a row names its subject's keys, most specific first.
     */
    static ModeRow.Subjects about(String... keys) {
        List<String> order = List.of(keys);
        return byName -> firstNamed(byName, order);
    }

    private static Set<String> firstNamed(Map<String, Object> byName, List<String> keys) {
        for (String key : keys) {
            Set<String> subjects = new LinkedHashSet<>();
            refsIn(byName.get(key), subjects);
            if (!subjects.isEmpty()) {
                return subjects;
            }
        }
        return Set.of();
    }

    /**
     * Subject keys, most specific first, for a mode no row reads: its event
     * is a {@code state_flag_change} carrying its mode, about the first of
     * these its map names.
     *
     * <p>Combat's own names are in the order that keeps each event about the
     * creature it is about. {@code CardLKI} is last, and only a last resort:
     * it is a copy of the card as it was, so wherever the live object is also
     * in the map one of the keys above names it.
     */
    static final ModeRow.Subjects UNREAD = about(
            "Affected", "Card", "Player", "DamageTarget", "Attacker",
            "Blocker", "Attackers", "Attached", "AttachSource", "CardLKI");

    /** {@code DamageDealtOnce}: the entities its {@code DamageMap} (target to amount) damaged. */
    static Set<String> damageMapTargets(Map<String, Object> byName) {
        Set<String> subjects = new LinkedHashSet<>();
        if (byName.get("DamageMap") instanceof Map<?, ?> map) {
            refsIn(map.keySet(), subjects);
        }
        return subjects;
    }

    /**
     * {@code DamageAll}: the entities its {@code DamageMap}, a {@code
     * CardDamageTable} of source, target and amount, damaged.
     */
    static Set<String> damageTableTargets(Map<String, Object> byName) {
        Set<String> subjects = new LinkedHashSet<>();
        if (byName.get("DamageMap") instanceof Table<?, ?, ?> table) {
            refsIn(table.columnKeySet(), subjects);
        }
        return subjects;
    }

    /**
     * {@code ChangesZoneAll}: every card a batch of moves moved ({@code Cards},
     * a {@code CardZoneTable} of origin, destination and cards), its moves
     * taken in the order of their origin's and then their destination's zone
     * name, so which cards the cap keeps never depends on the order the
     * engine made the moves in.
     */
    static Set<String> zoneTableCards(Map<String, Object> byName) {
        Set<String> moved = new LinkedHashSet<>();
        if (byName.get("Cards") instanceof Table<?, ?, ?> table) {
            table.cellSet().stream()
                    .sorted(BY_ZONE_NAMES)
                    .forEach(move -> refsIn(move.getValue(), moved));
        }
        return moved;
    }

    private static final Comparator<Table.Cell<?, ?, ?>> BY_ZONE_NAMES =
            Comparator.<Table.Cell<?, ?, ?>, String>comparing(move -> zoneName(move.getRowKey()))
                    .thenComparing(move -> zoneName(move.getColumnKey()));

    private static String zoneName(Object zone) {
        return String.valueOf(CardSpelling.zoneName(zone));
    }
}
