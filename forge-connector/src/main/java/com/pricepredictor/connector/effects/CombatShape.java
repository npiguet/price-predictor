package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A damage step's combat, as a record describes it.
 *
 * <p>Who attacked, who blocked whom, and how much damage the assignment gave
 * each target. The blocking relationship is what trample, deathtouch and first
 * strike act through, so a combat record without it says a creature was in a
 * fight and not what the fight was.
 *
 * <p>Shared by the observed record and the probe branch that mirrors it,
 * because gate 2 compares those two: a difference in how the two are read would
 * arrive as a difference the perturbation caused.
 *
 * <p><b>Read it when the step opens, not when the record is written.</b> The
 * record flushes at the phase boundary, by which point the damage has been
 * dealt and what it killed has left — an attacker that traded with its blocker
 * appears in neither list, and the record describes a combat nobody fought.
 */
final class CombatShape {

    private final List<String> attackers = new ArrayList<>();
    private final Map<String, List<String>> blocks = new LinkedHashMap<>();
    private final Map<String, Map<String, Integer>> assignment =
            new LinkedHashMap<>();

    private CombatShape() {
    }

    /** Who is attacking and who is blocking them, right now; empty outside combat. */
    static CombatShape of(Combat combat) {
        CombatShape shape = new CombatShape();
        if (combat == null) {
            return shape;
        }
        for (Card attacker : combat.getAttackers()) {
            String id = SnapshotBuilder.entityId(attacker);
            shape.attackers.add(id);
            List<String> blockers = new ArrayList<>();
            for (Card blocker : combat.getBlockers(attacker)) {
                blockers.add(SnapshotBuilder.entityId(blocker));
            }
            shape.blocks.put(id, blockers);
        }
        return shape;
    }

    /**
     * Add this damage step's assignment.
     *
     * <p>Called once the assignment has run, which can be a different moment
     * from {@link #of}: the assignment exists only after it. Forge keeps a copy
     * of each step's assignment as the players made it, before prevention or
     * redirection rewrote the damage, so a double striker's regular step
     * carries its own assignment and a prevented hit still shows what was
     * assigned. The observed record and its probe branch read the same thing.
     */
    void addAssignment(Combat combat) {
        if (combat == null) {
            return;
        }
        if (!(PatchHooks.ASSIGNED_DAMAGE.read(combat) instanceof Table<?, ?, ?> table)) {
            return;
        }
        for (Table.Cell<?, ?, ?> cell : table.cellSet()) {
            if (!(cell.getRowKey() instanceof Card source)
                    || !(cell.getValue() instanceof Integer amount)) {
                continue;
            }
            String target = targetId(cell.getColumnKey());
            if (target == null) {
                continue;
            }
            String from = SnapshotBuilder.entityId(source);
            assignment
                    .computeIfAbsent(from, key -> new LinkedHashMap<>())
                    .put(target, amount);
        }
    }

    private static String targetId(Object column) {
        if (column instanceof Card card) {
            return SnapshotBuilder.entityId(card);
        }
        if (column instanceof Player player) {
            return SnapshotBuilder.playerId(player);
        }
        return null;
    }

    /**
     * The three combat fields, in the order the payload writes them.
     *
     * <p>Handed over as fields rather than as a whole object so a caller can
     * add the fields only its own kind of record carries — the probe branch
     * adds what it perturbed.
     *
     * <p>A view, not a copy: the map is unmodifiable, but its values are this
     * shape's own collections, so whoever renders it later sees them as they
     * stand then, including an assignment added after this call.
     */
    Map<String, Object> fields() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("attackers", attackers);
        fields.put("blocks", blocks);
        fields.put("assignment_choices", assignment);
        return Collections.unmodifiableMap(fields);
    }
}
