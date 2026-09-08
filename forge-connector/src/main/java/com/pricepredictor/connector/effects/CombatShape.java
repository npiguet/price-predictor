package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.player.Player;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

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

    /** An empty shape, for a record written outside combat. */
    static CombatShape empty() {
        return new CombatShape();
    }

    /** Who is attacking and who is blocking them, right now. */
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
     * Add this step's share of the combat's damage assignment.
     *
     * <p>Called after the assignment has run, which is a different moment from
     * {@link #of}: the attackers have to be read before the damage and the
     * assignment only exists after it.
     *
     * <p>Forge accumulates the assignment over a whole combat and never clears
     * it, so at the regular step the table still holds what first strike
     * assigned. {@code alreadyWritten} carries what earlier steps of the same
     * combat recorded, and grows as this reads. A double striker that assigns
     * the same damage twice is counted once — the engine stores the latest
     * value per pair, so "assigned again, identically" and "not assigned again"
     * are the same table.
     *
     * @param alreadyWritten seen assignments for this combat, or null when the
     *                       combat is a fork's and has no earlier step
     */
    void addAssignment(Combat combat, Set<String> alreadyWritten) {
        if (combat == null) {
            return;
        }
        Object assigned = PatchHooks.read(combat, "getAssignedDamage");
        if (!(assigned instanceof Table<?, ?, ?> table)) {
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
            if (alreadyWritten != null
                    && !alreadyWritten.add(from + ">" + target + "=" + amount)) {
                continue;
            }
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
     * The three combat fields, without the surrounding braces.
     *
     * <p>Rendered as a fragment so a caller can add the fields only its own
     * kind of record carries — the probe branch adds what it perturbed.
     */
    String fields() {
        StringJoiner attackerJson = new StringJoiner(",", "[", "]");
        for (String attacker : attackers) {
            attackerJson.add(Json.string(attacker));
        }
        StringJoiner blockJson = new StringJoiner(",", "{", "}");
        for (Map.Entry<String, List<String>> block : blocks.entrySet()) {
            StringJoiner blockers = new StringJoiner(",", "[", "]");
            for (String blocker : block.getValue()) {
                blockers.add(Json.string(blocker));
            }
            blockJson.add(Json.string(block.getKey()) + ":" + blockers);
        }
        StringJoiner assignmentJson = new StringJoiner(",", "{", "}");
        for (Map.Entry<String, Map<String, Integer>> from : assignment.entrySet()) {
            StringJoiner targets = new StringJoiner(",", "{", "}");
            for (Map.Entry<String, Integer> to : from.getValue().entrySet()) {
                targets.add(Json.string(to.getKey()) + ":" + to.getValue());
            }
            assignmentJson.add(Json.string(from.getKey()) + ":" + targets);
        }
        return "\"attackers\":" + attackerJson
                + ",\"blocks\":" + blockJson
                + ",\"assignment_choices\":" + assignmentJson;
    }
}
