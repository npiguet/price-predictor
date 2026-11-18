package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.ProvenanceRecorder;
import forge.game.trigger.Trigger;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The lines of a face the converter leaves out as another line's duplicate,
 * and where their keys go.
 *
 * <p>A line is left out when another line of the face states its text: an
 * earlier line with the same description (the second trigger Forge registers
 * for "enters or attacks"), or, for a secondary trigger, any line restating
 * it ({@link RepeatedLines}: Coffin Queen's triggers repeat its activated
 * line). A rebuilt line takes the place of the one it rebuilds (a Class
 * card's level one). Every trait whose line is left out stays live at
 * runtime, so its keys go onto the line that stays, through
 * {@link ProvenanceRecorder#fold}; this class is the one place that folds.
 */
final class DuplicateLines {

    private final ProvenanceRecorder recorder;
    /** The lines of secondary triggers whose Execute$ no primary shares (identity set). */
    private final Set<Ability> secondaryLines = Collections.newSetFromMap(new IdentityHashMap<>());

    DuplicateLines(ProvenanceRecorder recorder) {
        this.recorder = recorder;
    }

    /** Note {@code line}, the line {@code trigger} printed, if the trigger is secondary. */
    void noteTriggerLine(Ability line, Trigger trigger) {
        if (line != null && trigger.isSecondary()) secondaryLines.add(line);
    }

    /** {@code rebuilt} in place of {@code line}, with its keys. */
    Ability replace(Ability line, Ability rebuilt) {
        recorder.fold(rebuilt, line);
        return rebuilt;
    }

    /**
     * {@code lines} less each whose description an earlier line has, which
     * takes its keys and is no secondary line if the duplicate was not.
     */
    List<Ability> withoutIdentical(List<Ability> lines) {
        Map<String, Ability> survivors = new HashMap<>();
        List<Ability> result = new ArrayList<>();
        for (Ability line : lines) {
            String description = line.descriptionText();
            Ability survivor = description == null ? null : survivors.putIfAbsent(description, line);
            if (survivor == null) {
                result.add(line);
            } else {
                foldIdentical(survivor, line);
            }
        }
        return result;
    }

    private void foldIdentical(Ability survivor, Ability duplicate) {
        if (!secondaryLines.contains(duplicate)) secondaryLines.remove(survivor);
        recorder.fold(survivor, duplicate);
    }

    /**
     * {@code lines} less each secondary trigger's line whose text another
     * line restates; the first restating line, in line order, takes its keys.
     * Forge's card view prints no secondary trait, and a line that says
     * something no other line does keeps its place (Free for All's upkeep
     * trigger).
     */
    List<Ability> withoutRestatedSecondaries(List<Ability> lines) {
        List<Ability> result = new ArrayList<>(lines);
        for (Ability line : lines) {
            Ability cover = secondaryLines.contains(line) ? restating(result, line) : null;
            if (cover != null) {
                result.removeIf(x -> x == line);
                recorder.fold(cover, line);
            }
        }
        return result;
    }

    /** The first of {@code lines} other than {@code line} restating it, or null. */
    private static Ability restating(List<Ability> lines, Ability line) {
        String text = line.descriptionText();
        if (text == null) return null;
        return lines.stream()
                .filter(other -> other != line && restates(text, other))
                .findFirst().orElse(null);
    }

    private static boolean restates(String text, Ability other) {
        return other.descriptionText() != null && RepeatedLines.repeats(text, other.descriptionText());
    }
}
