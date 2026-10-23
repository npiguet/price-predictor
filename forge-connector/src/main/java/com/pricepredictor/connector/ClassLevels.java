package com.pricepredictor.connector;

import com.pricepredictor.connector.ability.ClassLevelAbility;
import com.pricepredictor.connector.ability.TextAbility;
import com.pricepredictor.connector.effects.ProvenanceRecorder;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A Class face's ability list, rearranged so it reads as the printed card
 * does: one line per level, level one first, the levels in order.
 *
 * <p>Three steps, in this order: drop each level's duplicate trait entry,
 * re-type the remaining unlevelled abilities as level one, then sort the
 * levels ahead of everything else. The recorder follows every step, because
 * each one either removes or replaces an ability the recorder keys by
 * identity.
 */
final class ClassLevels {

    /**
     * Levels first, in level order; every other ability after them, in the
     * order it already had. {@link List#sort} is stable, so the zero the
     * second key gives every non-level ability keeps that order.
     */
    private static final Comparator<Ability> LEVELS_FIRST =
            Comparator.comparing((Ability a) -> !isLevel(a))
                    .thenComparingInt(a -> isLevel(a) ? a.ordinal() : 0);

    private final List<Ability> abilities;
    private final ProvenanceRecorder recorder;

    private ClassLevels(List<Ability> abilities, ProvenanceRecorder recorder) {
        this.abilities = new ArrayList<>(abilities);
        this.recorder = recorder;
    }

    /**
     * {@code abilities} rearranged as a Class face prints them.
     *
     * @param levelDescriptions each Class level's inner text, which is how its
     *                          duplicate trait entry is recognised
     */
    static List<Ability> arrange(List<Ability> abilities, Set<String> levelDescriptions,
                                 ProvenanceRecorder recorder) {
        ClassLevels levels = new ClassLevels(abilities, recorder);
        levels.mergeLevelDuplicates(levelDescriptions);
        levels.retypeLevelOne();
        levels.abilities.sort(LEVELS_FIRST);
        return levels.abilities;
    }

    /**
     * Remove each trait entry a level already prints.
     *
     * <p>A level's effect reaches the parser twice — once through the Class
     * keyword that prints the level, once as the trait itself — and only the
     * keyword's entry survives. It therefore takes over the removed entry's
     * keys, or a record fired by that trait names no line at all.
     */
    private void mergeLevelDuplicates(Set<String> levelDescriptions) {
        Map<String, Ability> levelByDescription = new HashMap<>();
        for (Ability a : abilities) {
            if (!isLevel(a)) continue;
            // A Class level prints its level-up cost ahead of its text, so the
            // description the removal below matches is the level's inner text,
            // not its printed line. Keyed by the printed line, every lookup
            // misses and the removed trait's keys are lost.
            levelByDescription.putIfAbsent(
                    a instanceof ClassLevelAbility level
                            ? level.innerDescription() : a.descriptionText(), a);
        }
        abilities.removeIf(a -> {
            boolean duplicate = !isLevel(a)
                    && levelDescriptions.contains(a.descriptionText());
            if (duplicate) {
                recorder.merge(levelByDescription.get(a.descriptionText()), a);
            }
            return duplicate;
        });
    }

    /**
     * Re-type each static, triggered or replacement ability as level one: on
     * a Class card, whatever no level keyword claims is the level the card
     * starts at.
     */
    private void retypeLevelOne() {
        for (int i = 0; i < abilities.size(); i++) {
            Ability a = abilities.get(i);
            if (a.type() == AbilityType.STATIC || a.type() == AbilityType.TRIGGERED
                    || a.type() == AbilityType.REPLACEMENT) {
                // The level-one ability is re-typed as a fresh object, and the
                // recorder keys by identity, so its attribution moves with it.
                TextAbility level =
                        new TextAbility(AbilityType.LEVEL, a.descriptionText(), 1);
                recorder.transfer(a, level);
                abilities.set(i, level);
            }
        }
    }

    private static boolean isLevel(Ability a) {
        return a.type() == AbilityType.LEVEL;
    }
}
