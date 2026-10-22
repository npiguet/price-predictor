package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Set;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Every outcome type the patched engine can send is one this collector knows.
 *
 * <p>The engine's names are read reflectively, because this module builds
 * against an unpatched checkout too; there the test is skipped. A name the
 * collector does not know would otherwise surface only as the unknown-outcome
 * warning in a collection run's log.
 */
class OutcomeVocabularyContractTest {

    private static Set<String> stringConstants(Class<?> owner) throws IllegalAccessException {
        Set<String> values = new TreeSet<>();
        for (Field field : owner.getFields()) {
            int modifiers = field.getModifiers();
            if (Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers)
                    && field.getType() == String.class) {
                values.add((String) field.get(null));
            }
        }
        return values;
    }

    @Test
    void everyEngineOutcomeTypeIsAnEffectEventType() throws IllegalAccessException {
        Class<?> engineOutcomes;
        try {
            engineOutcomes = Class.forName(PatchHooks.EFFECT_RECORD_OUTCOMES);
        } catch (ClassNotFoundException unpatched) {
            engineOutcomes = null;
        }
        assumeTrue(engineOutcomes != null, "the sibling Forge checkout is unpatched");

        Set<String> engineTypes = stringConstants(engineOutcomes);
        Set<String> unknown = new TreeSet<>(engineTypes);
        unknown.removeAll(stringConstants(EffectEvent.class));

        assertFalse(engineTypes.isEmpty(), "the engine declares its outcome type names");
        assertTrue(unknown.isEmpty(), "outcome types EffectEvent does not know: " + unknown);
    }
}
