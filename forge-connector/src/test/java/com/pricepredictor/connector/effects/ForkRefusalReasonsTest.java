package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.effects.directed.KeywordEdit;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every refusal reason reads back whole from the refusal tally's printed map
 * ({@link ForkRefusals}: {@code {<kind> <reason>=<count>, ...}}), whose
 * inside the supervisor matches between braces and splits on {@code ", "}
 * ({@code price_predictor.domain.worker_failures._java_map}): a reason
 * holding a comma would be cut into a wrong key and a lost count, one
 * holding a brace would lose the whole line.
 */
class ForkRefusalReasonsTest {

    /** The classes declaring the reasons a {@link ForkRefusal} is thrown with. */
    private static final List<Class<?>> HOLDERS = List.of(
            ForkRefusal.class, InterventionFork.class, DamageStepProbe.class,
            KeywordEdit.class, ReplaceCardOperation.class);

    /** Every {@code static final String} the holders declare: each reason, and kind names beside them. */
    private static List<String> constants() throws IllegalAccessException {
        List<String> values = new ArrayList<>();
        for (Field field : HOLDERS.stream().flatMap(holder -> Arrays.stream(holder.getDeclaredFields()))
                .filter(ForkRefusalReasonsTest::isStringConstant).toList()) {
            field.setAccessible(true);
            values.add((String) field.get(null));
        }
        return values;
    }

    private static boolean isStringConstant(Field field) {
        int modifiers = field.getModifiers();
        return Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers) && field.getType() == String.class;
    }

    @Test
    void noReasonHoldsACommaOrABrace() throws IllegalAccessException {
        List<String> reasons = constants();
        assertTrue(reasons.contains(ReplaceCardOperation.LINKED_CARDS), "the holders are read: " + reasons);
        assertEquals(List.of(), reasons.stream()
                .filter(reason -> reason.matches(".*[,{}].*")).toList());
    }
}
