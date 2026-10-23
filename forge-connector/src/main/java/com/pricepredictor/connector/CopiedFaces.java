package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardDb;
import forge.card.CardRules;
import forge.card.ICardFace;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fills in the faces a script copies from another card's with
 * {@code CopyFaceFrom:} rather than writing them out.
 *
 * <p>Forge's {@code CardRules.Reader} leaves such a face empty: the face is
 * only known once every script has been read, so {@code CardDb} supplies it
 * from its index of faces by name after loading the whole tree. The converter
 * reads one script at a time with its own reader, and without this step it
 * renders a card Forge never plays — Emeritus of Conflict without the Lightning
 * Bolt it prepares, and Bind // Liberate, whose halves are both copied, with
 * no front face at all, failing on the first thing that asks its name.
 *
 * <p>The faces come from the card database the converter's Forge environment
 * has already loaded, through the same package-private
 * {@code supplyPlaceholderFaces} {@code CardDb} itself calls, so the parsed
 * rules hold exactly the faces the engine's own copy of the card holds — which
 * is what a provenance key's face and trait ordinals are counted against.
 * Reflective because Forge exposes the step only to its own package.
 */
final class CopiedFaces {

    private static final String DIRECTIVE = "CopyFaceFrom:";

    private static final Method SUPPLY = supplyMethod();

    private CopiedFaces() {
    }

    /**
     * Supply every face {@code scriptLines} copies into {@code rules}, as
     * Forge's card database does.
     *
     * @throws java.util.NoSuchElementException naming the face, when the
     *         database holds no face by the name a script copies — Forge's own
     *         message for the same failure, so a script that names a face no
     *         other script defines reads the same here as in the engine
     */
    static void supply(CardRules rules, List<String> scriptLines) {
        Map<String, ICardFace> copied = copiedFaces(scriptLines);
        if (copied == null) {
            return;
        }
        try {
            SUPPLY.invoke(rules, copied);
        } catch (InvocationTargetException e) {
            if (e.getCause() instanceof RuntimeException cause) {
                throw cause;
            }
            throw new IllegalStateException(e.getCause());
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    /**
     * The faces the script copies, keyed by the name it copies them by, or null
     * for a script that copies none. A name the database does not hold is left
     * out, for {@code supplyPlaceholderFaces} to refuse by name.
     */
    private static Map<String, ICardFace> copiedFaces(List<String> scriptLines) {
        Map<String, ICardFace> copied = null;
        for (String line : scriptLines) {
            if (!line.startsWith(DIRECTIVE)) {
                continue;
            }
            if (copied == null) {
                copied = new HashMap<>(2);
            }
            String name = line.substring(DIRECTIVE.length()).trim();
            ICardFace face = cards().getFaceByName(name);
            if (face != null) {
                copied.put(name, face);
            }
        }
        return copied;
    }

    private static CardDb cards() {
        return StaticData.instance().getCommonCards();
    }

    private static Method supplyMethod() {
        try {
            Method method = CardRules.class.getDeclaredMethod(
                    "supplyPlaceholderFaces", Map.class);
            method.setAccessible(true);
            return method;
        } catch (NoSuchMethodException e) {
            // A Forge without the step would convert every copying script
            // wrongly, so it fails at class load rather than per card.
            throw new ExceptionInInitializerError(e);
        }
    }
}
