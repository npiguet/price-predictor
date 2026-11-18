package com.pricepredictor.connector;

import java.util.function.Supplier;

/**
 * The face whose own name the descriptions cased on this thread write as
 * CARDNAME ({@link OwnName}): the one being parsed. Forge hands each trait's
 * description to its own entry factory, and every one of them cases it through
 * {@link AbilityDescription#applyCasing}, so the name is known there without
 * every factory taking it.
 */
final class OwnNameScope {

    /** The face being parsed on this thread, or null outside a face. */
    private static final ThreadLocal<OwnName> CURRENT = new ThreadLocal<>();

    private OwnNameScope() {
    }

    /** What {@code parse} returns, run with {@code name} the face's own name. */
    static <T> T within(OwnName name, Supplier<T> parse) {
        OwnName outer = CURRENT.get();
        CURRENT.set(name);
        try {
            return parse.get();
        } finally {
            CURRENT.set(outer);
        }
    }

    /** {@code raw} with the name of the face being parsed written as the placeholder. */
    static String writtenIn(String raw) {
        OwnName current = CURRENT.get();
        return current == null ? raw : current.apply(raw);
    }
}
