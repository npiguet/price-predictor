package com.pricepredictor.connector;

/**
 * A closed vocabulary member with a spelling of its own on the wire.
 *
 * <p>Every file the workers write for the Python side spells its enumerated
 * values in that side's lower-case vocabulary, never as the Java constant
 * name, and the readers refuse a spelling they do not have. Implementing this
 * is what lets a writer hand the member itself to {@code effects.Json} and get
 * its wire name back, so forgetting a {@code .wireName()} at one call site
 * cannot leak {@code ZONE_CHANGE} into an append-only corpus.
 *
 * <p>It lives in the root package because both the effect collectors and the
 * root-package workers write such vocabularies, and the root package is the
 * one {@code effects} already depends on.
 */
public interface WireNamed {

    /** The spelling the Python side reads. */
    String wireName();

    /**
     * The member of {@code type} that {@code wireName} spells.
     *
     * @throws IllegalArgumentException for a spelling {@code type} has no
     *         member for, naming both, since a reader that silently mapped an
     *         unknown spelling to some default would mislabel a row
     */
    static <E extends Enum<E> & WireNamed> E fromWireName(Class<E> type, String wireName) {
        for (E member : type.getEnumConstants()) {
            if (member.wireName().equals(wireName)) {
                return member;
            }
        }
        throw new IllegalArgumentException(
                type.getSimpleName() + " has no member spelled '" + wireName + "'");
    }
}
