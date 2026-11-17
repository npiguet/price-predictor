package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * A provenance key as a value: its mode and built-trait components, and a
 * resolution's two halves. The JSON forms of a resolved key are pinned by
 * {@code ProvenanceKeyCharacterizationTest}; a built key's, which no resolver
 * emits yet, here.
 */
class ProvenanceKeyTest {

    /**
     * A mode is a different ability from its charm, and the only name Forge
     * leaves it is its position in {@code Choices$}; a key carrying that must
     * never collapse into the root's, or every mode line would share one row.
     */
    @Test
    void aModeKeyIsDistinctFromItsRootAndKnowsIt() {
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);
        ProvenanceKey mode = root.withOption(2);

        assertNotEquals(root, mode);
        assertNotEquals(root.hashCode(), mode.hashCode());
        assertNotEquals(mode, root.withOption(1));
        assertEquals(mode, root.withOption(2));
        assertEquals(root, mode.root());
        assertEquals(root, root.root());
        assertNull(root.option());
        assertEquals(2, mode.option());
    }

    /** A position is never negative: the component is a Choices$ index or nothing. */
    @Test
    void aNegativeOptionIsRefused() {
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);
        org.junit.jupiter.api.Assertions.assertThrows(
                IllegalArgumentException.class, () -> root.withOption(-1));
    }

    /** A resolution names a line or says why it has none, never both nor neither. */
    @Test
    void aResolutionIsALineOrAReason() {
        ProvenanceKey key = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);

        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new ProvenanceKey.Resolved(key, UnresolvedReason.UNKNOWN_KIND));
        org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                () -> new ProvenanceKey.Resolved(null, null));
    }

    /** A mode's option goes onto a line, and leaves a reason alone. */
    @Test
    void anOptionKeysTheModeOfALineAndLeavesAReasonAlone() {
        ProvenanceKey key = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);
        ProvenanceKey.Resolved line = new ProvenanceKey.Resolved(key, null);

        assertEquals(key.withOption(2), line.withOption(2).key());
        assertEquals(line, line.withOption(null));
        assertEquals(ProvenanceKey.Resolved.UNKNOWN, ProvenanceKey.Resolved.UNKNOWN.withOption(2));
    }

    // ── the trait a keyword built ──────────────────────────────────────

    /** Binding the Old Gods' saga keyword, and its chapter III trigger. */
    private static final ProvenanceKey SAGA = new ProvenanceKey(
            "cardsfolder/b/binding_the_old_gods.txt", 0, TraitKind.KEYWORD, 0);

    /**
     * The record form of {@code SAGA.withBuilt(TRIGGER, 2)}, byte for byte. The
     * Python codec's test decodes and re-encodes this same string.
     */
    private static final String CHAPTER_III_JSON =
            "{\"script_file\":\"cardsfolder/b/binding_the_old_gods.txt\",\"face\":0,"
            + "\"trait_kind\":\"keyword\",\"index_within_kind\":0,"
            + "\"built\":{\"kind\":\"trigger\",\"index\":2}}";

    /**
     * Each chapter is its own trait, so a key naming one must never collapse
     * into the saga's or another chapter's, or every chapter joins one line.
     */
    @Test
    void aBuiltKeyIsDistinctFromItsKeywordAndFromAnotherBuiltTrait() {
        ProvenanceKey chapterIII = SAGA.withBuilt(TraitKind.TRIGGER, 2);

        assertNotEquals(SAGA, chapterIII);
        assertNotEquals(SAGA.hashCode(), chapterIII.hashCode());
        assertNotEquals(chapterIII, SAGA.withBuilt(TraitKind.TRIGGER, 1));
        assertNotEquals(chapterIII, SAGA.withBuilt(TraitKind.STATIC, 2));
        assertEquals(chapterIII, SAGA.withBuilt(TraitKind.TRIGGER, 2));
        assertEquals(new ProvenanceKey.Built(TraitKind.TRIGGER, 2), chapterIII.built());
        assertEquals(SAGA, chapterIII.withoutBuilt());
        assertEquals(SAGA, SAGA.withoutBuilt());
        assertNull(SAGA.built());
    }

    /** The member is written only where set, in both forms. */
    @Test
    void aBuiltKeyWritesItsMemberAndABareKeyDoesNot() {
        ProvenanceKey chapterIII = SAGA.withBuilt(TraitKind.TRIGGER, 2);

        assertEquals(CHAPTER_III_JSON, chapterIII.toJson());
        assertEquals("{\"face\":0,\"trait_kind\":\"keyword\",\"index_within_kind\":0,"
                + "\"built\":{\"kind\":\"trigger\",\"index\":2}}", chapterIII.toSidecarJson());
        assertFalse(SAGA.toJson().contains("built"));
        assertEquals("(cardsfolder/b/binding_the_old_gods.txt) face0/keyword[0]/built trigger[2]",
                chapterIII.describe());
    }

    /** A mode of a built trait and its root both keep the built member. */
    @Test
    void anOptionAndItsRootKeepTheBuiltTrait() {
        ProvenanceKey chapterIII = SAGA.withBuilt(TraitKind.TRIGGER, 2);

        assertEquals(chapterIII.built(), chapterIII.withOption(1).built());
        assertEquals(chapterIII, chapterIII.withOption(1).root());
    }

    /**
     * Only a keyword builds traits, a built trait is never itself a keyword,
     * and its position is an index.
     */
    @Test
    void aBuiltMemberOutsideItsContractIsRefused() {
        ProvenanceKey spell = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);

        assertThrows(IllegalArgumentException.class, () -> spell.withBuilt(TraitKind.TRIGGER, 0));
        assertThrows(IllegalArgumentException.class, () -> SAGA.withBuilt(TraitKind.KEYWORD, 0));
        assertThrows(IllegalArgumentException.class, () -> SAGA.withBuilt(TraitKind.TRIGGER, -1));
    }
}
