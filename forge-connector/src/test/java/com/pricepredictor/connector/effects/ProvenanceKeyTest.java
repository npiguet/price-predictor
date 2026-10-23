package com.pricepredictor.connector.effects;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/** A provenance key as a value: its mode component, its JSON, and a resolution's two halves. */
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

    /**
     * The member is written only when set, so a root key serializes exactly as
     * it did before the component existed and a gen-1 reader never sees it.
     */
    @Test
    void aModeKeyWritesItsOptionAndARootKeyDoesNot() {
        ProvenanceKey root = new ProvenanceKey(
                "cardsfolder/c/cryptic_command.txt", 0, TraitKind.SPELL, 0);

        assertEquals("{\"face\":0,\"trait_kind\":\"spell\",\"index_within_kind\":0}",
                root.toSidecarJson());
        assertEquals("{\"face\":0,\"trait_kind\":\"spell\",\"index_within_kind\":0,\"option\":2}",
                root.withOption(2).toSidecarJson());
        assertEquals("{\"script_file\":\"cardsfolder/c/cryptic_command.txt\",\"face\":0,"
                        + "\"trait_kind\":\"spell\",\"index_within_kind\":0}",
                root.toJson());
        assertEquals("{\"script_file\":\"cardsfolder/c/cryptic_command.txt\",\"face\":0,"
                        + "\"trait_kind\":\"spell\",\"index_within_kind\":0,\"option\":2}",
                root.withOption(2).toJson());
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
}
