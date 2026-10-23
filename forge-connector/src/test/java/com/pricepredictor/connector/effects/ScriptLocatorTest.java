package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * The script path a live card's keys name: which tree, and which file in it,
 * for tokens rebuilt without a paper card and for copies of printed cards.
 */
@ExtendWith(ForgeExtension.class)
class ScriptLocatorTest {

    private static Card card(String name) {
        return TestCards.build(name);
    }

    /**
     * A forked game's tokens are rebuilt by {@code TokenInfo}, which drops the
     * {@code PaperToken}. Keyed by name they would name
     * {@code cardsfolder/z/zombie_token.txt}, a file no tree holds, so every
     * Food, Treasure and Zombie on a forked board would reach the model as no
     * text. The image key survives the copy and names the script stem (FR-150).
     */
    @Test
    void aCopiedTokenWithNoPaperCardStillKeysIntoTheTokenTree() {
        Card copy = TestCards.copiedToken("c_a_food_sac");

        assertNull(copy.getPaperCard(), "the fixture must reproduce the copier's shape");
        assertEquals("tokenscripts/c_a_food_sac.txt", ScriptLocator.scriptFileOf(copy));
    }

    /** And a real card's copy is not mistaken for a token: no image-key path for it. */
    @Test
    void aCopiedPrintedCardStillKeysIntoTheCardTree() {
        Card bolt = card("Lightning Bolt");
        Card copy = new Card(TestCards.nextCardId(), TestCards.game());
        copy.setName(bolt.getName());
        copy.setImageKey(bolt.getImageKey());

        assertEquals("cardsfolder/l/lightning_bolt.txt", ScriptLocator.scriptFileOf(copy));
    }

    /** A token image key the database does not know is not turned into a path. */
    @Test
    void anUnknownTokenImageKeyFallsThroughToNull() {
        Card copy = TestCards.copiedToken("c_a_food_sac");
        copy.setImageKey(forge.ImageKeys.getTokenKey("no_such_token_xyz"));

        // Falls through to the name-derived cardsfolder path, exactly as before
        // this change; what must not happen is a fabricated tokenscripts path.
        assertEquals("cardsfolder/f/food_token.txt", ScriptLocator.scriptFileOf(copy));
    }

    /**
     * A null or empty image key must not throw. {@code ImageKeys.getTokenImageName}
     * rejects a null key with an NPE that {@code tokenScriptStem} catches
     * internally, and an empty key never matches the token prefix either way —
     * both fall through to the same name-derived cardsfolder path as an
     * unknown token key.
     */
    @Test
    void aNullOrEmptyImageKeyFallsThroughToTheCardTreeWithoutThrowing() {
        Card withNullKey = TestCards.copiedToken("c_a_food_sac");
        withNullKey.setImageKey(null);
        assertEquals("cardsfolder/f/food_token.txt", ScriptLocator.scriptFileOf(withNullKey));

        Card withEmptyKey = TestCards.copiedToken("c_a_food_sac");
        withEmptyKey.setImageKey("");
        assertEquals("cardsfolder/f/food_token.txt", ScriptLocator.scriptFileOf(withEmptyKey));
    }

    /**
     * A token copy of a printed card: {@code isToken()} is true (the copier
     * sets {@code GamePieceType.TOKEN} on every copy it makes, real permanent
     * or not) but the image key is the printed card's, not a {@code t:} key.
     * {@code ImageKeys.getTokenImageName} rejects a non-token key outright, so
     * this must still reach the cardsfolder path — the case the class doc for
     * {@code tokenScriptStem} names explicitly.
     */
    @Test
    void aTokenCopyOfAPrintedCardStillKeysIntoTheCardTree() {
        Card copy = TestCards.copiedToken("c_a_food_sac");
        copy.setName("Lightning Bolt");
        copy.setImageKey(card("Lightning Bolt").getImageKey());

        assertEquals("cardsfolder/l/lightning_bolt.txt", ScriptLocator.scriptFileOf(copy));
    }

    /**
     * The join is a string comparison, so what the collector writes has to be
     * what the converter wrote: the source path relativised against the tree,
     * with forward slashes, on a platform whose own separator is neither.
     * Spelled out rather than recomputed, so a change to the shared derivation
     * cannot move both sides at once; {@code +2 Mace} is a card the tree files
     * under {@code p/} rather than where its name derives.
     */
    @Test
    void theKeyedPathIsTheOneTheConverterWouldHaveWritten() {
        assertEquals("cardsfolder/p/+2_mace.txt", ScriptLocator.scriptFileOf(card("+2 Mace")));
    }
}
