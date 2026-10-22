package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which converted tree a variant's provenance keys name.
 *
 * <p>A staged variant is an ordinary non-token card by the time Forge has
 * loaded it, so nothing on the card itself says it is synthetic. Registering
 * the staged names is the only thing that keeps a variant's records keyed under
 * {@code variant-scripts/} — keyed under {@code cardsfolder/} they would name a
 * sidecar that does not exist, and the reader fails loudly on a corpus that is
 * in fact correct.
 */
@ExtendWith(ForgeExtension.class)
class VariantRegistryTest {

    // The registry is JVM-wide, so each test names its variants uniquely.

    @Test
    void anUnregisteredNameIsNotAVariant() {
        assertFalse(VariantRegistry.isVariant("Lightning Bolt"));
    }

    @Test
    void registrationReadsTheNameLineOfEveryScript(@TempDir Path dir)
            throws IOException {
        write(dir, "lightning_bolt_variant_0.txt", "Lightning Bolt Variant 0");
        write(dir, "shock_variant_1.txt", "Shock Variant 1");

        VariantRegistry.registerAll(dir);

        assertTrue(VariantRegistry.isVariant("Lightning Bolt Variant 0"));
        assertTrue(VariantRegistry.isVariant("Shock Variant 1"));
        assertFalse(VariantRegistry.isVariant("Lightning Bolt"));
    }

    @Test
    void aNonScriptFileIsIgnored(@TempDir Path dir) throws IOException {
        write(dir, "bolt_variant_2.txt", "Bolt Variant 2");
        // Only the extension tells a script from these: each holds a Name: line.
        Files.writeString(dir.resolve("bolt_variant_2.provenance.json"),
                "Name:Sidecar Not A Script\n");
        Files.writeString(dir.resolve("loadable.lst"), "Name:Listed Not A Script\n");

        VariantRegistry.registerAll(dir);

        assertTrue(VariantRegistry.isVariant("Bolt Variant 2"));
        assertFalse(VariantRegistry.isVariant("Listed Not A Script"));
        assertFalse(VariantRegistry.isVariant("Sidecar Not A Script"));
    }

    @Test
    void aMissingOrNonDirectorySourceRegistersNothing(@TempDir Path dir) throws IOException {
        write(dir, "lonely.txt", "Lonely Variant 3");

        VariantRegistry.registerAll(dir.resolve("absent"));
        // A script file is not a tree to walk, even though it holds a name.
        VariantRegistry.registerAll(dir.resolve("lonely.txt"));

        assertFalse(VariantRegistry.isVariant("Lonely Variant 3"));
    }

    @Test
    void aRegisteredVariantsKeyNamesTheVariantTree(@TempDir Path dir)
            throws IOException {
        write(dir, "lightning_bolt_variant_4.txt", "Lightning Bolt Variant 4");
        VariantRegistry.registerAll(dir);
        Card variant = new Card(TestCards.nextCardId(), TestCards.game());
        variant.setName("Lightning Bolt Variant 4");

        assertEquals("variant-scripts/lightning_bolt_variant_4.txt",
                ScriptLocator.scriptFileOf(variant));
    }

    private static void write(Path dir, String file, String name)
            throws IOException {
        Files.writeString(
                dir.resolve(file),
                "Name:" + name + "\nManaCost:R\nTypes:Instant\n"
                        + "A:SP$ DealDamage | Cost$ R | ValidTgts$ Any | NumDmg$ 7\n");
    }
}
