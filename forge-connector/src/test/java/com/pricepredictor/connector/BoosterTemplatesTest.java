package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.item.SealedTemplate;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(ForgeExtension.class)
class BoosterTemplatesTest {

    /**
     * Every set Forge's booster map holds resolves to the same template here.
     *
     * <p>The map holds only Draft templates under a bare set code (other
     * booster types carry an affix), so the edition's Draft template is the
     * whole lookup, and no set the map knew is lost by reading the edition.
     */
    @Test
    void everySetTheBoosterMapHoldsResolvesToTheSameTemplate() {
        List<String> differing = new ArrayList<>();
        for (CardEdition edition : StaticData.instance().getEditions()) {
            String code = edition.getCode();
            SealedTemplate mapped = StaticData.instance().getBoosters().get(code);
            if (mapped != null && BoosterTemplates.forSet(code).orElse(null) != mapped) {
                differing.add(code);
            }
        }
        assertTrue(differing.isEmpty(), "resolved differently: " + differing);
    }

    /** An alias names the set it is an alias of: {@code CON} is Conflux, {@code CFX}. */
    @Test
    void anAliasResolvesToItsSetsTemplate() {
        SealedTemplate conflux = BoosterTemplates.forSet("CFX").orElseThrow();

        assertSame(conflux, BoosterTemplates.forSet("CON").orElseThrow());
    }

    @Test
    void anAliasIsCanonicalizedAndAPlainOrUnknownCodeIsNot() {
        assertEquals("CFX", BoosterTemplates.canonicalCode("CON"));
        assertEquals("CFX", BoosterTemplates.canonicalCode("CFX"));
        assertEquals("XXXXUNKNOWN", BoosterTemplates.canonicalCode("XXXXUNKNOWN"));
    }

    @Test
    void anUnknownSetHasNoTemplate() {
        assertEquals(Optional.empty(), BoosterTemplates.forSet("XXXXUNKNOWN"));
        assertEquals(Optional.empty(), BoosterTemplates.forSet(null));
    }

    @Test
    void eligibleSetsExcludeFunnySets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

        // Known un-sets that must be excluded
        List<String> unsets = List.of("UGL", "UNH", "UST", "UND", "UNF");
        for (String unset : unsets) {
            assertFalse(eligibleSets.contains(unset),
                    "Un-set " + unset + " must not be in eligible sets");
        }
    }

    @Test
    void eligibleSetsIncludeKnownSealedSets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

        // Well-known sets that should be in the eligible list
        List<String> expectedSets = List.of("RVR", "MH3", "BLB");
        for (String set : expectedSets) {
            assertTrue(eligibleSets.contains(set),
                    "Known set " + set + " should be in eligible sets");
        }
    }

    @Test
    void eligibleSetsExcludeSmallBoosterSets() {
        List<String> eligibleSets = BoosterTemplates.eligibleSealedSets();

        // Legacy sets whose original boosters held only 8 cards are too small
        // for a realistic sealed pool. Must be excluded.
        List<String> tinyBoosterSets = List.of("DRK", "FEM");
        for (String set : tinyBoosterSets) {
            assertFalse(eligibleSets.contains(set),
                    "Small-booster set " + set + " must not be in eligible sets");
        }
    }
}
