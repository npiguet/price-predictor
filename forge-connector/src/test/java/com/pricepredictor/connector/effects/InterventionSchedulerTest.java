package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Which lines an intervention may force.
 *
 * <p>Checked here rather than against a game's records: a protection fork plays
 * its turn on and records the lands it taps as a live game does, so a basic
 * land's mana record carrying {@code interventional = true} is no forced one.
 */
@ExtendWith(ForgeExtension.class)
class InterventionSchedulerTest {

    @Test
    void neitherALandNorItsManaAbilityIsForced() {
        Card forest = TestCards.build("Forest");
        List<SpellAbility> abilities = List.copyOf(forest.getAllSpellAbilities());

        assertTrue(abilities.stream().anyMatch(SpellAbility::isLandAbility), "no land ability to test");
        assertTrue(abilities.stream().anyMatch(SpellAbility::isManaAbility), "no mana ability to test");
        for (SpellAbility ability : abilities) {
            assertFalse(InterventionScheduler.worthIntervening(ability, new HashSet<>()),
                    "would force " + ability);
        }
    }

    @Test
    void aSpellIsForcedOncePerLine() {
        SpellAbility bolt = TestCards.build("Lightning Bolt").getFirstSpellAbility();
        Set<String> seen = new HashSet<>();

        assertTrue(InterventionScheduler.worthIntervening(bolt, seen));
        seen.add(AbilityLines.key(bolt));
        assertFalse(InterventionScheduler.worthIntervening(bolt, seen));
    }
}
