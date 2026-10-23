package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** What {@link ActivationLedger} says became of a cast that resolved. */
@ExtendWith(ForgeExtension.class)
class ActivationLedgerTest {

    /**
     * Losing some targets but not all is a partial fizzle.
     *
     * <p>{@code MagicStack.hasFizzled} strips the targets that became illegal
     * before the spell resolves, so the count at resolution against the count at
     * the cast is the engine's own answer. Losing every target without the engine
     * calling it a fizzle is {@code CantFizzle}, where the spell still does what
     * it does.
     */
    @Test
    void losingSomeTargetsButNotAllIsAPartialFizzle() {
        SpellAbility bolt = AbilityFactory.getAbility(
                "SP$ DealDamage | Cost$ R | ValidTgts$ Any | TgtPrompt$ Choose | NumDmg$ 3",
                TestCards.build("Lightning Bolt"));
        bolt.resetTargets();
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));
        bolt.getTargets().add(TestCards.build("Grizzly Bears"));

        assertEquals(ActivationOutcome.RESOLVED,
                ActivationLedger.outcomeOf(2, bolt));

        bolt.getTargets().remove(bolt.getTargets().get(0));
        assertEquals(ActivationOutcome.PARTIALLY_FIZZLED,
                ActivationLedger.outcomeOf(2, bolt));

        bolt.resetTargets();
        assertEquals(ActivationOutcome.RESOLVED,
                ActivationLedger.outcomeOf(2, bolt));
    }

    /** An untargeted spell never reads as partially fizzled. */
    @Test
    void anUntargetedSpellIsSimplyResolved() {
        SpellAbility gain = AbilityFactory.getAbility(
                "SP$ GainLife | Cost$ W | Defined$ You | LifeAmount$ 3",
                TestCards.build("Healing Salve"));

        assertEquals(ActivationOutcome.RESOLVED,
                ActivationLedger.outcomeOf(0, gain));
    }
}
