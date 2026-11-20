package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static com.pricepredictor.connector.effects.ApiClauseObservations.observe;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * The {@link HealGrantAndRetargetApiRules}: healed damage, granted
 * abilities and retargeted spells.
 */
@ExtendWith(ForgeExtension.class)
class ApiEventsHealGrantAndRetargetTest {

    /**
     * HealDamageEffect calls healDamage() and leaves zero behind, so the amount
     * exists only before the clause. This is the case the before-half of the
     * hook exists for.
     *
     * <p>The {@code HealDamage} clause itself carries no targets, because the
     * real engine never gives it any: {@code HealDamageEffect.resolve()} reads
     * its cards through {@code SpellAbilityEffect.getTargetCards(sa)}, which
     * only consults {@code sa.getTargets()} when {@code sa.usesTargeting()}
     * is true, and neither real cardsfolder usage declares any targeting on
     * the {@code HealDamage} clause -- {@code pyramids.txt}'s is {@code
     * Defined$ ReplacedCard}, {@code wolverine_fierce_fighter.txt}'s is
     * {@code Defined$ ReplacedTarget}. Targets added straight onto the clause
     * would let a {@code damageBefore} memo that reads {@code sa.getTargets()}
     * unconditionally pass.
     *
     * <p>Builds the shape production actually uses instead: no {@code
     * ValidTgts$}/{@code Tgt$} on the {@code HealDamage} clause itself
     * (matching both real cards), resolving its card through {@code
     * Defined$ Targeted} instead -- the same {@code getAllTargetChoices()}
     * mechanism {@link ApiEventsTurnAndPhaseTest#aSkippedTurnThroughDefinedTargetedNamesTheRootsTarget}
     * already exercises and pins for players, mirrored here for a card, so
     * the healed creature is a different object than the sub-ability's own
     * host and a hardcoded {@code sa.getHostCard()} shortcut could not pass
     * this by coincidence.
     */
    @Test
    void healedDamageIsReadFromBeforeTheClause() {
        SpellAbility root = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        SpellAbility sub = AbilityFactory.getAbility(
                "DB$ HealDamage | Defined$ Targeted", root.getHostCard());
        root.setSubAbility((AbilitySub) sub);
        Card target = TestCards.build("Runeclaw Bear");
        target.setDamage(3);
        root.resetTargets();
        root.getTargets().add(target);

        ApiEvents.ClauseObservation<?> observed = observe(sub);
        target.setDamage(0);

        EffectEvent event = TestCards.onlyEvent(observed);

        assertEquals(EventType.DAMAGE_HEALED, event.type());
        assertEquals(3, event.param(EventParam.AMOUNT));
    }

    /**
     * Dragonshift: {@code SP$ Animate | ... | RemoveAllAbilities$ True | ...},
     * a root ability. No reachable root-level Animate/AnimateAll script in
     * the cardsfolder pairs with an {@code Abilities$} grant list (the
     * common shapes use {@code Keywords$}, {@code staticAbilities$} or
     * {@code Triggers$} instead), so {@code abilities} here is the {@code ""}
     * fallback; {@code removed} is the distinct, identifiable value this
     * card actually pins.
     */
    @Test
    void anAnimateClauseNamesWhetherItStrippedAbilities() {
        SpellAbility sa = TestCards.scriptedAbility("Dragonshift", "Animate");

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.ABILITY_CHANGE, event.type());
        assertEquals(true, event.param(EventParam.REMOVED));
    }

    /**
     * Wild Ricochet: {@code SP$ ChangeTargets | TargetType$ Spell | ...}, a
     * root ability -- built and inspected directly rather than resolved,
     * since resolution needs a live {@code SpellAbilityStackInstance} this
     * harness does not construct. The retarget helper reads {@code
     * sa.getTargets().getTargetSpells()} (the spell(s) the clause itself
     * targets) and then each one's own, post-change targets, so wiring a
     * hand-built "changed" ability directly onto {@code sa}'s targets and
     * giving that ability its own new target exercises the same read a real
     * resolution would leave behind.
     */
    @Test
    void changeTargetsNamesWhatTheRetargetedSpellPointsAtNow() {
        SpellAbility sa = TestCards.scriptedAbility("Wild Ricochet", "ChangeTargets");
        SpellAbility changed = AbilityFactory.getAbility(
                "SP$ Pump | ValidTgts$ Creature | NumAtt$ +0 | NumDef$ +0",
                TestCards.build("Grizzly Bears"));
        Card newTarget = TestCards.build("Runeclaw Bear");
        changed.resetTargets();
        changed.getTargets().add(newTarget);
        sa.resetTargets();
        sa.getTargets().add(changed);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.TARGETS_CHANGED, event.type());
        assertEquals(List.of(Refs.of(newTarget)), event.param(EventParam.TARGETS));
    }

    /**
     * Redirecting a burn spell at a
     * <em>player</em> is the classic use of {@code ChangeTargets} --
     * Deflection, Misdirection and Bolt Bend are all real, and all reach
     * {@code ChangeTargetsEffect}'s default "choose any new legal target"
     * branch (no {@code RandomTarget}/{@code DefinedMagnet}/{@code
     * ChangeSingleTarget} param), which can retarget at a player as readily
     * as a card. A {@code retargeted()} iterating only {@code
     * getTargetCards()}, which {@code TargetChoices} filters to {@code Card}
     * instances, would drop a {@code Player} target silently, and the event
     * would still fire with {@code targets: []}, so the channel would look
     * wired while reporting nothing distinguishable from "retargeted at
     * nothing."
     */
    @Test
    void changeTargetsNamesARetargetedPlayerNotJustCards() {
        SpellAbility sa = TestCards.scriptedAbility("Deflection", "ChangeTargets");
        SpellAbility changed = AbilityFactory.getAbility(
                "SP$ DealDamage | ValidTgts$ Any | NumDmg$ 3",
                TestCards.build("Grizzly Bears"));
        Player newTarget = new Player("redirect-target", TestCards.game(), 94001);
        changed.resetTargets();
        changed.getTargets().add(newTarget);
        sa.resetTargets();
        sa.getTargets().add(changed);

        EffectEvent event = TestCards.onlyEvent(observe(sa));

        assertEquals(EventType.TARGETS_CHANGED, event.type());
        assertEquals(List.of(Refs.of(newTarget)), event.param(EventParam.TARGETS));
    }
}
