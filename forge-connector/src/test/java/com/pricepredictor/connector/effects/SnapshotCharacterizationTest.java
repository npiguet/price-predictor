package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.trigger.Trigger;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The state snapshot's rendered bytes, for each way a collector asks for one.
 *
 * <p>Four call shapes reach {@link SnapshotBuilder} in production: an acting
 * spell ability with its host and targets (the bracket's cast, a fork's
 * forced ability, a playability candidate); no acting line at all (combat,
 * the legality subkinds); no acting line minus one static (continuous); and
 * a trait's host with the event it is about to handle (rewrite, trigger).
 * Each renders the same board, which carries something in every inclusion
 * tier, so a refactor that drops or reorders any block or field shows here.
 */
@ExtendWith({ForgeExtension.class, Characterization.PinnedForgeIds.class})
class SnapshotCharacterizationTest {

    @BeforeEach
    void patchedCheckout() {
        Assumptions.assumeTrue(AttributionMode.detect() == AttributionMode.PATCHED,
                "the expected snapshots were captured against the patched ../forge");
    }

    /** {@code toJson(acting, referenced)}: the stack's bolt, its host and target. */
    @Test
    void anActingAbilityWithItsReferencedCards() {
        CharacterizationBoard board = CharacterizationBoard.full(PhaseType.MAIN1);
        SpellAbility bolt = board.game.getStack().peekAbility();
        List<Card> referenced = new ArrayList<>();
        referenced.add(bolt.getHostCard());
        referenced.add(board.named("Serra Angel"));
        // Tier 1 alone carries a card outside the zones the later tiers walk.
        referenced.add(board.game.getCardsIn(ZoneType.Exile).get(0));

        Characterization.assertMatches("snapshot-acting-ability.json",
                new SnapshotBuilder(board.game).toJson(bolt, referenced) + "\n");
    }

    /** {@code toJson(null, List.of())}: a damage step, with blocks declared. */
    @Test
    void noActingLineInACombatDamageStep() {
        CharacterizationBoard board = CharacterizationBoard.full(PhaseType.COMBAT_DAMAGE);
        Card bears = board.named("Grizzly Bears");
        Card soldier = board.named("Soldier Token");
        Card giant = board.named("Hill Giant");
        Combat combat = new Combat(board.you);
        combat.addAttacker(bears, board.opponent);
        combat.addAttacker(soldier, board.opponent);
        combat.addBlocker(bears, giant);
        combat.setBlocked(bears, true);
        combat.setBlocked(soldier, false);
        combat.orderBlockersForDamageAssignment();
        combat.orderAttackersForDamageAssignment();
        board.game.getPhaseHandler().setCombat(combat);

        Characterization.assertMatches("snapshot-no-acting-line.json",
                new SnapshotBuilder(board.game).toJson(null, List.of()) + "\n");
    }

    /** {@code toJson(null, List.of(), staticId)}: the board minus the anthem. */
    @Test
    void noActingLineWithoutTheAnthem() {
        Assumptions.assumeTrue(PatchedCollectors.CAN_SUPPRESS,
                "../forge cannot remove one static's contributions");
        CharacterizationBoard board = CharacterizationBoard.full(PhaseType.MAIN1);
        StaticAbility anthem = board.named("Glorious Anthem")
                .getStaticAbilities().iterator().next();

        Characterization.assertMatches("snapshot-without-static.json",
                new SnapshotBuilder(board.game).toJson(null, List.of(), (long) anthem.getId())
                        + "\n");
    }

    /** {@code toJsonForTrait(source, referenced, pending)}: a trigger about to fire. */
    @Test
    void aTraitsHostWithThePendingEvent() {
        CharacterizationBoard board = CharacterizationBoard.full(PhaseType.MAIN1);
        Card warden = board.battlefield("Soul Warden", board.you);
        Trigger trigger = warden.getTriggers().iterator().next();
        Map<AbilityKey, Object> runParams = AbilityKey.newMap();
        runParams.put(AbilityKey.Card, board.named("Grizzly Bears"));
        runParams.put(AbilityKey.Destination, ZoneType.Battlefield.name());
        EffectEvent pending = RunParamNormalizer.describeParams(trigger, runParams);

        Characterization.assertMatches("snapshot-for-trait.json",
                new SnapshotBuilder(board.game).toJsonForTrait(warden, List.of(warden), pending)
                        + "\n");
    }

    /**
     * {@code global.emblems} with an emblem that carries an activated ability.
     *
     * <p>The full board's Elspeth emblem holds only a static, so its emblem
     * list renders empty; Ob Nixilis's ultimate makes one whose ability is
     * keyed into the list.
     */
    @Test
    void anEmblemWithAnAbilityIsNamedInTheGlobalBlock() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card obNixilis = board.battlefield("Ob Nixilis of the Black Oath", board.you);
        board.resolve(board.ability(obNixilis, ApiType.Effect), board.you);
        board.battlefield("Grizzly Bears", board.you);

        Characterization.assertMatches("snapshot-with-emblem.json",
                new SnapshotBuilder(board.game).toJson(null, List.of()) + "\n");
    }
}
