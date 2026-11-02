package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.combat.Combat;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MustBlock}, one case per source of a block requirement Forge reads,
 * and the blockers record's {@code must_block} it fills (FR-037).
 */
@ExtendWith(ForgeExtension.class)
class MustBlockTest {

    @TempDir
    Path tempDir;

    private final Game game = TestBoard.twoPlayerGame(PhaseType.COMBAT_DECLARE_BLOCKERS);
    private final Player attacking = game.getPlayers().get(0);
    private final Player defending = game.getPlayers().get(1);

    private Card attacker(String name) {
        return TestBoard.inPlay(game, attacking, name);
    }

    private Card blocker(String name) {
        return TestBoard.inPlay(game, defending, name);
    }

    /** {@code attacker} attacking the defending player, with no blocks yet. */
    private Combat attackWith(Card attacker) {
        Combat combat = new Combat(attacking);
        combat.addAttacker(attacker, defending);
        game.getPhaseHandler().setCombat(combat);
        game.getAction().checkStaticAbilities();
        return combat;
    }

    private static void attach(Card attachment, Card to) {
        attachment.attachToEntity(to, null);
    }

    /** Provoke and "blocks CARDNAME this turn if able" name the attacker on the blocker. */
    @Test
    void theBlockersMustBlockCardsNameTheAttacker() {
        Card bears = attacker("Grizzly Bears");
        Card provoked = blocker("Hill Giant");
        Card free = blocker("Runeclaw Bear");
        provoked.addMustBlockCard(game.getNextTimestamp(), bears);
        Combat combat = attackWith(bears);

        assertTrue(MustBlock.required(provoked, bears, combat));
        assertFalse(MustBlock.required(free, bears, combat));
    }

    /** {@code MustBeBlockedBy}: Slayer's Cleaver's "must be blocked by an Eldrazi if able". */
    @Test
    void mustBeBlockedByNamesTheCreaturesOfItsKind() {
        Card bears = attacker("Grizzly Bears");
        attach(attacker("Slayer's Cleaver"), bears);
        Card eldrazi = blocker("Ulamog's Crusher");
        Card giant = blocker("Hill Giant");
        Combat combat = attackWith(bears);

        assertTrue(MustBlock.required(eldrazi, bears, combat));
        assertFalse(MustBlock.required(giant, bears, combat));
    }

    /** {@code MustBeBlockedByAll}: "All creatures with flying able to block CARDNAME do so." */
    @Test
    void mustBeBlockedByAllNamesEveryCreatureOfItsKind() {
        Card piper = attacker("Talruum Piper");
        Card angel = blocker("Serra Angel");
        Card giant = blocker("Hill Giant");
        Combat combat = attackWith(piper);

        assertTrue(MustBlock.required(angel, piper, combat));
        assertFalse(MustBlock.required(giant, piper, combat));
    }

    /** A lure: "All creatures able to block CARDNAME do so.", granted by Lure. */
    @Test
    void aLureNamesEveryCreature() {
        Card bears = attacker("Grizzly Bears");
        attach(attacker("Lure"), bears);
        Card giant = blocker("Hill Giant");
        Card angel = blocker("Serra Angel");
        Combat combat = attackWith(bears);

        assertTrue(MustBlock.required(giant, bears, combat));
        assertTrue(MustBlock.required(angel, bears, combat));
        Card plain = attacker("Runeclaw Bear");
        assertFalse(MustBlock.required(giant, plain, combat), "the lure is on the bears alone");
    }

    /** {@code StaticAbilityMustBlock}: Iron Golem blocks each combat if able. */
    @Test
    void aBlocksEachCombatStaticNamesEveryAttacker() {
        Card bears = attacker("Grizzly Bears");
        Card golem = blocker("Iron Golem");
        Card giant = blocker("Hill Giant");
        Combat combat = attackWith(bears);

        assertTrue(MustBlock.required(golem, bears, combat));
        assertFalse(MustBlock.required(giant, bears, combat));
    }

    /**
     * A requirement whose block has a cost is none (rule 509.1c): Archangel
     * of Tithes, attacking, makes every block cost {1}, so its lure requires
     * nothing.
     */
    @Test
    void aRequirementWhoseBlockCostsSomethingIsNone() {
        Card archangel = attacker("Archangel of Tithes");
        attach(attacker("Lure"), archangel);
        Card giant = blocker("Hill Giant");
        Combat combat = attackWith(archangel);

        assertFalse(MustBlock.required(giant, archangel, combat));
    }

    /**
     * The blockers record lists, in {@code must_block}, the creatures a
     * requirement makes block its anchor, the forbidden ones included: a
     * requirement is recorded whether or not the creature can obey it.
     */
    @Test
    void aBlockersRecordListsTheCreaturesThatMustBlockItsAnchor() throws Throwable {
        Card bears = attacker("Grizzly Bears");
        attach(attacker("Lure"), bears);
        Card giant = blocker("Hill Giant");
        Card tapped = blocker("Runeclaw Bear");
        tapped.setTapped(true);
        Card secondBears = blocker("Grizzly Bears");
        attackWith(bears);
        List<Card> candidates = List.of(giant, tapped, secondBears);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                HookShapes.proxy(HookShapes.CombatLegality.class,
                        collectors.playability().legalityHandler())
                        .onBlockersDeclarable(defending, bears, candidates,
                                List.of(giant, secondBears), 1));

        List<Map<String, Object>> blockers = lines.stream().map(RecordLines::parse)
                .filter(r -> "blockers".equals(r.get("subkind"))).toList();
        assertEquals(1, blockers.size(), lines.toString());
        Map<String, Object> payload = RecordLines.asMap(blockers.get(0).get("payload"));
        assertEquals(List.of(Refs.of(giant), Refs.of(tapped), Refs.of(secondBears)),
                payload.get("must_block"));
        String written = RecordLines.payloadOf(lines.stream()
                .filter(l -> l.contains("\"subkind\":\"blockers\"")).findFirst().orElseThrow());
        assertEquals(List.of("anchor_attacker", "legal_blockers", "forbidden", "must_block",
                        "min_blockers"), AttackersRecordTest.membersInOrder(written,
                        "anchor_attacker", "legal_blockers", "forbidden", "must_block",
                        "min_blockers"),
                "the member order the contract gives: " + written);
        assertEquals(false, blockers.get(0).get("what_if"));
    }

    /** No requirement anywhere: an empty {@code must_block}. */
    @Test
    void aBlockersRecordWithNoRequirementListsNoCreature() throws Throwable {
        Card bears = attacker("Grizzly Bears");
        Card giant = blocker("Hill Giant");
        attackWith(bears);

        List<String> lines = ShardRun.records(tempDir, game, (bracket, collectors) ->
                HookShapes.proxy(HookShapes.CombatLegality.class,
                        collectors.playability().legalityHandler())
                        .onBlockersDeclarable(defending, bears, List.of(giant), List.of(giant), 1));

        Map<String, Object> payload = RecordLines.payload(lines.stream()
                .filter(l -> l.contains("\"subkind\":\"blockers\"")).findFirst().orElseThrow());
        assertEquals(List.of(), payload.get("must_block"));
    }
}
