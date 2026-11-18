package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityKey;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardZoneTable;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The acting item's {@code stack_extras} on its effect half: its targets,
 * divided amounts and chosen count as cast, over its whole chain, spell
 * targets included, never what Forge's check for illegal targets left on it
 * (608.2b). Read off the thinned item, a fizzled half's extras would show no
 * target and hand the model the fizzle it is to predict (finding 4.1).
 */
@ExtendWith(ForgeExtension.class)
class ActingStackExtrasTest {

    private static final String WHY = "for the resolution hook's targets as cast";

    @TempDir
    Path tempDir;

    private static Map<String, Object> refsOf(String record) {
        return asMap(asMap(parse(record).get("state")).get("refs"));
    }

    /** The {@code stack_extras} of {@code card}'s entity on {@code record}'s board. */
    private static Map<String, Object> extrasOf(String record, Card card) {
        List<?> entities = (List<?>) asMap(parse(record).get("state")).get("entities");
        Map<String, Object> entity = entities.stream().map(RecordLines::asMap)
                .filter(each -> Refs.of(card).equals(each.get("id"))).findFirst()
                .orElseThrow(() -> new AssertionError("no entity " + Refs.of(card) + ": " + record));
        Map<String, Object> extras = asMap(entity.get("stack_extras"));
        assertNotNull(extras, "the acting item's host carries its extras: " + record);
        return extras;
    }

    /** Sacrifice {@code card} as an effect would, with the last state the move reads. */
    private static void sacrifice(Game game, Card card) {
        Map<AbilityKey, Object> params = AbilityKey.newMap();
        AbilityKey.addCardZoneTableParams(params, new CardZoneTable(
                game.copyLastStateBattlefield(), game.copyLastStateGraveyard()));
        game.getAction().sacrifice(new CardCollection(card), null, true, params);
    }

    private static void resolveAndMoveOn(Game game) {
        game.getStack().resolveStack();
        TestBoard.priority(game);
        TestBoard.nextStep(game);
    }

    /**
     * Lightning Bolt whose target was sacrificed in response fizzles; its
     * extras still name the creature as cast, as it last existed, with one
     * target chosen: the refs' {@code targets_at_cast}.
     */
    @Test
    void aFizzledBoltsExtrasHoldItsTargetAsCast() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card bolt = TestBoard.inHand(game, caster, "Lightning Bolt");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            cast[0] = TestBoard.cast(game, bolt, caster, bears);
            sacrifice(game, bears);
            resolveAndMoveOn(game);
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        assertEquals("fizzled", outcomeOf(half), half);
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        assertEquals(List.of("L" + bears.getId()), extras.get("targets"), half);
        assertEquals(refsOf(half).get("targets_at_cast"), extras.get("targets"), half);
        assertEquals(Map.of("chosen", 1L), extras.get("up_to_counts"), half);
    }

    /**
     * Arc Trail with its creature sacrificed in response partially fizzles;
     * its extras hold both targets, the sub-ability's player too, and two
     * chosen.
     */
    @Test
    void aPartiallyFizzledSpellsExtrasHoldBothTargets() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card arcTrail = TestBoard.inHand(game, caster, "Arc Trail");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            cast[0] = TestBoard.spellOf(game.getAction().moveToStack(arcTrail, null), caster);
            cast[0].getTargets().add(bears);
            cast[0].getSubAbility().getTargets().add(opponent);
            game.getStack().add(cast[0]);
            sacrifice(game, bears);
            resolveAndMoveOn(game);
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/a/arc_trail.txt");
        assertEquals("partially_fizzled", outcomeOf(half), half);
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        assertEquals(List.of("L" + bears.getId(), Refs.of(opponent)), extras.get("targets"), half);
        assertEquals(refsOf(half).get("targets_at_cast"), extras.get("targets"), half);
        assertEquals(Map.of("chosen", 2L), extras.get("up_to_counts"), half);
    }

    /**
     * Forked Bolt split 1 and 1, one of its creatures sacrificed in
     * response: Forge drops that target's share with the target, and the
     * extras give it back, what the other leaves of the 2 announced.
     */
    @Test
    void aPartiallyFizzledDividedSpellKeepsTheShareOfItsLostTarget() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card wurm = TestBoard.inPlay(game, opponent, "Craw Wurm");
        Card forkedBolt = TestBoard.inHand(game, caster, "Forked Bolt");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            cast[0] = TestBoard.spellOf(game.getAction().moveToStack(forkedBolt, null), caster);
            cast[0].clearTargets();
            cast[0].getTargets().add(bears);
            cast[0].addDividedAllocation(bears, 1);
            cast[0].getTargets().add(wurm);
            cast[0].addDividedAllocation(wurm, 1);
            game.getStack().add(cast[0]);
            sacrifice(game, bears);
            resolveAndMoveOn(game);
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/f/forked_bolt.txt");
        assertEquals("partially_fizzled", outcomeOf(half), half);
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        String lost = "L" + bears.getId();
        assertEquals(List.of(lost, Refs.of(wurm)), extras.get("targets"), half);
        assertEquals(Map.of(lost, 1L, Refs.of(wurm), 1L), extras.get("per_target_amounts"), half);
    }

    /**
     * Counterspell at a Shock on the stack: its extras name the Shock by its
     * card, the ref its {@code targets_at_cast} gives it, one target chosen.
     * The Shock is still a legal target, so this is the spell target counted
     * at all, as cast or not.
     */
    @Test
    void aCounterspellsExtrasHoldItsSpellTarget() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        Card shock = TestBoard.inHand(game, opponent, "Shock");
        Card counterspell = TestBoard.inHand(game, caster, "Counterspell");
        SpellAbility[] counter = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility spell = TestBoard.cast(game, shock, opponent, bears);
            counter[0] = TestBoard.spellOf(
                    game.getAction().moveToStack(counterspell, null), caster);
            counter[0].resetTargets();
            counter[0].getTargets().add(spell);
            game.getStack().add(counter[0]);
            resolveAndMoveOn(game);
        });

        String half = RecordLines.only(records, "resolution", "cardsfolder/c/counterspell.txt");
        Map<String, Object> extras = extrasOf(half, counter[0].getHostCard());
        String shockRef = Refs.of(game.getCardState(shock));
        assertEquals(List.of(shockRef), extras.get("targets"), half);
        assertEquals(refsOf(half).get("targets_at_cast"), extras.get("targets"), half);
        assertEquals(Map.of("chosen", 1L), extras.get("up_to_counts"), half);
    }
}
