package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.CardStateName;
import forge.game.Game;
import forge.game.GameObject;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
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
import java.util.Set;
import java.util.function.Supplier;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordAbilities.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * The acting item's {@code stack_extras} on its effect half: its targets,
 * divided amounts and chosen count as cast, over its whole chain, spell
 * targets included, never what Forge's check for illegal targets left on it
 * (608.2b). Read off the thinned item, a fizzled half's extras would show no
 * target and hand the model the fizzle it is to predict.
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

    /** The effect half of {@code script}'s spell, resolved once {@code play} has put it on the stack. */
    private String resolutionHalf(Game game, String script, Runnable play) throws Throwable {
        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            play.run();
            resolveAndMoveOn(game);
        });
        return RecordPicks.only(records, "resolution", script);
    }

    /**
     * The extras of {@code host} on {@code half} name {@code expected} as the
     * refs' {@code targets_at_cast} do, and count them as chosen.
     */
    private static void assertTargetsAsCast(String half, Card host, List<String> expected) {
        Map<String, Object> extras = extrasOf(half, host);
        assertEquals(expected, extras.get("targets"), half);
        assertEquals(refsOf(half).get("targets_at_cast"), extras.get("targets"), half);
        assertEquals(Map.of("chosen", (long) expected.size()), extras.get("up_to_counts"), half);
    }

    /**
     * The spell {@code cast} puts on the stack fizzles once {@code target} is
     * sacrificed in response; its host's extras still hold the target as cast.
     */
    private void assertFizzledKeepingItsTarget(
            Game game, String script, Card target, Supplier<SpellAbility> cast) throws Throwable {
        SpellAbility[] spell = {null};
        String half = resolutionHalf(game, script, () -> {
            spell[0] = cast.get();
            sacrifice(game, target);
        });
        assertEquals("fizzled", outcomeOf(half), half);
        assertTargetsAsCast(half, spell[0].getHostCard(), List.of("L" + target.getId()));
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
        assertFizzledKeepingItsTarget(game, "cardsfolder/l/lightning_bolt.txt", bears,
                () -> TestBoard.cast(game, bolt, caster, bears));
    }

    /**
     * {@code card}'s spell on {@code face}, cast as the engine's play path
     * casts it: the card moved to the stack with that ability as its cause,
     * which turns it to the face, then the ability put on the stack for its owner, aimed at
     * {@code target} from its first ability that takes one.
     */
    private static SpellAbility castOnFace(
            Game game, Card card, CardStateName face, GameObject target) {
        Player caster = card.getOwner();
        SpellAbility faceSpell = card.getAllSpellAbilities().stream()
                .filter(sa -> sa.isSpell() && sa.getCardState() != null
                        && sa.getCardState().getStateName() == face)
                .findFirst().orElseThrow(() -> new AssertionError("no spell on " + face));
        faceSpell.setActivatingPlayer(caster);
        SpellAbility spell = TestBoard.spellOf(game.getAction().moveToStack(card, faceSpell), caster);
        for (SpellAbility link = spell; link != null; link = link.getSubAbility()) {
            if (link.usesTargeting()) {
                link.resetTargets();
                link.getTargets().add(target);
                break;
            }
        }
        game.getStack().add(spell);
        return spell;
    }

    /**
     * Bonecrusher Giant's Stomp, the Adventure face cast from a hand that
     * holds the card as the Giant, aimed at a creature sacrificed in
     * response: the host shows its extras as cast like any other host.
     */
    @Test
    void anAdventureSpellsExtrasHoldItsTargetAsCast() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card giant = TestBoard.inHand(game, caster, "Bonecrusher Giant");
        assertFizzledKeepingItsTarget(game, "cardsfolder/b/bonecrusher_giant_stomp.txt", bears,
                () -> castOnFace(game, giant, CardStateName.Secondary, bears));
    }

    /**
     * Hagra Mauling, a modal double-faced card cast as its spell face,
     * aimed at a creature sacrificed in response: the same.
     */
    @Test
    void aModalDoubleFacedSpellsExtrasHoldItsTargetAsCast() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card mauling = TestBoard.inHand(game, caster, "Hagra Mauling");
        assertFizzledKeepingItsTarget(game, "cardsfolder/h/hagra_mauling_hagra_broodpit.txt", bears,
                () -> castOnFace(game, mauling, CardStateName.Original, bears));
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

        String half = resolutionHalf(game, "cardsfolder/a/arc_trail.txt", () -> {
            cast[0] = TestBoard.spellOf(game.getAction().moveToStack(arcTrail, null), caster);
            cast[0].getTargets().add(bears);
            cast[0].getSubAbility().getTargets().add(opponent);
            game.getStack().add(cast[0]);
            sacrifice(game, bears);
        });
        assertEquals("partially_fizzled", outcomeOf(half), half);
        assertTargetsAsCast(half, cast[0].getHostCard(),
                List.of("L" + bears.getId(), Refs.of(opponent)));
    }

    /**
     * Forked Bolt split 1 and 1, one of its creatures sacrificed in
     * response: Forge drops that target's share with the target, and the
     * extras keep it as announced.
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

        String half = RecordPicks.only(records, "resolution", "cardsfolder/f/forked_bolt.txt");
        assertEquals("partially_fizzled", outcomeOf(half), half);
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        String lost = "L" + bears.getId();
        assertEquals(List.of(lost, Refs.of(wurm)), extras.get("targets"), half);
        assertEquals(Map.of(lost, 1L, Refs.of(wurm), 1L), extras.get("per_target_amounts"), half);
    }

    /**
     * Arc Lightning split 1, 1 and 1 over three creatures, two of them
     * sacrificed in response: Forge drops each lost target's share with it,
     * and the extras keep all three shares as announced, which the lone-target
     * arithmetic could not give back for two.
     */
    @Test
    void aDividedSpellThatLostTwoTargetsKeepsAllThreeShares() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card wurm = TestBoard.inPlay(game, opponent, "Craw Wurm");
        Card giant = TestBoard.inPlay(game, opponent, "Hill Giant");
        Card arcLightning = TestBoard.inHand(game, caster, "Arc Lightning");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            cast[0] = TestBoard.spellOf(game.getAction().moveToStack(arcLightning, null), caster);
            cast[0].clearTargets();
            for (Card target : List.of(bears, wurm, giant)) {
                cast[0].getTargets().add(target);
                cast[0].addDividedAllocation(target, 1);
            }
            game.getStack().add(cast[0]);
            sacrifice(game, bears);
            sacrifice(game, wurm);
            resolveAndMoveOn(game);
        });

        String half = RecordPicks.only(records, "resolution", "cardsfolder/a/arc_lightning.txt");
        assertEquals("partially_fizzled", outcomeOf(half), half);
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        assertEquals(Map.of("L" + bears.getId(), 1L, "L" + wurm.getId(), 1L, Refs.of(giant), 1L),
                extras.get("per_target_amounts"), half);
    }

    /**
     * Arc Lightning split 1, 1 and 1 over three creatures, one of them
     * redirected by Spellskite's {@code ChangeTargets} (real card, its
     * {@code ChangeSingleTarget} branch): Forge moves the share to the new
     * target, and the effect half writes it as announced there, the old
     * target no longer a target and without a share.
     */
    @Test
    void aRetargetedDividedSpellKeepsTheShareItMovedToItsNewTarget() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card wurm = TestBoard.inPlay(game, opponent, "Craw Wurm");
        Card skite = TestBoard.inPlay(game, opponent, "Spellskite");
        Card arcLightning = TestBoard.inHand(game, caster, "Arc Lightning");
        SpellAbility[] cast = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            cast[0] = TestBoard.spellOf(game.getAction().moveToStack(arcLightning, null), caster);
            cast[0].clearTargets();
            for (Card target : List.of(bears, wurm)) {
                cast[0].getTargets().add(target);
                cast[0].addDividedAllocation(target, target == bears ? 2 : 1);
            }
            game.getStack().add(cast[0]);
            SpellAbility redirect = TestBoard.ability(skite, ApiType.ChangeTargets);
            redirect.setActivatingPlayer(opponent);
            redirect.resetTargets();
            redirect.getTargets().add(cast[0]);
            game.getStack().add(redirect);
            game.getStack().resolveStack();
            resolveAndMoveOn(game);
        });

        String half = RecordPicks.only(records, "resolution", "cardsfolder/a/arc_lightning.txt");
        Map<String, Object> extras = extrasOf(half, cast[0].getHostCard());
        assertEquals(Map.of(Refs.of(skite), 2L, Refs.of(wurm), 1L), extras.get("per_target_amounts"), half);
        assertEquals(Set.of(Refs.of(skite), Refs.of(wurm)), Set.copyOf((List<?>) extras.get("targets")), half);
    }

    /**
     * The same spell forced in a fork reads the shares it announced at its
     * forced cast: all of Arc Lightning's 3 damage, shared out over the
     * targets its extras name, as the live half's are.
     */
    @Test
    void aForcedDividedSpellsExtrasShareOutItsWholeTotal() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Hill Giant");
        TestBoard.inPlay(game, game.getPlayers().get(1), "Grizzly Bears");
        Card arcLightning = TestBoard.inHand(game, caster, "Arc Lightning");
        SpellAbility spell = arcLightning.getFirstSpellAbility();
        spell.setActivatingPlayer(caster);
        CollectionCaps caps = TestCaps.caps().manaCap(2000).interventions(1).probes(0).build();

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(tempDir, new ShardRun.ForkRun(game, caps, 1L, spell), WHY);

        String half = RecordPicks.only(fork.records(), "resolution", "cardsfolder/a/arc_lightning.txt");
        Map<String, Object> extras = extrasOf(half, arcLightning);
        Map<String, Object> amounts = asMap(extras.get("per_target_amounts"));
        assertEquals(3L, amounts.values().stream().mapToLong(each -> (Long) each).sum(), half);
        // A JSON object's members have no order (the parser hands back a
        // HashMap): the shares name the targets, whatever order each lists.
        assertEquals(Set.copyOf((List<?>) extras.get("targets")), amounts.keySet(), half);
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

        String half = RecordPicks.only(records, "resolution", "cardsfolder/c/counterspell.txt");
        Map<String, Object> extras = extrasOf(half, counter[0].getHostCard());
        String shockRef = Refs.of(game.getCardState(shock));
        assertEquals(List.of(shockRef), extras.get("targets"), half);
        assertEquals(refsOf(half).get("targets_at_cast"), extras.get("targets"), half);
        assertEquals(Map.of("chosen", 1L), extras.get("up_to_counts"), half);
    }
}
