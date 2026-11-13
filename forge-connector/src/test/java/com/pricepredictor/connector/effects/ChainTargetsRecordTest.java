package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AbilitySub;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.optionOf;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordLines.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A stack item's {@code refs.targets} and {@code refs.targets_at_cast}, read
 * over its whole chain, played through the engine's own resolution with the
 * resolution hook installed (X1 #2, X1 #14, X1 #15).
 */
@ExtendWith(ForgeExtension.class)
class ChainTargetsRecordTest {

    private static final String WHY = "for the resolution hook's targets as cast";

    @TempDir
    Path tempDir;

    private static Map<String, Object> refsOf(String record) {
        return asMap(asMap(parse(record).get("state")).get("refs"));
    }

    /**
     * Arc Trail's second target sits on its sub-ability: both halves name
     * both targets, and its effect half names them as cast too.
     */
    @Test
    void aSubAbilitysTargetIsAmongTheTargets() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card arcTrail = TestBoard.inHand(game, caster, "Arc Trail");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility spell = TestBoard.spellOf(
                    game.getAction().moveToStack(arcTrail, null), caster);
            spell.getTargets().add(bears);
            spell.getSubAbility().getTargets().add(opponent);
            game.getStack().add(spell);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        List<String> both = List.of(Refs.of(bears), Refs.of(opponent));
        String cost = records(records, "activation").get(0);
        assertEquals(both, refsOf(cost).get("targets"), cost);
        assertNull(refsOf(cost).get("targets_at_cast"), "only an effect half carries them: " + cost);
        String effect = records(records, "resolution").get(0);
        assertEquals("resolved", outcomeOf(effect), effect);
        assertEquals(both, refsOf(effect).get("targets"), effect);
        assertEquals(both, refsOf(effect).get("targets_at_cast"), effect);
    }

    /**
     * Each of Arc Trail's damages belongs to the link that targets its
     * subject (WP4's leftover): Forge deals both from one damage map as the
     * chain's last clause (DamageResolve) resolves, so both read its index before.
     */
    @Test
    void eachDamageOfAChainIsTheLinkTargetingItsSubject() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card wurm = TestBoard.inPlay(game, opponent, "Craw Wurm");
        Card arcTrail = TestBoard.inHand(game, caster, "Arc Trail");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility spell = TestBoard.spellOf(
                    game.getAction().moveToStack(arcTrail, null), caster);
            spell.getTargets().add(wurm);
            spell.getSubAbility().getTargets().add(opponent);
            game.getStack().add(spell);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String effect = records(records, "resolution").get(0);
        Map<String, String> bySubject = new java.util.HashMap<>();
        for (Map<String, Object> event : RecordLines.events(effect)) {
            if ("damage_dealt".equals(event.get("type"))) {
                bySubject.put((String) ((List<?>) event.get("subjects")).get(0),
                        String.valueOf(event.get("attributed_to")));
            }
        }
        assertEquals(Map.of(Refs.of(wurm), "root", Refs.of(opponent), "0"), bySubject, effect);
    }

    /**
     * Shock at a creature that left the battlefield before it resolved
     * fizzles: its effect half holds no target, names the creature as cast,
     * and carries the creature's entity, which the cast's referenced it.
     */
    @Test
    void aFizzledHalfNamesItsTargetAsCast() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, shock, caster, bears);
            game.getAction().exile(bears, null, null);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String effect = records(records, "resolution").get(0);
        assertEquals("fizzled", outcomeOf(effect), effect);
        assertEquals(List.of(), refsOf(effect).get("targets"), effect);
        List<?> atCast = (List<?>) refsOf(effect).get("targets_at_cast");
        assertEquals(1, atCast.size(), effect);
        String target = (String) atCast.get(0);
        List<?> entities = (List<?>) asMap(parse(effect).get("state")).get("entities");
        assertTrue(entities.stream().anyMatch(entity -> target.equals(asMap(entity).get("id"))),
                "the target as cast is an entity of the half's board: " + effect);
        List<?> links = (List<?>) asMap(parse(effect).get("state")).get("links");
        assertTrue(links.stream().map(RecordLines::asMap).anyMatch(link ->
                        "targets".equals(link.get("type")) && target.equals(link.get("to"))),
                "the half's targets link names its target as cast, not after the fizzle: "
                        + links);
    }

    /** The entity of {@code record}'s board whose id is {@code id}, or null. */
    private static Map<String, Object> entity(String record, String id) {
        List<?> entities = (List<?>) asMap(parse(record).get("state")).get("entities");
        return entities.stream().map(RecordLines::asMap)
                .filter(entity -> id.equals(entity.get("id"))).findFirst().orElse(null);
    }

    /**
     * Shock at a Soldier token destroyed in response: the token has ceased
     * to exist, so the fizzled half names it as it last existed, {@code L},
     * role {@code last_known}, in no zone, and holds no {@code E} entity for
     * a card that is gone.
     */
    @Test
    void aTargetThatCeasedToExistIsNamedAsItLastExisted() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card shock = TestBoard.inHand(game, caster, "Shock");
        Card[] soldier = {null};

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            soldier[0] = soldierToken(game, opponent);
            TestBoard.cast(game, shock, caster, soldier[0]);
            game.getAction().destroy(soldier[0], null, false, null);
            game.getAction().checkStateEffects(true);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String effect = records(records, "resolution").get(0);
        assertEquals("fizzled", outcomeOf(effect), effect);
        String lastKnown = "L" + soldier[0].getId();
        assertEquals(List.of(lastKnown), refsOf(effect).get("targets_at_cast"), effect);
        Map<String, Object> gone = entity(effect, lastKnown);
        assertEquals("last_known", gone.get("role"), effect);
        assertNull(gone.get("zone"), effect);
        assertNull(entity(effect, Refs.of(soldier[0])), "no entity for a card that is gone: " + effect);
    }

    /**
     * Shock at a creature flickered in response: the creature on the
     * battlefield is a new object, so the fizzled half names the target as
     * it last existed, {@code L}, beside the new object's {@code E}.
     */
    @Test
    void aTargetThatBecameANewObjectIsNamedAsItLastExisted() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        Card shock = TestBoard.inHand(game, caster, "Shock");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, shock, caster, bears);
            Card exiled = game.getAction().exile(bears, null, null);
            game.getAction().moveToPlay(exiled, opponent, null, null);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String effect = records(records, "resolution").get(0);
        assertEquals("fizzled", outcomeOf(effect), effect);
        String lastKnown = "L" + bears.getId();
        assertEquals(List.of(lastKnown), refsOf(effect).get("targets_at_cast"), effect);
        assertEquals("last_known", entity(effect, lastKnown).get("role"), effect);
        assertEquals("battlefield", entity(effect, Refs.of(bears)).get("zone"),
                "the new object is on the battlefield under its own E: " + effect);
    }

    /**
     * Counterspell at a Shock on the stack names the Shock by its card and
     * lists it among the targets on the stack; Shock at a creature lists none
     * (follow-up item 8).
     */
    @Test
    void aTargetedSpellIsListedAmongTheTargetsOnTheStack() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        Card shock = TestBoard.inHand(game, opponent, "Shock");
        Card counterspell = TestBoard.inHand(game, caster, "Counterspell");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility bolt = TestBoard.cast(game, shock, opponent, bears);
            SpellAbility counter = TestBoard.spellOf(
                    game.getAction().moveToStack(counterspell, null), caster);
            counter.resetTargets();
            counter.getTargets().add(bolt);
            game.getStack().add(counter);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String shockCost = RecordLines.only(records, "activation", "cardsfolder/s/shock.txt");
        assertEquals(List.of(Refs.of(bears)), refsOf(shockCost).get("targets"), shockCost);
        assertEquals(List.of(), refsOf(shockCost).get("targets_on_stack"), shockCost);
        String counterHalf = RecordLines.only(records, "resolution", "cardsfolder/c/counterspell.txt");
        String shockRef = Refs.of(game.getCardState(shock));
        assertEquals(List.of(shockRef), refsOf(counterHalf).get("targets_at_cast"), counterHalf);
        assertEquals(List.of(shockRef), refsOf(counterHalf).get("targets_on_stack"), counterHalf);
    }

    /** A 1/1 Soldier token for {@code owner}, on the battlefield. */
    private static Card soldierToken(Game game, Player owner) {
        Card token = forge.game.card.CardFactory.getCard(
                forge.StaticData.instance().getAllTokens().getToken("w_1_1_soldier"), owner, game);
        return game.getAction().moveToPlay(token, owner, null, null);
    }

    /**
     * Kolaghan's Command with a discard at the opponent and 2 damage at a
     * creature that left before it resolved: each mode's half names its own
     * mode's target, the damage half its creature as cast only, and the
     * resolution partially fizzled — a mode's target counts as the root's
     * would.
     */
    @Test
    void eachModesHalfNamesItsOwnTargets() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card bears = TestBoard.inPlay(game, opponent, "Grizzly Bears");
        TestBoard.inHand(game, opponent, "Mountain");
        Card command = TestBoard.inHand(game, caster, "Kolaghan's Command");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            Card onStack = game.getAction().moveToStack(command, null);
            SpellAbility charm = TestBoard.charmOf(onStack, caster);
            List<AbilitySub> modes = TestBoard.chain(charm, 1, 3);
            modes.get(0).getTargets().add(opponent);
            modes.get(1).getTargets().add(bears);
            game.getStack().add(charm);
            game.getAction().exile(bears, null, null);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String cost = records(records, "activation").get(0);
        assertEquals(List.of(Refs.of(opponent), Refs.of(bears)), refsOf(cost).get("targets"), cost);
        List<String> halves = records(records, "resolution");
        assertEquals(2, halves.size(), records.toString());
        String discard = halves.get(0);
        String damage = halves.get(1);
        assertEquals(1L, optionOf(discard), discard);
        assertEquals(3L, optionOf(damage), damage);
        assertEquals(List.of(Refs.of(opponent)), refsOf(discard).get("targets"), discard);
        assertEquals(List.of(Refs.of(opponent)), refsOf(discard).get("targets_at_cast"), discard);
        assertEquals(List.of(), refsOf(damage).get("targets"), damage);
        // Exiled, the bears are a new object: named as they last existed.
        assertEquals(List.of("L" + bears.getId()), refsOf(damage).get("targets_at_cast"), damage);
        assertEquals("partially_fizzled", outcomeOf(discard), discard);
    }
}
