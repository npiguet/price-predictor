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

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A modal resolution as one cost half and one effect half per chosen mode
 * (FR-029a–e).
 *
 * <p>The modes are chained exactly as the engine chains them —
 * {@code CharmEffect.chainAbilities} on a real charm's own {@code Choices$}
 * — and the clause hook is driven the way {@code AbilityUtils.resolveApiAbility}
 * drives it, with the clone that is about to resolve. What is not here is the
 * engine resolving the charm, which needs a running game and is the pilot's
 * job; what is here is everything the collectors decide about a clause once
 * they are handed one, which is where a wrong option or a merged half would
 * come from.
 */
@ExtendWith(ForgeExtension.class)
class ModalResolutionTest {

    @TempDir
    Path tempDir;

    /** Why a fork test skips on a stock checkout. */
    private static final String STOCK_FORK =
            "to split a fork's modes; a stock fork writes one root half";

    private static Method resolving() {
        return HookShapes.method(HookShapes.Clause.class, "onClauseResolving");
    }

    /**
     * The charm spell of a real card, activated by {@code owner}, in a game
     * with two players so a snapshot has a life total to move.
     */
    private static SpellAbility charmOf(Game game, Player owner, String cardName) {
        return TestBoard.charmOf(TestBoard.card(game, owner, cardName), owner);
    }

    private static String field(String json, String key) {
        int at = json.indexOf("\"" + key + "\":");
        assertTrue(at >= 0, key + " missing from " + json);
        int from = at + key.length() + 3;
        int to = json.indexOf(",\"", from);
        return json.substring(from, to < 0 ? json.length() - 1 : to);
    }

    // ── the keys ────────────────────────────────────────────────────────

    /** A chosen mode's clone keys to the charm's line plus its Choices$ position. */
    @Test
    void aModeCloneResolvesToTheRootKeyWithItsOption() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Boros Charm");
        List<AbilitySub> clones = TestBoard.chain(charm, 0, 2);

        ProvenanceKey root = ProvenanceKey.resolve(charm).key();
        assertNotNull(root);
        assertNull(root.option(), "the charm itself is the root line");
        assertEquals(root.withOption(0), ProvenanceKey.resolve(clones.get(0)).key());
        assertEquals(root.withOption(2), ProvenanceKey.resolve(clones.get(1)).key());
    }

    /** A mode's own sub-abilities resolve under the mode, and open no mode of their own. */
    @Test
    void aModesOwnSubAbilityCarriesTheModesOptionAndOpensNoMode() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Season of Gathering");
        // Mode 0 is "put a +1/+1 counter", which chains a pump and a cleanup.
        List<AbilitySub> clones = TestBoard.chain(charm, 0);
        assertTrue(clones.size() >= 2, "the mode's chain should follow its clone: " + clones);
        AbilitySub mode = clones.get(0);
        AbilitySub pump = clones.get(1);

        ProvenanceKey root = ProvenanceKey.resolve(charm).key();
        assertEquals(root.withOption(0), ProvenanceKey.resolve(pump).key());
        assertNotNull(CharmModes.modeStartOf(mode));
        assertNull(CharmModes.modeStartOf(pump), "a sub-ability of a mode opens nothing");
    }

    /** Spec Story 3 scenario 13: one mode chosen twice gives two clones under one option. */
    @Test
    void aRepeatedModeIsTwoClonesOfOneOption() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Brokers Confluence");
        List<AbilitySub> clones = TestBoard.chain(charm, 1, 1, 0);

        assertEquals(3, clones.size());
        assertEquals(0, CharmModes.modeStartOf(clones.get(0)).option());
        assertEquals(1, CharmModes.modeStartOf(clones.get(1)).option());
        assertEquals(1, CharmModes.modeStartOf(clones.get(2)).option());
    }

    // ── the halves ──────────────────────────────────────────────────────

    /**
     * Spec Story 3 scenario 12: a "choose two" charm cast with modes 1 and 3
     * writes one cost record through the root line and two effect halves
     * through the option lines, each carrying only its own mode's events, a
     * snapshot taken just before its mode resolved, and all three one link.
     */
    @Test
    void aPatchedWorkerWritesOneEffectHalfPerChosenMode() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        SpellAbility charm = charmOf(game, caster, "Boros Charm");
        List<AbilitySub> clones = TestBoard.chain(charm, 0, 2);

        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(charm);

            collectors.clauseHandler().invoke(null, resolving(), new Object[]{charm});
            collectors.clauseHandler().invoke(null, resolving(), new Object[]{clones.get(0)});
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.DAMAGE_DEALT).param("amount", 4));
            // The first mode's effect: a life total that moved, which the
            // second mode's snapshot must show.
            caster.getOpponents().get(0).setLife(16, null);
            collectors.clauseHandler().invoke(null, resolving(), new Object[]{clones.get(1)});
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.KEYWORD_CHANGE));
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        });

        assertEquals(3, records.size(), records.toString());
        String cost = records.get(0);
        String first = records.get(1);
        String second = records.get(2);
        assertTrue(cost.contains("\"moment\":\"activation\""), cost);
        assertFalse(cost.contains("\"option\""), "the cost half acts through the root: " + cost);

        String link = field(cost, "link_id");
        assertEquals(link, field(first, "link_id"));
        assertEquals(link, field(second, "link_id"));

        assertTrue(first.contains("\"option\":0}"), first);
        assertTrue(second.contains("\"option\":2}"), second);
        assertTrue(first.contains("damage_dealt") && !first.contains("keyword_change"),
                "the first half carries its own mode's events only: " + first);
        assertTrue(second.contains("keyword_change") && !second.contains("damage_dealt"),
                "the second half carries its own mode's events only: " + second);
        assertTrue(first.contains("\"life\":20"), "the first snapshot predates the damage: " + first);
        assertTrue(second.contains("\"life\":16"),
                "the second snapshot shows the first mode's effect: " + second);
    }

    /** Spec Story 3 scenario 13 on the record: a repeated mode acts in two halves. */
    @Test
    void aRepeatedModeActsInOneHalfPerResolution() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Brokers Confluence");
        List<AbilitySub> clones = TestBoard.chain(charm, 0, 0, 2);

        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(charm);
            for (AbilitySub clone : clones) {
                collectors.clauseHandler().invoke(null, resolving(), new Object[]{clone});
            }
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        });

        assertEquals(4, records.size(), records.toString());
        assertTrue(records.get(1).contains("\"option\":0}"), records.get(1));
        assertTrue(records.get(2).contains("\"option\":0}"), records.get(2));
        assertTrue(records.get(3).contains("\"option\":2}"), records.get(3));
    }

    /**
     * A mode that fizzled on its own — no target left when it resolved —
     * still gets its half, with nothing in it, under its own option: the
     * other modes' halves are unaffected (FR-029c).
     */
    @Test
    void aModeThatDidNothingKeepsItsOwnEmptyHalf() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Boros Charm");
        List<AbilitySub> clones = TestBoard.chain(charm, 0, 2);

        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(charm);
            collectors.clauseHandler().invoke(null, resolving(), new Object[]{clones.get(0)});
            collectors.clauseHandler().invoke(null, resolving(), new Object[]{clones.get(1)});
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.KEYWORD_CHANGE));
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        });

        assertEquals(3, records.size(), records.toString());
        assertTrue(records.get(1).contains("\"option\":0}"), records.get(1));
        assertTrue(records.get(1).contains("\"events\":[]"), records.get(1));
        assertTrue(records.get(2).contains("\"option\":2}"), records.get(2));
        assertTrue(records.get(2).contains("keyword_change"), records.get(2));
    }

    /**
     * Spec Story 3 scenario 14: a degraded worker has no clause hook, so no
     * mode ever opens and the one effect half acts through the root line with
     * every chosen mode's events (FR-029e). Disabling the hook is exactly
     * what degraded means here: the bracket is driven without it.
     */
    @Test
    void aDegradedWorkerWritesOneRootHalfWithEveryModesEvents() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Boros Charm");
        TestBoard.chain(charm, 0, 2);

        // The patched collectors are never handed a clause: the bracket alone
        // is driven, as on a checkout without the hook.
        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(charm);
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.DAMAGE_DEALT).param("amount", 4));
            bracket.recordClauseEvent(new EffectEvent(EffectEvent.KEYWORD_CHANGE));
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        });

        assertEquals(2, records.size(), records.toString());
        String effect = records.get(1);
        assertFalse(effect.contains("\"option\""), "the root line acts: " + effect);
        assertTrue(effect.contains("damage_dealt") && effect.contains("keyword_change"), effect);
        assertEquals(field(records.get(0), "link_id"), field(effect, "link_id"));
    }

    /**
     * FR-029f: an interventional fork of a charm writes one effect half per
     * chosen mode, by the live rule. Fiery Confluence chooses three with
     * repeats; on a board with no artifact its destroy mode has no target and
     * is not offered, so the three halves act through modes 0 and 1 only.
     * The fork resolves for real, through the engine's own clause hook, which
     * is why this test installs the patched collectors and skips on a stock
     * checkout, where no clause hook exists to split the modes and the fork
     * writes one root half (FR-029e).
     */
    @Test
    void aForkedCharmWritesOneEffectHalfPerChosenMode() throws IOException {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.onBattlefield(game, opponent, "Grizzly Bears");
        // In hand, where the fork's copier can find it: a card in no zone has
        // no counterpart in the copy.
        SpellAbility charm = TestBoard.charmOf(
                TestBoard.inHand(game, caster, "Fiery Confluence"), caster);
        // A game in play: GameAction.checkStateEffects freezes the stack and
        // returns early on a game that has not started, which leaves the
        // fork's stack frozen and the forced ability parked off it.
        game.setAge(forge.game.GameStage.Play);
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 2, 0, List.of(), 0.0, 0.0);

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 5L, charm, STOCK_FORK);

        assertTrue(fork.wrote(), "the forced resolution observed something");
        List<String> records = fork.records();
        assertEquals(3, records.size(), records.toString());
        ProvenanceKey root = ProvenanceKey.resolve(charm).key();
        for (String record : records) {
            assertTrue(record.contains("\"interventional\":true"), record);
            assertFalse(record.contains("\"link_id\":\""), "a forced half has no partner: " + record);
            assertTrue(record.contains("\"option\":0}") || record.contains("\"option\":1}"),
                    "each half acts through a mode the board offered: " + record);
            assertTrue(record.contains("\"index_within_kind\":" + root.indexWithinKind()), record);
        }
    }

    /**
     * A fork forcing a creature spell resolves the creature's own modal
     * enters trigger too, on the same host card. Those modes belong to the
     * trigger's line, not the spell's: split under the spell's key they would
     * name {@code spell[0]} with an option, a key no sidecar line carries.
     */
    @Test
    void aForkedSpellIsNotSplitByItsOwnTriggersModes() throws IOException {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card barbarian = TestBoard.inHand(game, caster, "Plundering Barbarian");
        SpellAbility creatureSpell = barbarian.getFirstSpellAbility();
        assertNotNull(creatureSpell);
        creatureSpell.setActivatingPlayer(caster);
        game.setAge(forge.game.GameStage.Play);
        CollectionCaps caps = new CollectionCaps(
                2000, 0.1, 2, 0, List.of(), 0.0, 0.0);

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, game, caps, 5L, creatureSpell, STOCK_FORK);

        assertTrue(fork.wrote(), "the forced resolution observed something");
        List<String> records = fork.records();
        assertEquals(1, records.size(), records.toString());
        String record = records.get(0);
        assertTrue(record.contains("\"trait_kind\":\"spell\""), record);
        assertFalse(record.contains("\"option\""), record);
        assertEquals(2, record.split("\"zone_change\"", -1).length - 1,
                "the trigger's Treasure still rides in the spell's half: " + record);
    }

    /** A mode of some other resolution cannot cut the open bracket. */
    @Test
    void anotherCharmsModeDoesNotCutTheOpenBracket() throws Throwable {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        SpellAbility charm = charmOf(game, game.getPlayers().get(0), "Boros Charm");
        SpellAbility other = charmOf(game, game.getPlayers().get(1), "Boros Charm");
        List<AbilitySub> otherClones = TestBoard.chain(other, 1);

        List<String> records = ShardRun.records(tempDir, game, (bracket, collectors) -> {
            bracket.beginBracket(charm);
            collectors.clauseHandler().invoke(null, resolving(), new Object[]{otherClones.get(0)});
            bracket.endBracket(charm.getId(), false);
            bracket.flushHeldResolution();
        });

        assertEquals(2, records.size(), records.toString());
        assertFalse(records.get(1).contains("\"option\""), records.get(1));
    }
}
