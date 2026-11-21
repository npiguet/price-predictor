package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static com.pricepredictor.connector.effects.RecordAbilities.abilityKeys;
import static com.pricepredictor.connector.effects.RecordAbilities.scriptOf;
import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordEvents.eventTypes;
import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordPicks.records;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A countered ability's fate is the countering clause's effect half, as a
 * countered spell's is (701.6a): an {@code ability_countered} naming the
 * ability as a {@code targets} link ends at it, its host's entity and its
 * line, with no subject, since the host itself was not countered.
 */
@ExtendWith(ForgeExtension.class)
class CounteredAbilityTest {

    private static final String WHY = "to record the countering clause's effect half";

    @TempDir
    Path tempDir;

    /** The ability {@code line} (a rendered link end) names, parsed. */
    private static Object end(String line) {
        return parse("{\"end\":" + line + "}").get("end");
    }

    /** The {@code ability} of each {@code ability_countered} the record holds. */
    private static List<Object> countered(String record) {
        return events(record).stream()
                .filter(event -> "ability_countered".equals(event.get("type")))
                .map(event -> {
                    assertEquals(List.of(), event.get("subjects"), "the host is no subject: " + record);
                    return asMap(event.get("params")).get("ability");
                })
                .toList();
    }

    /** The effect halves acting through a line of {@code traitKind} of {@code script}. */
    private static List<String> effectHalves(List<String> all, String script, String traitKind) {
        return records(all, "resolution").stream()
                .filter(record -> script.equals(scriptOf(record)) && abilityKeys(record).stream()
                        .anyMatch(key -> traitKind.equals(key.get("trait_kind"))))
                .toList();
    }

    /**
     * Azorius Guildmage's counter at Prodigal Sorcerer's ping on the stack:
     * the ping writes no effect half and deals no damage, the Guildmage's
     * half holds an {@code ability_countered} naming the ping's line on the
     * Sorcerer, and no {@code spell_countered}; the Sorcerer stays put.
     */
    @Test
    void anActivatedAbilityCounteredIsNamedOnTheCountersHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card sorcerer = TestBoard.inPlay(game, opponent, "Prodigal Sorcerer");
        SpellAbility ping = TestBoard.ability(sorcerer, ApiType.DealDamage);
        ping.setActivatingPlayer(opponent);
        ping.resetTargets();
        ping.getTargets().add(caster);
        Card guildmage = TestBoard.inPlay(game, caster, "Azorius Guildmage");
        SpellAbility counter = TestBoard.ability(guildmage, ApiType.Counter);
        counter.setActivatingPlayer(caster);

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            game.getStack().add(ping);
            counter.resetTargets();
            counter.getTargets().add(game.getStack().peekAbility());
            game.getStack().add(counter);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertEquals(20, caster.getLife(), "the ping never resolved");
        assertTrue(game.getStack().isEmpty(), "the ping left the stack");
        assertEquals(ZoneType.Battlefield, sorcerer.getZone().getZoneType());
        assertEquals(List.of(), effectHalves(all, "cardsfolder/p/prodigal_sorcerer.txt", "activated"),
                all.toString());
        String countering = only(all, "resolution", "cardsfolder/a/azorius_guildmage.txt");
        assertEquals(List.of(end(SnapshotLinks.line(sorcerer, ping))), countered(countering), countering);
        assertFalse(eventTypes(countering).contains("spell_countered"), countering);
    }

    /**
     * Stifle at Elvish Visionary's enters trigger: the draw never happens,
     * the trigger writes no effect half, and Stifle's half names the
     * trigger's line on the Visionary.
     */
    @Test
    void aTriggeredAbilityCounteredIsNamedOnTheCountersHalf() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        TestBoard.card(game, opponent, "Island");
        Card stifle = TestBoard.inHand(game, caster, "Stifle");
        SpellAbility[] trigger = {null};
        Card[] visionary = {null};
        int[] hand = {0};

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            visionary[0] = TestBoard.castUntargeted(game, opponent, "Elvish Visionary").getHostCard();
            game.getStack().resolveStack();
            TestBoard.priorityWithTriggers(game);
            trigger[0] = game.getStack().peekAbility();
            visionary[0] = game.getCardState(visionary[0]);
            hand[0] = opponent.getCardsIn(ZoneType.Hand).size();
            TestBoard.cast(game, stifle, caster, trigger[0]);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertTrue(trigger[0].isTrigger(), "the enters trigger was on the stack");
        assertEquals(hand[0], opponent.getCardsIn(ZoneType.Hand).size(), "the draw never happened");
        assertEquals(List.of(), effectHalves(all, "cardsfolder/e/elvish_visionary.txt", "trigger"),
                all.toString());
        String countering = only(all, "resolution", "cardsfolder/s/stifle.txt");
        Object named = end(SnapshotLinks.line(visionary[0], trigger[0]));
        assertEquals("trigger", asMap(asMap(named).get("key")).get("trait_kind"), named.toString());
        assertEquals(List.of(named), countered(countering), countering);
    }

    /**
     * Voidslime, which counters a spell or an ability, at a Lightning Bolt:
     * a spell countered is still its {@code spell_countered} and its card's
     * move off the stack, never an {@code ability_countered}.
     */
    @Test
    void aSpellCounteredByAnAbilityCounterIsStillASpellCountered() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card voidslime = TestBoard.inHand(game, opponent, "Voidslime");
        Card[] bolt = {null};

        List<String> all = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility cast = TestBoard.cast(game, TestBoard.inHand(game, caster, "Lightning Bolt"),
                    caster, opponent);
            bolt[0] = cast.getHostCard();
            TestBoard.cast(game, voidslime, opponent, game.getStack().peekAbility());
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        assertEquals(20, opponent.getLife(), "the Bolt never resolved");
        String countering = only(all, "resolution", "cardsfolder/v/voidslime.txt");
        List<Map<String, Object>> spellCountered = events(countering).stream()
                .filter(event -> "spell_countered".equals(event.get("type"))).toList();
        assertEquals(1, spellCountered.size(), countering);
        assertEquals(List.of(Refs.of(bolt[0])), spellCountered.get(0).get("subjects"), countering);
        assertEquals(List.of(), countered(countering), countering);
    }
}
