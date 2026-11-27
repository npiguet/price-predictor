package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.ai.ComputerUtil;
import forge.game.Game;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static com.pricepredictor.connector.effects.RecordEvents.events;
import static com.pricepredictor.connector.effects.RecordEvents.eventsNaming;
import static com.pricepredictor.connector.effects.RecordPicks.only;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A bus {@code attached} names what the attachment now hangs on
 * ({@code attached_to}), a bus {@code unattached} what it left
 * ({@code detached_from}). A move from one host to another
 * is an {@code unattached}, then an {@code attached}: Forge unattaches first
 * ({@code Card.attachToEntity}).
 */
@ExtendWith(ForgeExtension.class)
class AttachTargetsTest {

    private static final String WHY = "for the resolution hook";

    /**
     * The events of Holy Strength's half on Grizzly Bears, in the order a
     * resolution hook sees them: the Aura's ref, then the
     * bears'.
     */
    private static final String AURA_HALF = """
            {"events":[            {"type":"attached","subjects":["%1$s"],"params":{"attached_to":"%2$s","cause":"%1$s"},            "duration":"instant","attributed_to":"root"},            {"type":"pt_change","subjects":["%2$s"],"params":{"power_delta":1,"toughness_delta":2,"cause":"%1$s"},            "duration":"permanent","attributed_to":"root"},            {"type":"zone_change","subjects":["%1$s"],            "params":{"from_zone":"stack","to_zone":"battlefield","cause":"%1$s"},            "duration":"instant","attributed_to":"root"}]}""";

    @TempDir
    Path tempDir;

    /**
     * Bonesplitter moves from Grizzly Bears to Hill Giant: its equip half
     * holds the unattaching from the bears, then the attaching to the giant,
     * and the +2/+0 moving with it, which the static check after the equip's
     * one clause applies and attributes to that clause.
     */
    @Test
    void anEquipMoveLeavesOneHostAndJoinsTheOther() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        Card giant = TestBoard.inPlay(game, caster, "Hill Giant");
        TestBoard.inPlay(game, caster, "Mountain");
        Card splitter = TestBoard.inPlay(game, caster, "Bonesplitter");
        splitter.attachToEntity(bears, null);
        game.getAction().checkStateEffects(false);
        SpellAbility equip = TestBoard.ability(splitter, ApiType.Attach);
        equip.setActivatingPlayer(caster);
        equip.resetTargets();
        equip.getTargets().add(giant);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, equip, null),
                    "the AI activated the equip");
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        assertEquals(giant, splitter.getAttachedTo(), "the splitter moved to the giant");
        String half = only(records, "resolution", "cardsfolder/b/bonesplitter.txt");
        List<Map<String, Object>> attachments = attachmentsIn(half);
        assertEquals(List.of("unattached", "attached"),
                attachments.stream().map(event -> event.get("type")).toList(), half);
        for (Map<String, Object> event : attachments) {
            assertEquals(List.of(Refs.of(splitter)), event.get("subjects"), half);
            assertEquals("root", event.get("attributed_to"), half);
        }
        assertEquals(Refs.of(bears), params(attachments.get(0)).get("detached_from"), half);
        assertEquals(Refs.of(giant), params(attachments.get(1)).get("attached_to"), half);
        assertEquals(Map.of(Refs.of(bears), List.of(-2L, 0L), Refs.of(giant), List.of(2L, 0L)),
                ptChangesIn(half), "the +2/+0 leaves the bears and joins the giant: " + half);
        assertEquals(List.of("root", "root"), events(half).stream()
                .filter(event -> "pt_change".equals(event.get("type")))
                .map(event -> event.get("attributed_to")).toList(), half);
    }

    /**
     * Swiftfoot Boots equipped to Grizzly Bears: the equip half holds the
     * bears gaining hexproof and haste, which the static check after the
     * equip's last clause applies, attributed to that clause,
     * the equip's one.
     */
    @Test
    void anEquipsKeywordGrantIsItsHalfs() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        TestBoard.inPlay(game, caster, "Mountain");
        Card boots = TestBoard.inPlay(game, caster, "Swiftfoot Boots");
        SpellAbility equip = TestBoard.ability(boots, ApiType.Attach);
        equip.setActivatingPlayer(caster);
        equip.resetTargets();
        equip.getTargets().add(bears);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            assertTrue(ComputerUtil.handlePlayingSpellAbility(caster, equip, null),
                    "the AI activated the equip");
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/s/swiftfoot_boots.txt");
        List<Map<String, Object>> granted = eventsNaming(half, Refs.of(bears)).stream()
                .filter(event -> "keyword_change".equals(event.get("type"))).toList();
        assertEquals(1, granted.size(), half);
        assertEquals(Map.of("keywords", List.of("haste", "hexproof"), "removed", false, "cause", Refs.of(boots)),
                params(granted.get(0)), half);
        assertEquals("root", granted.get(0).get("attributed_to"), half);
    }

    /**
     * Holy Strength cast on Grizzly Bears: an Aura spell's move onto the
     * battlefield runs a static check inside its resolution, so its half
     * holds the +1/+2 beside the attachment, in the same events and the same
     * order with or without the resolution hook.
     */
    @Test
    void anAuraSpellsHalfIsUnchanged() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inPlay(game, caster, "Grizzly Bears");
        Card strength = TestBoard.inHand(game, caster, "Holy Strength");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, strength, caster, bears);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/h/holy_strength.txt");
        assertEquals(parse(AURA_HALF.formatted(Refs.of(strength), Refs.of(bears))).get("events"),
                events(half), half);
    }

    /** Each {@code pt_change}'s subject, with its power and toughness deltas. */
    private static Map<String, List<Long>> ptChangesIn(String record) {
        Map<String, List<Long>> changes = new TreeMap<>();
        for (Map<String, Object> event : events(record)) {
            if ("pt_change".equals(event.get("type"))) {
                Map<String, Object> deltas = params(event);
                changes.put(soleSubject((List<?>) event.get("subjects")),
                        List.of((Long) deltas.get("power_delta"), (Long) deltas.get("toughness_delta")));
            }
        }
        return changes;
    }

    private static String soleSubject(List<?> subjects) {
        assertEquals(1, subjects.size(), subjects.toString());
        return (String) subjects.get(0);
    }

    /** Curse of the Pierced Heart enchants a player: {@code attached_to} names the player. */
    @Test
    void aCurseIsAttachedToItsPlayer() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card curse = TestBoard.inHand(game, caster, "Curse of the Pierced Heart");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, curse, caster, opponent);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/c/curse_of_the_pierced_heart.txt");
        List<Map<String, Object>> events = events(half);
        int attached = -1;
        int arrived = -1;
        for (int i = 0; i < events.size(); i++) {
            Map<String, Object> event = events.get(i);
            if ("attached".equals(event.get("type"))) {
                attached = i;
                assertEquals(Refs.of(opponent), params(event).get("attached_to"), half);
            }
            if ("zone_change".equals(event.get("type"))
                    && "battlefield".equals(params(event).get("to_zone"))) {
                arrived = i;
            }
        }
        assertTrue(attached >= 0 && arrived > attached,
                "attached to the player, before its arrival on the battlefield: " + half);
    }

    /**
     * Holy Strength on Savannah Lions, which a Lightning Bolt kills: the
     * Aura's falling off names the creature it left, in the Bolt's held half.
     */
    @Test
    void anAuraFallingOffNamesTheCreatureItLeft() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Player opponent = game.getPlayers().get(1);
        Card lions = TestBoard.inPlay(game, opponent, "Savannah Lions");
        Card strength = TestBoard.inPlay(game, opponent, "Holy Strength");
        strength.attachToEntity(lions, null);
        game.getAction().checkStateEffects(false);
        Card bolt = TestBoard.inHand(game, caster, "Lightning Bolt");

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            TestBoard.cast(game, bolt, caster, lions);
            game.getStack().resolveStack();
            game.getAction().checkStateEffects(false);
            TestBoard.nextStep(game);
        });

        String half = only(records, "resolution", "cardsfolder/l/lightning_bolt.txt");
        List<Map<String, Object>> fell = attachmentsIn(half).stream()
                .filter(event -> "unattached".equals(event.get("type"))).toList();
        assertEquals(1, fell.size(), half);
        assertEquals(List.of(Refs.of(strength)), fell.get(0).get("subjects"), half);
        assertEquals(Refs.of(lions), params(fell.get(0)).get("detached_from"), half);
    }

    private static List<Map<String, Object>> attachmentsIn(String record) {
        return events(record).stream()
                .filter(event -> "attached".equals(event.get("type"))
                        || "unattached".equals(event.get("type"))).toList();
    }
}
