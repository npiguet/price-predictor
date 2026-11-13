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

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A bus {@code attached} names what the attachment now hangs on
 * ({@code attached_to}), a bus {@code unattached} what it left
 * ({@code detached_from}) (C1 #5, V2 #8). A move from one host to another
 * is an {@code unattached}, then an {@code attached}: Forge unattaches first
 * ({@code Card.attachToEntity}).
 */
@ExtendWith(ForgeExtension.class)
class AttachTargetsTest {

    private static final String WHY = "for the resolution hook";

    @TempDir
    Path tempDir;

    /**
     * Bonesplitter moves from Grizzly Bears to Hill Giant: its equip half
     * holds the unattaching from the bears, then the attaching to the giant.
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
