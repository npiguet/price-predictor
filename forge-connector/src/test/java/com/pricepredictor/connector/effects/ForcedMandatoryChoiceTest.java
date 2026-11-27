package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A mandatory choice inside a forced resolution is made.
 *
 * <p>Forge's {@code AttachAi} answers the choice of what an Aura or an
 * Equipment attaches to with nothing when it finds the attachment useless,
 * whether or not the choice is optional. The effect then does nothing, and an
 * Aura spell stays in the stack zone: forced Phantasmal Terrain on an Island
 * resolved with no event, its {@code ChangeType} logic passing over a land
 * that is already an Island. The forced actor answers such a choice with an
 * option, and a forced Aura spell left in the stack zone is refused.
 */
@ExtendWith(ForgeExtension.class)
class ForcedMandatoryChoiceTest {

    @TempDir
    Path tempDir;

    @Test
    void forcedPhantasmalTerrainOnAnIslandAttaches() throws Exception {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        TestBoard.inPlay(game, game.getPlayers().get(1), "Island");
        SpellAbility terrain = TestBoard.spellOf(TestBoard.inHand(game, holder, "Phantasmal Terrain"), holder);

        ShardRun.Fork fork = ShardRun.forkedWithClauseHook(
                tempDir, new ShardRun.ForkRun(game, TestCaps.caps().interventions(1).probes(0).build(), 1L, terrain), "to record the forced Aura's resolution");

        assertTrue(fork.wrote(), "the forced Aura is written: " + ForkRefusals.counts());
        List<String> records = ShardRun.readShard(fork.shard());
        assertEquals(1, records.size(), records.toString());
        assertTrue(RecordEvents.eventTypes(records.get(0)).contains(EventType.ATTACHED.wireName()),
                "the Aura attaches to the Island: " + records.get(0));
    }

    /** The belt behind the forced actor's answer: an Aura spell the stack let go of while it stayed in the stack zone. */
    @Test
    void anAuraSpellStillInTheStackZoneIsStranded() {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player holder = game.getPlayers().get(0);
        Card pacifism = TestBoard.inHand(game, holder, "Pacifism");
        SpellAbility spell = TestBoard.spellOf(pacifism, holder);

        assertFalse(ForcedResolution.strandedAura(game, spell), "an Aura in hand is not stranded");
        game.getAction().moveTo(ZoneType.Stack, pacifism, null, null);
        assertTrue(ForcedResolution.strandedAura(game, spell), "an Aura left in the stack zone is");
        game.getAction().moveTo(ZoneType.Graveyard, game.getCardState(pacifism), null, null);
        assertFalse(ForcedResolution.strandedAura(game, spell), "an Aura that fizzled is not");
        SpellAbility bearsCast = TestBoard.spellOf(TestBoard.inHand(game, holder, "Grizzly Bears"), holder);
        game.getAction().moveTo(ZoneType.Stack, bearsCast.getHostCard(), null, null);
        assertFalse(ForcedResolution.strandedAura(game, bearsCast), "nor is any spell but an Aura");
    }
}
