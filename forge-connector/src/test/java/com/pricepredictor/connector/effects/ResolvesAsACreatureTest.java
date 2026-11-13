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
import java.util.function.Predicate;

import static com.pricepredictor.connector.effects.RecordLines.events;
import static com.pricepredictor.connector.effects.RecordLines.only;
import static com.pricepredictor.connector.effects.RecordLines.outcomeOf;
import static com.pricepredictor.connector.effects.RecordLines.params;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * A bestowed Aura spell or a mutating creature spell whose target is gone by
 * its resolution is no Aura or mutating spell any more and resolves as a
 * creature spell (702.103e, 702.140b): Forge reports it fizzled, and its
 * permanent still enters. Its effect half is a resolution's, with the
 * permanent's arrival in it.
 */
@ExtendWith(ForgeExtension.class)
class ResolvesAsACreatureTest {

    private static final String NAIAD = "cardsfolder/n/nimbus_naiad.txt";
    private static final String GEMRAZER = "cardsfolder/g/gemrazer.txt";

    @TempDir
    Path tempDir;

    @Test
    void aBestowSpellWhoseTargetIsGoneEntersAsACreature() throws Throwable {
        assertEntersAsACreature("Nimbus Naiad", NAIAD, SpellAbility::isBestow);
    }

    @Test
    void aMutatingSpellWhoseTargetIsGoneEntersAsACreature() throws Throwable {
        assertEntersAsACreature("Gemrazer", GEMRAZER, SpellAbility::isMutate);
    }

    /**
     * A permanent spell's own arrival is its root clause's: Forge resolves a
     * creature spell through its {@code PermanentCreature} clause, with the
     * resolving-clause pointer set (C1 #6, V2 #9).
     */
    @Test
    void aPermanentSpellsArrivalIsItsRootClauses() throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card bears = TestBoard.inHand(game, caster, "Grizzly Bears");

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the clause hook", () -> {
                    EffectEndingTheTurnTest.castWithoutTarget(game, bears, caster);
                    game.getStack().resolveStack();
                    game.getAction().checkStateEffects(false);
                    TestBoard.nextStep(game);
                });

        String half = only(records, "resolution", "cardsfolder/g/grizzly_bears.txt");
        List<java.util.Map<String, Object>> arrivals = events(half).stream()
                .filter(event -> "zone_change".equals(event.get("type"))
                        && "battlefield".equals(params(event).get("to_zone"))).toList();
        assertEquals(1, arrivals.size(), half);
        assertEquals("root", arrivals.get(0).get("attributed_to"), half);
    }

    private void assertEntersAsACreature(String name, String script, Predicate<SpellAbility> kind)
            throws Throwable {
        Game game = TestBoard.startedTwoPlayerGame(PhaseType.MAIN1);
        Player caster = game.getPlayers().get(0);
        Card target = TestBoard.inPlay(game, caster, "Grizzly Bears");
        Card card = TestBoard.inHand(game, caster, name);

        List<String> records = ShardRun.recordsWithClauseHook(
                tempDir, game, "for the resolution hook", () -> {
                    SpellAbility chosen = spellOfKind(card, kind);
                    card.setSplitStateToPlayAbility(chosen);
                    Card onStack = game.getAction().moveToStack(card, chosen);
                    SpellAbility spell = spellOfKind(onStack, kind);
                    spell.setActivatingPlayer(caster);
                    spell.resetTargets();
                    spell.getTargets().add(target);
                    game.getStack().add(spell);
                    // The target leaves before the spell resolves.
                    game.getAction().exile(target, null, null);
                    game.getStack().resolveStack();
                    game.getAction().checkStateEffects(false);
                    TestBoard.nextStep(game);
                });

        assertTrue(game.getCardsIn(ZoneType.Battlefield).stream()
                .anyMatch(c -> c.getId() == card.getId()), name + " entered the battlefield");
        String half = only(records, "resolution", script);
        assertEquals("resolved", outcomeOf(half), half);
        assertTrue(events(half).stream().anyMatch(event -> "zone_change".equals(event.get("type"))
                        && "stack".equals(params(event).get("from_zone"))
                        && "battlefield".equals(params(event).get("to_zone"))),
                "its permanent's arrival: " + half);
    }

    private static SpellAbility spellOfKind(Card card, Predicate<SpellAbility> kind) {
        for (SpellAbility sa : card.getAllSpellAbilities()) {
            if (sa.isSpell() && kind.test(sa)) {
                return sa;
            }
        }
        assertNotNull(null, "no such spell on " + card);
        return null;
    }
}
