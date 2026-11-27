package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityKey;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardZoneTable;
import forge.game.combat.Combat;
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

import static com.pricepredictor.connector.effects.RecordLines.asMap;
import static com.pricepredictor.connector.effects.RecordLines.parse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What a snapshot writes for an object it names says what that object is:
 * a token that has ceased to exist is no graveyard card, a
 * library card the record's events name has its owner and controller like
 * every other card entity, and an attacker's defender is a
 * ref, as every other object a record names.
 */
@ExtendWith(ForgeExtension.class)
class SnapshotNamedObjectsTest {

    private static final String WHY = "for the effect half read when the resolution starts";

    @TempDir
    Path tempDir;

    private static List<Map<String, Object>> listOf(Map<String, Object> state, String member) {
        return ((List<?>) state.get(member)).stream().map(RecordLines::asMap).toList();
    }

    private static Map<String, Object> entityIn(Map<String, Object> state, String id) {
        return listOf(state, "entities").stream().filter(entity -> id.equals(entity.get("id")))
                .findFirst().orElseThrow(() -> new AssertionError("no entity " + id + ": " + state));
    }

    private static List<Map<String, Object>> linksOf(Map<String, Object> state) {
        return listOf(state, "links");
    }

    private static boolean hasLink(Map<String, Object> state, String type, String from, String to) {
        return linksOf(state).stream().anyMatch(link -> type.equals(link.get("type"))
                && from.equals(link.get("from")) && to.equals(link.get("to")));
    }

    /** Sacrifice {@code card} as an effect would, with the last state the move reads. */
    private static void sacrifice(Game game, Card card) {
        Map<AbilityKey, Object> params = AbilityKey.newMap();
        AbilityKey.addCardZoneTableParams(params, new CardZoneTable(
                game.copyLastStateBattlefield(), game.copyLastStateGraveyard()));
        game.getAction().sacrifice(new CardCollection(card), null, true, params);
    }

    // ── a token that ceased to exist ────────────────────────────────────

    /**
     * A Food whose ability is on the stack is sacrificed, as its own cost
     * sacrifices it: the effect half's board holds the token, which no zone
     * holds any more, as it last existed, role {@code last_known}, with no
     * zone and no {@code order_rank}, under the {@code E<id>} that
     * {@code refs.source} and its line name; its {@code source_of} link and
     * its owner's links are written, as for any card entity.
     */
    @Test
    void aSacrificedFoodsAbilityListsTheTokenAsLastKnownAndZoneless() throws Throwable {
        CharacterizationBoard board = CharacterizationBoard.on(
                TestBoard.startedTwoPlayerGame(PhaseType.MAIN1));
        Game game = board.game;
        Card food = board.token("c_a_food_sac", board.you);
        String ref = Refs.of(food);

        List<String> records = ShardRun.recordsWithClauseHook(tempDir, game, WHY, () -> {
            SpellAbility gain = TestBoard.ability(food, ApiType.GainLife);
            gain.setActivatingPlayer(board.you);
            game.getStack().add(gain);
            sacrifice(game, food);
            game.getStack().resolveStack();
            TestBoard.priority(game);
            TestBoard.nextStep(game);
        });

        String half = RecordPicks.only(records, "resolution", "tokenscripts/c_a_food_sac.txt");
        Map<String, Object> state = asMap(parse(half).get("state"));
        Map<String, Object> token = entityIn(state, ref);
        assertEquals("last_known", token.get("role"), String.valueOf(state));
        assertNull(token.get("zone"), String.valueOf(state));
        assertNull(token.get("order_rank"), String.valueOf(state));
        assertEquals(ref, asMap(state.get("refs")).get("source"), state.toString());
        assertTrue(linksOf(state).stream().anyMatch(link -> "source_of".equals(link.get("type"))
                && ref.equals(link.get("from"))), "the token hosts its ability: " + state);
        assertTrue(hasLink(state, "owns", Refs.of(board.you), ref), String.valueOf(state));
        assertTrue(hasLink(state, "controls", Refs.of(board.you), ref), String.valueOf(state));
    }

    // ── a library card the events name ──────────────────────────────────

    /**
     * A card milled from below the top joins the board as it sat in the
     * library, with its owner's {@code owns} and {@code controls} links, as
     * the library's top card has them.
     */
    @Test
    void aMilledCardTheEventsNameHasItsOwnersLinks() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.MAIN1);
        Card top = board.game.getAction().moveToLibrary(board.card("Forest", board.you), null, null);
        Card milled = board.game.getAction().moveToLibrary(
                board.card("Grizzly Bears", board.you), -1, null, null);
        Snapshot before = new SnapshotBuilder(board.game).ofBoard();
        board.game.getAction().moveTo(ZoneType.Graveyard, milled, null, null);

        Map<String, Object> state = parse(before.withEventCards(List.of(
                new EffectEvent(EventType.ZONE_CHANGE).subject(Refs.of(milled)))).render());

        String player = Refs.of(board.you);
        assertEquals("library_card", entityIn(state, Refs.of(milled)).get("role"), String.valueOf(state));
        assertTrue(hasLink(state, "owns", player, Refs.of(milled)), String.valueOf(state));
        assertTrue(hasLink(state, "controls", player, Refs.of(milled)), String.valueOf(state));
        assertTrue(hasLink(state, "owns", player, Refs.of(top)), "the top card's, as before: " + state);
        List<Object> types = linksOf(state).stream().map(link -> link.get("type")).toList();
        assertEquals(types.lastIndexOf("controls") + 1, types.indexOf("owns"),
                "the links stay listed by family, controls before owns: " + types);
    }

    // ── an attacker's defender ──────────────────────────────────────────

    /**
     * {@code combat.attacking} is the defender's ref: the player's
     * {@code P<id>}, a planeswalker's {@code E<id>}, as the {@code attacks}
     * link names it; never Forge's rendering of the object.
     */
    @Test
    void anAttackersDefenderIsItsRef() {
        CharacterizationBoard board = CharacterizationBoard.empty(PhaseType.COMBAT_DECLARE_BLOCKERS);
        Card bears = board.battlefield("Grizzly Bears", board.you);
        Card giant = board.battlefield("Hill Giant", board.you);
        Card jace = board.battlefield("Jace Beleren", board.opponent);
        Combat combat = new Combat(board.you);
        combat.addAttacker(bears, board.opponent);
        combat.addAttacker(giant, jace);
        board.game.getPhaseHandler().setCombat(combat);

        Map<String, Object> state = parse(new SnapshotBuilder(board.game).ofBoard().render());

        assertEquals(Refs.of(board.opponent), combatOf(state, bears).get("attacking"), String.valueOf(state));
        assertEquals(Refs.of(jace), combatOf(state, giant).get("attacking"), String.valueOf(state));
        assertTrue(hasLink(state, "attacks", Refs.of(giant), Refs.of(jace)), String.valueOf(state));
    }

    private static Map<String, Object> combatOf(Map<String, Object> state, Card card) {
        return asMap(entityIn(state, Refs.of(card)).get("combat"));
    }
}
