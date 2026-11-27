package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.AbilityUtils;
import forge.game.card.Card;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

/**
 * A library search resolved by the real engine is one {@code card_looked_at}
 * named {@code SearchedLibrary}, whoever's library it looks into: Forge
 * counts only a search of the searcher's own library, and the rule reads the
 * rest from the clause. The clauses are scripted in the shape of
 * the cards they stand for (Thada Adel's needs a combat-damage trigger's
 * context for its {@code TriggeredTarget}, which a bare clause cannot have).
 */
@ExtendWith(ForgeExtension.class)
class LibrarySearchEventTest {

    private static final String OPPONENTS_ARTIFACT = "SP$ ChangeZone | Origin$ Library | Destination$ Exile"
            + " | DefinedPlayer$ Player.Opponent | Chooser$ You | ChangeType$ Artifact | ChangeNum$ 1";

    private static void stock(Game game, Player owner, String... names) {
        for (String name : names) {
            game.getAction().moveTo(ZoneType.Library, TestBoard.card(game, owner, name), null, null);
        }
    }

    /** The clause, resolved by the engine between the rule's two readings. */
    private static List<EffectEvent> resolved(String script, Card host, Player activator) {
        SpellAbility sa = AbilityFactory.getAbility(script, host);
        sa.setActivatingPlayer(activator);
        ApiEvents.ClauseObservation<?> observed = ApiEvents.observe(sa, ApiEvents::ruleFor);
        AbilityUtils.resolve(sa);
        return observed.finish();
    }

    private static void assertOneSearchOf(List<EffectEvent> events, Player library) {
        assertEquals(1, events.size(), events.toString());
        EffectEvent event = events.get(0);
        assertEquals(EventType.CARD_LOOKED_AT, event.type());
        assertEquals(List.of(Refs.of(library)), event.subjects());
        assertEquals("SearchedLibrary", event.param(EventParam.MODE));
        assertEquals("library", event.param(EventParam.FROM_ZONE));
    }

    /** Thada Adel's shape: the opponent's library, searched by its attacker. */
    @Test
    void aSearchOfAnotherPlayersLibraryIsALookAtThatLibrary() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card host = TestBoard.inPlay(game, a, "Thada Adel, Acquisitor");
        stock(game, b, "Sol Ring", "Forest");

        assertOneSearchOf(resolved(OPPONENTS_ARTIFACT, host, a), b);
    }

    /** A search of one's own library is still one look, not two. */
    @Test
    void aSearchOfOneOwnLibraryIsStillOneLook() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        stock(game, a, "Sol Ring", "Forest");

        assertOneSearchOf(resolved(
                "SP$ ChangeZone | Origin$ Library | Destination$ Hand | ChangeType$ Card", host, a), a);
    }

    /** Each player searching their own library with no chooser is counted by Forge: one look each, none doubled. */
    @Test
    void everyPlayerSearchingTheirOwnLibraryIsOneLookEach() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Player b = game.getPlayers().get(1);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        stock(game, a, "Forest");
        stock(game, b, "Forest");

        List<EffectEvent> events = resolved(
                "SP$ ChangeZone | Origin$ Library | Destination$ Hand | DefinedPlayer$ Player"
                        + " | ChangeType$ Card", host, a);

        assertEquals(1, events.size(), events.toString());
        assertNotNull(events.get(0));
        assertEquals(List.of(Refs.of(a), Refs.of(b)), events.get(0).subjects());
    }

    /** A clause that only looks at a known zone searches nothing. */
    @Test
    void aMoveFromTheGraveyardIsNoSearch() {
        Game game = TestBoard.twoPlayerGame(PhaseType.MAIN1);
        Player a = game.getPlayers().get(0);
        Card host = TestBoard.inHand(game, a, "Grizzly Bears");
        TestBoard.inGraveyard(game, game.getPlayers().get(1), "Forest");

        List<EffectEvent> events = resolved(
                "SP$ ChangeZone | Origin$ Graveyard | Destination$ Exile | DefinedPlayer$ Player.Opponent"
                        + " | Chooser$ You | ChangeType$ Card", host, a);

        assertNull(events.isEmpty() ? null : events.get(0), events.toString());
    }
}
