package com.pricepredictor.connector.effects;

import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The libraries a {@code ChangeZone} clause searches that Forge does not
 * count: a search whose chooser is not the library's owner ("search target
 * player's library ... exile it", Thada Adel, Cranial Extraction).
 * {@code ChangeZoneEffect} counts a search only when the searching player
 * owns the library ({@code Player.incLibrarySearched}), and runs a trigger
 * no event reports, so the clause is read the way Forge reads it: its
 * fetchers, its chooser and its guards, before it resolves.
 *
 * <p>Read from the clause, so a search no one could carry out is not told
 * apart from one made: the guards Forge checks before it looks are repeated
 * here, and an {@code Optional$} search only counts once a card left the
 * library (a declined one, like a search that found nothing, is
 * indistinguishable afterwards). A clause with several possible choosers,
 * whom Forge asks, is not read.
 */
final class UncountedSearches {

    private UncountedSearches() {
    }

    /** Each library such a clause searches, owner to its size before the clause. */
    static Map<Player, Integer> before(SpellAbility sa, Card host) {
        Map<Player, Integer> libraries = new LinkedHashMap<>();
        Player chooser = soleChooser(sa, host);
        if (chooser == null) {
            return libraries;
        }
        for (Player owner : AbilityUtils.getDefinedPlayers(host, sa.getParam("DefinedPlayer"), sa)) {
            Player library = searched(sa, owner);
            if (isSearchedBy(chooser, library, sa)) {
                libraries.put(library, librarySize(library));
            }
        }
        return libraries;
    }

    /** The library {@code fetcher} names: a target replaces it where the clause has no {@code DefinedPlayer$}. */
    private static Player searched(SpellAbility sa, Player fetcher) {
        return sa.usesTargeting() && !sa.hasParam("DefinedPlayer")
                ? sa.getTargets().getFirstTargetedPlayer() : fetcher;
    }

    /** Whether {@code chooser} searches {@code library}, which is not their own and which they may search. */
    private static boolean isSearchedBy(Player chooser, Player library, SpellAbility sa) {
        return library != null && library.isInGame() && !library.equals(chooser)
                && chooser.canSearchLibraryWith(sa, library);
    }

    /** The owners among {@code before} whose search happened, by what the clause left of their library. */
    static List<Player> happened(SpellAbility sa, Map<Player, Integer> before) {
        return before.entrySet().stream()
                .filter(each -> !sa.hasParam("Optional") || librarySize(each.getKey()) < each.getValue())
                .map(Map.Entry::getKey).toList();
    }

    /** The hidden path of {@code ChangeZone} that looks at a library: no known card, no {@code NoLooking$}. */
    private static boolean searchesALibrary(SpellAbility sa) {
        return sa.isHidden() && sa.hasParam("Origin")
                && ZoneType.listValueOf(sa.getParam("Origin")).contains(ZoneType.Library)
                && !sa.hasParam("Defined") && !sa.hasParam("ChooseFromDefined")
                && !sa.hasParam("NoLooking") && sa.hasParam("Chooser");
    }

    /** The one player {@code Chooser$} names of a library search, else null. */
    private static Player soleChooser(SpellAbility sa, Card host) {
        if (sa.getApi() != ApiType.ChangeZone || !searchesALibrary(sa)) {
            return null;
        }
        List<Player> choosers = AbilityUtils.getDefinedPlayers(host, sa.getParam("Chooser"), sa);
        return choosers.size() == 1 ? choosers.get(0) : null;
    }

    private static int librarySize(Player owner) {
        return owner.getCardsIn(ZoneType.Library).size();
    }
}
