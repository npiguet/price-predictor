package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardEdition;
import forge.card.CardRules;
import forge.card.CardType;
import forge.card.ICardFace;
import forge.game.keyword.Keyword;
import forge.token.TokenDb;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/** What Forge's loaded data says about a card name found in a line's text. */
final class ForgeCardNames {

    /**
     * The keyword actions of the Comprehensive Rules, 701.2 to 701.71
     * (Exile, Regenerate, Manifest Dread, ...), lower-cased. Forge keeps no
     * list of them: a card script writes one as a verb.
     */
    private static final Set<String> KEYWORD_ACTIONS = Set.of(
            "activate", "attach", "behold", "cast", "counter", "create", "destroy", "discard",
            "double", "triple", "exchange", "exile", "fight", "goad", "investigate", "mill",
            "play", "regenerate", "reveal", "sacrifice", "scry", "search", "shuffle", "surveil",
            "tap", "untap", "transform", "convert", "fateseal", "clash", "planeswalk",
            "set in motion", "abandon", "proliferate", "detain", "populate", "monstrosity",
            "vote", "bolster", "manifest", "support", "meld", "exert", "explore", "assemble",
            "adapt", "amass", "learn", "venture into the dungeon", "connive",
            "open an attraction", "roll to visit your attractions", "incubate",
            "the ring tempts you", "face a villainous choice", "time travel", "discover",
            "cloak", "collect evidence", "suspect", "forage", "manifest dread", "endure",
            "harness", "airbend", "earthbend", "waterbend", "blight", "heal", "recruit",
            "empower jace");

    private ForgeCardNames() {
    }

    /**
     * Whether {@code name} (any case) is a word Forge's scripts write in title
     * case as something other than a card: a creature type (Aurochs,
     * Assembly-Worker), a keyword (Fear), a keyword action (Exile) or a set's
     * name (War of the Spark).
     */
    static boolean namesSomethingElse(String name) {
        return isCreatureType(name) || Keyword.smartValueOf(name) != Keyword.UNDEFINED
                || KEYWORD_ACTIONS.contains(name.toLowerCase(Locale.ROOT)) || isSetName(name);
    }

    private static boolean isCreatureType(String name) {
        return CardType.Constant.CREATURE_TYPES.stream().anyMatch(name::equalsIgnoreCase);
    }

    /** Whether {@code name} (any case) is a set's name; false when no card data is loaded. */
    private static boolean isSetName(String name) {
        StaticData data = StaticData.instance();
        if (data == null) return false;
        for (CardEdition edition : data.getEditions()) {
            if (name.equalsIgnoreCase(edition.getName())) return true;
        }
        return false;
    }

    /**
     * The lower-cased names of the cards and tokens in Forge's database that
     * hold {@code name} as whole words and are longer ("keeper of kookus" for
     * "kookus", "stangg twin" for "stangg"). Empty when no card data is loaded.
     */
    static List<String> longerNamesHolding(String name) {
        StaticData data = StaticData.instance();
        if (data == null) return List.of();
        String lowered = name.toLowerCase(Locale.ROOT);
        Pattern whole = Pattern.compile(
                "(?<![\\p{L}\\p{N}_])" + Pattern.quote(lowered) + "(?![\\p{L}\\p{N}_])");
        List<String> longer = new ArrayList<>();
        for (String other : cardAndTokenNames(data)) {
            if (other.length() > lowered.length() && whole.matcher(other).find()) {
                longer.add(other);
            }
        }
        return longer;
    }

    /** The lower-cased name of every card face and token Forge has loaded. */
    private static List<String> cardAndTokenNames(StaticData data) {
        List<String> names = new ArrayList<>();
        for (ICardFace face : data.getCommonCards().getAllFaces()) {
            names.add(face.getName().toLowerCase(Locale.ROOT));
        }
        TokenDb tokens = data.getAllTokens();
        if (tokens == null) return names;
        for (CardRules token : tokens.getRules().values()) {
            names.add(token.getName().toLowerCase(Locale.ROOT));
        }
        return names;
    }
}
