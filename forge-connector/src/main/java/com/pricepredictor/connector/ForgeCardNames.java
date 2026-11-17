package com.pricepredictor.connector;

import forge.StaticData;
import forge.card.CardType;
import forge.card.ICardFace;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

/** What Forge's loaded data says about a card name found in a line's text. */
final class ForgeCardNames {

    private ForgeCardNames() {
    }

    /** Whether {@code name} (any case) is a creature type: Aurochs, Assembly-Worker. */
    static boolean isCreatureType(String name) {
        return CardType.Constant.CREATURE_TYPES.stream().anyMatch(name::equalsIgnoreCase);
    }

    /**
     * The lower-cased names of the cards in Forge's database that hold
     * {@code name} as whole words and are longer ("keeper of kookus" for
     * "kookus"). Empty when no card data is loaded.
     */
    static List<String> longerNamesHolding(String name) {
        StaticData data = StaticData.instance();
        if (data == null) return List.of();
        String lowered = name.toLowerCase(Locale.ROOT);
        Pattern whole = Pattern.compile(
                "(?<![\\p{L}\\p{N}_])" + Pattern.quote(lowered) + "(?![\\p{L}\\p{N}_])");
        List<String> longer = new ArrayList<>();
        for (ICardFace face : data.getCommonCards().getAllFaces()) {
            String other = face.getName().toLowerCase(Locale.ROOT);
            if (other.length() > lowered.length() && whole.matcher(other).find()) {
                longer.add(other);
            }
        }
        return longer;
    }
}
