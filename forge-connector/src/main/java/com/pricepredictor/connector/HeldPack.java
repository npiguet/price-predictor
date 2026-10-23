package com.pricepredictor.connector;

import forge.gamemodes.limited.DraftPack;
import forge.item.PaperCard;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The cards of the pack a seat holds at a pick, by name: what a pick request
 * lists and what a supervisor's chosen name is looked up in.
 *
 * <p>Names, because that is all the protocol and the transcript carry; a
 * pack holding two copies of a card gives the first, which is the same card.
 */
record HeldPack(DraftPack pack) {

    /** The pack's card names, in pack order, duplicates kept. */
    List<String> names() {
        List<String> names = new ArrayList<>(pack.size());
        for (int i = 0; i < pack.size(); i++) {
            names.add(pack.get(i).getName());
        }
        return names;
    }

    /** The first card in the pack named {@code name}, or empty when the pack holds none. */
    Optional<PaperCard> card(String name) {
        for (int i = 0; i < pack.size(); i++) {
            if (pack.get(i).getName().equals(name)) {
                return Optional.of(pack.get(i));
            }
        }
        return Optional.empty();
    }
}
