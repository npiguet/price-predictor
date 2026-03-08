package com.pricepredictor.connector;

import lombok.experimental.UtilityClass;

/**
 * Formats ConvertedCard and MultiCard into text output.
 */
@UtilityClass
public class OutputFormatter {

    /**
     * Format a single card face.
     */
    public String formatCard(ConvertedCard card) {
        StringBuilder sb = new StringBuilder();
        sb.append("name: ").append(card.name());
        if (card.manaCost() != null) {
            sb.append('\n').append("mana cost: ").append(card.manaCost());
        }
        sb.append('\n').append("types: ").append(card.types());
        if (card.powerToughness() != null) {
            sb.append('\n').append("power toughness: ").append(card.powerToughness());
        }
        if (card.loyalty() != null) {
            sb.append('\n').append("loyalty: ").append(card.loyalty());
        }
        if (card.defense() != null) {
            sb.append('\n').append("defense: ").append(card.defense());
        }
        if (card.colors() != null) {
            sb.append('\n').append("colors: ").append(card.colors());
        }
        if (card.text() != null) {
            sb.append('\n').append("text: ").append(card.text());
        }
        for (AbilityLine ability : card.abilities()) {
            sb.append('\n').append(ability.formatLine());
        }
        return sb.toString();
    }

    /**
     * Format a complete card (single or multi-face).
     */
    public String formatMultiCard(MultiCard card) {
        if (card.layout() == null) {
            // Single-face card — no layout line
            return formatCard(card.faces().get(0));
        }

        StringBuilder sb = new StringBuilder();
        sb.append("layout: ").append(card.layout());
        for (int i = 0; i < card.faces().size(); i++) {
            sb.append('\n');
            if (i > 0) {
                sb.append("\nALTERNATE\n\n");
            }
            sb.append(formatCard(card.faces().get(i)));
        }
        return sb.toString();
    }
}
