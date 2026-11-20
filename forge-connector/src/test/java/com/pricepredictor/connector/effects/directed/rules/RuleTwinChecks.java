package com.pricepredictor.connector.effects.directed.rules;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Set;
import java.util.TreeSet;

import static com.pricepredictor.connector.effects.directed.BoardChanges.changes;
import static com.pricepredictor.connector.effects.directed.BoardChanges.id;
import static com.pricepredictor.connector.effects.directed.BoardChanges.renamed;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The checks the {@code RegisteredRules*Test} classes make of a
 * {@link RuleTwin}: the fork differs from the live board by the variant's
 * one removed, added, replaced, swapped or changed card alone.
 */
final class RuleTwinChecks {

    private RuleTwinChecks() {
    }

    // ── shared checks ───────────────────────────────────────────────────

    /**
     * The fork differs from the live board by {@code removed} alone: it is
     * gone, with its links, the battlefield's ranks after it and the mana its
     * controller has untapped.
     */
    static void assertOnlyRemoved(RuleTwin twin, Card removed, String role) {
        String gone = id(removed);
        assertTrue(twin.changes().contains("-" + gone), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + gone) || change.endsWith(".order_rank")
                            || change.endsWith(".untapped_production")
                            || change.startsWith("-link ") && change.contains("=" + gone),
                    "removing changed " + change + "; all: " + twin.changes());
        }
        assertNull(twin.fork().card(role).getZone());
    }

    /**
     * The fork differs from the live board by a new {@code name} under
     * {@code player} alone: it is there, with its links.
     */
    static void assertOnlyAdded(RuleTwin twin, Player player, String name) {
        Card card = player.getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(name)).findFirst().orElseThrow();
        String added = id(card);
        assertTrue(twin.changes().contains("+" + added), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("+" + added) || change.startsWith("+link ") && change.contains("=" + added),
                    "adding changed " + change + "; all: " + twin.changes());
        }
    }

    /**
     * The fork differs from the live board by a {@code name} in
     * {@code replaced}'s place alone: read with the new card's id wherever the
     * old one's stood, what is left differs only in what makes the card that
     * card, and in the {@code alsoChanged} members the variant names besides
     * (a token's {@code token_script_id}, where a card takes its place).
     * Returns the new card.
     */
    static Card assertOnlyReplaced(RuleTwin twin, Card replaced, String name, String... alsoChanged) {
        Card card = twin.forkGame().getCardsIn(ZoneType.Battlefield).stream()
                .filter(c -> c.getName().equals(name) && twin.changes().contains("+" + id(c)))
                .findFirst().orElseThrow(() -> new AssertionError("no new " + name + "; all: " + twin.changes()));
        Set<String> left = changes(renamed(twin.real(), id(replaced), id(card)), twin.forked());
        Set<String> identity = new TreeSet<>(List.of("name", "colors", "types", "subtypes", "mana_value",
                "mana_cost", "base_pt", "lines", "granted_temporary"));
        identity.addAll(List.of(alsoChanged));
        for (String change : left) {
            assertTrue(change.startsWith(id(card) + ".")
                            && identity.contains(change.substring(change.indexOf('.') + 1)),
                    "replacing changed " + change + "; all: " + left);
        }
        return card;
    }

    /**
     * The fork differs from the live board by {@code fresh} in {@code old}'s
     * place in a hand or library alone, with their links.
     */
    static void assertOnlySwapped(RuleTwin twin, Card old, Card fresh) {
        String gone = id(old);
        String added = id(fresh);
        assertTrue(twin.changes().containsAll(Set.of("-" + gone, "+" + added)), twin.changes().toString());
        for (String change : twin.changes()) {
            assertTrue(change.equals("-" + gone) || change.equals("+" + added)
                            || change.startsWith("-link ") && change.contains("=" + gone)
                            || change.startsWith("+link ") && change.contains("=" + added),
                    "replacing changed " + change + "; all: " + twin.changes());
        }
    }

    /**
     * The fork differs from the live board by {@code gone} leaving the
     * snapshot and {@code added} joining it alone (the same card for a move),
     * with their links and own members, and the player members named.
     */
    static void assertOnlyChanged(RuleTwin twin, Card gone, Card added, String... members) {
        Set<String> ids = new TreeSet<>(List.of(id(gone), id(added)));
        java.util.function.Predicate<String> own = change -> ids.stream().anyMatch(
                card -> change.equals("-" + card) || change.equals("+" + card)
                        || change.startsWith(card + ".")
                        || change.startsWith("-link ") && change.contains("=" + card)
                        || change.startsWith("+link ") && change.contains("=" + card));
        // Something of the edited cards changed: an edit that did nothing
        // passes every "only" check below.
        assertTrue(twin.changes().stream().anyMatch(own),
                "the edit changed nothing of " + ids + "; all: " + twin.changes());
        for (String change : twin.changes()) {
            assertTrue(own.test(change) || List.of(members).contains(change),
                    "changed " + change + "; all: " + twin.changes());
        }
    }
}
