package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.player.Player;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Where every library card sat when a snapshot was captured, for the library
 * cards the record's events name (FR-024, {@code in.library_cards}).
 *
 * <p>A snapshot is taken before its events, so which library cards they will
 * name — the card drawn, the cards milled, revealed or searched out — is
 * known only when the record is written. Rendering every library card into
 * every snapshot would multiply a record's size for the few it needs, so the
 * capture keeps only each card and its position, and a card is rendered when
 * an event names it.
 *
 * <p>Rendered as the card was in the library (FR-042e, root § 8.14): a fresh
 * object built from the card's printing under the same id, as Forge builds a
 * card leaving the library for any zone but the battlefield. So it reads
 * untapped, with no counters, attached to nothing, in no combat, controlled
 * by its owner, face up, not entered this turn and with nothing chosen, its
 * name, types and lines its library face's — whatever the record's events
 * then did to it. Forge keeps the library's own object for a move onto the
 * battlefield, so reading that object at the write would describe it after
 * the move: tapped, sick, face down for a manifest, a copy for a clone. A
 * card with no printing to build from (a token, a copy) is read as it is.
 * Its zone and rank are written as captured.
 *
 * @param present the ids already among the snapshot's entities
 */
record LibraryCapture(
        Game game, StaticSuppression board, Map<String, Slot> byId, Set<String> present) {

    /** One library card and its position, 0 the top. */
    record Slot(Card card, int position) {
    }

    /** Every card of every library as it sits now. */
    static LibraryCapture of(Game game, StaticSuppression board, Set<String> present) {
        Map<String, Slot> byId = new LinkedHashMap<>();
        for (Player player : game.getPlayers()) {
            int position = 0;
            for (Card card : player.getCardsIn(ZoneType.Library)) {
                byId.put(Refs.of(card), new Slot(card, position++));
            }
        }
        return new LibraryCapture(game, board, byId, present);
    }

    /**
     * The captured library cards {@code events} name and the snapshot does
     * not already hold, in the order first named.
     */
    List<Slot> named(Iterable<EffectEvent> events) {
        List<Slot> named = new ArrayList<>();
        Set<String> added = new java.util.HashSet<>(present);
        for (EffectEvent event : events) {
            for (String subject : event.subjects()) {
                Slot slot = byId.get(subject);
                if (slot != null && added.add(subject)) {
                    named.add(slot);
                }
            }
        }
        return named;
    }

    /** The entities of {@code slots}, each where it sat. */
    List<String> render(List<Slot> slots) {
        List<String> rendered = new ArrayList<>();
        for (Slot slot : slots) {
            rendered.add(EntityJson.render(game, asInLibrary(slot.card()), board,
                    new EntityJson.Beyond(null, "library", slot.position(),
                            EntityRole.LIBRARY_CARD)));
        }
        return rendered;
    }

    /**
     * The card as it was in the library: a fresh object from its printing,
     * under its own id and owner, or the card itself where it has no
     * printing to build from.
     */
    static Card asInLibrary(Card card) {
        if (hasNoPrintingToBuild(card)) {
            return card;
        }
        Map<Integer, Card> built = builtFor(card.getGame());
        Card cached = cachedBuild(built, card);
        if (cached != null) {
            return cached;
        }
        try {
            Card fresh = CardFactory.getCard(
                    card.getPaperCard(), card.getOwner(), card.getId(), card.getGame());
            // Not entered this turn: a new object takes the current turn as
            // the one it came into its zone, and a library card is in no
            // zone a turn could name.
            fresh.setTurnInZone(-1);
            synchronized (built) {
                built.put(card.getId(), fresh);
            }
            return fresh;
        } catch (RuntimeException e) {
            return card;
        }
    }

    /** Whether the card is a token, a copy or has no printing: read as it is. */
    private static boolean hasNoPrintingToBuild(Card card) {
        return card.getPaperCard() == null || isCopyOrToken(card);
    }

    private static boolean isCopyOrToken(Card card) {
        return card.isToken() || card.isCopiedSpell() || card.getCopiedPermanent() != null;
    }

    private static Map<Integer, Card> builtFor(Game game) {
        synchronized (BUILT) {
            return BUILT.computeIfAbsent(game, g -> new java.util.HashMap<>());
        }
    }

    /** The build of the card kept for its game, or null when none matches its printing and owner. */
    private static Card cachedBuild(Map<Integer, Card> built, Card card) {
        synchronized (built) {
            Card cached = built.get(card.getId());
            boolean same = cached != null && cached.getPaperCard() == card.getPaperCard()
                    && cached.getOwner() == card.getOwner();
            return same ? cached : null;
        }
    }

    /**
     * Each game's library cards as built from their printings, by card id.
     *
     * <p>Built once per card rather than once per snapshot: parsing a card
     * advances Forge's global trait id counters, so a build per snapshot —
     * every library's top card, in every record — renumbered every trait
     * the game created afterwards, and cost a script parse each time. The
     * object is never put in a zone, so nothing changes it.
     *
     * <p>Weak keys alone never let a game go: each built card holds its game
     * and its owner, so the entry's value kept its own key reachable, and
     * every game and fork a worker recorded stayed in its heap until the
     * JVM filled it (games slowing tenfold after some thirty, then an
     * {@code OutOfMemoryError}). An entry is dropped when its game's
     * recording closes and when its fork ends ({@link #release}).
     */
    private static final Map<Game, Map<Integer, Card>> BUILT = new java.util.WeakHashMap<>();

    /** Forget {@code game}'s built cards: its recording has closed, or its fork has ended. */
    static void release(Game game) {
        if (game == null) {
            return;
        }
        synchronized (BUILT) {
            BUILT.remove(game);
        }
    }

    /** Whether cards of {@code game} are held, for a test. */
    static boolean holds(Game game) {
        synchronized (BUILT) {
            return BUILT.containsKey(game);
        }
    }

    /** How many games' cards are held, for a test. */
    static int gamesHeld() {
        synchronized (BUILT) {
            return BUILT.size();
        }
    }
}
