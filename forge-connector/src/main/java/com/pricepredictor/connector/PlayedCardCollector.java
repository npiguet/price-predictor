package com.pricepredictor.connector;

import com.google.common.eventbus.Subscribe;
import forge.game.card.Card;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.IGameEventVisitor;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Per-game eventbus visitor that records every non-basic, non-token card
 * played by each side. Mirrors the working pattern in
 * {@code ../jumpstart-tierlist/.../JumpstartMatch.java#CardCollector}
 * (research.md Decision D-3).
 *
 * <p>Listens to two events:
 * <ol>
 *   <li>{@link GameEventCardChangeZone} filtered to
 *       {@code ZoneType.Battlefield} — catches permanents that resolved.</li>
 *   <li>{@link GameEventSpellAbilityCast} — catches every cast spell,
 *       including instants/sorceries that never reach the battlefield.</li>
 * </ol>
 *
 * <p>Filters applied at observation time:
 * <ul>
 *   <li>{@code card.getController() == card.getOwner()} (drops stolen cards).</li>
 *   <li>{@code !card.isToken()}.</li>
 *   <li>{@code !card.getType().isBasicLand()} (FR-004a).</li>
 *   <li>The recorded name is {@code card.getPaperCard().getName()} so
 *       copy/clone effects credit the cast card, not the copied permanent.</li>
 * </ul>
 *
 * <p>Cards are bucketed by {@link Card#getOwner()}'s lobby name, then mapped
 * to {@code 'A'} / {@code 'B'} via {@code GamePlayer}'s
 * {@code LOBBY_NAME_A} / {@code LOBBY_NAME_B} constants.
 *
 * <p>The collector is created fresh per game; flicker / re-enter effects
 * collapse to set membership.
 */
public class PlayedCardCollector extends IGameEventVisitor.Base<Void> {

    private final Map<Character, List<String>> cardsBySide = new HashMap<>();

    /** Return the recorded card names for one side ({@code 'A'} or {@code 'B'}). */
    public List<String> getCards(char side) {
        return cardsBySide.getOrDefault(side, List.of());
    }

    @Override
    @Subscribe
    public Void visit(GameEventCardChangeZone event) {
        if (event.to() == null || event.to().getZoneType() != ZoneType.Battlefield) {
            return super.visit(event);
        }
        recordIfEligible(event.card());
        return super.visit(event);
    }

    @Override
    @Subscribe
    public Void visit(GameEventSpellAbilityCast event) {
        if (event.sa() == null) {
            return super.visit(event);
        }
        recordIfEligible(event.sa().getHostCard());
        return super.visit(event);
    }

    private void recordIfEligible(Card card) {
        if (card == null || card.getOwner() == null) {
            return;
        }
        boolean controllerEqualsOwner = card.getController() == card.getOwner();
        boolean isToken = card.isToken();
        boolean isBasicLand = card.getType().isBasicLand();
        if (!shouldRecord(controllerEqualsOwner, isToken, isBasicLand)) {
            return;
        }
        // Use paperCard.getName() so copy/clone effects credit the cast card,
        // not the copied permanent (research.md Decision D-3).
        if (card.getPaperCard() == null) {
            return;
        }
        String name = card.getPaperCard().getName();
        char side;
        try {
            side = lobbyNameToSide(card.getOwner().getName());
        } catch (IllegalArgumentException e) {
            // Unknown lobby name — should not happen in practice; skip.
            return;
        }
        cardsBySide.computeIfAbsent(side, s -> new ArrayList<>()).add(name);
    }

    /** Static filter: returns {@code true} iff the card should be recorded. */
    static boolean shouldRecord(boolean controllerEqualsOwner, boolean isToken, boolean isBasicLand) {
        return controllerEqualsOwner && !isToken && !isBasicLand;
    }

    /**
     * Map a Forge lobby player name to its match side. Reads
     * {@code LOBBY_NAME_A} / {@code LOBBY_NAME_B} from {@link GamePlayer}.
     *
     * @throws IllegalArgumentException for any other input.
     */
    static char lobbyNameToSide(String lobbyName) {
        if (GamePlayer.LOBBY_NAME_A.equals(lobbyName)) {
            return 'A';
        }
        if (GamePlayer.LOBBY_NAME_B.equals(lobbyName)) {
            return 'B';
        }
        throw new IllegalArgumentException("Unknown lobby name: " + lobbyName);
    }
}
