package com.pricepredictor.connector;

import com.google.common.eventbus.Subscribe;
import forge.game.card.CardView;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.IGameEventVisitor;
import forge.game.player.PlayerView;
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
 *   <li>{@code controller == owner} (drops stolen cards).</li>
 *   <li>{@code !isToken()}.</li>
 *   <li>{@code !isBasicLand()} on the current state (FR-004a).</li>
 *   <li>The recorded name comes from {@code getBackup()} when available so
 *       copy/clone effects credit the cast card, not the copied permanent.</li>
 * </ul>
 *
 * <p>Cards are bucketed by the owner's lobby player name, then mapped to
 * {@code 'A'} / {@code 'B'} via {@link GamePlayer}'s {@code LOBBY_NAME_A}
 * / {@code LOBBY_NAME_B} constants.
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
        if (event.to() == null || event.to().zoneType() != ZoneType.Battlefield) {
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

    private void recordIfEligible(CardView card) {
        if (card == null) {
            return;
        }
        PlayerView owner = card.getOwner();
        if (owner == null) {
            return;
        }
        boolean controllerEqualsOwner = owner.equals(card.getController());
        boolean isToken = card.isToken();
        boolean isBasicLand = card.getCurrentState() != null
                && card.getCurrentState().getType() != null
                && card.getCurrentState().getType().isBasicLand();
        if (!shouldRecord(controllerEqualsOwner, isToken, isBasicLand)) {
            return;
        }
        String name = resolveRecordedName(card);
        if (name == null || name.isEmpty()) {
            return;
        }
        char side;
        try {
            side = lobbyNameToSide(owner.getLobbyPlayerName());
        } catch (IllegalArgumentException e) {
            // Unknown lobby name — should not happen in practice; skip.
            return;
        }
        cardsBySide.computeIfAbsent(side, s -> new ArrayList<>()).add(name);
    }

    /**
     * Use the paper-card backup's name when present so copy/clone and
     * face-down (manifested) cards credit the underlying card. Falls back
     * to the current state's name when no backup exists (e.g. dungeons).
     */
    private static String resolveRecordedName(CardView card) {
        CardView backup = card.getBackup();
        if (backup != null && backup.getCurrentState() != null) {
            String name = backup.getCurrentState().getName();
            if (name != null && !name.isEmpty()) {
                return name;
            }
        }
        return card.getCurrentState() != null
                ? card.getCurrentState().getName()
                : null;
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
