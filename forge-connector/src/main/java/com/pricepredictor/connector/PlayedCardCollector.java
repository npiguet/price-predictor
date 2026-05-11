package com.pricepredictor.connector;

import com.google.common.eventbus.Subscribe;
import forge.card.GamePieceType;
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
 *   <li>{@code gamePieceType == CARD} — drops emblems / "X's effect"
 *       trackers (e.g. The Ring, Monarch), dungeons (e.g. Undercity),
 *       copied spells, schemes, planes, and any other non-deck game
 *       object Forge models as a Card.</li>
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
        // Skip face-down entries to the battlefield. Self-induced face-down
        // play (Morph / Disguise) still gets counted via the cast-event
        // branch below, since casting for a morph cost fires
        // GameEventSpellAbilityCast with the morph card as host. Externally
        // forced face-down play (Manifest / Cloak / Manifest Dread) has no
        // such cast event, so it stays unrecorded — matching the rule that
        // a card "is played" only when its identity actually shows up.
        //
        // TODO (face-up reveal): when an externally-manifested card later
        // turns face-up, no event we currently subscribe to fires for the
        // affected card with its real identity. To capture that the card
        // was eventually played we'd need to subscribe to
        // GameEventCardStatsChanged and detect the face-down -> face-up
        // transition. Skipped for now — most manifested 2/2s die face-down,
        // so the missed-signal volume should be small.
        CardView card = event.card();
        if (card != null && card.isFaceDown()) {
            return super.visit(event);
        }
        recordIfEligible(card);
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
        boolean isRealDeckCard = card.getGamePieceType() == GamePieceType.CARD;
        if (!shouldRecord(controllerEqualsOwner, isToken, isBasicLand, isRealDeckCard)) {
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
     * Use the paper-card backup's oracle name when present so copy/clone
     * and face-down (manifested) cards credit the underlying card. Falls
     * back to the current state's oracle name when no backup exists
     * (e.g. dungeons). Oracle name is the canonical MTG name (unaltered
     * by flavor names or {@code SetName} effects); display name is not.
     */
    private static String resolveRecordedName(CardView card) {
        CardView backup = card.getBackup();
        if (backup != null && backup.getCurrentState() != null) {
            String name = backup.getCurrentState().getOracleName();
            if (name != null && !name.isEmpty()) {
                return name;
            }
        }
        return card.getCurrentState() != null
                ? card.getCurrentState().getOracleName()
                : null;
    }

    /** Static filter: returns {@code true} iff the card should be recorded. */
    static boolean shouldRecord(
            boolean controllerEqualsOwner,
            boolean isToken,
            boolean isBasicLand,
            boolean isRealDeckCard) {
        return controllerEqualsOwner && !isToken && !isBasicLand && isRealDeckCard;
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
