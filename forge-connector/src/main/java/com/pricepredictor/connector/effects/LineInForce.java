package com.pricepredictor.connector.effects;

import forge.card.CardStateName;
import forge.game.CardTraitBase;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.keyword.KeywordInterface;
import forge.game.staticability.StaticAbility;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A card line's {@code active}: whether the line is in force (root spec
 * § 9.2). A line of a face not up is not. Of the face up, a static is while
 * Forge's condition for it holds, a {@code Class:N} level line is once the
 * Class has reached level N (written whether or not it has), and so is a
 * trait that level grants, which keys to the level's keyword; a Room's line
 * is while its door is unlocked. Every other line says nothing, and reads as
 * in force. A static's condition is read on the board the lines are read on:
 * on a {@code continuous} record's board, without the record's static
 * ({@link LentBoard}).
 *
 * <p>A Room's doors are its two split halves. Forge plays the doors
 * unlocked as the state up: both halves' {@code Original}, one half's own
 * state, or the empty room with none. A locked door is a face not up, so
 * its lines are listed {@code active: false}; off the battlefield the state
 * up is {@code Original} (both halves, 709.4) or, on the stack, the half
 * cast.
 */
final class LineInForce {

    private static final String CLASS_LEVEL = "Class:";
    private static final List<CardStateName> DOORS =
            List.of(CardStateName.LeftSplit, CardStateName.RightSplit);
    /** The states a Room is in with at least one door up. */
    private static final Set<CardStateName> DOORS_UP = Set.of(
            CardStateName.Original, CardStateName.LeftSplit, CardStateName.RightSplit);

    private final Card card;
    /** The board the lines are read on. */
    private final StaticSuppression board;
    /** The level of each {@code Class:N} keyword of the face up, by its key. */
    private final Map<ProvenanceKey, Integer> levels = new HashMap<>();
    /** True on a Room with a door up: every line of its face up is a door's. */
    private final Boolean doorUp;

    LineInForce(Card card, StaticSuppression board) {
        this.card = card;
        this.board = board;
        this.doorUp = isRoom(card) && DOORS_UP.contains(card.getCurrentStateName()) ? Boolean.TRUE : null;
        CardState up = card.getCurrentState();
        for (KeywordInterface keyword : up.getIntrinsicKeywords()) {
            addLevel(up, keyword);
        }
    }

    private void addLevel(CardState up, KeywordInterface keyword) {
        String original = keyword.getOriginal();
        if (original == null || !original.startsWith(CLASS_LEVEL)) {
            return;
        }
        ProvenanceKey key = ProvenanceResolver.keyOfKeyword(up, keyword);
        if (key != null) {
            levels.put(key.withoutBuilt(), levelOf(original));
        }
    }

    /** A keyword line's {@code active}, or null for none. */
    Boolean ofKeyword(KeywordInterface keyword, boolean faceUp) {
        if (!faceUp) {
            return Boolean.FALSE;
        }
        String original = keyword.getOriginal();
        if (original.startsWith(CLASS_LEVEL)) {
            return levelOf(original) <= card.getClassLevel();
        }
        return doorUp;
    }

    /**
     * A trigger, ability, static or replacement line's {@code active}, or
     * null for none. A static a level grants keeps its own condition: it is
     * there only once the level is reached.
     */
    Boolean ofTrait(CardTraitBase trait, ProvenanceKey key, boolean faceUp) {
        if (!faceUp) {
            return Boolean.FALSE;
        }
        if (trait instanceof StaticAbility stat) {
            return holds(stat);
        }
        Integer level = key == null ? null : levels.get(key.withoutBuilt());
        return level == null ? doorUp : Boolean.valueOf(level <= card.getClassLevel());
    }

    /** A Room's doors that are not up, in face order: none for any other card. */
    List<CardState> lockedDoors() {
        List<CardState> locked = new ArrayList<>();
        if (!isRoom(card)) {
            return locked;
        }
        CardStateName up = card.getCurrentStateName();
        for (CardStateName door : DOORS) {
            if (up != door && up != CardStateName.Original) {
                locked.add(card.getState(door));
            }
        }
        return locked;
    }

    /** The level of a {@code Class:<level>:<cost>:...} keyword. */
    private static int levelOf(String original) {
        return Integer.parseInt(original.split(":", 3)[1]);
    }

    /** Whether the card's printed halves are a Room's doors, whichever state is up. */
    private static boolean isRoom(Card card) {
        return card.hasState(CardStateName.LeftSplit)
                && card.getState(CardStateName.LeftSplit).getType().hasSubtype("Room");
    }

    /**
     * Forge's answer to whether a static's condition holds now on the board;
     * false when the question cannot be asked of this card, such as a
     * static on a card no player controls.
     */
    private Boolean holds(StaticAbility stat) {
        try {
            return LentBoard.read(card.getGame(), board, stat::checkConditions);
        } catch (RuntimeException e) {
            return Boolean.FALSE;
        }
    }
}
