package com.pricepredictor.connector.effects;

import forge.StaticData;
import forge.card.GamePieceType;
import forge.game.Game;
import forge.game.GameRules;
import forge.game.GameType;
import forge.game.Match;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.card.CardFactory;
import forge.game.spellability.SpellAbility;
import forge.item.IPaperCard;
import forge.item.PaperToken;

import java.util.List;

/**
 * Real Forge cards for the provenance tests.
 *
 * <p>Built through {@code CardFactory} rather than hand-assembled, because what
 * these tests are about is the shape of the objects the engine actually hands
 * the collectors — a hand-built {@code Card} would have exactly the tidy trait
 * lists whose absence is the defect under test.
 *
 * <p>Requires {@code ForgeExtension} to have initialised the card database.
 */
final class TestCards {

    private TestCards() {
    }

    /**
     * A game exists only because {@code CardFactory} will not build a card
     * without one. No turn is ever taken in it.
     */
    private static final Game GAME = newGame();

    private static int nextCardId = 1;

    private static Game newGame() {
        GameRules rules = new GameRules(GameType.Constructed);
        return new Game(List.of(), rules, new Match(rules, List.of(), "ProvenanceTest"));
    }

    static Game game() {
        return GAME;
    }

    /** A fresh card id, for a test that builds a {@code Card} by hand. */
    static int nextCardId() {
        return nextCardId++;
    }

    static Card build(String name) {
        IPaperCard paper = StaticData.instance().getCommonCards().getCard(name);
        if (paper == null) {
            throw new AssertionError("card not in the database: " + name);
        }
        return CardFactory.getCard(paper, null, nextCardId++, GAME);
    }

    /**
     * A token, named by its script stem rather than by its printed name: a
     * token is filed under what it is, so the two 1/1 Eldrazi Scions that
     * differ only in a sacrifice ability do not collide.
     */
    static Card token(String scriptStem) {
        PaperToken paper = StaticData.instance().getAllTokens().getToken(scriptStem);
        if (paper == null) {
            throw new AssertionError("token script not in the database: " + scriptStem);
        }
        return CardFactory.getCard(paper, null, nextCardId++, GAME);
    }

    /**
     * A token the way {@code GameCopier} rebuilds one in a forked game:
     * {@code TokenInfo.toCard} makes a bare {@code Card}, copies the name,
     * image key, colour, types and P/T, and never sets a paper card. Every
     * ability-bearing token in a forked combat record arrives like this.
     */
    static Card copiedToken(String scriptStem) {
        Card original = token(scriptStem);
        Card copy = new Card(nextCardId++, GAME);
        copy.setName(original.getName());
        copy.setImageKey(original.getImageKey());
        copy.setGamePieceType(GamePieceType.TOKEN);
        for (forge.card.CardType.CoreType type : original.getType().getCoreTypes()) {
            copy.addType(type.toString());
        }
        for (String subtype : original.getType().getSubtypes()) {
            copy.addType(subtype);
        }
        copy.setBasePower(original.getBasePower());
        copy.setBaseToughness(original.getBaseToughness());
        return copy;
    }

    /** The first ability on a card whose API matches, for the emitter tests. */
    static SpellAbility scriptedAbility(String cardName, String api) {
        Card card = build(cardName);
        for (SpellAbility sa : card.getCurrentState().getSpellAbilities()) {
            if (sa.getApi() != null && api.equals(sa.getApi().name())) {
                return sa;
            }
            for (SpellAbility sub = sa.getSubAbility(); sub != null; sub = sub.getSubAbility()) {
                if (sub.getApi() != null && api.equals(sub.getApi().name())) {
                    return sub;
                }
            }
        }
        throw new AssertionError("no " + api + " ability on " + cardName);
    }

    /**
     * A {@code SpellAbility} built directly from script text, for an API with
     * no route through {@link #scriptedAbility} at all.
     *
     * <p>{@code HealDamage} is the case this exists for (Ruling R15,
     * task-8-brief.md): it appears in exactly two cards in the whole
     * cardsfolder, and neither is reachable this way -- {@code
     * pyramids.txt}'s is buried behind a {@code Charm}'s {@code Choices$} into
     * a {@code DB$ Effect}'s {@code ReplacementEffects$}, and {@code
     * wolverine_fierce_fighter.txt}'s sits behind a replacement effect's
     * {@code ReplaceWith$}; {@code scriptedAbility} walks root abilities and
     * {@code SubAbility$} chains only, and reaches neither.
     *
     * <p>A thin wrapper over {@code AbilityFactory.getAbility(text, host)} --
     * the same call {@link #scriptedAbility} itself ends in -- against a
     * generic host, since the text alone (no {@code Cost$}, no targeting) is
     * everything the API under test needs.
     */
    static SpellAbility abilityFromText(String text) {
        return AbilityFactory.getAbility(text, build("Grizzly Bears"));
    }
}
