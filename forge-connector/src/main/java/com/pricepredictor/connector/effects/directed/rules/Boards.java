package com.pricepredictor.connector.effects.directed.rules;

import com.pricepredictor.connector.effects.PlayabilityVerdict;
import com.pricepredictor.connector.effects.directed.ForcedPlay;
import com.pricepredictor.connector.effects.directed.operations.ReplaceCardOperation;
import forge.game.Game;
import forge.game.ability.AbilityFactory;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.card.CardState;
import forge.game.combat.Combat;
import forge.game.cost.Cost;
import forge.game.keyword.Keyword;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordWithCostAndAmount;
import forge.game.keyword.KeywordWithCostInterface;
import forge.game.keyword.KeywordsChange;
import forge.game.phase.PhaseType;
import forge.game.player.Player;
import forge.game.spellability.AlternativeCost;
import forge.game.spellability.SpellAbility;
import forge.game.trigger.Trigger;
import forge.game.trigger.TriggerType;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * What the board preconditions read off a live game, in the words the
 * scenarios use: the phase a boundary begins, the player whose turn it is,
 * the creatures a player controls. Read only: a precondition changes nothing
 * in the game it is asked about.
 */
final class Boards {

    private Boards() {
    }

    /** Whether the boundary the live game is at begins one of {@code phases}. */
    static boolean at(Game live, PhaseType... phases) {
        return Set.of(phases).contains(live.getPhaseHandler().getPhase());
    }

    /** Whether the boundary begins a main phase, where a sorcery-speed decision is next. */
    static boolean atMainPhase(Game live) {
        return at(live, PhaseType.MAIN1, PhaseType.MAIN2);
    }

    /** The player whose turn it is. */
    static Player active(Game live) {
        return live.getPhaseHandler().getPlayerTurn();
    }

    /**
     * Whether {@code holder} can cast {@code card} at this boundary as far as
     * timing goes: the active player at a main phase can cast anything, any
     * other player only an instant or a card with flash. A sorcery in the
     * non-active player's hand cannot be cast this turn, so a fork waiting
     * for it would be spent on nothing.
     */
    static boolean castableAtItsSpeed(Player holder, Card card) {
        return holder == active(holder.getGame()) || card.isInstant() || card.hasKeyword(Keyword.FLASH);
    }

    /**
     * Whether {@code player} cannot add a land to its mana this turn: it holds
     * no card it could play as a land, or it has played as many lands this
     * turn as it may. A "one land short" variant forked at a main phase
     * before the turn's land drop is undone by that drop.
     */
    static boolean noLandToPlay(Player player) {
        boolean dropLeft = player.getMaxLandPlaysInfinite()
                || player.getLandsPlayedThisTurn() < player.getMaxLandPlays();
        return !dropLeft || player.getCardsIn(ZoneType.Hand).stream().noneMatch(Card::hasPlayableLandFace);
    }

    /**
     * Whether {@code replace_card} puts {@code cardName} in {@code permanent}'s
     * place, refusing none of what the permanent holds (a token, a merged or
     * melded permanent, a face-down one: {@link ReplaceCardOperation#carries}),
     * and the new card survives the fork's first state-based check, carrying
     * the permanent's marked damage, deathtouch damage, counters, boosts and
     * resolved effects: the edit's own readings
     * ({@link ReplaceCardOperation#replacementDies}), so a precondition binds
     * no board the edit would refuse.
     */
    static boolean replacementSurvives(Card permanent, String cardName) {
        return ReplaceCardOperation.carries(permanent) && !ReplaceCardOperation.replacementDies(permanent, cardName);
    }

    /**
     * Whether every instance of {@code keyword} the card has is its own text's:
     * none is granted by a static ability (an Equipment, an Aura, an anthem)
     * or by a resolved effect. {@code replace_card} relinks the attachments
     * and carries the resolved effects, and the statics apply to the new card
     * anew, so a granted keyword would stay on the card put in its place.
     */
    static boolean ownKeywordOnly(Card card, Keyword keyword) {
        for (KeywordsChange change : card.getChangedCardKeywords().values()) {
            if (change.getKeywords().stream().anyMatch(granted -> granted.getKeyword() == keyword)) {
                return false;
            }
        }
        return true;
    }

    /** Forge's spelling of trample over planeswalkers, a trample instance with a detail. */
    static final String TRAMPLE_OVER_PLANESWALKERS = "Trample:Planeswalker";

    /** Whether the card has trample over planeswalkers ({@link #TRAMPLE_OVER_PLANESWALKERS}). */
    static boolean tramplesOverPlaneswalkers(Card card) {
        return card.hasKeyword(TRAMPLE_OVER_PLANESWALKERS);
    }

    /**
     * The attackers of the live combat that deal their combat damage in the
     * regular damage step and none blocks: neither first strike nor double
     * strike, power above 0, and no blocker declared against them.
     */
    static List<Card> unblockedRegularStrikers(Combat combat) {
        return combat.getAttackers().stream()
                .filter(attacker -> !attacker.hasFirstStrike() && !attacker.hasDoubleStrike())
                .filter(attacker -> attacker.getNetCombatDamage() > 0)
                .filter(attacker -> !combat.isBlocked(attacker) && combat.getBlockers(attacker).isEmpty())
                .toList();
    }

    /**
     * The unblocked regular-step attackers ({@link #unblockedRegularStrikers})
     * whose attacked planeswalker or battle has left combat: Forge keeps them
     * attacking a placeholder of no zone (506.4c, {@code Combat.isRemovedDefender}).
     */
    static List<Card> attackersOfAGoneDefender(Combat combat) {
        return unblockedRegularStrikers(combat).stream()
                .filter(attacker -> Combat.isRemovedDefender(combat.getDefenderByAttacker(attacker)))
                .toList();
    }

    /**
     * Whether a combatant of the live combat other than {@code attacker} deals
     * combat damage in the regular damage step: an attacker or blocker with
     * power above 0 that strikes in it (double strike, or no first strike)
     * and has something to hit (an attacker its blockers, or unblocked a
     * player or a planeswalker or battle still in combat; a blocker the
     * attackers it blocks). A damage step in which no combat damage is dealt
     * writes no combat record ({@code CombatStepBracket}), so an attacker
     * dealing none is recorded only beside another's damage.
     */
    static boolean anotherDealsRegularDamage(Combat combat, Card attacker) {
        for (Card other : combat.getAttackers()) {
            if (other == attacker || !strikesInTheRegularStep(other)) {
                continue;
            }
            if (!combat.getBlockers(other).isEmpty()
                    || !combat.isBlocked(other) && !Combat.isRemovedDefender(combat.getDefenderByAttacker(other))) {
                return true;
            }
        }
        return combat.getAllBlockers().stream().anyMatch(blocker -> strikesInTheRegularStep(blocker)
                && !combat.getAttackersBlockedBy(blocker).isEmpty());
    }

    private static boolean strikesInTheRegularStep(Card combatant) {
        return combatant.getNetCombatDamage() > 0 && (combatant.hasDoubleStrike() || !combatant.hasFirstStrike());
    }

    /** The planeswalker on the battlefield {@code attacker} attacks, or empty when it attacks none. */
    static Optional<Card> attackedPlaneswalker(Combat combat, Card attacker) {
        return combat.getDefenderByAttacker(attacker) instanceof Card defender
                && defender.isPlaneswalker() && defender.isInPlay()
                ? Optional.of(defender) : Optional.empty();
    }

    /** The creatures {@code player} controls that are untapped. */
    static List<Card> untappedCreatures(Player player) {
        return player.getCreaturesInPlay().stream().filter(card -> !card.isTapped()).toList();
    }

    /** The highest life total among the game's players other than {@code player}. */
    static int mostLifeBesides(Player player) {
        return player.getGame().getPlayers().stream()
                .filter(other -> other != player)
                .mapToInt(Player::getLife)
                .max()
                .orElseThrow(() -> new IllegalStateException("no other player"));
    }

    /**
     * Whether {@code player} can pay one of {@code card}'s own casts
     * ({@link ForcedPlay.Line#SPELL}) now, colours included
     * ({@link #affordable(Player, SpellAbility)}).
     */
    static boolean affordable(Player player, Card card) {
        return card.getSpells().stream().anyMatch(spell -> ForcedPlay.Line.SPELL.matches(spell)
                && affordable(player, spell));
    }

    /**
     * Whether {@code player} can pay {@code ability}'s cost now, by the rules
     * the collector's playability reads ({@link PlayabilityVerdict#affordable}):
     * the mana its sources make, colours included, and the additional costs.
     * Asked of a copy activated by {@code player}, so the live ability stays
     * as it is; a cost that cannot be read on this board is not affordable.
     */
    static boolean affordable(Player player, SpellAbility ability) {
        try {
            return PlayabilityVerdict.affordable(asPlayedBy(ability, player), player);
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** A copy of {@code ability} as {@code player} would play it: what a precondition asks questions of. */
    static SpellAbility asPlayedBy(SpellAbility ability, Player player) {
        return ability.copy(ability.getHostCard(), player, true);
    }

    /**
     * How much mana one activation of the permanent's most productive mana
     * ability makes: two for a bounce land, two for Sol Ring, one for a basic
     * land.
     */
    static int manaUnits(Card source) {
        return source.getManaAbilities().stream()
                .mapToInt(ability -> ability.amountOfManaGenerated(true))
                .max().orElse(0);
    }

    /**
     * The untapped permanents {@code player} controls that can make mana now.
     * A summoning-sick creature's tap ability makes none.
     */
    static List<Card> manaSources(Player player) {
        return player.getCardsIn(ZoneType.Battlefield).stream()
                .filter(card -> !card.isTapped() && !card.getManaAbilities().isEmpty())
                .filter(card -> !(card.isCreature() && card.isSick()))
                .toList();
    }

    /**
     * The one untapped permanent {@code player} controls that can make mana
     * now, or empty when it controls none or several: the last mana it has
     * for a cost.
     */
    static Optional<Card> soleManaSource(Player player) {
        List<Card> sources = manaSources(player);
        return sources.size() == 1 ? Optional.of(sources.get(0)) : Optional.empty();
    }

    /**
     * A land {@code player} needs every one of for a cost of {@code value}
     * mana: when the permanents it can make mana with now ({@link #manaSources})
     * are all lands and number exactly {@code value}, the first of them; else
     * empty. Without it the cost is one mana short.
     */
    static Optional<Card> lastLandFor(Player player, int value) {
        List<Card> sources = manaSources(player);
        if (value <= 0 || sources.size() != value || !sources.stream().allMatch(Card::isLand)) {
            return Optional.empty();
        }
        return Optional.of(sources.get(0));
    }

    /** The mana value of a cost's mana part, 0 for none. */
    static int manaValue(Cost cost) {
        return cost == null || cost.getTotalMana() == null ? 0 : cost.getTotalMana().getCMC();
    }

    /** A spell of the card that searches a library ({@link #searches}), the line its forced cast plays. */
    static final ForcedPlay.Line LIBRARY_SEARCH_SPELL = ForcedPlay.Line.spell("library search", Boards::searches);

    /**
     * The cards in {@code player}'s hand it has the mana for whose spell
     * searches its library ({@link #LIBRARY_SEARCH_SPELL}) and that it can
     * cast at this boundary ({@link #castableAtItsSpeed}: a sorcery only as
     * the active player).
     */
    static List<Card> searchersInHand(Player player) {
        return castableInHand(player, LIBRARY_SEARCH_SPELL).stream()
                .filter(card -> castableAtItsSpeed(player, card))
                .toList();
    }

    /** Whether the spell searches a library: it moves a card of a stated type out of one, its contents hidden. */
    static boolean searches(SpellAbility spell) {
        return spell.getApi() == ApiType.ChangeZone && lists(spell.getParam("Origin"), "Library")
                && spell.hasParam("ChangeType");
    }

    /**
     * The spells of the cards in {@code player}'s hand that are {@code line}
     * and that it has the mana for ({@link #affordable(Player, SpellAbility)}),
     * in hand order. A variant forcing the cast forces the same line, so a
     * split card's matching half is the one cast.
     */
    static List<SpellAbility> castableSpellsInHand(Player player, ForcedPlay.Line line) {
        return player.getCardsIn(ZoneType.Hand).stream()
                .flatMap(card -> card.getSpells().stream())
                .filter(spell -> line.matches(spell) && affordable(player, spell))
                .toList();
    }

    /** The cards of {@link #castableSpellsInHand}, each once, in hand order. */
    static List<Card> castableInHand(Player player, ForcedPlay.Line line) {
        return castableSpellsInHand(player, line).stream().map(SpellAbility::getHostCard).distinct().toList();
    }

    /** The costs of the card's instances of {@code keyword} (Transmute, Unearth, Suspend), in the card's order. */
    static List<Cost> keywordCosts(Card card, Keyword keyword) {
        List<Cost> costs = new ArrayList<>();
        for (KeywordInterface instance : card.getKeywords(keyword)) {
            if (instance instanceof KeywordWithCostInterface withCost) {
                costs.add(withCost.getCost());
            } else if (instance instanceof KeywordWithCostAndAmount withCost) {
                costs.add(withCost.getCost());
            }
        }
        return costs;
    }

    /**
     * Whether {@code player} can pay one of the card's abilities of
     * {@code keyword} now ({@link #affordable(Player, SpellAbility)}): the
     * activated ability of Transmute, Unearth or Aura swap, or the alternative
     * cost of the spell that Bestow or Web-slinging makes. Keywords that pay
     * by tapping creatures (Improvise, Convoke) are not asked here.
     */
    static boolean canPayKeyword(Player player, Card card, Keyword keyword) {
        return card.getAllSpellAbilities().stream()
                .filter(ability -> viaKeyword(ability, keyword))
                .anyMatch(ability -> affordable(player, ability));
    }

    private static boolean viaKeyword(SpellAbility ability, Keyword keyword) {
        return ability.isKeyword(keyword)
                || keyword == Keyword.BESTOW && ability.isBestow()
                || keyword == Keyword.WEB_SLINGING && ability.isAlternativeCost(AlternativeCost.WebSlinging);
    }

    /**
     * The card's own cast of a spell that acts with {@code api}, an
     * {@code ApiType} name: the spell has it at its root, or the permanent it
     * makes has a trigger executing it (a creature that explores, connives or
     * discovers as it enters). Named by string: the registry builds it where
     * Forge is not initialised.
     */
    static ForcedPlay.Line acting(String api) {
        return ForcedPlay.Line.spell(api, spell -> acts(spell, api));
    }

    private static boolean acts(SpellAbility spell, String api) {
        if (spell.getApi() != null && spell.getApi().name().equals(api)) {
            return true;
        }
        CardState state = spell.getCardState();
        return state != null && state.getTriggers().stream()
                .anyMatch(trigger -> Triggers.entersItself(trigger) && api.equals(api(executed(trigger))));
    }

    /** The first card {@code player} holds with a castable spell of {@code line} ({@link #castableInHand}). */
    static Optional<Card> acting(Player player, ForcedPlay.Line line) {
        return castableInHand(player, line).stream().findFirst();
    }

    /**
     * Whether a permanent on the battlefield has a trigger of {@code mode}
     * (Explores, Connives, Discover) that {@code active}'s {@code actor}
     * sets off ({@link Triggers#watchedFor}).
     */
    static boolean watched(Game live, TriggerType mode, Player active, Card actor) {
        return Triggers.watchedFor(live.getCardsIn(ZoneType.Battlefield), mode, active, actor);
    }

    /** Whether a comma-separated script value lists {@code item}. */
    static boolean lists(String value, String item) {
        return value != null && List.of(value.split(",")).contains(item);
    }

    /** The parameters of the ability script {@code host} holds as {@code svar}, empty when it holds none. */
    static Map<String, String> script(Card host, String svar) {
        if (svar == null || !host.hasSVar(svar)) {
            return Map.of();
        }
        return AbilityFactory.getMapParams(host.getSVar(svar));
    }

    /** The parameters of the ability {@code trigger} executes, empty when it names none. */
    static Map<String, String> executed(Trigger trigger) {
        return trigger.hasParam("Execute") ? script(trigger.getHostCard(), trigger.getParam("Execute")) : Map.of();
    }

    /** The API a script's parameters name, whether it is a spell, an ability or a sub-ability. */
    static String api(Map<String, String> script) {
        for (String kind : List.of("SP", "AB", "DB")) {
            if (script.containsKey(kind)) {
                return script.get(kind);
            }
        }
        return "";
    }

    /** {@code value} as a number, or empty when it is an X, a count or absent. */
    static Optional<Integer> number(String value) {
        if (value == null) {
            return Optional.empty();
        }
        try {
            return Optional.of(Integer.parseInt(value.trim()));
        } catch (NumberFormatException e) {
            return Optional.empty();
        }
    }
}
