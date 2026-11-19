package com.pricepredictor.connector.effects;

import forge.card.mana.ManaAtom;
import forge.card.mana.ManaCostShard;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.mana.Mana;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.List;

/**
 * Whether a player can pay a candidate's mana cost by the rules alone (C5).
 *
 * <p>Not {@code ComputerUtilMana.canPayManaCost}, the AI's payment planner:
 * it keeps sources the AI reserved for a later spell, for a combat trick or
 * for its second main phase out of the payment, applies AI-only filters (a
 * life cost it would rather not pay, a drawback it dislikes), draws from
 * Forge's global {@code MyRandom} for every source it checks, and writes the
 * AI's card memory ({@code UNPAID_COSTS}, the tap and sacrifice sets, the
 * held-for-blockers sets it clears). A label read that way says what the AI
 * would do, and reading it changes the game's random sequence.
 *
 * <p>Here the cost the rules charge now ({@link RulesCost}) is matched
 * against the mana the player could make: what floats in the pool and, from
 * each battlefield or hand card, one mana ability that can be activated and
 * whose non-mana costs can be paid. A mana ability with a mana cost of its
 * own (a filter land) is not counted, nor one producing special mana; X is
 * zero, as nothing is announced at a priority stop. Phyrexian shards may be
 * paid with two life each, a {@code {2/W}} shard with two generic. Each card
 * makes one activation's mana, which is a slight undercount for a card whose
 * several mana abilities need no tap.
 *
 * <p>Beside that mana, what the candidate's keywords let pay
 * ({@link KeywordPayers}), the payment chosen freely: a card delve may exile
 * is one more generic mana, and a permanent that may tap for convoke,
 * improvise or waterbend is one more choice of what that card gives, so a
 * mana creature either convokes or makes its mana, never both. No more
 * waterbend tappers pay than its amount. A keyword's {1} may pay half of a
 * {@code {2/W}} shard's two generic here, which the rules do not allow.
 */
final class ManaAffordability {

    /** More choices of activation than this across the sources and each card keeps its first. */
    private static final int MAX_COMBINATIONS = 4096;

    private final Player payer;
    private final SpellAbility candidate;
    private final ShardDemand demand;
    private final KeywordPayers payers;
    private final List<List<List<ManaUnit>>> sources;

    private ManaAffordability(Player payer, SpellAbility candidate, ShardDemand demand, KeywordPayers payers) {
        this.payer = payer;
        this.candidate = candidate;
        this.demand = demand;
        this.payers = payers;
        this.sources = withinBudget(sources(candidate, payer, payers));
    }

    /**
     * Whether {@code payer} can pay {@code candidate}'s mana cost now.
     *
     * @param charged what the rules charge, or null when it has no cost
     */
    static boolean canPay(SpellAbility candidate, Player payer, RulesCost charged) {
        if (charged == null) {
            return true;
        }
        ShardDemand demand = ShardDemand.of(charged.total());
        if (demand.isEmpty()) {
            return true;
        }
        List<ManaUnit> units = floating(candidate, payer);
        units.addAll(charged.payers().delved(demand.generic()));
        return new ManaAffordability(payer, candidate, demand, charged.payers()).payable(units, 0);
    }

    /** Each source's choices, or only its first where all of them together are too many to try. */
    private static List<List<List<ManaUnit>>> withinBudget(List<List<List<ManaUnit>>> sources) {
        long combinations = 1;
        for (List<List<ManaUnit>> options : sources) {
            combinations *= options.size();
            if (combinations > MAX_COMBINATIONS) {
                return sources.stream().map(choices -> List.of(choices.get(0))).toList();
            }
        }
        return sources;
    }

    /** Choose one activation per source, then try the shards against the mana that makes. */
    private boolean payable(List<ManaUnit> units, int next) {
        if (next == sources.size()) {
            return covers(units);
        }
        for (List<ManaUnit> option : sources.get(next)) {
            units.addAll(option);
            boolean paid = payable(units, next + 1);
            units.subList(units.size() - option.size(), units.size()).clear();
            if (paid) {
                return true;
            }
        }
        return false;
    }

    /** Whether these units pay the shards, some phyrexian ones with life and some {2/C} ones with two generic. */
    private boolean covers(List<ManaUnit> units) {
        int usable = units.size() - payers.beyondLimit(units);
        for (int withLife = 0; withLife <= demand.phyrexian().size(); withLife++) {
            if (withLife > 0 && !payer.canPayLife(2 * withLife, false, candidate)) {
                return false;
            }
            if (coversWithLife(units, usable, withLife)) {
                return true;
            }
        }
        return false;
    }

    private boolean coversWithLife(List<ManaUnit> units, int usable, int withLife) {
        for (int asGeneric = 0; asGeneric <= demand.twoGeneric().size(); asGeneric++) {
            List<ManaCostShard> slots = demand.slots(withLife, asGeneric);
            int needed = demand.generic() + 2 * asGeneric;
            if (slots.size() + needed <= usable && ShardMatching.size(units, slots) == slots.size()) {
                return true;
            }
        }
        return false;
    }

    /** The mana floating in the payer's pool that may be spent on the candidate. */
    private static List<ManaUnit> floating(SpellAbility candidate, Player payer) {
        List<ManaUnit> units = new ArrayList<>();
        for (Mana mana : payer.getManaPool()) {
            if (mana.meetsManaRestrictions(candidate)
                    && candidate.allowsPayingWithShard(mana.getSourceCard(), mana.getColor())) {
                units.add(new ManaUnit(mana.getColor(), mana.isSnow(), false));
            }
        }
        return units;
    }

    /**
     * Each card's choices of what it pays: the mana each of its mana
     * abilities would make, and the unit it pays as a keyword's tapper; a
     * card the candidate's own tap cost taps is no source.
     */
    private static List<List<List<ManaUnit>>> sources(SpellAbility candidate, Player payer, KeywordPayers payers) {
        Cost cost = candidate.getPayCosts();
        Card host = candidate.getHostCard();
        boolean hostTaps = cost != null && cost.hasTapCost();
        List<List<List<ManaUnit>>> sources = new ArrayList<>();
        List<Card> cards = new ArrayList<>(payer.getCardsIn(ZoneType.Battlefield));
        cards.addAll(payer.getCardsIn(ZoneType.Hand));
        for (Card card : cards) {
            List<List<ManaUnit>> options = hostTaps && card == host
                    ? List.of() : optionsOf(card, candidate, payer, payers);
            if (!options.isEmpty()) {
                sources.add(options);
            }
        }
        return sources;
    }

    private static List<List<ManaUnit>> optionsOf(Card card, SpellAbility candidate, Player payer, KeywordPayers payers) {
        List<List<ManaUnit>> options = new ArrayList<>();
        for (SpellAbility ability : card.getManaAbilities()) {
            List<ManaUnit> made = ManaAbilityUnits.made(ability, candidate, payer);
            if (!made.isEmpty()) {
                options.add(made);
            }
        }
        ManaUnit tapper = payers.tappers().get(card);
        if (tapper != null) {
            options.add(List.of(tapper));
        }
        return options;
    }
}
