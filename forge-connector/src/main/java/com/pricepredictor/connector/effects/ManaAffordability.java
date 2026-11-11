package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.card.mana.ManaAtom;
import forge.card.mana.ManaCostShard;
import forge.game.ability.ApiType;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostPayment;
import forge.game.mana.Mana;
import forge.game.mana.ManaCostBeingPaid;
import forge.game.player.Player;
import forge.game.spellability.AbilityManaPart;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Arrays;
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
 * <p>Here the cost Forge would charge now ({@link DecisionCandidate#charged})
 * is matched against the mana the player could make: what floats in the pool
 * and, from each battlefield or hand card, one mana ability that can be
 * activated and whose non-mana costs can be paid. A mana ability with a mana
 * cost of its own (a filter land) is not counted, nor one producing special
 * mana; X is zero, as nothing is announced at a priority stop. Phyrexian
 * shards may be paid with two life each, a {@code {2/W}} shard with two
 * generic. Each card makes one activation's mana, which is a slight
 * undercount for a card whose several mana abilities need no tap.
 */
final class ManaAffordability {

    /** More choices of activation than this across the sources and each card keeps its first. */
    private static final int MAX_COMBINATIONS = 4096;

    private ManaAffordability() {
    }

    /** One mana the player could make: its possible colours, and whether it is snow. */
    private record Unit(byte colors, boolean snow) {
    }

    /** The colours one unit can pay a shard with. */
    private static boolean pays(Unit unit, ManaCostShard shard) {
        if (shard.isSnow() && !unit.snow()) {
            return false;
        }
        if (shard == ManaCostShard.GENERIC || shard == ManaCostShard.S) {
            return true;
        }
        for (byte color : ManaAtom.MANATYPES) {
            if ((unit.colors() & color) != 0 && shard.canBePaidWithManaOfColor(color)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Whether {@code payer} can pay {@code candidate}'s mana cost now.
     *
     * @param charged the mana Forge would charge, or null when it has no cost
     */
    static boolean canPay(SpellAbility candidate, Player payer, ManaCostBeingPaid charged) {
        if (charged == null) {
            return true;
        }
        List<ManaCostShard> colored = new ArrayList<>();
        List<ManaCostShard> phyrexian = new ArrayList<>();
        List<ManaCostShard> twoGeneric = new ArrayList<>();
        int generic = 0;
        for (ManaCostShard shard : charged.getUnpaidShards()) {
            if (shard.isOfKind(ManaAtom.IS_X)) {
                continue; // nothing is announced at a priority stop
            } else if (shard == ManaCostShard.GENERIC) {
                generic++;
            } else if (shard.isPhyrexian()) {
                phyrexian.add(shard);
            } else if (shard.isOr2Generic()) {
                twoGeneric.add(shard);
            } else {
                colored.add(shard);
            }
        }
        if (colored.isEmpty() && phyrexian.isEmpty() && twoGeneric.isEmpty() && generic == 0) {
            return true;
        }
        List<Unit> floating = floating(candidate, payer);
        List<List<List<Unit>>> sources = sources(candidate, payer);
        long combinations = 1;
        for (List<List<Unit>> options : sources) {
            combinations *= options.size();
            if (combinations > MAX_COMBINATIONS) {
                break;
            }
        }
        if (combinations > MAX_COMBINATIONS) {
            sources = sources.stream().map(options -> List.of(options.get(0))).toList();
        }
        return payable(new ArrayList<>(floating), sources, 0, payer, candidate,
                colored, phyrexian, twoGeneric, generic);
    }

    /** Choose one activation per source, then try the shards against the mana that makes. */
    private static boolean payable(
            List<Unit> units, List<List<List<Unit>>> sources, int next, Player payer,
            SpellAbility candidate, List<ManaCostShard> colored, List<ManaCostShard> phyrexian,
            List<ManaCostShard> twoGeneric, int generic) {
        if (next == sources.size()) {
            return covers(units, payer, candidate, colored, phyrexian, twoGeneric, generic);
        }
        for (List<Unit> option : sources.get(next)) {
            units.addAll(option);
            boolean paid = payable(units, sources, next + 1, payer, candidate,
                    colored, phyrexian, twoGeneric, generic);
            units.subList(units.size() - option.size(), units.size()).clear();
            if (paid) {
                return true;
            }
        }
        return false;
    }

    /** Whether these units pay the shards, some phyrexian ones with life and some {2/C} ones with two generic. */
    private static boolean covers(
            List<Unit> units, Player payer, SpellAbility candidate, List<ManaCostShard> colored,
            List<ManaCostShard> phyrexian, List<ManaCostShard> twoGeneric, int generic) {
        for (int withLife = 0; withLife <= phyrexian.size(); withLife++) {
            if (withLife > 0 && !payer.canPayLife(2 * withLife, false, candidate)) {
                break;
            }
            for (int asGeneric = 0; asGeneric <= twoGeneric.size(); asGeneric++) {
                List<ManaCostShard> slots = new ArrayList<>(colored);
                slots.addAll(phyrexian.subList(withLife, phyrexian.size()));
                slots.addAll(twoGeneric.subList(asGeneric, twoGeneric.size()));
                int needed = generic + 2 * asGeneric;
                if (slots.size() + needed > units.size()) {
                    continue;
                }
                if (matched(units, slots) == slots.size()) {
                    return true;
                }
            }
        }
        return false;
    }

    /** The size of a maximum matching of shards to units (Kuhn's augmenting paths). */
    private static int matched(List<Unit> units, List<ManaCostShard> slots) {
        int[] unitOf = new int[slots.size()];
        int[] slotOf = new int[units.size()];
        Arrays.fill(unitOf, -1);
        Arrays.fill(slotOf, -1);
        int count = 0;
        for (int slot = 0; slot < slots.size(); slot++) {
            if (augment(slot, units, slots, slotOf, unitOf, new boolean[units.size()])) {
                count++;
            }
        }
        return count;
    }

    private static boolean augment(int slot, List<Unit> units, List<ManaCostShard> slots,
                                   int[] slotOf, int[] unitOf, boolean[] seen) {
        for (int unit = 0; unit < units.size(); unit++) {
            if (seen[unit] || !pays(units.get(unit), slots.get(slot))) {
                continue;
            }
            seen[unit] = true;
            if (slotOf[unit] < 0 || augment(slotOf[unit], units, slots, slotOf, unitOf, seen)) {
                slotOf[unit] = slot;
                unitOf[slot] = unit;
                return true;
            }
        }
        return false;
    }

    /** The mana floating in the payer's pool that may be spent on the candidate. */
    private static List<Unit> floating(SpellAbility candidate, Player payer) {
        List<Unit> units = new ArrayList<>();
        for (Mana mana : payer.getManaPool()) {
            if (mana.meetsManaRestrictions(candidate)
                    && candidate.allowsPayingWithShard(mana.getSourceCard(), mana.getColor())) {
                units.add(new Unit(mana.getColor(), mana.isSnow()));
            }
        }
        return units;
    }

    /**
     * Each card's choices of activation, as the mana each would make; a card
     * the candidate's own tap cost taps is no source.
     */
    private static List<List<List<Unit>>> sources(SpellAbility candidate, Player payer) {
        Cost cost = candidate.getPayCosts();
        Card host = candidate.getHostCard();
        boolean hostTaps = cost != null && cost.hasTapCost();
        List<List<List<Unit>>> sources = new ArrayList<>();
        List<Card> cards = new ArrayList<>(payer.getCardsIn(ZoneType.Battlefield));
        cards.addAll(payer.getCardsIn(ZoneType.Hand));
        for (Card card : cards) {
            if (hostTaps && card == host) {
                continue;
            }
            List<List<Unit>> options = new ArrayList<>();
            for (SpellAbility ability : card.getManaAbilities()) {
                List<Unit> made = made(ability, candidate, payer);
                if (!made.isEmpty()) {
                    options.add(made);
                }
            }
            if (!options.isEmpty()) {
                sources.add(options);
            }
        }
        return sources;
    }

    /** The mana one activation of {@code ability} would make for {@code candidate}, or none when it cannot be activated. */
    private static List<Unit> made(SpellAbility ability, SpellAbility candidate, Player payer) {
        AbilityManaPart part = ability.getManaPart();
        if (part == null || part.isSpecialMana()) {
            return List.of();
        }
        ability.setActivatingPlayer(payer);
        Cost own = ability.getPayCosts();
        if (own != null && own.hasManaCost()) {
            return List.of(); // a filter: its mana comes from other sources
        }
        if (!ability.canPlay() || !ability.checkRestrictions(payer)
                || !CostPayment.canPayAdditionalCosts(own, ability, false, payer)
                || !ability.metConditions() || !part.meetsManaRestrictions(candidate)) {
            return List.of();
        }
        boolean snow = part.getSourceCard() != null && part.getSourceCard().isSnow();
        List<Unit> units = new ArrayList<>();
        if (part.isAnyMana() || part.isComboMana() || ability.getApi() == ApiType.ManaReflected) {
            byte colors = 0;
            for (byte color : ManaAtom.MANATYPES) {
                if (ability.isManaAbilityFor(candidate, color)) {
                    colors |= color;
                }
            }
            if (colors != 0) {
                for (int i = ability.amountOfManaGenerated(false); i > 0; i--) {
                    units.add(new Unit(colors, snow));
                }
            }
            return units;
        }
        int amount = ability.hasParam("Amount")
                ? forge.game.ability.AbilityUtils.calculateAmount(
                        ability.getHostCard(), ability.getParam("Amount"), ability)
                : 1;
        for (int i = 0; i < amount; i++) {
            for (String token : part.mana(ability).split(" ")) {
                byte color = token.equals("C") ? (byte) ManaAtom.COLORLESS
                        : MagicColor.fromName(token);
                if (color != 0 && ability.isManaAbilityFor(candidate, color)) {
                    units.add(new Unit(color, snow));
                }
            }
        }
        return units;
    }
}
