package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * A stack item's targets over its whole sub-ability chain, root first, split
 * into the chosen modes they belong to.
 *
 * <p>Forge keeps a target on the clause that chose it: a charm's root targets
 * nothing, and its modes' targets sit on the mode clones {@code
 * CharmEffect.chainAbilities} appended to it; Arc Trail's second target sits
 * on its {@code SubAbility$}. Reading the root alone names none of them.
 *
 * <p>A link belongs to the last mode clone at or before it in the chain
 * ({@link CharmModes#isModeClone}); the links before the first clone are the
 * root's, which every mode's half shares.
 */
final class ChainTargets {

    /** The root's links' targets, before any mode clone. */
    private final List<String> root;
    /** Each chosen mode's targets, in chain order. */
    private final List<ModeTargets> modes;
    /** How many targets the chain holds, counted per link. */
    private final int count;
    /** The cards among the targets, every link's, root first. */
    private final List<Card> cards;

    private record ModeTargets(Integer option, List<String> targets) {
    }

    private ChainTargets(List<String> root, List<ModeTargets> modes, int count, List<Card> cards) {
        this.root = root;
        this.modes = modes;
        this.count = count;
        this.cards = cards;
    }

    /** The targets the chain holds now. */
    static ChainTargets now(SpellAbility item) {
        List<Collection<?>> links = new ArrayList<>();
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            links.add(link.getTargets() == null ? List.of() : List.copyOf(link.getTargets()));
        }
        return of(item, links);
    }

    /**
     * The targets {@code perLink} lists, one list per link of {@code item}'s
     * chain, root first, as the resolution hook's {@code targetsAtCast} gives
     * them; null when the hook gave none (it could not copy them) or a list
     * that does not match the chain.
     */
    static ChainTargets atCast(SpellAbility item, Object perLink) {
        if (item == null || !(perLink instanceof List<?> lists)) {
            return null;
        }
        List<Collection<?>> links = new ArrayList<>();
        for (Object list : lists) {
            if (!(list instanceof Collection<?> targets)) {
                return null;
            }
            links.add(targets);
        }
        int chain = 0;
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            chain++;
        }
        return chain == links.size() ? of(item, links) : null;
    }

    private static ChainTargets of(SpellAbility item, List<Collection<?>> links) {
        List<String> root = new ArrayList<>();
        List<ModeTargets> modes = new ArrayList<>();
        List<String> into = root;
        List<Card> cards = new ArrayList<>();
        int count = 0;
        int index = 0;
        for (SpellAbility link = item; link != null; link = link.getSubAbility(), index++) {
            CharmModes.ModeStart start = CharmModes.modeStartOf(link);
            if (start != null) {
                into = new ArrayList<>();
                modes.add(new ModeTargets(start.option(), into));
            }
            for (Object target : links.get(index)) {
                count++;
                if (target instanceof Card card) {
                    cards.add(card);
                }
                String ref = refOf(target);
                if (ref != null) {
                    into.add(ref);
                }
            }
        }
        return new ChainTargets(root, modes, count, List.copyOf(cards));
    }

    /**
     * A target as a ref: a card or a player as itself, a spell or an ability
     * on the stack by its host card, which is the entity the snapshot holds
     * for it.
     */
    static String refOf(Object target) {
        if (target instanceof SpellAbility ability) {
            return Refs.ofNullable(ability.getHostCard());
        }
        if (target instanceof Card || target instanceof Player) {
            return Refs.ofAny(target).orElse(null);
        }
        return null;
    }

    /** Every link's targets, root first, each ref once. */
    List<String> all() {
        Set<String> all = new LinkedHashSet<>(root);
        for (ModeTargets mode : modes) {
            all.addAll(mode.targets());
        }
        return List.copyOf(all);
    }

    /** How many targets the chain holds, a target two links share counted twice. */
    int count() {
        return count;
    }

    /**
     * The targets of one effect half: the root's and its own mode's, or
     * every link's for a half that is no mode's (the item is not modal, or
     * its modes cannot be told apart).
     *
     * @param half   the half's position among the item's halves
     * @param halves how many halves the item writes
     * @param option the half's {@code Choices$} position, or null
     */
    List<String> ofHalf(int half, int halves, Integer option) {
        if (modes.isEmpty()) {
            return all();
        }
        ModeTargets mode = null;
        if (halves == modes.size()) {
            mode = modes.get(half);
        } else if (option != null) {
            for (ModeTargets each : modes) {
                if (option.equals(each.option())) {
                    mode = each;
                    break;
                }
            }
        }
        if (mode == null) {
            return all();
        }
        Set<String> targets = new LinkedHashSet<>(root);
        targets.addAll(mode.targets());
        return List.copyOf(targets);
    }

    /** The cards among the targets, every link's, root first. */
    List<Card> cards() {
        return cards;
    }

    /** The cards among a chain's targets now, every link's, root first. */
    static List<Card> cardsOf(SpellAbility item) {
        return now(item).cards();
    }
}
