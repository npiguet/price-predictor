package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.GameObject;
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
 *
 * <p>A card target is named {@code E<id>} while the game's card for it is
 * still the object that was targeted; one that has ceased to exist (a token
 * that died) or become a new object (a creature flickered) is the card as it
 * last existed, {@code L<id>}, Forge's own test for a target gone illegal
 * ({@code Card.equalsWithGameTimestamp}, {@code MagicStack.hasFizzled}). A
 * spell or ability on the stack is named by its host card's {@code E<id>},
 * and listed among {@link #onStack} too, which is how it is told from that
 * card targeted itself.
 */
final class ChainTargets {

    /** The root's links' targets, before any mode clone. */
    private final List<String> root;
    /** Each chosen mode's targets, in chain order. */
    private final List<ModeTargets> modes;
    /** How many targets the chain holds, counted per link. */
    private final int count;
    /** The cards the targets name that still are those objects, hosts of stack targets included. */
    private final List<Card> cards;
    /** The targeted cards that ceased to exist or became new objects, as targeted. */
    private final List<Card> lastKnown;
    /** The refs that name a spell or ability on the stack, by its host card. */
    private final Set<String> onStack;
    /** Each link with the targets of it that have a ref, root first. */
    private final List<Link> links;

    private record ModeTargets(Integer option, List<String> targets) {
    }

    /** One link of the chain and its targets that have a ref, in the order chosen. */
    record Link(SpellAbility ability, List<Target> targets) {
    }

    /** One target of a link: the object Forge holds, and the ref it is written as. */
    record Target(GameObject object, String ref) {
    }

    private ChainTargets(Reading reading) {
        this.root = List.copyOf(reading.root);
        this.modes = List.copyOf(reading.modes);
        this.count = reading.count;
        this.cards = List.copyOf(reading.cards);
        this.lastKnown = List.copyOf(reading.lastKnown);
        this.onStack = Set.copyOf(reading.onStack);
        this.links = List.copyOf(reading.links);
    }

    /** The targets the chain holds now. */
    static ChainTargets now(SpellAbility item) {
        List<Collection<?>> links = new ArrayList<>();
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            links.add(link.getTargets() == null ? List.of() : List.copyOf(link.getTargets()));
        }
        // Targets an item holds now passed Forge's check, or were just chosen.
        return of(item, links, false);
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
        return chain == links.size() ? of(item, links, true) : null;
    }

    private static ChainTargets of(
            SpellAbility item, List<Collection<?>> links, boolean asCast) {
        Game game = item.getHostCard() == null ? null : item.getHostCard().getGame();
        Reading reading = new Reading(game, asCast);
        int index = 0;
        for (SpellAbility link = item; link != null; link = link.getSubAbility(), index++) {
            reading.link(link, links.get(index));
        }
        return new ChainTargets(reading);
    }

    /** The chain read link by link, root first: what a {@link ChainTargets} holds. */
    private static final class Reading {
        private final Game game;
        private final boolean asCast;
        private final List<String> root = new ArrayList<>();
        private final List<ModeTargets> modes = new ArrayList<>();
        private final List<Card> cards = new ArrayList<>();
        private final List<Card> lastKnown = new ArrayList<>();
        private final Set<String> onStack = new LinkedHashSet<>();
        private final List<Link> links = new ArrayList<>();
        /** Where the current link's refs go: the root's, or its mode's. */
        private List<String> into = root;
        private int count;

        Reading(Game game, boolean asCast) {
            this.game = game;
            this.asCast = asCast;
        }

        void link(SpellAbility link, Collection<?> targets) {
            CharmModes.ModeStart start = CharmModes.modeStartOf(link);
            if (start != null) {
                into = new ArrayList<>();
                modes.add(new ModeTargets(start.option(), into));
            }
            List<Target> named = new ArrayList<>();
            for (Object target : targets) {
                count++;
                String ref = nameOf(target);
                if (ref != null) {
                    into.add(ref);
                    named.add(new Target((GameObject) target, ref));
                }
            }
            links.add(new Link(link, List.copyOf(named)));
        }

        /** A target's ref, the card or host it holds noted; null for anything else. */
        private String nameOf(Object target) {
            if (target instanceof Card card) {
                return cardRef(card);
            }
            if (target instanceof SpellAbility ability && ability.getHostCard() != null) {
                return stackRef(ability.getHostCard());
            }
            return target instanceof Player player ? Refs.of(player) : null;
        }

        private String cardRef(Card card) {
            if (asCast && isGone(game, card)) {
                lastKnown.add(card);
                return lastKnownRef(card);
            }
            cards.add(card);
            return Refs.of(card);
        }

        private String stackRef(Card host) {
            String ref = Refs.of(host);
            onStack.add(ref);
            if (game == null || game.getCardState(host, null) != null) {
                cards.add(host);
            }
            return ref;
        }
    }

    /**
     * Whether a targeted card is no longer the game's object for it: it has
     * ceased to exist, or it moved and the game's card is a new object.
     */
    static boolean isGone(Game game, Card target) {
        if (game == null) {
            return false;
        }
        Card current = game.getCardState(target, null);
        return current == null || !current.equalsWithGameTimestamp(target);
    }

    /** The ref of a card as it last existed: {@code L<id>}. */
    static String lastKnownRef(Card card) {
        return "L" + card.getId();
    }

    /**
     * A target as a ref, a game's state aside: a card or a player as itself,
     * a spell or an ability on the stack by its host card.
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

    /**
     * The cards a snapshot holds for these targets: each targeted card that
     * is still that object, and the host card of each targeted spell or
     * ability, root first.
     */
    List<Card> cards() {
        return cards;
    }

    /** The targeted cards that ceased to exist or became new objects, as they were targeted. */
    List<Card> lastKnown() {
        return lastKnown;
    }

    /** The refs that name a spell or ability on the stack by its host card. */
    Set<String> onStack() {
        return onStack;
    }

    /**
     * Each link of the chain, root first, with its targets that have a ref:
     * where a target's divided share is read ({@link StackExtras}).
     */
    List<Link> links() {
        return links;
    }

    /** The cards a snapshot holds for a chain's targets now, root first. */
    static List<Card> cardsOf(SpellAbility item) {
        return now(item).cards();
    }

    /**
     * Of {@code written}, the refs that name a spell or ability on the
     * stack in either reading, in the order written.
     */
    static List<String> onStackOf(List<String> written, ChainTargets... readings) {
        Set<String> stack = new LinkedHashSet<>();
        for (ChainTargets reading : readings) {
            if (reading != null) {
                stack.addAll(reading.onStack());
            }
        }
        List<String> named = new ArrayList<>();
        for (String ref : written) {
            if (stack.contains(ref) && !named.contains(ref)) {
                named.add(ref);
            }
        }
        return named;
    }
}
