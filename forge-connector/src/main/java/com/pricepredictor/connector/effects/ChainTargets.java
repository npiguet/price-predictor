package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.GameObject;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
 *
 * <p>Each target is read once, as a {@link Target}: what the refs list, what
 * a snapshot's {@code targets} links name and index, and what the stack
 * extras share out all come from that one reading.
 */
final class ChainTargets {

    /** The root's links' targets, before any mode clone. */
    private final List<Target> root;
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
    /** The targets a record's links name: every link's, or one half's; each ref once. */
    private final List<Target> linked;

    private record ModeTargets(Integer option, List<Target> targets) {
    }

    /** One link of the chain and its targets that have a ref, in the order chosen. */
    record Link(SpellAbility ability, List<Target> targets) {
    }

    /**
     * One effect half of an item: its position among the item's halves and
     * the {@code Choices$} position of its mode, null where it has none.
     *
     * @param index  the half's position among the item's halves
     * @param count  how many halves the item writes
     * @param option the half's {@code Choices$} position, or null
     */
    record Half(int index, int count, Integer option) {
    }

    /** What a target is, which says how a link names it. */
    enum Kind {
        /** A card that is still the object targeted: {@code E<id>}. */
        CARD,
        /** A card that ceased to exist or became a new object: {@code L<id>}. */
        LAST_KNOWN,
        /** A player: {@code P<id>}. */
        PLAYER,
        /** A spell on the stack: its card's {@code E<id>}, its card the link's end. */
        SPELL,
        /** An ability on the stack: its host card's {@code E<id>}, its line the link's end. */
        ABILITY
    }

    /**
     * One target of a link: the object Forge holds, the ref the refs block
     * writes it as, what it is, and its share of the link's divided total
     * ({@link DividedAllocation}), null where the link divides nothing.
     */
    record Target(GameObject object, String ref, Kind kind, Integer share) {

        /** This target with its divided share on the link that chose it, null where none. */
        Target withShare(Integer divided) {
            return new Target(object, ref, kind, divided);
        }

        /**
         * The card a snapshot holds for it: the targeted card itself, the
         * host of a targeted spell or ability; null for a player.
         */
        Card card() {
            if (object instanceof SpellAbility ability) {
                return ability.getHostCard();
            }
            return object instanceof Card card ? card : null;
        }
    }

    private ChainTargets(Reading reading) {
        this.root = List.copyOf(reading.root);
        this.modes = List.copyOf(reading.modes);
        this.count = reading.count;
        this.cards = List.copyOf(reading.cards);
        this.lastKnown = List.copyOf(reading.lastKnown);
        this.onStack = Set.copyOf(reading.onStack);
        this.links = List.copyOf(reading.links);
        this.linked = distinct(root, modes);
    }

    /** {@code whole} with the targets its links name narrowed to {@code linked}. */
    private ChainTargets(ChainTargets whole, List<Target> linked) {
        this.root = whole.root;
        this.modes = whole.modes;
        this.count = whole.count;
        this.cards = whole.cards;
        this.lastKnown = whole.lastKnown;
        this.onStack = whole.onStack;
        this.links = whole.links;
        this.linked = linked;
    }

    /** The targets the chain holds now. */
    static ChainTargets now(SpellAbility item) {
        List<Collection<?>> links = new ArrayList<>();
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            links.add(link.getTargets() == null ? List.of() : List.copyOf(link.getTargets()));
        }
        // Targets an item holds now passed Forge's check, or were just chosen.
        return of(item, links, false, DividedAllocation.of(item));
    }

    /**
     * The targets {@code perLink} lists, one list per link of {@code item}'s
     * chain, root first, as the resolution hook's {@code targetsAtCast} gives
     * them; null when the hook gave none (it could not copy them) or a list
     * that does not match the chain.
     *
     * @param allocation the divided shares read when the item went on the
     *                   stack, or null where none was read
     */
    static ChainTargets atCast(SpellAbility item, Object perLink, DividedAllocation allocation) {
        List<Collection<?>> links = item == null ? null : listsOf(perLink);
        return links != null && chainLength(item) == links.size()
                ? of(item, links, true, allocation == null ? DividedAllocation.NONE : allocation)
                : null;
    }

    /** {@code perLink} as one collection per link, or null where it is no such list. */
    private static List<Collection<?>> listsOf(Object perLink) {
        if (!(perLink instanceof List<?> lists)) {
            return null;
        }
        List<Collection<?>> links = new ArrayList<>();
        for (Object list : lists) {
            if (!(list instanceof Collection<?> targets)) {
                return null;
            }
            links.add(targets);
        }
        return links;
    }

    private static int chainLength(SpellAbility item) {
        int chain = 0;
        for (SpellAbility link = item; link != null; link = link.getSubAbility()) {
            chain++;
        }
        return chain;
    }

    private static ChainTargets of(
            SpellAbility item, List<Collection<?>> links, boolean asCast,
            DividedAllocation allocation) {
        Game game = item.getHostCard() == null ? null : item.getHostCard().getGame();
        Reading reading = new Reading(game, asCast);
        int index = 0;
        for (SpellAbility link = item; link != null; link = link.getSubAbility(), index++) {
            reading.link(link, links.get(index), allocation.ofLink(index));
        }
        return new ChainTargets(reading);
    }

    /** The chain read link by link, root first: what a {@link ChainTargets} holds. */
    private static final class Reading {
        private final Game game;
        private final boolean asCast;
        private final List<Target> root = new ArrayList<>();
        private final List<ModeTargets> modes = new ArrayList<>();
        private final List<Card> cards = new ArrayList<>();
        private final List<Card> lastKnown = new ArrayList<>();
        private final Set<String> onStack = new LinkedHashSet<>();
        private final List<Link> links = new ArrayList<>();
        /** Where the current link's targets go: the root's, or its mode's. */
        private List<Target> into = root;
        private int count;

        Reading(Game game, boolean asCast) {
            this.game = game;
            this.asCast = asCast;
        }

        void link(SpellAbility link, Collection<?> targets, Map<GameObject, Integer> shares) {
            CharmModes.ModeStart start = CharmModes.modeStartOf(link);
            if (start != null) {
                into = new ArrayList<>();
                modes.add(new ModeTargets(start.option(), into));
            }
            List<Target> named = new ArrayList<>();
            for (Object target : targets) {
                count++;
                Target read = targetOf(target, asCast ? game : null);
                if (read != null) {
                    read = read.withShare(shares.get(read.object()));
                    note(read);
                    into.add(read);
                    named.add(read);
                }
            }
            links.add(new Link(link, List.copyOf(named)));
        }

        /** The card a snapshot holds for {@code target}, or the one it last was, noted. */
        private void note(Target target) {
            switch (target.kind()) {
                case LAST_KNOWN -> lastKnown.add(target.card());
                case CARD -> cards.add(target.card());
                case SPELL, ABILITY -> noteStack(target);
                default -> {
                    // A player is no card.
                }
            }
        }

        private void noteStack(Target target) {
            onStack.add(target.ref());
            if (game == null || game.getCardState(target.card(), null) != null) {
                cards.add(target.card());
            }
        }
    }

    /**
     * A target as one {@link Target}, or null for anything no ref names: a
     * card or a player as itself, a spell or an ability on the stack by its
     * host card. A card is {@link Kind#LAST_KNOWN} when {@code asCastIn}, the
     * game of targets as cast, no longer holds it as the object targeted;
     * null reads every card as the object it is.
     */
    static Target targetOf(Object target, Game asCastIn) {
        if (target instanceof Card card) {
            return isGone(asCastIn, card)
                    ? new Target(card, lastKnownRef(card), Kind.LAST_KNOWN, null)
                    : new Target(card, Refs.of(card), Kind.CARD, null);
        }
        if (target instanceof SpellAbility ability && ability.getHostCard() != null) {
            return new Target(ability, Refs.of(ability.getHostCard()),
                    ability.isSpell() ? Kind.SPELL : Kind.ABILITY, null);
        }
        return target instanceof Player player ? new Target(player, Refs.of(player), Kind.PLAYER, null) : null;
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
        Target read = targetOf(target, null);
        return read == null ? null : read.ref();
    }

    /** Every link's targets, root first, each ref once. */
    List<String> all() {
        return refsOf(distinct(root, modes));
    }

    /**
     * The targets a record's {@code targets} links name, each ref once, a
     * link's {@code index} its position here: every link's, root first, or
     * one half's ({@link #half}), in the order of the refs the record lists
     * for them.
     */
    List<Target> linked() {
        return linked;
    }

    /** The first target of each ref, root first. */
    private static List<Target> distinct(List<Target> root, List<ModeTargets> modes) {
        Map<String, Target> byRef = new LinkedHashMap<>();
        root.forEach(target -> byRef.putIfAbsent(target.ref(), target));
        for (ModeTargets mode : modes) {
            mode.targets().forEach(target -> byRef.putIfAbsent(target.ref(), target));
        }
        return List.copyOf(byRef.values());
    }

    private static List<String> refsOf(List<Target> targets) {
        return targets.stream().map(Target::ref).toList();
    }

    /** How many targets the chain holds, a target two links share counted twice. */
    int count() {
        return count;
    }

    /**
     * The targets of one effect half: the root's and its own mode's, or
     * every link's for a half that is no mode's (the item is not modal, or
     * its modes cannot be told apart).
     */
    List<String> ofHalf(Half half) {
        return refsOf(half(half).linked);
    }

    /**
     * These targets with the links narrowed to one effect half's ({@link
     * #ofHalf}): what that half's board links, in the order its refs list
     * them.
     */
    ChainTargets half(Half half) {
        ModeTargets mode = half.count() == modes.size() && half.index() < half.count()
                ? modes.get(half.index()) : modeOf(half.option());
        return new ChainTargets(this, distinct(root, mode == null ? modes : List.of(mode)));
    }

    /**
     * {@link #half} for a mode's half as it begins, the {@code begun}-th
     * mode begun, counted from 0: a resolution writes one half per mode
     * begun, each chosen mode beginning in chain order, so its half is the
     * {@code begun}-th of as many halves as chosen modes.
     */
    ChainTargets beginningMode(int begun, Integer option) {
        return half(new Half(begun, modes.size(), option));
    }

    /** The first chosen mode at {@code option}, or null where none is or the option is unknown. */
    private ModeTargets modeOf(Integer option) {
        if (option == null) {
            return null;
        }
        return modes.stream().filter(mode -> option.equals(mode.option())).findFirst().orElse(null);
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
     * where the extras read each target's divided share ({@link StackExtras}).
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
        Arrays.stream(readings).filter(Objects::nonNull)
                .forEach(reading -> stack.addAll(reading.onStack()));
        return written.stream().filter(stack::contains).distinct().toList();
    }
}
