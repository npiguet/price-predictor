package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.GameObject;
import forge.game.StaticEffect;
import forge.game.card.Card;
import forge.game.card.CardCloneStates;
import forge.game.combat.AttackingBand;
import forge.game.combat.Combat;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.spellability.SpellAbilityStackInstance;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Set;

/**
 * A snapshot's {@code links}: the typed links between its objects that the
 * relation-aware trunk reads (FR-026, FR-050, root § 10.3), each read off the
 * Forge source its family's table row names.
 *
 * <p>A link is {@code {from, to, type, index?}}, written once in its family's
 * forward direction ({@link LinkFamily}); {@code index} is a target's
 * position, on {@code targets} only. An end is a ref ({@code E<id>},
 * {@code P<id>}, a last-known copy's {@code L<id>}, a floating effect's
 * {@code F<timestamp>}, a delayed trigger's {@code D<id>}) or a line,
 * {@code {entity, key}}: a link an ability made ends at that ability's line,
 * not at its card. Where Forge keeps no record of which ability made a link
 * (an imprint, a remembered object, a chosen object, a control grant) the
 * end is the card.
 *
 * <p>The links are read from the snapshot's cards and the game's combat,
 * stack and command zone; an end the snapshot carries no entity for (a card
 * a cipher encoded in exile, a creature damaged and since exiled) is still
 * named, for the reader to drop. Families the reader builds from its own
 * layout are never written ({@link LinkFamily#collected}).
 *
 * <p>Links are listed by family in the table's order, each once.
 */
final class SnapshotLinks {

    private final Game game;
    private final Map<LinkFamily, Set<String>> byFamily = new EnumMap<>(LinkFamily.class);

    private SnapshotLinks(Game game) {
        this.game = game;
    }

    /**
     * The links of one snapshot.
     *
     * @param cards         every card entity the snapshot renders, in order
     * @param tiers         the cards of the four tiers, whose rows the global
     *                      block's timestamped floating effects are read from
     * @param acting        the acting spell ability, whose targets are links
     *                      even before it is on the stack; or null
     * @param actingTargets whether {@code acting}'s targets are its own: false
     *                      for a playability candidate, whose persistent
     *                      ability may hold an earlier evaluation's
     * @param pending       the event a trait is about to handle, or null
     * @param withoutStatic the static a {@code continuous} record's board
     *                      removes, or null; such a board carries no
     *                      {@code affects} from a static, whose answer it is
     * @param floating      the clause notes the floating effects are read by
     */
    static String render(Game game, Collection<Card> cards, Collection<Card> tiers,
                         SpellAbility acting, boolean actingTargets, EffectEvent pending,
                         Long withoutStatic, FloatingEffects floating) {
        SnapshotLinks links = new SnapshotLinks(game);
        links.cardLinks(cards, withoutStatic == null);
        links.combat(game.getCombat());
        links.stack(acting, actingTargets);
        links.commandZone(cards, tiers, floating);
        if (pending != null && pending.lastKnown() != null) {
            Card lastKnown = pending.lastKnown();
            links.add(LinkFamily.LAST_KNOWN_STATE_OF,
                    Json.string("L" + lastKnown.getId()), card(lastKnown));
        }
        List<String> all = new ArrayList<>();
        for (LinkFamily family : LinkFamily.values()) {
            all.addAll(links.byFamily.getOrDefault(family, Set.of()));
        }
        return "[" + String.join(",", all) + "]";
    }

    // ── the families ────────────────────────────────────────────────────

    /** Every family a card's own fields hold. */
    private void cardLinks(Collection<Card> cards, boolean withStaticEffects) {
        for (Card card : cards) {
            mergedWith(card);
            add(LinkFamily.ATTACHED_TO, card(card), card(card.getAttachedTo()));
            add(LinkFamily.ATTACHED_TO, card(card), player(card.getPlayerAttachedTo()));
            for (Card attacker : card.getMustBlockCards()) {
                add(LinkFamily.MUST_BLOCK, card(card), card(attacker));
            }
            for (Player goader : card.getGoaded()) {
                add(LinkFamily.GOADED_BY, card(card), player(goader));
            }
            if (withStaticEffects) {
                affects(card);
            }
            linkedByText(card);
            boundTo(card);
            add(LinkFamily.CONTROLS, player(card.getController()), card(card));
            add(LinkFamily.OWNS, player(card.getOwner()), card(card));
            for (Card given : card.getGainControlTargets()) {
                add(LinkFamily.CONTROL_GRANTED_BY, card(given), card(card));
            }
            copyOf(card);
            pair(LinkFamily.PAIRED_WITH, card, card.getPairedWith());
            add(LinkFamily.PROTECTS, player(card.getProtectingPlayer()), card(card));
            for (Card chosen : card.getChosenCards()) {
                add(LinkFamily.CHOSE, card(card), card(chosen));
            }
            add(LinkFamily.CHOSE, card(card), player(card.getChosenPlayer()));
            add(LinkFamily.CHOSE, card(card), player(card.getPromisedGift()));
            thisTurn(card);
            if (card.getType().isDungeon() && card.isInZone(ZoneType.Command)) {
                add(LinkFamily.IN_DUNGEON, card(card), player(card.getOwner()));
            }
        }
    }

    /** A mutated permanent's other cards, and the card a melded permanent melded with. */
    private void mergedWith(Card card) {
        if (card.isInPlay() && card.isMutated()) {
            for (Card component : card.getMergedCards()) {
                if (component.getId() != card.getId()) {
                    add(LinkFamily.MERGED_WITH, card(component), card(card));
                }
            }
        }
        if (card.getMeldedWith() != null) {
            add(LinkFamily.MERGED_WITH, card(card.getMeldedWith()), card(card));
        }
    }

    /** Attacks, blocks and bands, from the combat Forge is running. */
    private void combat(Combat combat) {
        if (combat == null) {
            return;
        }
        for (Card attacker : combat.getAttackers()) {
            add(LinkFamily.ATTACKS, card(attacker), entity(combat.getDefenderByAttacker(attacker)));
        }
        for (Card blocker : combat.getAllBlockers()) {
            for (Card attacker : combat.getAttackersBlockedBy(blocker)) {
                add(LinkFamily.BLOCKS, card(blocker), card(attacker));
            }
        }
        for (AttackingBand band : combat.getAttackingBands()) {
            List<Card> members = new ArrayList<>(band.getAttackers());
            for (int i = 0; i < members.size(); i++) {
                for (int j = i + 1; j < members.size(); j++) {
                    pair(LinkFamily.IN_BAND_WITH, members.get(i), members.get(j));
                }
            }
        }
    }

    /**
     * Each stack object's targets, by index across its chain (the root's
     * first, then each sub-ability's), its source and what triggered it; the
     * acting ability's too where it is not on the stack yet, its targets only
     * when they are its own.
     */
    private void stack(SpellAbility acting, boolean actingTargets) {
        Set<SpellAbility> objects = Collections.newSetFromMap(new IdentityHashMap<>());
        List<SpellAbility> ordered = new ArrayList<>();
        for (SpellAbilityStackInstance instance : game.getStack()) {
            SpellAbility ability = instance.getSpellAbility();
            if (ability != null && objects.add(ability)) {
                ordered.add(ability);
            }
        }
        // A candidate off the stack: its persistent ability's targets are an
        // earlier evaluation's, not chosen for this record.
        boolean staleTargets = acting != null && !actingTargets && !objects.contains(acting);
        if (acting != null && objects.add(acting)) {
            ordered.add(acting);
        }
        for (SpellAbility ability : ordered) {
            Card host = ability.getHostCard();
            String line = line(host, ability);
            int index = 0;
            boolean targets = !(staleTargets && ability == acting);
            for (SpellAbility clause = ability; targets && clause != null;
                    clause = clause.getSubAbility()) {
                if (clause.getTargets() == null) {
                    continue;
                }
                for (GameObject target : clause.getTargets()) {
                    addTarget(line, target(target), index++);
                }
            }
            if (line != null && !line.equals(card(host))) {
                add(LinkFamily.SOURCE_OF, card(host), line);
            }
            if (ability.isTrigger() && ability.getTriggeringObjects() != null) {
                for (Object about : ability.getTriggeringObjects().values()) {
                    for (String to : entities(about)) {
                        add(LinkFamily.TRIGGERED_BY, line, to);
                    }
                }
            }
        }
    }

    /**
     * The permanents, then the players, each static applies to, as Forge's
     * own static effect lists them (the same list as the line's
     * {@code affected}): Exploration's line affects its controller.
     */
    private void affects(Card card) {
        for (StaticAbility stat : card.getStaticAbilities()) {
            StaticEffect effect = ForgeFields.staticEffectOf(game, stat);
            if (effect != null) {
                for (String affected : Refs.affectedBy(effect)) {
                    add(LinkFamily.AFFECTS, line(card, stat), Json.string(affected));
                }
            }
        }
    }

    /**
     * Rule 607's links, from the line that made them: the cards a permanent
     * exiled, imprinted or exiled until it leaves, and its lasting remembered
     * objects (C8).
     */
    private void linkedByText(Card card) {
        for (Card exiled : card.getExiledCards()) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card, exiled), card(exiled));
        }
        if (card.getExiledWith() != null) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card.getExiledWith(), card), card(card));
        }
        for (Card imprinted : card.getImprintedCards()) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card, imprinted), card(imprinted));
        }
        for (Card held : card.getUntilLeavesBattlefield()) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card, held), card(held));
        }
        if (remembersLastingly(card)) {
            for (Object remembered : card.getRemembered()) {
                if (remembered instanceof Card || remembered instanceof Player) {
                    add(LinkFamily.LINKED_BY_TEXT, maker(card, remembered), entity(remembered));
                }
            }
        }
    }

    /**
     * Whether the card's remembered list holds only what outlived its
     * resolutions (C8): no ability of the card is on the stack or resolving,
     * so every {@code ClearRemembered$} that would have emptied it has run.
     */
    private boolean remembersLastingly(Card card) {
        if (game.getStack().isResolving(card)) {
            return false;
        }
        for (SpellAbilityStackInstance instance : game.getStack()) {
            Card host = instance.getSourceCard();
            if (host != null && host.getId() == card.getId()) {
                return false;
            }
        }
        return true;
    }

    /** Cipher and haunt: the exiled card to the permanent it acts through. */
    private void boundTo(Card card) {
        add(LinkFamily.BOUND_TO, card(card), card(card.getEncodingCard()));
        for (Card encoded : card.getEncodedCards()) {
            add(LinkFamily.BOUND_TO, card(encoded), card(card));
        }
        add(LinkFamily.BOUND_TO, card(card), card(card.getHaunting()));
        for (Card haunter : card.getHauntedBy()) {
            add(LinkFamily.BOUND_TO, card(haunter), card(card));
        }
    }

    /**
     * What a copy copies: the permanent or spell a copy was made from
     * ({@code getCopiedPermanent}), and the card a copy effect such as a
     * Clone's copies (its latest clone state's origin).
     */
    private void copyOf(Card card) {
        Card copied = card.getCopiedPermanent();
        if (copied != null && copied.getId() != card.getId()) {
            add(LinkFamily.COPY_OF, card(card), card(copied));
        }
        if (card.isCloned() && card.getCloneStates() instanceof NavigableMap<Long, CardCloneStates> states
                && !states.isEmpty()) {
            Card origin = states.lastEntry().getValue().getOrigin();
            if (origin != null && origin.getId() != card.getId()) {
                add(LinkFamily.COPY_OF, card(card), card(origin));
            }
        }
    }

    /** This turn's damage, blocks, crews and saddles. */
    private void thisTurn(Card card) {
        if (card.getDamageHistory() != null) {
            // The source's own history: Forge registers each damage there, and
            // keys the (source, target) pair by the damage's identity.
            for (Pair<Integer, Boolean> damage : card.getDamageHistory().getAllDmgInstances()) {
                Pair<Card, GameEntity> sourceToTarget = game.getDamageLKI(damage);
                if (sourceToTarget != null) {
                    add(LinkFamily.DAMAGED_THIS_TURN, card(card),
                            entity(sourceToTarget.getRight()));
                }
            }
        }
        for (Card attacker : card.getBlockedThisTurn()) {
            add(LinkFamily.FOUGHT_THIS_TURN, card(card), card(attacker));
        }
        for (Card blocker : card.getBlockedByThisTurn()) {
            add(LinkFamily.FOUGHT_THIS_TURN, card(blocker), card(card));
        }
        for (Card crew : card.getCrewedByThisTurn()) {
            add(LinkFamily.CREWED_BY, card(card), card(crew));
        }
        if (card.getSaddledByThisTurn() != null) {
            for (Card rider : card.getSaddledByThisTurn()) {
                add(LinkFamily.CREWED_BY, card(card), card(rider));
            }
        }
    }

    /**
     * Command-zone objects: what made each effect card, emblem, timestamped
     * floating effect and delayed trigger; what a floating effect applies to;
     * what a delayed trigger waits on.
     */
    private void commandZone(Collection<Card> cards, Collection<Card> tiers,
                             FloatingEffects floating) {
        for (Card card : cards) {
            if (card.isInZone(ZoneType.Command) && card.isImmutable()
                    && !card.getType().isDungeon()) {
                SpellAbility source = card.getEffectSourceAbility();
                add(LinkFamily.CREATED_BY, card(card), source != null
                        ? line(source.getHostCard(), source) : card(card.getEffectSource()));
            }
        }
        for (FloatingEffects.Timestamped row : floating.timestamped(tiers)) {
            for (String affected : row.affected()) {
                add(LinkFamily.AFFECTS, Json.string(row.id()), Json.string(affected));
            }
            add(LinkFamily.CREATED_BY, Json.string(row.id()), lineOf(row.host(), row.line()));
        }
        for (FloatingEffects.Pending pending : FloatingEffects.pendingDelayedTriggers(game)) {
            String id = Json.string(pending.id());
            add(LinkFamily.CREATED_BY, id,
                    line(pending.trigger().getHostCard(), pending.trigger()));
            for (String held : pending.remembered()) {
                add(LinkFamily.PENDING_ON, id, Json.string(held));
            }
        }
    }

    // ── ends ────────────────────────────────────────────────────────────

    private static String card(Card card) {
        return card == null ? null : Json.string(Refs.of(card));
    }

    private static String player(Player player) {
        return player == null ? null : Json.string(Refs.of(player));
    }

    /** A card's or player's ref, or null for anything else. */
    private static String entity(Object value) {
        return Refs.ofAny(value).map(Json::string).orElse(null);
    }

    /** The refs of every card and player a triggering object holds. */
    private static List<String> entities(Object value) {
        List<String> refs = new ArrayList<>();
        if (value instanceof Iterable<?> many) {
            for (Object each : many) {
                Refs.ofAny(each).ifPresent(ref -> refs.add(Json.string(ref)));
            }
        } else {
            Refs.ofAny(value).ifPresent(ref -> refs.add(Json.string(ref)));
        }
        return refs;
    }

    /**
     * A target: a card or a player by its ref, a spell by its card, and an
     * ability on the stack by its line.
     */
    private static String target(GameObject target) {
        if (target instanceof SpellAbility ability) {
            return ability.isSpell() ? card(ability.getHostCard())
                    : line(ability.getHostCard(), ability);
        }
        return entity(target);
    }

    /** A trait's line on its host, or the host where the trait keys to no line. */
    static String line(Card host, CardTraitBase trait) {
        if (host == null) {
            return null;
        }
        return lineOf(Refs.of(host), trait == null ? null : ProvenanceResolver.of(trait));
    }

    private static String lineOf(String host, ProvenanceKey key) {
        if (host == null) {
            return null;
        }
        if (key == null) {
            return Json.string(host);
        }
        return Json.object().value("entity", host).raw("key", key.toJson()).render();
    }

    /**
     * Where a rule 607 link to {@code object} starts on {@code holder}: the
     * line whose ability exiled it with the holder, which Forge keeps on the
     * exiled card; the holder itself where Forge keeps no ability.
     */
    private static String maker(Card holder, Object object) {
        if (object instanceof Card exiled && exiled.getExiledWith() != null
                && exiled.getExiledWith().getId() == holder.getId()
                && exiled.getExiledSA() != null) {
            return line(holder, exiled.getExiledSA());
        }
        return card(holder);
    }

    // ── writing ─────────────────────────────────────────────────────────

    private void add(LinkFamily family, String from, String to) {
        if (from != null && to != null) {
            byFamily.computeIfAbsent(family, f -> new LinkedHashSet<>()).add(Json.object()
                    .raw("from", from).raw("to", to).value("type", family).render());
        }
    }

    private void addTarget(String from, String to, int index) {
        if (from != null && to != null) {
            byFamily.computeIfAbsent(LinkFamily.TARGETS, f -> new LinkedHashSet<>()).add(Json.object()
                    .raw("from", from).raw("to", to).value("type", LinkFamily.TARGETS)
                    .value("index", index).render());
        }
    }

    /** A symmetric family's link, once per pair, from the lower card id. */
    private void pair(LinkFamily family, Card one, Card other) {
        if (one == null || other == null || one.getId() == other.getId()) {
            return;
        }
        boolean oneFirst = one.getId() < other.getId();
        add(family, card(oneFirst ? one : other), card(oneFirst ? other : one));
    }
}
