package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.WireNamed;
import com.pricepredictor.connector.json.Json;
import com.pricepredictor.connector.json.JsonParts;
import forge.game.CardTraitBase;
import forge.game.Game;
import forge.game.GameEntity;
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
import java.util.Comparator;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
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
 * position among its stack object's targets, each ref once, on
 * {@code targets} only ({@link ChainTargets#linked}). An end is a ref ({@code E<id>},
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
 * named, for the reader to drop. A card that has left the game (a token
 * that died) is no end unless the snapshot writes it as an entity (a
 * referenced token as it last existed, {@code SnapshotBuilder}): Forge's
 * lists may still hold it (a control card's gain-control targets, a
 * source's damage history), but it describes nothing on the board, and a
 * game copy, which holds no such card, could not write the link. Families
 * the reader builds from its own layout are never written ({@link
 * LinkFamily#collected}).
 *
 * <p>Links are listed by family in the table's order, each once.
 */
final class SnapshotLinks {

    private final Game game;
    private final Map<LinkFamily, Set<String>> byFamily = new EnumMap<>(LinkFamily.class);
    /** The ref of every card in the game or among the snapshot's entities, each a card end may name. */
    private final Set<String> inGame = new HashSet<>();

    private SnapshotLinks(Game game, Collection<Card> entities) {
        this.game = game;
        for (Card entity : entities) {
            inGame.add(card(entity));
        }
        game.forEachCardInGame(card -> {
            inGame.add(card(card));
            return true;
        });
        // A mutated permanent's other cards, which Forge's walk leaves out.
        for (Player player : game.getPlayers()) {
            for (Card merged : player.getCardsIn(ZoneType.Merged)) {
                inGame.add(card(merged));
            }
        }
    }

    /**
     * A snapshot's acting spell ability, and the targets its links name.
     *
     * @param ability the acting spell ability, whose targets are links even
     *                before it is on the stack; or null
     * @param chosen  whether its targets are its own: false for a
     *                playability candidate, whose persistent ability may
     *                hold an earlier evaluation's
     * @param targets the targets its links name ({@link ChainTargets#linked}):
     *                an effect half's as cast, before Forge's check for
     *                illegal targets, a mode's half's its mode's; or null
     *                for those it holds now
     */
    record Acting(SpellAbility ability, boolean chosen, ChainTargets targets) {

        /** No acting ability: a board's snapshot. */
        static final Acting NONE = new Acting(null, true, null);

        /**
         * The targets the acting ability's links name: those given, else
         * those it holds now, unless they are a candidate's, an earlier
         * evaluation's, off the stack.
         */
        List<ChainTargets.Target> linked(boolean onStack) {
            if (targets != null) {
                return targets.linked();
            }
            return chosen || onStack ? ChainTargets.now(ability).linked() : List.of();
        }
    }

    /**
     * The links of one snapshot.
     *
     * @param request  what the snapshot is of: its acting spell ability and
     *                 the targets its links name, the event a trait is about
     *                 to handle, and the board as it stands or without one
     *                 static
     * @param entities the entities the snapshot renders: every card entity in
     *                 order, and the cards of the four tiers, whose rows the
     *                 global block's timestamped floating effects are read from
     * @param notes    the clause notes the floating effects are read by
     */
    static String render(
            Game game, SnapshotRequest request, SnapshotEntities entities, ClauseNotes notes) {
        Collection<Card> cards = entities.cards();
        SnapshotLinks links = new SnapshotLinks(game, cards);
        links.cardLinks(cards, request.board());
        links.combat(game.getCombat());
        links.stack(request.acting());
        links.commandZone(cards, entities.tiers(), notes);
        EffectEvent pending = request.pending();
        if (pending != null && pending.lastKnown() != null) {
            Card lastKnown = pending.lastKnown();
            links.add(LinkFamily.LAST_KNOWN_STATE_OF,
                    Json.string("L" + lastKnown.getId()), card(lastKnown));
        }
        return joined(links.byFamily);
    }

    /**
     * {@code links}, a snapshot's rendered links, with the {@code controls}
     * and {@code owns} links of {@code libraryCards} added in their
     * families' places: the library cards the record's events name, which
     * join the entities at the write ({@link Snapshot#withEventCards}),
     * each controlled by its owner, as in the library.
     */
    static String withLibraryCards(String links, Collection<Card> libraryCards) {
        Map<LinkFamily, Set<String>> byFamily = byFamily(links);
        for (Card card : libraryCards) {
            String owner = player(card.getOwner());
            byFamily.computeIfAbsent(LinkFamily.CONTROLS, f -> new LinkedHashSet<>())
                    .add(linkJson(LinkFamily.CONTROLS, owner, card(card)));
            byFamily.computeIfAbsent(LinkFamily.OWNS, f -> new LinkedHashSet<>())
                    .add(linkJson(LinkFamily.OWNS, owner, card(card)));
        }
        return joined(byFamily);
    }

    /**
     * {@code links}, a cast's links read as it started, with the
     * {@code targets} family read again now, its modes and targets chosen
     * and {@code acting} on the stack: each stack object's targets as the
     * cast's refs block reads them. {@code added} are the card entities the
     * snapshot gained for the targets chosen since.
     */
    static String withTargetsNow(
            String links, Game game, Collection<Card> added, SpellAbility acting) {
        SnapshotLinks now = new SnapshotLinks(game, added);
        now.stack(new Acting(acting, true, null));
        Map<LinkFamily, Set<String>> byFamily = byFamily(links);
        byFamily.put(LinkFamily.TARGETS, now.byFamily.getOrDefault(LinkFamily.TARGETS, Set.of()));
        return joined(byFamily);
    }

    /** Rendered links, by family, each family's in the order given. */
    private static Map<LinkFamily, Set<String>> byFamily(String links) {
        Map<LinkFamily, Set<String>> byFamily = new EnumMap<>(LinkFamily.class);
        for (String link : JsonParts.elements(links)) {
            byFamily.computeIfAbsent(familyOf(link), f -> new LinkedHashSet<>()).add(link);
        }
        return byFamily;
    }

    /** A rendered link's family, from its {@code type}. */
    private static LinkFamily familyOf(String link) {
        String type = JsonParts.fields(link).get("type");
        return WireNamed.fromWireName(LinkFamily.class, type.substring(1, type.length() - 1));
    }

    /** Every link, by family in the table's order. */
    private static String joined(Map<LinkFamily, Set<String>> byFamily) {
        List<String> all = new ArrayList<>();
        for (LinkFamily family : LinkFamily.values()) {
            all.addAll(byFamily.getOrDefault(family, Set.of()));
        }
        return "[" + String.join(",", all) + "]";
    }

    // ── the families ────────────────────────────────────────────────────

    /** Every family a card's own fields hold. */
    private void cardLinks(Collection<Card> cards, StaticSuppression board) {
        for (Card card : cards) {
            mergedWith(card);
            add(LinkFamily.ATTACHED_TO, card(card), card(card.getAttachedTo()));
            add(LinkFamily.ATTACHED_TO, card(card), player(card.getPlayerAttachedTo()));
            mustBlock(card);
            for (Player goader : card.getGoaded()) {
                add(LinkFamily.GOADED_BY, card(card), player(goader));
            }
            if (board.showsAffected()) {
                affects(card);
            }
            linkedByText(card);
            boundTo(card);
            add(LinkFamily.CONTROLS, player(board.controller(card)), card(card));
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

    /**
     * The attackers a creature must block, by card id: Forge keeps them in
     * a hash map's order, which no rule of the game fixes.
     */
    private void mustBlock(Card card) {
        List<Card> attackers = new ArrayList<>(card.getMustBlockCards());
        attackers.sort(Comparator.comparingInt(Card::getId));
        for (Card attacker : attackers) {
            add(LinkFamily.MUST_BLOCK, card(card), card(attacker));
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
        combat.getAttackingBands().forEach(this::band);
    }

    /** Every pair of an attacking band's members. */
    private void band(AttackingBand band) {
        List<Card> members = new ArrayList<>(band.getAttackers());
        for (int i = 0; i < members.size(); i++) {
            for (int j = i + 1; j < members.size(); j++) {
                pair(LinkFamily.IN_BAND_WITH, members.get(i), members.get(j));
            }
        }
    }

    /**
     * Each stack object's targets, its source and what triggered it; the
     * acting ability's too where it is not on the stack yet. A target's
     * {@code index} is its position among its object's targets, each ref
     * once ({@link ChainTargets#linked}): the acting ability's those its
     * record names ({@link Acting#linked}), every other object's those it
     * holds now.
     */
    private void stack(Acting acting) {
        List<SpellAbility> ordered = stackObjects();
        SpellAbility actor = acting.ability();
        boolean onStack = ordered.stream().anyMatch(each -> each == actor);
        if (actor != null && !onStack) {
            ordered.add(actor);
        }
        for (SpellAbility ability : ordered) {
            String line = line(ability.getHostCard(), ability);
            targets(line, ability == actor
                    ? acting.linked(onStack) : ChainTargets.now(ability).linked());
            sourceOf(ability, line);
            triggeredBy(ability, line);
        }
    }

    /** The spell abilities on the stack, top first, each once. */
    private List<SpellAbility> stackObjects() {
        Set<SpellAbility> objects = Collections.newSetFromMap(new IdentityHashMap<>());
        List<SpellAbility> ordered = new ArrayList<>();
        for (SpellAbilityStackInstance instance : game.getStack()) {
            SpellAbility ability = instance.getSpellAbility();
            if (ability != null && objects.add(ability)) {
                ordered.add(ability);
            }
        }
        return ordered;
    }

    /** A stack object's targets, each indexed by its position in {@code targets}. */
    private void targets(String line, List<ChainTargets.Target> targets) {
        for (int index = 0; index < targets.size(); index++) {
            addTarget(line, target(targets.get(index)), index);
        }
    }

    private void sourceOf(SpellAbility ability, String line) {
        Card host = ability.getHostCard();
        if (line != null && !line.equals(card(host))) {
            add(LinkFamily.SOURCE_OF, card(host), line);
        }
    }

    private void triggeredBy(SpellAbility ability, String line) {
        if (!ability.isTrigger() || ability.getTriggeringObjects() == null) {
            return;
        }
        for (Object about : ability.getTriggeringObjects().values()) {
            for (String to : entities(about)) {
                add(LinkFamily.TRIGGERED_BY, line, to);
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
            add(LinkFamily.LINKED_BY_TEXT, maker(card, current(exiled)), card(exiled));
        }
        if (card.getExiledWith() != null) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card.getExiledWith(), card), card(card));
        }
        for (Card imprinted : card.getImprintedCards()) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card, current(imprinted)), card(imprinted));
        }
        for (Card held : card.getUntilLeavesBattlefield()) {
            add(LinkFamily.LINKED_BY_TEXT, maker(card, current(held)), card(held));
        }
        if (remembersLastingly(card)) {
            remembered(card);
        }
    }

    /** The cards and players a permanent lastingly remembers. */
    private void remembered(Card card) {
        for (Object remembered : card.getRemembered()) {
            if (remembered instanceof Card || remembered instanceof Player) {
                add(LinkFamily.LINKED_BY_TEXT, maker(card, current(remembered)), entity(remembered));
            }
        }
    }

    /**
     * The card as the game holds it now, for a card a list kept as it was:
     * Spell Queller remembers the spell it exiled as the stack's card, which
     * names nothing that exiled it, while the exiled card does. Read stale,
     * the one link came out twice, keyed to Spell Queller's line and plain;
     * a game copy, holding only the exiled card, wrote it once. A last-known
     * copy, or a card no longer in the game, is kept as it is.
     */
    private Object current(Object object) {
        if (object instanceof Card kept && !kept.isLKI()) {
            return game.getCardState(kept, kept);
        }
        return object;
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
        addCopyOf(card, card.getCopiedPermanent());
        if (card.isCloned()) {
            addCopyOf(card, cloneOrigin(card));
        }
    }

    /** The card a copy effect such as a Clone's copies: its latest clone state's origin, or null. */
    private static Card cloneOrigin(Card card) {
        return card.getCloneStates() instanceof NavigableMap<Long, CardCloneStates> states
                && !states.isEmpty() ? states.lastEntry().getValue().getOrigin() : null;
    }

    /** {@code card} copies {@code original}, unless that is itself or nothing. */
    private void addCopyOf(Card card, Card original) {
        if (original != null && original.getId() != card.getId()) {
            add(LinkFamily.COPY_OF, card(card), card(original));
        }
    }

    /** This turn's damage, blocks, crews and saddles. */
    private void thisTurn(Card card) {
        damagedThisTurn(card);
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
     * What a card damaged this turn: the source's own history, where Forge
     * registers each damage and keys the (source, target) pair by the
     * damage's identity.
     */
    private void damagedThisTurn(Card card) {
        if (card.getDamageHistory() == null) {
            return;
        }
        for (Pair<Integer, Boolean> damage : card.getDamageHistory().getAllDmgInstances()) {
            Pair<Card, GameEntity> sourceToTarget = game.getDamageLKI(damage);
            if (sourceToTarget != null) {
                add(LinkFamily.DAMAGED_THIS_TURN, card(card), entity(sourceToTarget.getRight()));
            }
        }
    }

    /**
     * Command-zone objects: what made each effect card, emblem, timestamped
     * floating effect and delayed trigger; what a floating effect applies to;
     * what a delayed trigger waits on.
     */
    private void commandZone(Collection<Card> cards, Collection<Card> tiers,
                             ClauseNotes notes) {
        cards.stream().filter(SnapshotLinks::madeByAnEffect).forEach(this::effectCard);
        FloatingEffects.timestamped(notes, tiers).forEach(this::floatingEffect);
        FloatingEffects.pendingDelayedTriggers(game).forEach(this::delayedTrigger);
    }

    /** What made an effect card: its line, or its card where it has none. */
    private void effectCard(Card card) {
        SpellAbility source = card.getEffectSourceAbility();
        add(LinkFamily.CREATED_BY, card(card), source != null
                ? line(source.getHostCard(), source) : card(card.getEffectSource()));
    }

    /** What a timestamped floating effect applies to, and what made it. */
    private void floatingEffect(FloatingEffects.Timestamped row) {
        for (String affected : row.affected()) {
            add(LinkFamily.AFFECTS, Json.string(row.id()), Json.string(affected));
        }
        add(LinkFamily.CREATED_BY, Json.string(row.id()), lineOf(row.host(), row.line()));
    }

    /** An effect card or emblem in the command zone, which a line or a card made. */
    private static boolean madeByAnEffect(Card card) {
        return card.isInZone(ZoneType.Command) && card.isImmutable()
                && !card.getType().isDungeon();
    }

    /** What made a delayed trigger, and what it waits on. */
    private void delayedTrigger(FloatingEffects.Pending pending) {
        String id = Json.string(pending.id());
        add(LinkFamily.CREATED_BY, id,
                line(pending.trigger().getHostCard(), pending.trigger()));
        for (String held : pending.remembered()) {
            add(LinkFamily.PENDING_ON, id, Json.string(held));
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
     * A target's end: its ref, which names a card, a player, a card as it
     * last existed and a spell by its card; an ability on the stack by its
     * line.
     */
    private static String target(ChainTargets.Target target) {
        if (target.kind() == ChainTargets.Kind.ABILITY) {
            return line(target.card(), (SpellAbility) target.object());
        }
        return Json.string(target.ref());
    }

    /** A trait's line on its host, or the host where the trait keys to no line. */
    static String line(Card host, CardTraitBase trait) {
        Object end = lineEnd(host, trait);
        return end == null ? null : Json.value(end);
    }

    /**
     * The same end as a value written inside another value (an event's
     * {@code ability}): the line as an {@code {entity, key}} map, the host's
     * ref where the trait keys to no line, null for no host.
     */
    static Object lineEnd(Card host, CardTraitBase trait) {
        if (host == null) {
            return null;
        }
        return endOf(Refs.of(host), trait == null ? null : ProvenanceResolver.of(trait));
    }

    private static String lineOf(String host, ProvenanceKey key) {
        Object end = endOf(host, key);
        return end == null ? null : Json.value(end);
    }

    private static Object endOf(String host, ProvenanceKey key) {
        if (host == null || key == null) {
            return host;
        }
        Map<String, Object> end = new LinkedHashMap<>();
        end.put("entity", host);
        end.put("key", key.toJsonMap());
        return end;
    }

    /**
     * Where a rule 607 link to {@code object} starts on {@code holder}: the
     * line whose ability exiled it with the holder, which Forge keeps on the
     * exiled card; the holder itself where Forge keeps no ability.
     */
    private static String maker(Card holder, Object object) {
        if (object instanceof Card exiled && isExiledBy(exiled, holder)) {
            return line(holder, exiled.getExiledSA());
        }
        return card(holder);
    }

    /** Whether Forge keeps the ability that exiled {@code exiled} with {@code holder}. */
    private static boolean isExiledBy(Card exiled, Card holder) {
        Card exiledWith = exiled.getExiledWith();
        return exiledWith != null && exiledWith.getId() == holder.getId()
                && exiled.getExiledSA() != null;
    }

    // ── writing ─────────────────────────────────────────────────────────

    private void add(LinkFamily family, String from, String to) {
        if (isEnd(from) && isEnd(to)) {
            byFamily.computeIfAbsent(family, f -> new LinkedHashSet<>()).add(linkJson(family, from, to));
        }
    }

    private static String linkJson(LinkFamily family, String from, String to) {
        return Json.object().raw("from", from).raw("to", to).value("type", family).render();
    }

    private void addTarget(String from, String to, int index) {
        if (isEnd(from) && isEnd(to)) {
            byFamily.computeIfAbsent(LinkFamily.TARGETS, f -> new LinkedHashSet<>()).add(Json.object()
                    .raw("from", from).raw("to", to).value("type", LinkFamily.TARGETS)
                    .value("index", index).render());
        }
    }

    /**
     * Whether a rendered end names something: not null, and, for a card's
     * ref, a card still in the game or an entity of the snapshot. A line, a player and a command-zone
     * object's ref are ends as they are.
     */
    private boolean isEnd(String end) {
        if (end == null) {
            return false;
        }
        return !CARD_REF.matcher(end).matches() || inGame.contains(end);
    }

    private static final java.util.regex.Pattern CARD_REF = java.util.regex.Pattern.compile("\"E\\d+\"");

    /** A symmetric family's link, once per pair, from the lower card id. */
    private void pair(LinkFamily family, Card one, Card other) {
        if (!arePair(one, other)) {
            return;
        }
        boolean oneFirst = one.getId() < other.getId();
        add(family, card(oneFirst ? one : other), card(oneFirst ? other : one));
    }

    private static boolean arePair(Card one, Card other) {
        return one != null && other != null && one.getId() != other.getId();
    }
}
