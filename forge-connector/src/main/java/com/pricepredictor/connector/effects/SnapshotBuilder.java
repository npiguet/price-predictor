package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.json.Json;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Builds a record's pre-event state snapshot.
 *
 * <p>Three rules shape every snapshot, and each exists because getting it wrong
 * is invisible until the model is trained:
 *
 * <ul>
 *   <li><b>Characteristics are computed, not printed.</b> The model has to
 *       predict what happens to a 4/4 that an anthem made a 5/5; recording the
 *       4/4 would ask it to predict from a board that does not exist.</li>
 *   <li><b>Perspective is not stored.</b> Controllers are absolute player ids;
 *       mine and opponent derive at training time from the record's actor. One
 *       snapshot then serves records with different actors.</li>
 *   <li><b>An absent inclusion tier means uncollected, not empty.</b> The
 *       collected tiers are written into every record, because an empty entity
 *       list and an uncollected tier are otherwise identical.</li>
 * </ul>
 *
 * <p>Every snapshot renders all four inclusion tiers: every referenced object
 * in whatever zone it sits (1), the global state, the battlefield and
 * command-zone effect cards (2), the unreferenced stack (3), and unreferenced
 * hands and graveyards (4). The depth is the same on every record, whatever
 * collector or run wrote it, because a depth that varies makes
 * {@code state.tiers} a proxy for how a record was collected: a tier only the
 * interventional forks rendered would identify {@code interventional} exactly,
 * and a flag the schema bars from ever being a model input would have a
 * perfect proxy sitting in the state.
 *
 * <p>The tiers and the global block are decided here; each entity, each
 * player and the refs block are rendered by {@link EntityJson},
 * {@link PlayerJson} and {@link RefsJson}, and the links between them by
 * {@link SnapshotLinks}.
 */
public final class SnapshotBuilder {

    private final Game game;
    /** Whose clause notes join a timestamped floating effect to its line. */
    private final FloatingEffects floating;

    public SnapshotBuilder(Game game) {
        this(game, game);
    }

    /**
     * A builder over a fork's copy, reading the live game's clause notes: the
     * copy keeps the rows' timestamps, and only the live game's clauses were
     * noted.
     */
    SnapshotBuilder(Game game, Game notesOf) {
        this.game = game;
        this.floating = FloatingEffects.of(notesOf);
    }

    /**
     * An ability's tier-1 referenced list: its host and the cards it targets.
     *
     * <p>One list for every collector that snapshots an ability, observed or
     * forked, because a target outside the zones the other tiers sweep — an
     * exiled card, say — is in a snapshot only through this tier, and a
     * collector that left it out wrote a record naming a target its own state
     * did not hold.
     */
    private static List<Card> referencedOf(SpellAbility ability) {
        List<Card> referenced = new ArrayList<>();
        if (ability == null) {
            return referenced;
        }
        if (ability.getHostCard() != null) {
            referenced.add(ability.getHostCard());
        }
        if (ability.getTargets() != null) {
            for (Object target : ability.getTargets()) {
                if (target instanceof Card card) {
                    referenced.add(card);
                }
            }
        }
        return referenced;
    }

    /**
     * The snapshot for a spell ability that acts: its refs name the ability's
     * host, targets and announced values.
     *
     * @param acting     the ability, or null — a bracket opened for nothing, a
     *                   legality question with no candidate — whose refs then
     *                   render empty, as {@link #ofBoard}'s do
     * @param referenced the cards the record points at, carried whatever zone
     *                   they sit in (tier 1)
     */
    Snapshot ofAbility(SpellAbility acting, Iterable<Card> referenced) {
        return snapshot(referenced, null, acting, null, null, true);
    }

    /**
     * The snapshot for a playability candidate: its host is the one card
     * referenced and its refs name that host alone (C4).
     *
     * <p>A candidate is the card's persistent ability, which keeps whatever
     * an earlier evaluation or an abandoned cast set on it — the AI's
     * {@code canPlaySa} chooses targets and X before it asks whether the
     * cost can be paid. A decision is taken before any of that is chosen, so
     * targets, X, modes and choices are empty, and no stale target is linked;
     * the host is still linked as the candidate line's source.
     */
    Snapshot ofCandidate(SpellAbility candidate) {
        Card host = candidate.getHostCard();
        return snapshot(host == null ? List.of() : List.of(host), null, candidate, host, null, false);
    }

    /**
     * The snapshot for an ability being cast or forced: {@link #ofAbility}
     * with its host and targets as the referenced list.
     */
    Snapshot ofResolving(SpellAbility acting) {
        return ofAbility(acting, referencedOf(acting));
    }

    /**
     * The snapshot for a record no single line acts in — a damage step, a
     * legality question, a mana activation, an orphaned outcome — so its refs
     * block is the empty shape.
     */
    Snapshot ofBoard() {
        return snapshot(List.of(), null, null, null, null, true);
    }

    /**
     * The board-only snapshot with one static's contributions removed from
     * every entity: a {@code continuous} record's state, which must not
     * already contain the answer the record labels.
     */
    Snapshot ofBoardWithout(long staticId) {
        return snapshot(List.of(), staticId, null, null, null, true);
    }

    /**
     * The snapshot for a trait — a trigger or a replacement — about to handle
     * an event.
     *
     * <p>A trait is not a {@link SpellAbility}, so it cannot be the acting
     * ability -- but it has a host card, and that is what {@code refs.source}
     * names: the join that lets the record say which permanent's text acted.
     * Neither kind has targets or announced values at the moment it is
     * recorded, so the rest of the refs block is legitimately empty.
     *
     * <p>The host is the tier-1 referenced list too. Without it a host outside
     * every zone the other tiers sweep — one in exile, say — is in no
     * {@code state.entities}, and {@code refs.source} would name an entity id
     * the reader cannot follow.
     *
     * @param source  the trait's host, or null where the trait has none
     * @param pending the event as the trait received it
     */
    Snapshot ofTrait(Card source, EffectEvent pending) {
        return snapshot(source == null ? List.of() : List.of(source), null, null, source, pending, true);
    }

    /**
     * The one assembly behind every call shape, private so that no caller
     * passes the combination of nulls that picks its shape.
     *
     * @param withoutStatic the id of a static whose changes are removed from
     *                      every entity, or null to show the board as it stands
     * @param source        the host of an acting line that is not a spell
     *                      ability, or null
     * @param pending       the event a trait is about to handle, or null
     * @param chosen        whether {@code acting}'s targets and announced
     *                      values are its own, chosen for this record: false
     *                      for a playability candidate, whose refs name its
     *                      host alone and whose targets are no links
     */
    private Snapshot snapshot(
            Iterable<Card> referenced, Long withoutStatic, SpellAbility acting, Card source,
            EffectEvent pending, boolean chosen) {
        Set<Card> tiers = collectEntities(referenced);
        List<String> entities = new ArrayList<>();
        Set<String> present = new HashSet<>();
        for (Card card : tiers) {
            entities.add(EntityJson.render(game, card, withoutStatic));
            present.add(Refs.of(card));
        }
        LibraryCapture libraries = LibraryCapture.of(game, withoutStatic, present);
        List<Card> cards = new ArrayList<>(tiers);
        addBeyondTheTiers(entities, present, tiers, cards, withoutStatic, libraries, pending);
        String players = Json.array(game.getPlayers(), player -> PlayerJson.render(game, player));
        return new Snapshot(
                globalToJson(tiers), players, "[" + String.join(",", entities) + "]",
                SnapshotLinks.render(
                        game, cards, tiers, acting, chosen, pending, withoutStatic, floating),
                RefsJson.render(chosen ? acting : null, source),
                pending == null ? "null" : pending.toJson(),
                libraries);
    }

    static final String ROLE_LIBRARY_CARD = "library_card";
    static final String ROLE_EXILED_WITH = "exiled_with";
    static final String ROLE_MUTATE_COMPONENT = "mutate_component";
    static final String ROLE_LAST_KNOWN = "last_known";

    /**
     * The card entities beyond the four tiers (FR-024), each once, after
     * them and with its {@code role}: each library's top card; the library
     * cards a pending event names; the cards exiled with a permanent; a
     * mutated permanent's components; and the pending event's subject as it
     * last existed, under id {@code L<card id>} because the live card keeps
     * its own. The library cards the record's later events name join at the
     * write ({@link Snapshot#withEventCards}).
     *
     * @param cards the card entities rendered so far, which each added card
     *              joins (a last-known copy is no card of the game, and does
     *              not)
     */
    private void addBeyondTheTiers(
            List<String> entities, Set<String> present, Set<Card> tiers, List<Card> cards,
            Long withoutStatic, LibraryCapture libraries, EffectEvent pending) {
        for (var player : game.getPlayers()) {
            Card top = player.getCardsIn(ZoneType.Library).isEmpty()
                    ? null : player.getCardsIn(ZoneType.Library).get(0);
            addLibraryCard(entities, present, cards, top, 0, withoutStatic);
        }
        if (pending != null) {
            for (String subject : pending.subjects()) {
                LibraryCapture.Slot slot = libraries.byId().get(subject);
                if (slot != null) {
                    addLibraryCard(entities, present, cards, slot.card(), slot.position(),
                            withoutStatic);
                }
            }
        }
        for (Card card : tiers) {
            if (card.isInPlay()) {
                for (Card exiled : card.getExiledCards()) {
                    addBeyond(entities, present, cards, exiled, withoutStatic,
                            ROLE_EXILED_WITH);
                }
            }
        }
        for (Card card : tiers) {
            if (card.isInPlay() && card.isMutated()) {
                for (Card component : card.getMergedCards()) {
                    addBeyond(entities, present, cards, component, withoutStatic,
                            ROLE_MUTATE_COMPONENT);
                }
            }
        }
        Card lastKnown = pending == null ? null : pending.lastKnown();
        if (lastKnown != null) {
            // A last-known copy is in no zone; it names the one it was in.
            var zone = lastKnown.getZone() != null ? lastKnown.getZone() : lastKnown.getLastKnownZone();
            entities.add(EntityJson.render(game, lastKnown, withoutStatic, new EntityJson.Beyond(
                    "L" + lastKnown.getId(),
                    zone == null ? null : CardSpelling.zoneName(zone.getZoneType()),
                    null, ROLE_LAST_KNOWN)));
        }
    }

    private void addBeyond(
            List<String> entities, Set<String> present, List<Card> cards, Card card,
            Long withoutStatic, String role) {
        if (card != null && present.add(Refs.of(card))) {
            cards.add(card);
            entities.add(EntityJson.render(game, card, withoutStatic,
                    new EntityJson.Beyond(null, null, null, role)));
        }
    }

    /**
     * A library card, read as it is in the library (FR-042e), at its
     * position: the same rendering whether the snapshot holds it as a
     * library's top card or the record's events name it ({@link
     * LibraryCapture}).
     */
    private void addLibraryCard(
            List<String> entities, Set<String> present, List<Card> cards, Card card,
            int position, Long withoutStatic) {
        if (card != null && present.add(Refs.of(card))) {
            cards.add(card);
            entities.add(EntityJson.render(game, LibraryCapture.asInLibrary(card), withoutStatic,
                    new EntityJson.Beyond(null, "library", position, ROLE_LIBRARY_CARD)));
        }
    }

    /**
     * Every entity the snapshot carries, tier by tier, each once and in the
     * first tier that reaches it: the referenced cards wherever they sit (1),
     * the battlefield and the command zone's effect cards (2), the stack (3),
     * then hands and graveyards (4).
     */
    private Set<Card> collectEntities(Iterable<Card> referenced) {
        Set<Card> entities = new LinkedHashSet<>();
        for (Card card : referenced) {
            if (card != null) {
                entities.add(card);
            }
        }
        entities.addAll(game.getCardsIn(EnumSet.of(ZoneType.Battlefield, ZoneType.Command)));
        entities.addAll(game.getCardsIn(ZoneType.Stack));
        entities.addAll(game.getCardsIn(EnumSet.of(ZoneType.Hand, ZoneType.Graveyard)));
        return entities;
    }

    private String globalToJson(Collection<Card> cards) {
        var phase = game.getPhaseHandler();
        Boolean night = game.getDayTime();
        return Json.object()
                .value("turn", phase.getTurn())
                .value("phase", phaseName(phase.getPhase()))
                .value("active", Refs.ofNullable(phase.getPlayerTurn()))
                .value("priority", Refs.ofNullable(phase.getPriorityPlayer()))
                .value("stack_size", game.getStack().size())
                // Null outside combat damage. A first-strike combat produces
                // two damage steps and so two combat records, which only the
                // substep tells apart.
                .value("combat_substep", DamageStep.of(phase))
                .raw("emblems", ProvenanceKey.toJsonArray(emblems()))
                // Forge's daytime field is true at night, null for neither.
                .value("day_night", night == null ? null : night ? "night" : "day")
                .raw("floating_effects",
                        floating.floatingEffectsJson(game, cards))
                .raw("delayed_triggers", FloatingEffects.delayedTriggersJson(game))
                .raw("dungeons", FloatingEffects.dungeonsJson(game))
                .render();
    }

    /**
     * Command-zone emblems, as the key of the line that made each.
     *
     * <p>An emblem is a continuous effect with no permanent to hang on, so it
     * appears nowhere else in the snapshot; without this the board says nothing
     * about an effect that changes every turn of the rest of the game.
     */
    private List<ProvenanceKey> emblems() {
        Set<ProvenanceKey> keys = new LinkedHashSet<>();
        for (Card card : game.getCardsIn(ZoneType.Command)) {
            if (card.isEmblem()) {
                // An emblem's lines are mostly statics and triggers.
                EntityJson.addKeys(keys, card.getSpellAbilities());
                EntityJson.addKeys(keys, card.getTriggers());
                EntityJson.addKeys(keys, card.getStaticAbilities());
                EntityJson.addKeys(keys, card.getReplacementEffects());
            }
        }
        return List.copyOf(keys);
    }

    private static String phaseName(Object phase) {
        return phase == null
                ? null
                : phase.toString().toLowerCase(java.util.Locale.ROOT);
    }
}
