package com.pricepredictor.connector.effects;

import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.ZoneType;

import java.util.ArrayList;
import java.util.EnumSet;
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
 * {@link PlayerJson} and {@link RefsJson}.
 */
public final class SnapshotBuilder {

    private final Game game;

    public SnapshotBuilder(Game game) {
        this.game = game;
    }

    /**
     * {@link Refs#of(Player)}, kept for {@code GamePlayer} until it names the
     * random seat through {@link Refs} directly.
     */
    public static String playerId(Player player) {
        return Refs.of(player);
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
     *                   they sit in (tier 1) — the host alone for a playability
     *                   candidate, which has no targets yet; an ability being
     *                   cast or forced takes {@link #ofResolving} instead
     */
    Snapshot ofAbility(SpellAbility acting, Iterable<Card> referenced) {
        return snapshot(referenced, null, acting, null, null);
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
        return snapshot(List.of(), null, null, null, null);
    }

    /**
     * The board-only snapshot with one static's contributions removed from
     * every entity: a {@code continuous} record's state, which must not
     * already contain the answer the record labels.
     */
    Snapshot ofBoardWithout(long staticId) {
        return snapshot(List.of(), staticId, null, null, null);
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
        return snapshot(source == null ? List.of() : List.of(source), null, null, source, pending);
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
     */
    private Snapshot snapshot(
            Iterable<Card> referenced, Long withoutStatic, SpellAbility acting, Card source,
            EffectEvent pending) {
        String entities = Json.array(collectEntities(referenced),
                card -> EntityJson.render(game, card, withoutStatic));
        String players = Json.array(game.getPlayers(), player -> PlayerJson.render(game, player));
        return new Snapshot(
                globalToJson(), players, entities,
                RefsJson.render(acting, source), pending == null ? "null" : pending.toJson());
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

    private String globalToJson() {
        var phase = game.getPhaseHandler();
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
                EntityJson.addKeys(keys, card.getSpellAbilities());
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
