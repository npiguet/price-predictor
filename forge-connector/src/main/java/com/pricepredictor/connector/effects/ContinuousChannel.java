package com.pricepredictor.connector.effects;

import forge.game.card.Card;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * One {@code continuous} record per static ability that is currently changing
 * the board.
 *
 * <p>Collected at a phase boundary, which is the cheapest moment the board is
 * stable, from the layer tables {@link LayerContributions} reads.
 *
 * <p>The snapshot a continuous record carries has the acting static's own
 * contributions removed, because a model asked to predict the effect must not
 * be handed a board that already contains it. That removal is the record's
 * whole premise, so a checkout that cannot do it collects nothing here rather
 * than a corpus whose inputs contain their own labels.
 *
 * <p>No cap at all: coalescing per stable board is itself the cap. A static
 * that applies on every recompute would otherwise write a record per priority
 * pass, and none of them would say anything the first did not.
 */
final class ContinuousChannel {

    /**
     * Whether the engine can say what a card would be without one static.
     *
     * <p>Separate from {@link AttributionMode}, which probes the sub-ability
     * hook. These three arrived later, so a checkout patched against an older
     * revision of the patch reports {@code patched} and still cannot suppress.
     */
    static final boolean CAN_SUPPRESS = PatchHooks.TYPE_WITHOUT.present()
            && PatchHooks.COLOR_WITHOUT.present()
            && PatchHooks.KEYWORDS_WITHOUT.present();

    private final LiveGame live;
    private final CollectorContext context;
    private final RecordDedup dedup;
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    /** Static ability by layer-table id, resolved once per game. */
    private final Map<Long, StaticAbility> staticsById = new HashMap<>();

    ContinuousChannel(LiveGame live, CollectorContext context, RecordDedup dedup) {
        this.live = live;
        this.context = context;
        this.dedup = dedup;
    }

    /** Record every static changing the board now. */
    void collect() {
        if (context.mode() == AttributionMode.DEGRADED || !CAN_SUPPRESS) {
            return;
        }
        LayerContributions contributions = new LayerContributions();
        for (Card card : live.game().getCardsIn(ZoneType.Battlefield)) {
            contributions.addCard(card);
        }
        String boardHash = boardHash();
        for (Map.Entry<Long, Map<String, LayerContributions.Contribution>> entry
                : contributions.byStatic().entrySet()) {
            record(entry.getKey(), entry.getValue().values(), boardHash);
        }
    }

    /** One static's record, unless it has nothing new to say. */
    private void record(
            Long staticId, Collection<LayerContributions.Contribution> contributions,
            String boardHash) {
        // Id 0 is Forge's "no static" column: a temporary pump written by a
        // resolving ability, which the resolution bracket already records.
        if (staticId == 0L) {
            return;
        }
        StaticAbility source = staticById(staticId);
        if (source == null) {
            return;
        }
        if (!allowContinuousRecord(String.valueOf(staticId), boardHash)) {
            return;
        }
        StringJoiner rendered = new StringJoiner(",", "[", "]");
        for (LayerContributions.Contribution contribution : contributions) {
            rendered.add(contribution.toJson());
        }
        ProvenanceKey.Resolved acting = ProvenanceResolver.resolve(source);
        String state = live.snapshots().ofBoardWithout(staticId).render();
        String payload = "{\"contributions\":" + rendered
                + ",\"board_hash\":" + Json.string(boardHash) + "}";
        // A second gate behind the per-board one, which cannot see this:
        // two *distinct* static ids -- a second copy of the same anthem,
        // two copies of one aura -- resolve to the same key, the same
        // contributions and the same suppressed board, so they render
        // identically and the board hash says they are different statics.
        if (!dedup.allowDistinct(RecordKind.CONTINUOUS.wireName(), payload, state)) {
            return;
        }
        context.emit(context.newRecord(RecordKind.CONTINUOUS)
                .actor(SnapshotBuilder.playerId(source.getHostCard().getController()))
                .ability(acting)
                .state(state)
                .payload(payload));
    }

    /**
     * What the board looks like, for coalescing.
     *
     * <p>Names and computed power/toughness of everything in play. Two boards
     * that hash the same produce the same contributions, so the second record
     * would repeat the first exactly.
     */
    private String boardHash() {
        StringBuilder shape = new StringBuilder();
        for (Card card : live.game().getCardsIn(ZoneType.Battlefield)) {
            shape.append(card.getId()).append(':').append(card.getName())
                    .append(':').append(card.getNetPower())
                    .append('/').append(card.getNetToughness()).append(';');
        }
        return Integer.toHexString(shape.toString().hashCode());
    }

    /**
     * The static ability an id names, searched once per game and remembered.
     *
     * <p>The layer tables carry the id rather than the object, and nothing in
     * the engine maps one back, so this walks what is in play. Cached because a
     * phase boundary asks for the same handful of ids every time.
     */
    private StaticAbility staticById(Long id) {
        if (staticsById.containsKey(id)) {
            return staticsById.get(id);
        }
        StaticAbility found = null;
        for (Card card : live.game().getCardsInGame()) {
            for (StaticAbility candidate : card.getStaticAbilities()) {
                if (candidate.getId() == id) {
                    found = candidate;
                    break;
                }
            }
            if (found != null) {
                break;
            }
        }
        staticsById.put(id, found);
        return found;
    }

    /**
     * Whether a continuous record for this static on this board is new.
     *
     * <p>Coalescing per {@code (game, static, board hash)} is the cap: a static
     * applies on every recompute, and a second record for the same board would
     * repeat the first exactly.
     */
    boolean allowContinuousRecord(String staticKey, String boardHash) {
        return coalescedBoards.add(staticKey + "@" + boardHash);
    }
}
