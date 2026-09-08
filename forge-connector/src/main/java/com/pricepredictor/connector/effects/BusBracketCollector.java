package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.player.Player;
import forge.game.zone.ZoneType;
import forge.game.card.CardView;
import forge.game.event.GameEventCardAttachment;
import forge.game.event.GameEventCardChangeZone;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardStatsChanged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventScry;
import forge.game.event.GameEventSurveil;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Stage-one collection: the public event bus plus a bracket around resolution.
 *
 * <p>No patched Forge. Events arriving on the bus are attributed to whatever the
 * stack said was resolving when the bracket opened — correct for the common case
 * of one thing resolving at a time, and approximate where two resolve together.
 * Every record it writes is stamped {@code degraded} so a corpus never claims
 * more precision than it has.
 *
 * <p>The bracket is delimited by the bus itself: a cast opens one, a resolution
 * closes it. The resolving ability is read from {@code game.getStack()} rather
 * than from the event, because the event carries a view and the provenance key
 * needs the model object.
 *
 * <p>Damage steps get their own brackets, closed at the phase boundary, so a
 * first-strike combat writes two records — the only way a keyword that acts by
 * splitting the step is visible in the corpus at all.
 */
public final class BusBracketCollector {

    private final Game game;
    private final RecordShardWriter writer;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final String gameId;

    /** The ability the open bracket attributes to, or null between brackets. */
    private SpellAbility resolving;
    private List<ProvenanceKey> resolvingKeys = List.of();
    private String resolvingActor;
    private String openBracketState;
    private final List<EffectEvent> bracketEvents = new ArrayList<>();

    /** Combat damage accumulates into its own bracket, closed at the phase end. */
    private final List<EffectEvent> combatEvents = new ArrayList<>();
    /** The step's combat, read when its bracket opens rather than at flush. */
    private CombatShape combatShape = CombatShape.empty();
    /**
     * Computed characteristics, so a stats-changed event can say what changed.
     *
     * <p>Lives for the game rather than the bracket: a pump in one bracket and
     * its wearing off in another are both differences from what was there
     * before, and a per-bracket baseline would report the second as nothing.
     */
    private final StatDiffer stats = new StatDiffer();
    /**
     * Assignments already written, for the length of one combat.
     *
     * <p>Forge's assignment table is never cleared between the first-strike and
     * regular steps, so without this the regular step's record repeats first
     * strike's. Cleared when the combat ends, not when a step does.
     */
    private final Set<String> assignmentsSeen = new LinkedHashSet<>();
    private final Set<String> combatParticipants = new LinkedHashSet<>();
    private String combatSubstep;
    private String combatState;

    private long recordsWritten;

    /** Told each combat record's id, for a probe that mirrors it. */
    private java.util.function.Consumer<String> onCombatRecord;

    public BusBracketCollector(
            Game game, RecordShardWriter writer, String gameId) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.snapshots = new SnapshotBuilder(game);
        this.mode = AttributionMode.detect();
    }

    /**
     * Be told the id of each {@code combat} record as it is written.
     *
     * <p>A damage-step probe forks before the step and cannot know what it
     * mirrors until the real record exists, so the pairing is handed over here
     * rather than reconstructed. Reconstructing it from board state would not
     * work: two combats in one turn can look identical, and a wrong pairing
     * makes the difference between them meaningless.
     */
    public void onCombatRecord(java.util.function.Consumer<String> listener) {
        this.onCombatRecord = listener;
    }

    public long recordsWritten() {
        return recordsWritten;
    }

    // ── brackets ────────────────────────────────────────────────────────

    /**
     * A cast opens a bracket and writes the cost half.
     *
     * <p>The cost half is written now rather than at resolution because a
     * countered or fizzled spell never resolves, and the corpus needs the record
     * that says what was paid regardless.
     */
    @Subscribe
    public void onCast(GameEventSpellAbilityCast event) {
        SpellAbility ability = game.getStack().peekAbility();
        openBracket(ability);
        if (ability == null) {
            return;
        }
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                gameId, EffectRecord.KIND_RESOLUTION, mode)
                .moment(EffectRecord.MOMENT_ACTIVATION)
                .actor(resolvingActor)
                .ability(resolvingKeys)
                .linkId(bracketLinkId())
                .state(openBracketState)
                .payload(EffectRecord.costPayload(ability, EffectRecord.OUTCOME_RESOLVED)));
    }

    /** A resolution closes the bracket and writes the effect half. */
    @Subscribe
    public void onResolved(GameEventSpellResolved event) {
        if (resolving == null) {
            bracketEvents.clear();
            return;
        }
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(), RecordShardWriter.timestamp(),
                gameId, EffectRecord.KIND_RESOLUTION, mode)
                .moment(EffectRecord.MOMENT_RESOLUTION)
                .actor(resolvingActor)
                .ability(resolvingKeys)
                .linkId(event.hasFizzled() ? null : bracketLinkId())
                .state(openBracketState)
                .payload(EffectRecord.eventsPayload(bracketEvents)));
        closeBracket();
    }

    /**
     * A phase change closes the combat bracket, if one is open.
     *
     * <p>One record per damage step: first strike and regular damage are
     * separate phases, so the phase boundary is exactly the step boundary.
     */
    @Subscribe
    public void onPhase(GameEventTurnPhase event) {
        flushCombat();
    }

    @Subscribe
    public void onCombatEnded(GameEventCombatEnded event) {
        flushCombat();
        // The next combat's assignment table starts empty, so what this one
        // wrote must stop suppressing entries in it.
        assignmentsSeen.clear();
    }

    // ── outcome events ──────────────────────────────────────────────────

    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        String subject = "E" + event.card().getId();
        EffectEvent damage = EventAttribution.stamp(
                BusEvents.cardDamaged(event, isCombatDamage()), resolving);
        if (isCombatDamage()) {
            openCombatBracket();
            combatParticipants.add(subject);
            combatEvents.add(damage);
        } else {
            bracketEvents.add(damage);
        }
    }

    @Subscribe
    public void onPlayerDamaged(GameEventPlayerDamaged event) {
        String subject = "P" + event.target().getId();
        EffectEvent damage = EventAttribution.stamp(
                BusEvents.playerDamaged(event), resolving);
        if (event.combat()) {
            openCombatBracket();
            combatParticipants.add(subject);
            combatEvents.add(damage);
        } else {
            bracketEvents.add(damage);
        }
    }

    @Subscribe
    public void onLifeChanged(GameEventPlayerLivesChanged event) {
        record(BusEvents.lifeChanged(event));
    }

    @Subscribe
    public void onPoisoned(GameEventPlayerPoisoned event) {
        record(BusEvents.poisoned(event));
    }

    @Subscribe
    public void onCounters(GameEventCardCounters event) {
        record(BusEvents.counters(event));
    }

    @Subscribe
    public void onTapped(GameEventCardTapped event) {
        record(BusEvents.tapped(event));
    }

    /** Zone changes, the workhorse outcome. */
    @Subscribe
    public void onZone(GameEventZone event) {
        EffectEvent moved = BusEvents.zone(event);
        if (moved != null) {
            record(moved);
        }
    }

    /**
     * A draw, a discard or a mill, which the bus has no event for.
     *
     * <p>Each is a card moving between two zones and which of the three it is
     * depends on where it came from, so it has to be classified from the pair.
     * The head counts all three per player and none of those counters had ever
     * fired.
     */
    @Subscribe
    public void onCardChangeZone(GameEventCardChangeZone event) {
        EffectEvent named = BusEvents.libraryMovement(event);
        if (named != null) {
            record(named);
        }
        if (event.card() != null && event.from() != null
                && event.from().zoneType() == ZoneType.Battlefield) {
            // Off the battlefield a card's computed characteristics stop
            // meaning anything, and one that returns is a new object.
            stats.forget(event.card().getId());
        }
    }

    /**
     * The engine says a card's stats moved; this says what moved.
     *
     * <p>The bus publishes one event for a pump, an anthem recompute, an
     * animation and a colour change alike, and it names only the card. Without
     * the difference, the whole "gets +2/+2 and gains flying" family — most of
     * what a limited deck does — resolved into an empty event list.
     */
    @Subscribe
    public void onStatsChanged(GameEventCardStatsChanged event) {
        if (event.cards() == null) {
            return;
        }
        for (CardView view : event.cards()) {
            if (view == null) {
                continue;
            }
            Card card = cardById(view.getId());
            if (card != null) {
                for (EffectEvent changed : stats.diff(card)) {
                    record(changed);
                }
            }
        }
    }

    @Subscribe
    public void onAttachment(GameEventCardAttachment event) {
        EffectEvent attached = BusEvents.attachment(event);
        if (attached != null) {
            record(attached);
        }
    }

    @Subscribe
    public void onScry(GameEventScry event) {
        for (EffectEvent looked : BusEvents.scry(event)) {
            record(looked);
        }
    }

    @Subscribe
    public void onSurveil(GameEventSurveil event) {
        for (EffectEvent looked : BusEvents.surveil(event)) {
            record(looked);
        }
    }

    /**
     * The battlefield card an id names, or null.
     *
     * <p>The bus hands out views rather than cards, and a view carries the id
     * and not the object. The battlefield is small enough that a scan costs
     * less than an index kept in step with every zone change.
     */
    private Card cardById(int id) {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (card.getId() == id) {
                return card;
            }
        }
        return null;
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Route an event into whichever bracket is open.
     *
     * <p>State-based-action deaths arrive with no ability resolving; they
     * attribute to the bracket they follow, which is the one that caused them.
     */
    private void record(EffectEvent event) {
        // Stamped here rather than where the event is built: which clause
        // produced it and how long it lasts are properties of the bracket it
        // landed in, and only this side knows that.
        EventAttribution.stamp(event, resolving);
        if (isCombatDamage()) {
            openCombatBracket();
            combatEvents.add(event);
            combatParticipants.addAll(event.subjects());
        } else {
            bracketEvents.add(event);
        }
    }

    private void openBracket(SpellAbility ability) {
        this.resolving = ability;
        this.resolvingKeys = keysOf(ability);
        this.resolvingActor = ability == null || ability.getActivatingPlayer() == null
                ? null
                : SnapshotBuilder.playerId(ability.getActivatingPlayer());
        this.openBracketState = snapshots.toJson(ability, referencedOf(ability));
        this.bracketEvents.clear();
    }

    private void closeBracket() {
        this.resolving = null;
        this.resolvingKeys = List.of();
        this.resolvingActor = null;
        this.openBracketState = null;
        this.bracketEvents.clear();
    }

    private String bracketLinkId() {
        return resolving == null ? null : gameId + ".link." + resolving.getId();
    }

    private static List<ProvenanceKey> keysOf(SpellAbility ability) {
        if (ability == null) {
            return List.of();
        }
        ProvenanceKey key = ProvenanceKey.of(ability);
        return key == null ? List.of() : List.of(key);
    }

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

    private boolean isCombatDamage() {
        var phase = game.getPhaseHandler();
        if (phase == null || phase.getPhase() == null) {
            return false;
        }
        return phase.getPhase().toString().contains("COMBAT_DAMAGE");
    }

    /**
     * Start a damage step's bracket, and capture the combat it is about.
     *
     * <p>The structure is read here rather than at the flush, because the flush
     * happens at the phase boundary — after the damage has been dealt and the
     * creatures it killed have left. Read there, an attacker that traded with
     * its blocker is in neither list, and the record describes a combat nobody
     * fought. Read here it matches the snapshot beside it, which is taken in
     * the same call.
     */
    private void openCombatBracket() {
        if (combatState != null) {
            return;
        }
        var phase = game.getPhaseHandler();
        combatSubstep = phase != null && phase.getPhase() != null
                && phase.getPhase().toString().contains("FIRST_STRIKE")
                ? "first_strike" : "regular";
        combatState = snapshots.toJson(null, List.of());
        // The damage has been assigned by now -- the first damage event is what
        // opened this bracket -- but nothing has died yet, so this is the one
        // moment both halves of the shape are readable.
        combatShape = CombatShape.of(game.getCombat());
        combatShape.addAssignment(game.getCombat(), assignmentsSeen);
    }

    /**
     * Write the open combat bracket, if any, as one {@code combat} record.
     *
     * <p>{@code ability} is absent: no single line acts in a damage step, and
     * inventing one would attribute the whole step to whichever creature
     * happened to be first.
     */
    private void flushCombat() {
        if (combatState == null) {
            return;
        }
        var phase = game.getPhaseHandler();
        String actor = phase == null || phase.getPlayerTurn() == null
                ? null : SnapshotBuilder.playerId(phase.getPlayerTurn());
        String recordId = writer.nextRecordId();
        emit(new EffectRecord(
                recordId, writer.runId(), RecordShardWriter.timestamp(),
                gameId, EffectRecord.KIND_COMBAT, mode)
                .actor(actor)
                .state(combatState)
                .payload(combatPayload()));
        // A damage-step probe forked before this step and has been holding its
        // branch since: it needs this record's id to say which combat it
        // mirrors, and the id does not exist until the record is written.
        if (onCombatRecord != null) {
            onCombatRecord.accept(recordId);
        }
        combatEvents.clear();
        combatParticipants.clear();
        combatShape = CombatShape.empty();
        combatState = null;
        combatSubstep = null;
    }

    private String combatPayload() {
        StringJoiner events = new StringJoiner(",", "[", "]");
        for (EffectEvent event : combatEvents) {
            events.add(event.toJson());
        }
        return "{" + combatShape.fields() + ",\"events\":" + events + "}";
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
