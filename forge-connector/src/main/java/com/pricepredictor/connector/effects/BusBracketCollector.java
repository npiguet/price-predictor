package com.pricepredictor.connector.effects;

import com.google.common.eventbus.Subscribe;
import forge.game.Game;
import forge.game.card.Card;
import forge.game.event.GameEventCardCounters;
import forge.game.event.GameEventCardDamaged;
import forge.game.event.GameEventCardTapped;
import forge.game.event.GameEventCombatEnded;
import forge.game.event.GameEventPlayerDamaged;
import forge.game.event.GameEventPlayerLivesChanged;
import forge.game.event.GameEventPlayerPoisoned;
import forge.game.event.GameEventSpellAbilityCast;
import forge.game.event.GameEventSpellResolved;
import forge.game.event.GameEventTurnPhase;
import forge.game.event.GameEventZone;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
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
                .payload(EffectRecord.costPayload(EffectRecord.OUTCOME_RESOLVED)));
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
    }

    // ── outcome events ──────────────────────────────────────────────────

    @Subscribe
    public void onCardDamaged(GameEventCardDamaged event) {
        String subject = "E" + event.card().getId();
        EffectEvent damage = BusEvents.cardDamaged(event, isCombatDamage());
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
        EffectEvent damage = BusEvents.playerDamaged(event);
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

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * Route an event into whichever bracket is open.
     *
     * <p>State-based-action deaths arrive with no ability resolving; they
     * attribute to the bracket they follow, which is the one that caused them.
     */
    private void record(EffectEvent event) {
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

    private void openCombatBracket() {
        if (combatState != null) {
            return;
        }
        var phase = game.getPhaseHandler();
        combatSubstep = phase != null && phase.getPhase() != null
                && phase.getPhase().toString().contains("FIRST_STRIKE")
                ? "first_strike" : "regular";
        combatState = snapshots.toJson(null, List.of());
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
        combatState = null;
        combatSubstep = null;
    }

    private String combatPayload() {
        StringJoiner attackers = new StringJoiner(",", "[", "]");
        var combat = game.getCombat();
        if (combat != null) {
            for (Card attacker : combat.getAttackers()) {
                attackers.add(Json.string(SnapshotBuilder.entityId(attacker)));
            }
        }
        StringJoiner events = new StringJoiner(",", "[", "]");
        for (EffectEvent event : combatEvents) {
            events.add(event.toJson());
        }
        return "{\"attackers\":" + attackers
                + ",\"blocks\":{}"
                + ",\"assignment_choices\":{}"
                + ",\"events\":" + events + "}";
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
