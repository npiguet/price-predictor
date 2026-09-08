package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.game.Game;
import forge.game.GameEntity;
import forge.game.card.Card;
import forge.game.keyword.KeywordInterface;
import forge.game.keyword.KeywordsChange;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.staticability.StaticAbility;
import forge.game.zone.ZoneType;
import org.apache.commons.lang3.tuple.Pair;

import java.lang.reflect.InvocationHandler;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;

/**
 * The four record kinds the engine patch unlocks.
 *
 * <p>{@code rewrite}, {@code trigger}, {@code playability} and {@code mana}
 * records all need a hook at a point the bus never publishes: what a replacement
 * received before it rewrote it, a trigger condition that evaluated to false, a
 * legality verdict the AI computed and discarded, a mana ability that resolved
 * on the inline path. Each is installed reflectively and simply does not run on
 * an unpatched checkout.
 *
 * <p>Two caps live here rather than in the supervisor, because both are
 * per-worker-process quantities the Python side cannot see: {@code --mana-cap}
 * counts resolution records per unique mana-ability text in this JVM, and
 * {@code --playability-rate} samples the {@code decision}-subkind logging points
 * (the {@code attackers} and {@code blockers} ones are always logged, because
 * they fire once per combat rather than once per candidate).
 *
 * <p>{@code continuous} records take no cap at all: coalescing per stable board
 * is itself the cap. A static that applies on every recompute would otherwise
 * write a record per priority pass, and none of them would say anything the
 * first did not.
 */
public final class PatchedCollectors implements AutoCloseable {

    private final Game game;
    private final RecordShardWriter writer;
    private final SnapshotBuilder snapshots;
    private final AttributionMode mode;
    private final String gameId;
    private final CollectionCaps caps;

    /** Activations seen per unique mana text, the reservoir's {@code n}. */
    private final Map<String, Integer> manaRecords = new HashMap<>();
    /** The sampled mana records, held until the game ends. */
    private final Map<String, List<EffectRecord>> manaReservoir = new LinkedHashMap<>();
    /** Lines already offered an intervention, so a redraw does not repeat one. */
    private final Set<String> interveneSeen = new LinkedHashSet<>();
    /** Stage three's forks, or null when the run takes none. */
    private ForkCollector forks;
    /** Probe branches waiting for the combat record they mirror. */
    private final List<ForkCollector.HeldProbe> heldProbes = new ArrayList<>();
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    /** Legality answers already recorded this game, keyed by rendered payload. */
    private final Set<String> coalescedLegality = new LinkedHashSet<>();
    private final List<String> installed = new ArrayList<>();
    /** Static ability by layer-table id, resolved once per game. */
    private final Map<Long, StaticAbility> staticsById = new HashMap<>();
    private final java.util.Random sampler;

    private long recordsWritten;

    /** The caps and budgets every collecting supervisor shares (FR-028). */
    public record CollectionCaps(
            int manaCap,
            double playabilityRate,
            int interventionsPerGame,
            int probesPerGame,
            List<String> probeKeywords) {

        public static CollectionCaps defaults() {
            return new CollectionCaps(1, 0.1, 2, 2, List.of());
        }

        /**
         * Read from the {@code effect.*} system properties the supervisor sets.
         *
         * <p>Each falls back to its default rather than failing, because a
         * worker started without instrumentation sets none of them and one
         * started by an older supervisor may set only some.
         */
        public static CollectionCaps fromSystemProperties() {
            CollectionCaps defaults = defaults();
            return new CollectionCaps(
                    intProperty("effect.mana.cap", defaults.manaCap()),
                    doubleProperty("effect.playability.rate", defaults.playabilityRate()),
                    intProperty("effect.interventions.per.game",
                            defaults.interventionsPerGame()),
                    intProperty("effect.probes.per.game", defaults.probesPerGame()),
                    listProperty("effect.probe.keywords"));
        }

        private static int intProperty(String name, int fallback) {
            try {
                String value = System.getProperty(name);
                return value == null || value.isBlank()
                        ? fallback : Integer.parseInt(value.trim());
            } catch (NumberFormatException e) {
                return fallback;
            }
        }

        private static double doubleProperty(String name, double fallback) {
            try {
                String value = System.getProperty(name);
                return value == null || value.isBlank()
                        ? fallback : Double.parseDouble(value.trim());
            } catch (NumberFormatException e) {
                return fallback;
            }
        }

        private static List<String> listProperty(String name) {
            String value = System.getProperty(name);
            if (value == null || value.isBlank()) {
                return List.of();
            }
            List<String> parsed = new ArrayList<>();
            for (String part : value.split(",")) {
                if (!part.isBlank()) {
                    parsed.add(part.trim());
                }
            }
            return List.copyOf(parsed);
        }

        /**
         * Whether any probe fork should be taken at all.
         *
         * <p>With no keywords named, none is — whatever the build state. The
         * flag is the runtime switch, separate from the build decision gate 2
         * drives.
         */
        public boolean probesEnabled() {
            return !probeKeywords.isEmpty();
        }
    }

    public PatchedCollectors(
            Game game, RecordShardWriter writer, String gameId,
            CollectionCaps caps, long seed) {
        this.game = game;
        this.writer = writer;
        this.gameId = gameId;
        this.caps = caps;
        this.snapshots = new SnapshotBuilder(
                game, new int[]{
                        SnapshotBuilder.TIER_REFERENCED,
                        SnapshotBuilder.TIER_CORE,
                        // Tier 3 arrives with the patch: unreferenced stack
                        // contents, which a rewrite or trigger record needs to
                        // describe what else was waiting to resolve.
                        SnapshotBuilder.TIER_UNREFERENCED_STACK,
                });
        this.mode = AttributionMode.detect();
        this.sampler = new java.util.Random(scramble(seed));
    }

    /**
     * Spread out a seed before it reaches {@link java.util.Random}.
     *
     * <p>Games are seeded consecutively, and a linear congruential generator's
     * first output is close to a linear function of its seed — so consecutive
     * seeds make consecutive games take nearly the same first sampling
     * decisions. That would correlate the mana reservoir and the playability
     * sample across a whole run, in a way no test of one game could see.
     *
     * <p>SplitMix64's finalizer, which is the standard mixer for exactly this.
     */
    private static long scramble(long seed) {
        long z = seed + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Attach stage three's fork collector.
     *
     * <p>Optional and off by default: with none attached, or with
     * {@code --interventions-per-game 0}, no game is ever copied. Separate from
     * the constructor because forking is a build decision gate 2 informs, not
     * something every collecting run should pay for.
     */
    public void withForks(ForkCollector forks) {
        this.forks = forks;
    }

    public long recordsWritten() {
        return recordsWritten + (forks == null ? 0 : forks.recordsWritten());
    }

    /**
     * Subscribes {@link #collectContinuous()} to the event bus.
     *
     * <p>A separate object rather than {@code @Subscribe} on the collector
     * itself, because Forge's bus reflects over every public method of a
     * subscriber and the collector's are the handler factories and the caps.
     */
    public final class PhaseBridge {
        @com.google.common.eventbus.Subscribe
        public void onPhase(forge.game.event.GameEventTurnPhase event) {
            collectContinuous();
            collectInterventions();
            // Before the turn-based action that deals the damage: the phase
            // event fires ahead of it, which is the only moment a probe can
            // fork from a board where nothing has been dealt yet.
            collectProbes();
        }
    }

    /** A bus subscriber that records continuous effects at each phase. */
    public PhaseBridge phaseBridge() {
        return new PhaseBridge();
    }

    /** Hooks that were found and installed; empty on a stock checkout. */
    public List<String> installedHooks() {
        return List.copyOf(installed);
    }

    /**
     * Install every hook the checkout offers.
     *
     * @return how many were found
     */
    public int install() {
        if (mode == AttributionMode.DEGRADED) {
            return 0;
        }
        if (PatchHooks.install(
                PatchHooks.REPLACEMENT_HANDLER, "setEffectRecordListener",
                rewriteHandler())) {
            installed.add("replacement");
        }
        if (PatchHooks.install(
                PatchHooks.TRIGGER_HANDLER, "setEffectRecordTriggerListener",
                triggerFireHandler())) {
            installed.add("trigger-fire");
        }
        if (PatchHooks.install(
                PatchHooks.AI_CONTROLLER, "setEffectRecordPlayabilityListener",
                playabilityHandler())) {
            installed.add("playability");
        }
        if (PatchHooks.install(
                PatchHooks.ABILITY_MANA_PART, "setEffectRecordManaListener",
                manaHandler())) {
            installed.add("mana");
        }
        if (PatchHooks.install(
                PatchHooks.AI_CONTROLLER, "setEffectRecordCombatListener",
                combatLegalityHandler())) {
            installed.add("combat-legality");
        }
        return installed.size();
    }

    @Override
    public void close() {
        // Before the hooks come down: this is the last moment the game can be
        // said to have ended, and the mana reservoir is only decided then.
        flushMana();
        PatchHooks.uninstall(
                PatchHooks.REPLACEMENT_HANDLER, "setEffectRecordListener");
        PatchHooks.uninstall(
                PatchHooks.TRIGGER_HANDLER, "setEffectRecordTriggerListener");
        PatchHooks.uninstall(
                PatchHooks.AI_CONTROLLER, "setEffectRecordPlayabilityListener");
        PatchHooks.uninstall(
                PatchHooks.ABILITY_MANA_PART, "setEffectRecordManaListener");
        PatchHooks.uninstall(
                PatchHooks.AI_CONTROLLER, "setEffectRecordCombatListener");
        installed.clear();
    }

    // ── rewrite records ─────────────────────────────────────────────────

    /**
     * One record per replacement, carrying the event it received and the one it
     * produced.
     *
     * <p>Two replacements stacked on one event each get their own record, and
     * the second's incoming is the first's outgoing — which only holds because
     * the hook deep-copies the parameter map before the call.
     */
    private InvocationHandler rewriteHandler() {
        return (proxy, method, args) -> {
            if (!"onReplacement".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            EffectEvent incoming = describeParams(args[1]);
            EffectEvent outgoing = describeParams(args[2]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_REWRITE, mode)
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"incoming\":" + incoming.toJson()
                            + ",\"outgoing\":" + outgoing.toJson() + "}"));
            return null;
        };
    }

    // ── trigger-fire records ────────────────────────────────────────────

    /**
     * One record per evaluated trigger condition, fired or not.
     *
     * <p>Non-fired negatives are what teach the condition; without them the
     * model would learn the base rate of a trigger type firing and nothing
     * about when. Drawn at roughly 1:1 by sampling the negatives, since a turn
     * evaluates far more conditions than it fires.
     */
    private InvocationHandler triggerFireHandler() {
        return (proxy, method, args) -> {
            if (!"onConditionEvaluated".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            boolean fired = Boolean.TRUE.equals(args[2]);
            if (!fired && sampler.nextDouble() > NEGATIVE_SAMPLE_RATE) {
                return null;
            }
            EffectEvent event = describeParams(args[1]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_TRIGGER, mode)
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"event\":" + event.toJson()
                            + ",\"fired\":" + fired + "}"));
            return null;
        };
    }

    /**
     * Share of non-fired evaluations kept.
     *
     * <p>Tuned to land near 1:1 against the fired ones: a turn evaluates every
     * registered trigger against every event and fires a handful.
     */
    private static final double NEGATIVE_SAMPLE_RATE = 0.02;

    // ── playability records ─────────────────────────────────────────────

    /**
     * Rules-level verdicts only.
     *
     * <p>The AI's policy judgments never reach a record: they are its opinion
     * rather than the game's rules, and training on them would teach the model
     * Forge's play style instead of Magic.
     *
     * <p>Sampled at {@code --playability-rate} because the AI evaluates every
     * candidate at every priority, which would otherwise swamp the corpus. The
     * {@code attackers} and {@code blockers} subkinds are always logged: they
     * fire once per combat, not once per candidate.
     */
    private InvocationHandler playabilityHandler() {
        return (proxy, method, args) -> {
            if (!"onCandidate".equals(method.getName()) || args == null
                    || args.length < 4) {
                return null;
            }
            if (sampler.nextDouble() > caps.playabilityRate()) {
                return null;
            }
            // Snapshot defensively: a verdict can be abandoned mid-evaluation,
            // and the legality check mutates the checked ability's targets, so
            // nothing here may keep a reference to the ability object.
            boolean canPlay = Boolean.TRUE.equals(args[1]);
            boolean affordable = Boolean.TRUE.equals(args[2]);
            boolean hasLegalTarget = Boolean.TRUE.equals(args[3]);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_PLAYABILITY, mode)
                    .subkind("decision")
                    .actor(activePlayerId())
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"candidates\":[{"
                            + "\"ability\":[]"
                            + ",\"verdict\":{\"can_play\":" + canPlay
                            + ",\"affordable\":" + affordable
                            + ",\"has_legal_target\":" + hasLegalTarget + "}"
                            + ",\"legal_targets\":[]"
                            + ",\"cost_after_adjustment\":{}"
                            + ",\"responsible_static\":[]}]}"));
            return null;
        };
    }

    // ── playability records: the legality subkinds ──────────────────────

    /**
     * The legal attacker and blocker sets, and what keeps the rest out.
     *
     * <p>The eighth sampling class, and the only one that says what the rules
     * *forbid* rather than what happened.
     *
     * <p>Deduplicated by payload rather than sampled. The AI asks these
     * questions repeatedly while it evaluates a combat — once per candidate
     * block assignment, once per defender it considers — and the board does not
     * move while it does, so the answers repeat verbatim. Left alone they were
     * four fifths of the corpus at thirteen hundred records a game. Two records
     * that would be byte-identical carry the same observation once, so
     * collapsing them loses nothing, which is what makes this a cap rather than
     * a sample.
     */
    private InvocationHandler combatLegalityHandler() {
        return (proxy, method, args) -> {
            if (args == null) {
                return null;
            }
            if ("onAttackersComputed".equals(method.getName()) && args.length >= 3) {
                emitAttackers(args[0], asCards(args[1]), asCards(args[2]));
            } else if ("onBlockersComputed".equals(method.getName())
                    && args.length >= 4) {
                emitBlockers(
                        args[0] instanceof Card attacker ? attacker : null,
                        asCards(args[1]), asCards(args[2]),
                        args[3] instanceof Integer min ? min : 0);
            }
            return null;
        };
    }

    private void emitAttackers(Object defender, List<Card> candidates, List<Card> legal) {
        if (candidates.isEmpty()) {
            return;
        }
        GameEntity target = defender instanceof GameEntity entity ? entity : null;
        StringJoiner attackers = new StringJoiner(",", "[", "]");
        for (Card card : legal) {
            attackers.add(Json.string(SnapshotBuilder.entityId(card)));
        }
        StringJoiner forbidden = new StringJoiner(",", "[", "]");
        for (Card card : candidates) {
            if (legal.contains(card)) {
                continue;
            }
            forbidden.add(forbiddenJson(
                    card, target == null ? null : cantAttackStatic(card, target)));
        }
        String payload = "{\"legal_attackers\":" + attackers
                + ",\"forbidden\":" + forbidden + "}";
        if (!allowLegalityRecord("attackers", payload)) {
            return;
        }
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_PLAYABILITY, mode)
                .subkind("attackers")
                .actor(activePlayerId())
                .state(snapshots.toJson(null, List.of()))
                .payload(payload));
    }

    private void emitBlockers(
            Card attacker, List<Card> candidates, List<Card> legal, int minBlockers) {
        if (attacker == null || candidates.isEmpty()) {
            return;
        }
        StringJoiner blockers = new StringJoiner(",", "[", "]");
        for (Card card : legal) {
            blockers.add(Json.string(SnapshotBuilder.entityId(card)));
        }
        StringJoiner forbidden = new StringJoiner(",", "[", "]");
        for (Card card : candidates) {
            if (legal.contains(card)) {
                continue;
            }
            forbidden.add(forbiddenJson(card, cantBlockByStatic(attacker, card)));
        }
        String payload = "{\"anchor_attacker\":"
                + Json.string(SnapshotBuilder.entityId(attacker))
                + ",\"legal_blockers\":" + blockers
                + ",\"forbidden\":" + forbidden
                + ",\"min_blockers\":" + minBlockers + "}";
        if (!allowLegalityRecord("blockers", payload)) {
            return;
        }
        emit(new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_PLAYABILITY, mode)
                .subkind("blockers")
                // Anchored on the attacker: "who may block" has no answer
                // without saying what they would be blocking.
                .actor(SnapshotBuilder.playerId(attacker.getController()))
                .state(snapshots.toJson(null, List.of()))
                .payload(payload));
    }

    /**
     * Whether this legality answer is new in this game.
     *
     * <p>Keyed on the rendered payload, so only a byte-identical record is
     * dropped. A legal set that genuinely changed — a creature untapped, an
     * anthem resolved — differs in the payload and records again.
     */
    public boolean allowLegalityRecord(String subkind, String payload) {
        return coalescedLegality.add(subkind + "@" + payload.hashCode());
    }

    /**
     * One entity kept out of a legal set, and the static responsible.
     *
     * <p>The static is often absent, and that is a real answer rather than a
     * gap: a creature that is tapped, or summoning sick, or has "can't attack"
     * printed on it, is stopped by the rules themselves and no static ability
     * is to blame.
     */
    private static String forbiddenJson(Card card, StaticAbility responsible) {
        ProvenanceKey key = responsible == null ? null : ProvenanceKey.of(responsible);
        return "{\"entity\":" + Json.string(SnapshotBuilder.entityId(card))
                + ",\"responsible_static\":["
                + (key == null ? "" : key.toJson()) + "]}";
    }

    /** Reflective, because the engine exposes these only on a patched checkout. */
    private static StaticAbility cantAttackStatic(Card attacker, GameEntity defender) {
        return (StaticAbility) PatchHooks.invokeStatic(
                PatchHooks.CANT_ATTACK_BLOCK, "cantAttackStatic",
                new Class<?>[]{Card.class, GameEntity.class}, attacker, defender);
    }

    private static StaticAbility cantBlockByStatic(Card attacker, Card blocker) {
        return (StaticAbility) PatchHooks.invokeStatic(
                PatchHooks.CANT_ATTACK_BLOCK, "cantBlockByStatic",
                new Class<?>[]{Card.class, Card.class}, attacker, blocker);
    }

    @SuppressWarnings("unchecked")
    private static List<Card> asCards(Object value) {
        return value instanceof List<?> list ? (List<Card>) list : List.of();
    }

    // ── interventional resolutions (stage three) ────────────────────────

    /**
     * Force an ability nobody played, at a phase boundary.
     *
     * <p>Some abilities appear in no record however long collection runs: the
     * AI judges them bad, cannot afford them, or never draws them into a board
     * where they do anything. Observation cannot reach those, so the corpus
     * forks the game and makes one happen.
     *
     * <p>Driven from a phase boundary rather than from the playability hook,
     * even though that hook knows exactly which candidates were declined. The
     * hook fires inside the AI's own evaluation, and forking the game there
     * means copying a game that is mid-decision and re-entering the rules
     * engine underneath it. A phase boundary is the same board with nothing in
     * flight.
     *
     * <p>Costs a game copy and a stack resolution each, which is why
     * {@code --interventions-per-game} defaults low and why nothing happens at
     * all when it is zero.
     */
    public void collectInterventions() {
        if (forks == null || mode == AttributionMode.DEGRADED) {
            return;
        }
        Player actor = game.getPhaseHandler() == null
                ? null : game.getPhaseHandler().getPlayerTurn();
        if (actor == null) {
            return;
        }
        for (Card card : actor.getCardsIn(ZoneType.Hand)) {
            for (SpellAbility candidate : card.getSpellAbilities()) {
                if (!worthIntervening(candidate)) {
                    continue;
                }
                if (!forks.mayIntervene(candidate.getId())) {
                    return;
                }
                interveneSeen.add(manaAbilityText(candidate, ""));
                forks.intervene(candidate, candidate.getId());
            }
        }
    }

    /**
     * Is this line one observation cannot reach anyway?
     *
     * <p>Lands and mana abilities are excluded because every game plays them —
     * forking to force a Mountain onto the battlefield spends a game copy to
     * observe the most common event in the corpus. What is left is the spells
     * and activated abilities the AI passed over, which is the population the
     * intervention exists for.
     *
     * <p>Once per line per game: a card redrawn or returned to hand is the same
     * line, and forcing it twice buys the second copy of one observation.
     */
    private boolean worthIntervening(SpellAbility candidate) {
        if (candidate.isLandAbility() || candidate.isManaAbility()) {
            return false;
        }
        return !interveneSeen.contains(manaAbilityText(candidate, ""));
    }

    // ── damage-step probes (stage three) ────────────────────────────────

    /**
     * Fork this damage step for each probed keyword a combatant carries.
     *
     * <p>Gate 2 perturbs the keyword <b>model-side</b> and asks whether the
     * prediction moves the right way. A probe answers the same question against
     * the engine: the same combat, actually re-run without the keyword, so the
     * model's response has a ground truth to be checked against. It is built
     * only for the keywords gate 2 routed to it, which is why
     * {@code --probe-keywords} is empty by default.
     *
     * <p>The branches are held rather than written. Each mirrors the real combat
     * record, which does not exist until the step this ran ahead of has
     * finished.
     */
    public void collectProbes() {
        if (forks == null || !caps.probesEnabled() || mode == AttributionMode.DEGRADED) {
            return;
        }
        var phase = game.getPhaseHandler();
        if (phase == null || phase.getPhase() == null) {
            return;
        }
        String step = phase.getPhase().toString();
        if (!step.contains("COMBAT_DAMAGE") && !step.contains("FIRST_STRIKE_DAMAGE")) {
            return;
        }
        boolean firstStrike = step.contains("FIRST_STRIKE");
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (!inCombat(card)) {
                continue;
            }
            for (KeywordInterface keyword : card.getKeywords()) {
                String original = keyword.getOriginal();
                if (original == null || original.isEmpty()) {
                    continue;
                }
                // The flag names keywords in the corpus's spelling
                // (first_strike); the card carries Forge's (First Strike).
                // Matched on the normalized form and stripped by the card's own,
                // so neither side has to guess the other's.
                if (!caps.probeKeywords().contains(normalizeKeyword(original))) {
                    continue;
                }
                ForkCollector.HeldProbe held =
                        forks.probe(original, card, firstStrike);
                if (held != null) {
                    heldProbes.add(held);
                }
            }
        }
    }

    /** Is this creature in the combat about to deal damage? */
    private boolean inCombat(Card card) {
        var combat = game.getCombat();
        return combat != null
                && (combat.isAttacking(card) || combat.isBlocking(card));
    }

    /**
     * Forge's spelling of a keyword in the one every reader uses.
     *
     * <p>Mirrors {@code effects.domain.state_snapshot.normalize_keyword}: the
     * parameter after {@code :} goes, spaces become underscores. Two spellings
     * of first strike is exactly the mismatch that made gate 2 report zero
     * observations of it.
     */
    static String normalizeKeyword(String original) {
        int parameter = original.indexOf(':');
        String base = parameter < 0 ? original : original.substring(0, parameter);
        return base.trim().toLowerCase(java.util.Locale.ROOT).replace(' ', '_');
    }

    /** Complete every held branch against the combat record it mirrors. */
    public void writeHeldProbes(String combatRecordId) {
        if (forks == null || heldProbes.isEmpty()) {
            return;
        }
        for (ForkCollector.HeldProbe held : heldProbes) {
            forks.writeHeldProbe(held, combatRecordId);
        }
        heldProbes.clear();
    }

    // ── mana records ────────────────────────────────────────────────────

    /**
     * One resolution record per mana ability that produced mana.
     *
     * <p>A mana ability never uses the stack, so the bracket collector sees no
     * cast and no resolution for it and the corpus has no effect half at all.
     * That is what leaves the role-polarity probe unrunnable: paying {@code R}
     * is observable from stage one, producing it is not.
     */
    private InvocationHandler manaHandler() {
        return (proxy, method, args) -> {
            if (!"onManaProduced".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            SpellAbility ability = args[0] instanceof SpellAbility sa ? sa : null;
            String produced = String.valueOf(args[2]);
            if (ability == null || produced.isEmpty()) {
                return null;
            }
            String key = manaAbilityText(ability, produced);
            int slot = manaReservoirSlot(key);
            if (slot < 0) {
                return null;
            }
            String player = args[1] instanceof Player p
                    ? SnapshotBuilder.playerId(p) : activePlayerId();
            EffectEvent event = new EffectEvent(EffectEvent.MANA_PRODUCED)
                    .subject(player)
                    .param("mana_by_color", manaByColor(produced));
            // Held rather than written: which activation survives is not known
            // until the game ends. The state is snapshotted here, at the moment
            // the mana was made, so a record that does survive describes the
            // board it was actually produced on.
            EffectRecord record = new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_RESOLUTION, mode)
                    .moment("resolution")
                    .actor(player)
                    .ability(keysOf(ability))
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"events\":[" + event.toJson() + "]}");
            List<EffectRecord> held =
                    manaReservoir.computeIfAbsent(key, k -> new ArrayList<>());
            if (slot == held.size()) {
                held.add(record);
            } else {
                held.set(slot, record);
            }
            return null;
        };
    }

    /**
     * The cap's key: the ability, not the instance.
     *
     * <p>Every Mountain shares one text, so counting by text caps the whole
     * class rather than each copy. The produced string rides along because a
     * filter land producing {@code W} and the same land producing {@code U} are
     * different observations of the same line.
     */
    private static String manaAbilityText(SpellAbility ability, String produced) {
        Object api = ability.getApi();
        return (api == null ? "?" : api.toString()) + "|"
                + ability.getHostCard().getName() + "|" + produced;
    }

    /**
     * Forge's produced-mana string as counts per colour letter.
     *
     * <p>Returned as a map rather than rendered here, because
     * {@link EffectEvent#param} renders a map as a JSON object and a string as
     * a JSON string — and the reader expects {@code {"B":1}}, not {@code "{...}"}.
     */
    static Map<String, Integer> manaByColor(String produced) {
        Map<String, Integer> counts = new LinkedHashMap<>();
        for (String atom : produced.split(" ")) {
            if (atom.isEmpty()) {
                continue;
            }
            // A bare number is that many generic; anything else is one mana of
            // the colour it names, "C" included.
            if (atom.chars().allMatch(Character::isDigit)) {
                counts.merge("C", Integer.parseInt(atom), Integer::sum);
            } else {
                counts.merge(atom.toUpperCase(java.util.Locale.ROOT), 1, Integer::sum);
            }
        }
        return counts;
    }

    /**
     * Which reservoir slot this activation takes, or -1 to skip it (FR-037).
     *
     * <p>A Mountain taps a dozen times a game for the same R, and the repeats
     * observe a board that barely moved, so the corpus keeps
     * {@code --mana-cap} of them per game. <b>Which</b> ones is the question
     * this answers, and taking the first would be the wrong answer: a land's
     * first tap is almost always turn one against an empty board, so every mana
     * record would describe the same early game and the model would see a
     * board-independent effect on a board that never varied.
     *
     * <p>Reservoir sampling (Algorithm R) instead: the first {@code cap} fill
     * the reservoir, and the {@code n}th after that replaces a uniformly chosen
     * one with probability {@code cap/n}. Every activation in the game ends up
     * equally likely to survive, without knowing in advance how many there will
     * be.
     *
     * <p>The record is built only when this returns a slot, so the expensive
     * part — the state snapshot — is paid a logarithmic number of times rather
     * than once per activation.
     *
     * <p>The produced mana is part of the key rather than of the count, so a
     * dual land making G and the same land making U each get a reservoir.
     */
    public int manaReservoirSlot(String abilityText) {
        int cap = caps.manaCap();
        if (cap <= 0) {
            return -1;
        }
        int seen = manaRecords.merge(abilityText, 1, Integer::sum);
        if (seen <= cap) {
            return seen - 1;
        }
        int candidate = sampler.nextInt(seen);
        return candidate < cap ? candidate : -1;
    }

    /**
     * Write the sampled mana records, once the game can produce no more.
     *
     * <p>They are held until here because reservoir sampling does not know
     * which activation won until the last one has happened. A game whose JVM
     * dies loses its reservoir — at most {@code --mana-cap} records per mana
     * ability, against a crash that already costs the rest of the game.
     */
    private void flushMana() {
        for (List<EffectRecord> held : manaReservoir.values()) {
            for (EffectRecord record : held) {
                if (record != null) {
                    emit(record);
                }
            }
        }
        manaReservoir.clear();
        manaRecords.clear();
    }

    // ── continuous records ──────────────────────────────────────────────

    /**
     * One record per static ability that is currently changing the board.
     *
     * <p>Called at a phase boundary, which is the cheapest moment the board is
     * stable. The layer tables are keyed by (timestamp, applying static's id),
     * so each entity's share of an effect is a read rather than an inference —
     * an anthem's +1/+1 is recorded against the anthem, on each creature it
     * touches, without diffing anything.
     *
     * <p>The snapshot a continuous record carries has the acting static's own
     * contributions removed, because a model asked to predict the effect must
     * not be handed a board that already contains it.
     */
    public void collectContinuous() {
        if (mode == AttributionMode.DEGRADED) {
            return;
        }
        Map<Long, Map<String, Contribution>> byStatic = new LinkedHashMap<>();
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            String entity = SnapshotBuilder.entityId(card);
            for (Table.Cell<Long, Long, Pair<Integer, Integer>> cell
                    : card.getPTBoostTable().cellSet()) {
                contribution(byStatic, cell.getColumnKey(), entity)
                        .boost(cell.getValue().getLeft(), cell.getValue().getRight());
            }
            for (Table.Cell<Long, Long, KeywordsChange> cell
                    : card.getChangedCardKeywords().cellSet()) {
                Contribution into = contribution(byStatic, cell.getColumnKey(), entity);
                for (KeywordInterface keyword : cell.getValue().getKeywords()) {
                    into.keyword(keyword.getOriginal());
                }
            }
        }

        String boardHash = boardHash();
        for (Map.Entry<Long, Map<String, Contribution>> entry : byStatic.entrySet()) {
            // Id 0 is Forge's "no static" column: a temporary pump written by a
            // resolving ability, which the resolution bracket already records.
            if (entry.getKey() == 0L) {
                continue;
            }
            StaticAbility source = staticById(entry.getKey());
            if (source == null) {
                continue;
            }
            String staticKey = String.valueOf(entry.getKey());
            if (!allowContinuousRecord(staticKey, boardHash)) {
                continue;
            }
            StringJoiner contributions = new StringJoiner(",", "[", "]");
            for (Contribution contribution : entry.getValue().values()) {
                contributions.add(contribution.toJson());
            }
            ProvenanceKey key = ProvenanceKey.of(source);
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_CONTINUOUS, mode)
                    .actor(SnapshotBuilder.playerId(source.getHostCard().getController()))
                    .ability(key == null ? List.of() : List.of(key))
                    .state(snapshots.toJson(null, List.of()))
                    .payload("{\"contributions\":" + contributions
                            + ",\"board_hash\":" + Json.string(boardHash) + "}"));
        }
    }

    private static Contribution contribution(
            Map<Long, Map<String, Contribution>> byStatic, Long staticId, String entity) {
        return byStatic
                .computeIfAbsent(staticId, id -> new LinkedHashMap<>())
                .computeIfAbsent(entity, Contribution::new);
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
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
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
        for (Card card : game.getCardsInGame()) {
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

    /** One entity's share of one static's effect, accumulated across layers. */
    private static final class Contribution {
        private final String entity;
        private int power;
        private int toughness;
        private final Set<String> keywords = new LinkedHashSet<>();

        Contribution(String entity) {
            this.entity = entity;
        }

        void boost(int addPower, int addToughness) {
            power += addPower;
            toughness += addToughness;
        }

        void keyword(String keyword) {
            if (keyword != null && !keyword.isEmpty()) {
                keywords.add(keyword.toLowerCase(java.util.Locale.ROOT));
            }
        }

        String toJson() {
            StringJoiner words = new StringJoiner(",", "[", "]");
            for (String keyword : keywords) {
                words.add(Json.string(keyword));
            }
            return "{\"entity\":" + Json.string(entity)
                    + ",\"pt_boost\":[" + power + "," + toughness + "]"
                    + ",\"keywords\":" + words
                    + ",\"types\":[],\"colors\":[],\"name\":null}";
        }
    }

    /**
     * Whether a continuous record for this static on this board is new.
     *
     * <p>Coalescing per {@code (game, static, board hash)} is the cap: a static
     * applies on every recompute, and a second record for the same board would
     * repeat the first exactly.
     */
    public boolean allowContinuousRecord(String staticKey, String boardHash) {
        return coalescedBoards.add(staticKey + "@" + boardHash);
    }

    // ── plumbing ────────────────────────────────────────────────────────

    /**
     * A patched hook's parameter map as an event.
     *
     * <p>The map's contents are Forge's own {@code AbilityKey} enum, which this
     * side does not link against, so the description is by string. That loses
     * type information the Python reader does not use.
     */
    private EffectEvent describeParams(Object params) {
        EffectEvent event = new EffectEvent(EffectEvent.ZONE_CHANGE);
        if (params instanceof Map<?, ?> map) {
            StringJoiner keys = new StringJoiner(",");
            for (Object key : map.keySet()) {
                keys.add(String.valueOf(key));
            }
            event.param("cause", keys.toString());
        }
        return event;
    }

    private static List<ProvenanceKey> keysOf(SpellAbility ability) {
        if (ability == null) {
            return List.of();
        }
        ProvenanceKey key = ProvenanceKey.of(ability);
        return key == null ? List.of() : List.of(key);
    }

    private String activePlayerId() {
        var phase = game.getPhaseHandler();
        if (phase == null || phase.getPlayerTurn() == null) {
            return null;
        }
        return SnapshotBuilder.playerId(phase.getPlayerTurn());
    }

    private void emit(EffectRecord record) {
        writer.write(record.toJson());
        recordsWritten++;
    }
}
