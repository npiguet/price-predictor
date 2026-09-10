package com.pricepredictor.connector.effects;

import com.google.common.collect.Table;
import forge.card.CardChangedType;
import forge.card.CardTypeView;
import forge.card.ColorSet;
import forge.card.RemoveType;
import forge.card.StateChangedType;
import forge.card.WordChangedType;
import forge.game.CardTraitBase;
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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;
import java.util.concurrent.atomic.AtomicBoolean;

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
    /**
     * The bracket this game's clause events are filed into, or null when none
     * is wired: {@code recordClauseEvent} is the door, and only the bracket
     * collector for this same game knows which resolution is open.
     */
    private BusBracketCollector bracket;
    /** Probe branches waiting for the combat record they mirror. */
    private final List<ForkCollector.HeldProbe> heldProbes = new ArrayList<>();
    /** Boards a continuous static has already been recorded on, for coalescing. */
    private final Set<String> coalescedBoards = new LinkedHashSet<>();
    /** Records already written this game, by a hash of what they say. */
    private final Set<Long> coalescedRecords = new java.util.HashSet<>();
    /** Kept trigger evaluations per mode: {fired, not fired}. */
    private final Map<String, int[]> triggerKept = new HashMap<>();
    /** Per mode, negatives offered since the last record that mode wrote. */
    private final Map<String, int[]> triggerOffers = new HashMap<>();
    /** Records written per acting line and verdict, for the flood cap. */
    private final Map<String, int[]> triggerPerLine = new HashMap<>();
    private final List<String> installed = new ArrayList<>();
    /** Static ability by layer-table id, resolved once per game. */
    private final Map<Long, StaticAbility> staticsById = new HashMap<>();
    private final java.util.Random sampler;

    /**
     * Whether the engine can say what a card would be without one static.
     *
     * <p>Separate from {@link AttributionMode}, which probes the trigger-cause
     * hook. These three arrived later, so a checkout patched against an older
     * revision of the patch reports {@code patched} and still cannot suppress.
     */
    static final boolean CAN_SUPPRESS = PatchHooks.present(
            "forge.game.card.Card",
            "getTypeWithout", "getColorWithout", "getKeywordsWithout");

    private long recordsWritten;

    /** The caps and budgets every collecting supervisor shares (FR-028). */
    public record CollectionCaps(
            int manaCap,
            double playabilityRate,
            int interventionsPerGame,
            int probesPerGame,
            List<String> probeKeywords,
            List<Integer> snapshotTiers,
            double legalityRate) {

        public static CollectionCaps defaults() {
            return new CollectionCaps(
                    1, 0.1, 2, 2, List.of(), List.of(1, 2, 3), 0.1);
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
                    listProperty("effect.probe.keywords"),
                    tierProperty("effect.snapshot.tiers", defaults.snapshotTiers()),
                    doubleProperty("effect.legality.rate", defaults.legalityRate()));
        }

        /**
         * The snapshot depth, as the builder wants it.
         *
         * <p>One run-level value read once, so no collector can pick its own.
         * The builder refuses a vector that is not a prefix of {1,2,3,4}, which
         * fails the worker at startup rather than writing a corpus whose tier
         * list means something different on each kind of record.
         */
        public int[] snapshotTierArray() {
            int[] tiers = new int[snapshotTiers.size()];
            for (int i = 0; i < tiers.length; i++) {
                tiers[i] = snapshotTiers.get(i);
            }
            return tiers;
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

        /** A comma-separated tier vector, falling back rather than failing. */
        private static List<Integer> tierProperty(String name, List<Integer> fallback) {
            String value = System.getProperty(name);
            if (value == null || value.isBlank()) {
                return fallback;
            }
            List<Integer> parsed = new ArrayList<>();
            for (String part : value.split(",")) {
                if (part.isBlank()) {
                    continue;
                }
                try {
                    parsed.add(Integer.parseInt(part.trim()));
                } catch (NumberFormatException e) {
                    return fallback;
                }
            }
            return parsed.isEmpty() ? fallback : List.copyOf(parsed);
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
        // The run's vector, not this collector's own. Every collector in the
        // worker takes the same one, because a tier list that varies by record
        // kind is collection metadata a model can read: in the first corpus
        // tier 4 appeared on the interventional records and on nothing else.
        this.snapshots = new SnapshotBuilder(game, caps.snapshotTierArray());
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

    /**
     * Give the clause-event hook somewhere to file what it builds.
     *
     * <p>Optional in the same sense {@link #withForks} is: on a checkout
     * without the hook, or a caller that never wires this, {@code install()}
     * still installs the listener where the patch offers it, and the handler's
     * null check just means nothing is ever filed.
     */
    public void withBracket(BusBracketCollector bracket) {
        this.bracket = bracket;
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
        if (PatchHooks.install(
                PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener",
                clauseHandler())) {
            installed.add("clause-events");
        }
        if (PatchHooks.install(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener",
                outcomeHandler())) {
            installed.add("effect-outcome");
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
        PatchHooks.uninstall(
                PatchHooks.ABILITY_UTILS, "setEffectRecordClauseListener");
        PatchHooks.uninstall(
                PatchHooks.EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener");
        installed.clear();
        // Last, and outside the hook teardown: the question the tally answers
        // is about the run, so it is asked once per game whatever the game did.
        JVM_REWRITES.endOfGame();
    }

    // ── rewrite records ─────────────────────────────────────────────────

    /**
     * One record per replacement, carrying the event it received, the one it
     * produced, which of the five results it returned and what ran instead.
     *
     * <p>Two replacements stacked on one event each get their own record, and
     * the second's incoming is the first's outgoing — which only holds because
     * the hook deep-copies the parameter map before the call.
     *
     * <p>The last two arguments are read by length rather than assumed present.
     * The listener is installed reflectively over whatever interface the patched
     * checkout declares, so its arity is a runtime fact: against a Forge jar
     * built before the result and the substituted ability were added, this is
     * still a three-argument call and the record honestly says it learnt
     * neither. Indexing {@code args[3]} unguarded would instead throw inside a
     * dynamic proxy, and the engine would surface that in the middle of a
     * replacement as an UndeclaredThrowableException.
     *
     * <p>Package-private rather than private, which none of the other handlers
     * needs to be: that argument order is a runtime contract with no compiler
     * behind it, so {@code RewriteContractTest} calls this with a synthesized
     * argument array in the compiler's place.
     */
    InvocationHandler rewriteHandler() {
        return (proxy, method, args) -> {
            if (!"onReplacement".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            // final-fix-3.md item 2: a fork's own replacements resolve
            // through this same JVM-static hook (ForkCollector.forceResolution
            // runs the fork's forced ability through the real resolution
            // pipeline, which is where replacements apply). args[0] is
            // declared ReplacementEffect on Forge's side -- always a
            // CardTraitBase with a real host -- so a positively-identified
            // other game is dropped; anything this cannot identify is left
            // alone rather than newly dropped.
            if (args[0] instanceof CardTraitBase trait && !belongsToLiveGame(trait)) {
                return null;
            }
            EffectRecord record = rewriteRecord(
                    args[0], args[1], args[2],
                    args.length > 3 ? args[3] : null,
                    args.length > 4 ? args[4] : null);
            if (!allowDistinctRecord(
                    "rewrite", record.payloadJson(), record.stateJson())) {
                return null;
            }
            emit(record);
            return null;
        };
    }

    /**
     * One replacement's record, built from the effect the hook hands over.
     *
     * <p>Package-private rather than inlined into the handler: what the record
     * says about the acting line is the part worth testing, and a dynamic proxy
     * over an interface that only exists on a patched checkout is not a thing a
     * unit test can call.
     *
     * <p>The host is taken as delivered, deliberately not re-resolved through
     * {@code game.getCardState(host)} the way {@code executeReplacementInternal}
     * does: that answers an alternate-state card, and the entity id in
     * {@code state.entities} is the one this object carries.
     *
     * <p><b>Every replacement writes a record, including the ones that declined.</b>
     * There used to be a rule here dropping a pair whose halves read alike, and
     * it took 87% of them, because a substitution leaves the map untouched and
     * looked exactly like a replacement that did nothing. The result argument is
     * what tells those apart, so there is no longer such a thing as an
     * uninformative rewrite record: a {@code not_replaced} is this channel's
     * negative, the same role a non-fired evaluation plays on the trigger
     * channel, and it is what teaches when a replacement applies rather than
     * only what it does when it has.
     *
     * @param result     Forge's {@code ReplacementResult}, or null where the
     *                   replacement threw or the listener predates the argument
     * @param replacedBy the {@code ReplaceWith$} ability that stood in for the
     *                   event, or null where none ran
     */
    EffectRecord rewriteRecord(
            Object trait, Object before, Object after, Object result,
            Object replacedBy) {
        CardTraitBase acting = trait instanceof CardTraitBase ctb ? ctb : null;
        Card source = acting == null ? null : acting.getHostCard();
        EffectEvent incoming = describeParams(trait, before);
        EffectEvent rendered = describeParams(trait, after);
        // Null rather than a copy, and this is the whole of the contract. The
        // two halves were byte-identical on 87% of the candidates because a
        // substitution does not edit the map -- it runs another ability and
        // returns a result -- so a copy here says "changed nothing observable"
        // about a replacement that exiled the card instead of killing it. What
        // survives the comparison is the genuine in-place edit:
        // ReplaceCounterEffect.setCount, a halved draw, Orim's Cure taking five
        // combat damage down to one.
        EffectEvent outgoing =
                incoming.toJson().equals(rendered.toJson()) ? null : rendered;
        CardTraitBase ran = replacedBy instanceof CardTraitBase ctb ? ctb : null;
        // Keyed through the same resolver as the record's own acting line, so
        // both join the provenance sidecar identically. Usually it answers that
        // same printed line, because ReplaceWith$ names an SVar on the card the
        // replacement is printed on -- what the field carries that `ability`
        // does not is that an ability ran at all, and the rarer case where the
        // overriding ability was granted from somewhere else.
        List<ProvenanceKey> ranKeys = ran == null ? List.of() : keysOf(ran);
        String wireResult = EffectRecord.rewriteResult(result);
        JVM_REWRITES.record(
                modeOf(trait), wireResult, outgoing != null, !ranKeys.isEmpty(),
                before, after);
        return new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_REWRITE, mode)
                .actor(controllerOf(source))
                .ability(resolveKey(acting))
                .state(snapshots.toJsonForTrait(source, referencedOf(source), incoming))
                .payload(EffectRecord.rewritePayload(
                        incoming, outgoing, wireResult, ranKeys));
    }

    /**
     * What the rewrite channel wrote, for the whole worker process.
     *
     * <p>It counted drops until the payload could say which of the five
     * {@code ReplacementResult}s happened. That rule took 87% of the
     * candidates, and it had to: a substitution runs another ability rather
     * than editing the parameter map, so "exile it instead" and "the
     * replacement declined" arrived as the same pair of identical halves. With
     * the result recorded there is nothing left to drop, so this counts what
     * the channel produced instead — and the question it now answers, in the
     * first hour of a run rather than at the end of one, is whether the channel
     * filled: how many records, and in what proportions of result.
     *
     * <p>The raw-parameter breakdown survives the change of subject, aimed at
     * the records that carry no {@code outgoing}. A key whose <em>value</em>
     * moved while the rendered halves stayed identical still names a parameter
     * this collector does not read, and that is still the difference between
     * "this replacement rewrote nothing" and "this normaliser cannot express
     * what it rewrote". What is new is that such a record is now written rather
     * than discarded, so the diagnostic says what a written record is missing
     * rather than what a discarded one contained.
     *
     * <p>Per JVM rather than per game: one game's replacements are too few to
     * read a share off, and a worker plays many games. Synchronized because
     * nothing promises Forge keeps one thread for a whole worker, and the cost
     * is paid once per replacement rather than once per event.
     */
    static final class RewriteTally {

        /** Entries kept per breakdown before it stops taking new ones. */
        private static final int MAX_KEYS = 512;
        /** Entries printed per breakdown. */
        private static final int TOP = 8;
        /** Games between two summaries, after the first. */
        private static final int GAMES_PER_SUMMARY = 10;

        /**
         * The result of a record that carries none.
         *
         * <p>Two causes, and the record cannot tell them apart: the replacement
         * threw, or this worker is running against a Forge jar whose listener
         * predates the argument. A run where this is the whole breakdown is the
         * second one, and it is the shape to look for in the first summary a
         * fresh deployment prints.
         */
        private static final String NO_RESULT = "?";

        private final Map<String, long[]> byResult = new java.util.TreeMap<>();
        private final Map<String, long[]> byModeAndResult = new java.util.TreeMap<>();
        private final Map<String, long[]> rewrittenByMode = new java.util.TreeMap<>();
        private final Map<String, long[]> movedButUnread = new java.util.TreeMap<>();
        private final Map<String, long[]> unreadableByMode = new java.util.TreeMap<>();
        private long records;
        private long rewritten;
        private long named;
        private long unreadable;
        private long games;

        /**
         * One replacement, and what its record says.
         *
         * <p>The raw maps are compared as rendered strings rather than by
         * {@code equals}: the engine deep-copies the containers of the incoming
         * map and leaves the leaves as identity references, so a copied
         * {@code Multiset} that was rewritten in place is a different object
         * reading the same text when nothing changed and a different text when
         * something did. Two distinct cards sharing one name read alike, which
         * understates rather than invents — the right way round for a diagnostic
         * whose job is to say "there is more here than the record shows".
         *
         * @param rewritten whether the record carries an {@code outgoing}
         * @param named     whether it names the ability that ran instead
         */
        synchronized void record(
                String mode, String result, boolean rewritten, boolean named,
                Object beforeParams, Object afterParams) {
            records++;
            String modeName = mode == null ? "?" : mode;
            String resultName = result == null ? NO_RESULT : result;
            count(byResult, resultName);
            count(byModeAndResult, modeName + "." + resultName);
            if (named) {
                this.named++;
            }
            if (rewritten) {
                this.rewritten++;
                count(rewrittenByMode, modeName);
                // A record that rewrote something in place already shows what
                // moved, so there is nothing for the gap diagnostic to find.
                return;
            }
            Map<String, Object> before = paramsByName(beforeParams);
            Map<String, Object> after = paramsByName(afterParams);
            Set<String> keys = new LinkedHashSet<>(before.keySet());
            keys.addAll(after.keySet());
            boolean moved = false;
            for (String key : keys) {
                if (String.valueOf(before.get(key))
                        .equals(String.valueOf(after.get(key)))) {
                    continue;
                }
                moved = true;
                String entry = modeName + "." + key;
                if (movedButUnread.size() < MAX_KEYS
                        || movedButUnread.containsKey(entry)) {
                    count(movedButUnread, entry);
                }
            }
            if (moved) {
                unreadable++;
                count(unreadableByMode, modeName);
            }
        }

        /**
         * Close a game, printing the tally at a cadence the first hour answers.
         *
         * <p>The first game always, so a run that is wrong is visibly wrong a
         * minute after it starts rather than eight hours later; every tenth
         * after that, so a worker's log stays readable. A process that saw no
         * replacement at all prints nothing — that is what
         * {@code PatchHooks.report()} already says, and repeating it per game
         * would be noise.
         */
        synchronized void endOfGame() {
            games++;
            if (records == 0) {
                return;
            }
            if (games == 1 || games % GAMES_PER_SUMMARY == 0) {
                System.out.println(summary());
                System.out.flush();
            }
        }

        synchronized long records() {
            return records;
        }

        synchronized long rewritten() {
            return rewritten;
        }

        synchronized long named() {
            return named;
        }

        /** The rendered summary, its own method so a test can read it. */
        synchronized String summary() {
            StringBuilder out = new StringBuilder();
            out.append("Effect rewrites [worker, ").append(games)
                    .append(games == 1 ? " game]: " : " games]: ")
                    .append(records).append(" records, ")
                    .append(rewritten).append(" rewrote a parameter in place (")
                    .append(share(rewritten, records))
                    .append("%), ").append(named)
                    .append(" named the ability that ran instead");
            // The headline of the run. A first summary whose results are all
            // "?" is a listener the patched jar does not have; one that is all
            // not_replaced is a format whose replacements never apply; one with
            // no records at all never reaches here.
            out.append("\n  by result: ").append(top(byResult));
            out.append("\n  by mode and result: ").append(top(byModeAndResult));
            out.append("\n  rewrote in place, by mode: ").append(top(rewrittenByMode));
            // The normaliser gap, and the list of keys that would close it.
            out.append("\n  ").append(unreadable)
                    .append(" carried no outgoing while a raw parameter moved: ")
                    .append(top(movedButUnread));
            out.append("\n  of those, by mode: ").append(top(unreadableByMode));
            return out.toString();
        }

        /** A percentage to one decimal, or zero rather than a division by it. */
        private static double share(long part, long whole) {
            return whole == 0 ? 0.0 : Math.round(1000.0 * part / whole) / 10.0;
        }

        private static void count(Map<String, long[]> into, String key) {
            into.computeIfAbsent(key, k -> new long[1])[0]++;
        }

        /** The heaviest entries, commonest first; the tail is a count. */
        private static String top(Map<String, long[]> counts) {
            if (counts.isEmpty()) {
                return "none";
            }
            List<Map.Entry<String, long[]>> entries =
                    new ArrayList<>(counts.entrySet());
            entries.sort((a, b) -> {
                int byCount = Long.compare(b.getValue()[0], a.getValue()[0]);
                return byCount != 0 ? byCount : a.getKey().compareTo(b.getKey());
            });
            StringJoiner joiner = new StringJoiner(", ");
            for (int i = 0; i < Math.min(TOP, entries.size()); i++) {
                joiner.add(entries.get(i).getKey() + "="
                        + entries.get(i).getValue()[0]);
            }
            if (entries.size() > TOP) {
                joiner.add("and " + (entries.size() - TOP) + " more");
            }
            return joiner.toString();
        }
    }

    /** The worker's rewrite tally, shared by every game it plays. */
    static final RewriteTally JVM_REWRITES = new RewriteTally();

    // ── trigger-fire records ────────────────────────────────────────────

    /**
     * One record per evaluated trigger condition, fired or not.
     *
     * <p>Non-fired negatives are what teach the condition; without them the
     * model would learn the base rate of a trigger type firing and nothing
     * about when. Drawn at roughly 1:1 by sampling the negatives, since a turn
     * evaluates far more conditions than it fires.
     *
     *
     * <p>Package-private rather than private (final-fix-3.md item 2), the same
     * reason {@link #rewriteHandler()} already is: the argument order has no
     * compiler behind it, so this hook's own game-identity test calls it with
     * a synthesized argument array.
     */
    InvocationHandler triggerFireHandler() {
        return (proxy, method, args) -> {
            if (!"onConditionEvaluated".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            // final-fix-3.md item 2: checked first, before this evaluation can
            // spend any of the mode's negative-sampling budget (offer/keep
            // below) -- a fork's trigger evaluations must not skew the live
            // game's ratio bookkeeping any more than they must reach its
            // shard. args[0] is declared Trigger on Forge's side, always a
            // CardTraitBase with a real host; anything this cannot identify
            // is left alone rather than newly dropped.
            if (args[0] instanceof CardTraitBase trait && !belongsToLiveGame(trait)) {
                return null;
            }
            boolean fired = Boolean.TRUE.equals(args[2]);
            String mode = modeOf(args[0]);
            if (!offerTriggerEvaluation(mode == null ? "?" : mode, fired)) {
                return null;
            }
            EffectRecord record = triggerRecord(args[0], args[1], fired);
            if (!allowDistinctRecord(
                    "trigger", record.payloadJson(), record.stateJson())
                    || !allowTriggerLine(record.abilityJson(), fired)) {
                return null;
            }
            // Counted only for a record that is actually written, so a
            // suppressed duplicate does not spend the negatives' quota and the
            // ratio holds over what reaches the shard.
            keepTriggerEvaluation(mode == null ? "?" : mode, fired);
            emit(record);
            return null;
        };
    }

    /** Negatives kept per positive kept, per trigger mode. */
    private static final double TARGET_NEGATIVE_RATIO = 1.0;
    /**
     * Negatives a mode may be offered at full rate before it backs off.
     *
     * <p>Only reached by a mode whose offered negatives are being refused
     * downstream -- by the duplicate check or the per-line flood cap -- so it
     * is a bound on wasted snapshot building, not a sampling parameter.
     */
    private static final int NEGATIVE_OFFERS_AT_FULL_RATE = 4;
    /** No negative is ever kept with probability below this. */
    private static final double MIN_NEGATIVE_RATE = 1.0 / 4096.0;
    /** Records one acting line may write per verdict in one game. */
    private static final int TRIGGER_RECORDS_PER_LINE = 24;

    /**
     * Whether this evaluation should be offered to the writer.
     *
     * <p>A fired evaluation always is. A non-fired one is offered whenever this
     * mode's kept negatives are behind its kept positives -- the deficit is the
     * whole rule, with no rate in front of it.
     *
     * <p>Three designs, and why this is the third. A fixed 2% rate landed at
     * 74:26 instead of 1:1, because <b>a fixed rate cannot hold a ratio</b>:
     * the population ratio varies by mode, board size and turn, and it measured
     * 1:17.7 over that run. A deficit rule with a probability in front of it --
     * keep a negative with probability {@code deficit / estimated run length},
     * so the kept ones spread through the run rather than bunching behind the
     * last firing -- landed at 58:42. That rate is a <b>lagging</b> controller:
     * over a run of the estimated length it catches a negative only about
     * 1 - 1/e of the time, and one game gives a mode too few firings for the
     * carried deficit ever to be repaid. Offering on the deficit alone removes
     * the lag, which is the whole of the remaining eight points.
     *
     * <p>Bunching behind the firing is not a loss worth a controller. The
     * negatives that reach the shard are spread by
     * {@link #allowDistinctRecord} instead: two consecutive non-fired
     * evaluations on an unmoved board render the same record and the second is
     * refused, so this walks forward until it finds one that differs. The
     * negative it lands on is then the one closest in time to the positive it
     * balances, which is the more useful contrast rather than the less.
     *
     * <p>What the deterministic rule needs in exchange is a bound on wasted
     * work, because a refusal downstream leaves the deficit standing and the
     * next evaluation is offered again. A mode whose negatives are all being
     * refused -- every acting line at its per-line cap, say -- would otherwise
     * build a snapshot per evaluation of the busiest hook in the collector.
     * {@code triggerOffers} counts offers since that mode last wrote anything
     * and decays the rate harmonically past
     * {@link #NEGATIVE_OFFERS_AT_FULL_RATE}, so an unfillable deficit costs a
     * few offers rather than all of them.
     *
     * <p>Per mode, and per game -- the maps live on this collector, which is
     * built per game. A mode that never fires contributes no negatives, which
     * is the intended reading of "negatives drawn from same-event-type
     * evaluations". The aggregate ratio is therefore near 1:1 rather than
     * exactly it, and the per-mode ratios are the ones to validate.
     */
    boolean offerTriggerEvaluation(String mode, boolean fired) {
        if (fired) {
            return true;
        }
        int[] kept = triggerKept.computeIfAbsent(mode, m -> new int[2]);
        if (kept[0] * TARGET_NEGATIVE_RATIO - kept[1] <= 0.0) {
            return false;
        }
        int[] offers = triggerOffers.computeIfAbsent(mode, m -> new int[1]);
        offers[0]++;
        if (offers[0] <= NEGATIVE_OFFERS_AT_FULL_RATE) {
            return true;
        }
        double rate = Math.max(
                MIN_NEGATIVE_RATE,
                (double) NEGATIVE_OFFERS_AT_FULL_RATE / offers[0]);
        return sampler.nextDouble() < rate;
    }

    /**
     * Count a written evaluation against its mode's balance.
     *
     * <p>Called only for a record that reached the shard, which is what makes
     * the ratio hold over the corpus rather than over the offers: a negative
     * refused as a duplicate has not been drawn, and the deficit it leaves
     * standing is repaid by the next evaluation of its mode.
     */
    void keepTriggerEvaluation(String mode, boolean fired) {
        triggerKept.computeIfAbsent(mode, m -> new int[2])[fired ? 0 : 1]++;
        // A written record of either verdict opens a fresh window: a positive
        // has just created a debt and a negative has just paid one, and either
        // way the offers spent finding this record are spent.
        triggerOffers.remove(mode);
    }

    /**
     * A flood cap per acting line, beside the duplicate check.
     *
     * <p>The duplicate check collapses records that say the same thing; this
     * bounds the ones that differ only in a board that moved a little. One
     * line's condition is worth a couple of dozen looks in a game, not the
     * hundreds an aura on a creature that keeps being pumped would write.
     */
    boolean allowTriggerLine(String abilityJson, boolean fired) {
        int[] written = triggerPerLine.computeIfAbsent(
                abilityJson + "|" + fired, key -> new int[1]);
        if (written[0] >= TRIGGER_RECORDS_PER_LINE) {
            return false;
        }
        written[0]++;
        return true;
    }

    /**
     * One evaluated condition's record, built from the trigger itself.
     *
     * <p>Built after the sampling gate, so a non-fired evaluation that will not
     * be written never pays for resolving its key — which matters, because this
     * is the hook that fires most often in the whole collector.
     */
    EffectRecord triggerRecord(Object trait, Object runParams, boolean fired) {
        CardTraitBase acting = trait instanceof CardTraitBase ctb ? ctb : null;
        Card source = acting == null ? null : acting.getHostCard();
        EffectEvent event = describeParams(trait, runParams);
        return new EffectRecord(
                writer.nextRecordId(), writer.runId(),
                RecordShardWriter.timestamp(), gameId,
                EffectRecord.KIND_TRIGGER, mode)
                .actor(controllerOf(source))
                .ability(resolveKey(acting))
                .state(snapshots.toJsonForTrait(source, referencedOf(source), event))
                .payload("{\"event\":" + event.toJson()
                        + ",\"fired\":" + fired + "}");
    }

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
     *
     * <p>Package-private rather than private (final-fix-3.md item 2), the same
     * reason {@link #rewriteHandler()} already is: this hook's own
     * game-identity test calls it with a synthesized argument array.
     */
    InvocationHandler playabilityHandler() {
        return (proxy, method, args) -> {
            if (!"onCandidate".equals(method.getName()) || args == null
                    || args.length < 4) {
                return null;
            }
            // final-fix-3.md item 2 / final-fix-4.md item 5: checked before the
            // sampler draw below, not after -- a fork's own candidates are
            // evaluated for playability too (ForkCollector.forceResolution's
            // chooseTargets, and GameSimulator's own AI decisions, both ask the
            // same AiController this hook is installed on), and letting one
            // consume from the shared `sampler` before being dropped would
            // re-randomise the live sample stream for every other subkind that
            // draws from it (legality, trigger negatives, the mana reservoir) --
            // exactly the cost triggerFireHandler's own identity check already
            // avoids by running first. A null candidate is pre-existing,
            // unrelated behaviour (the record still names no ability) and is
            // left alone.
            SpellAbility candidate =
                    args[0] instanceof SpellAbility sa ? sa : null;
            if (candidate != null && !belongsToLiveGame(candidate)) {
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
            // Read now, into values. The comment above is the reason: the
            // ability object is not safe to keep, but what it says at this
            // instant is, and without it the record names no ability at all.
            List<ProvenanceKey> candidateKeys = List.of();
            StringJoiner legalTargets = new StringJoiner(",", "[", "]");
            String manaCost = null;
            if (candidate != null) {
                candidateKeys = keysOf(candidate);
                for (String id : legalTargetsOf(candidate)) {
                    legalTargets.add(Json.string(id));
                }
                if (candidate.getPayCosts() != null) {
                    manaCost = String.valueOf(
                            candidate.getPayCosts().getTotalMana());
                }
            }
            // The candidate goes into the snapshot: refs.source was null on all
            // 11.2 million playability records, so a decision record did not say
            // which card the candidate was even on except through its key.
            String state = snapshots.toJson(
                    candidate,
                    candidate == null || candidate.getHostCard() == null
                            ? List.of() : List.of(candidate.getHostCard()));
            String payload = "{\"candidates\":[{"
                            + "\"ability\":" + keyListJson(candidateKeys)
                            + ",\"verdict\":{\"can_play\":" + canPlay
                            + ",\"affordable\":" + affordable
                            + ",\"has_legal_target\":" + hasLegalTarget + "}"
                            + ",\"legal_targets\":" + legalTargets
                            + ",\"cost_after_adjustment\":"
                            + (manaCost == null
                                    ? "{}"
                                    : "{\"mana\":" + Json.string(manaCost) + "}")
                            // responsible_static needs a cantBeCastStatic hook
                            // that does not exist; the attacker and blocker
                            // subkinds carry theirs, this one does not.
                            + ",\"responsible_static\":[]}]}";
            // The decision path had no dedup at all, and every duplicate
            // playability record measured in the first corpus was this subkind:
            // the AI re-asks the same question about the same board, and the
            // answer repeats verbatim.
            if (!allowDistinctRecord("decision", payload, state)) {
                return null;
            }
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_PLAYABILITY, mode)
                    .subkind("decision")
                    .actor(activePlayerId())
                    .state(state)
                    .payload(payload));
            return null;
        };
    }

    /** Legal-target refs one decision may carry before the payload is capped. */
    private static final int MAX_LEGAL_TARGETS = 32;

    /**
     * What the rules allowed this candidate to target, not what the AI picked.
     *
     * <p>The field was read off {@code getTargets()}, which is the chosen set --
     * so it was present on 1.8% of candidates and, where it was present, meant
     * the wrong thing. The per-entity target-legality head is specified to train
     * on the legal set, and an AI's choice is a policy judgment the spec keeps
     * out of the corpus.
     *
     * <p>Capped, because a wide board can make this list longer than the record
     * around it.
     */
    private static List<String> legalTargetsOf(SpellAbility candidate) {
        if (!candidate.usesTargeting() || candidate.getTargetRestrictions() == null) {
            return List.of();
        }
        List<String> ids = new ArrayList<>();
        for (GameEntity entity
                : candidate.getTargetRestrictions().getAllCandidates(candidate)) {
            String id = entity instanceof Card card
                    ? SnapshotBuilder.entityId(card)
                    : entity instanceof Player player
                            ? SnapshotBuilder.playerId(player) : null;
            if (id != null) {
                ids.add(id);
            }
            if (ids.size() >= MAX_LEGAL_TARGETS) {
                break;
            }
        }
        return ids;
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
     *
     * <p>And then sampled at {@code --legality-rate}, <b>after</b> the dedup so
     * the retained set is a uniform sample of distinct board answers rather
     * than one weighted by how often the AI re-asked. The cap alone still left
     * these at 34.4% of the first corpus against a 5% training share.
     *
     * <p>Package-private rather than private (final-fix-3.md item 2), the same
     * reason {@link #rewriteHandler()} already is: this hook's own
     * game-identity test calls it with a synthesized argument array.
     */
    InvocationHandler combatLegalityHandler() {
        return (proxy, method, args) -> {
            if (args == null) {
                return null;
            }
            // final-fix-3.md item 2: no CardTraitBase here -- the acting
            // argument is the GameEntity itself (defender/attacker), so
            // identity comes straight off it rather than through a host card.
            // Wrong-type or null is left alone rather than newly dropped;
            // emitAttackers/emitBlockers already tolerate both.
            if ("onAttackersComputed".equals(method.getName()) && args.length >= 3) {
                if (!(args[0] instanceof GameEntity defender) || belongsToLiveGame(defender)) {
                    emitAttackers(args[0], asCards(args[1]), asCards(args[2]));
                }
            } else if ("onBlockersComputed".equals(method.getName())
                    && args.length >= 4) {
                if (!(args[0] instanceof GameEntity attacker) || belongsToLiveGame(attacker)) {
                    emitBlockers(
                            args[0] instanceof Card attackerCard ? attackerCard : null,
                            asCards(args[1]), asCards(args[2]),
                            args[3] instanceof Integer min ? min : 0);
                }
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
        if (!allowLegalityRecord("attackers", payload)
                || sampler.nextDouble() > caps.legalityRate()) {
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
        if (!allowLegalityRecord("blockers", payload)
                || sampler.nextDouble() > caps.legalityRate()) {
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
     * <p>Keyed on the rendered payload alone, deliberately: the AI asks these
     * questions repeatedly while it evaluates one combat and the board does not
     * move while it does, so the state adds nothing and would only weaken the
     * cap.
     */
    public boolean allowLegalityRecord(String subkind, String payload) {
        return allowDistinctRecord(subkind, payload, null);
    }

    /** Distinct rendered records this game may hold before dedup stops. */
    private static final int COALESCE_MAX_KEYS = 200_000;

    /**
     * Whether a record saying exactly this has already been written this game.
     *
     * <p>The generalisation of the legality cap to the kinds that had none.
     * 624,679 records of the first corpus — 4.3% of it — were byte-identical to
     * another record of the same game, and dedup existed only for the attacker
     * and blocker subkinds: every duplicate playability record measured was the
     * {@code decision} subkind, which is exactly what an unguarded path
     * predicts. Two records that would be byte-identical carry one observation
     * once, so collapsing them loses nothing.
     *
     * <p>A 64-bit hash rather than the 32-bit {@code String.hashCode} the
     * legality cap used: over the hundred thousand keys one game can produce,
     * a 32-bit space has a real chance of colliding, and a collision here
     * silently drops a record that was not a duplicate.
     *
     * <p>Bounded: past the cap it stops deduplicating rather than growing, so a
     * pathological game reverts to writing duplicates instead of exhausting a
     * 1200 MB heap. The duplicate rate of a run is therefore something to
     * measure rather than assume.
     */
    public boolean allowDistinctRecord(String tag, String payload, String state) {
        if (coalescedRecords.size() >= COALESCE_MAX_KEYS) {
            return true;
        }
        long digest = com.google.common.hash.Hashing.murmur3_128().newHasher()
                .putUnencodedChars(tag).putChar('@')
                .putUnencodedChars(payload == null ? "" : payload).putChar('#')
                .putUnencodedChars(state == null ? "" : state)
                .hash().asLong();
        return coalescedRecords.add(digest);
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
        if (firstStrike && !anyFirstStriker()) {
            // Forge always enters the first-strike step and skips it when no
            // combatant has first or double strike, so the phase event alone is
            // no evidence that a step will happen. Probing anyway spent half the
            // budget on forks that assigned no damage: every empty probe fork in
            // the sampled corpus was a first-strike-step fork, and every
            // first-strike-step fork was empty. Worse, the real game writes no
            // combat record for a step that assigns nothing, so those branches
            // survived and were completed against the *regular* step's record.
            return;
        }
        String substep = firstStrike
                ? ForkCollector.SUBSTEP_FIRST_STRIKE : ForkCollector.SUBSTEP_REGULAR;
        discardStaleProbes(substep);
        // One keyword per game rather than whichever the battlefield order
        // reaches first. With eight probed keywords and a budget of two, the
        // first corpus probed lifelink and trample eight times each and
        // double strike, indestructible, wither and infect not at all -- and
        // gate 2 is a per-keyword decision, so an unprobed keyword gets no
        // check whatever.
        String focus = probeFocus();
        if (probeCombatants(focus, firstStrike, substep) == 0) {
            // The focus is a rotation, not a restriction: a game whose combat
            // carries none of it still spends its budget rather than saving it
            // for a keyword that is not there.
            probeCombatants(null, firstStrike, substep);
        }
    }

    /**
     * Hold a branch until the record it mirrors exists.
     *
     * <p>The one place a branch enters the queue, which is also what lets the
     * pairing rules below be exercised without a live combat to fork.
     */
    void hold(ForkCollector.HeldProbe held) {
        heldProbes.add(held);
    }

    /** This game's turn in the probe-keyword rotation, or null when none. */
    private String probeFocus() {
        List<String> keywords = caps.probeKeywords();
        if (keywords.isEmpty()) {
            return null;
        }
        return keywords.get(
                Math.floorMod(String.valueOf(gameId).hashCode(), keywords.size()));
    }

    /**
     * Take a probe for each combatant carrying a probed keyword.
     *
     * @param focus the one keyword to spend on, or null for any probed one
     * @return how many branches were taken
     */
    private int probeCombatants(String focus, boolean firstStrike, String substep) {
        int taken = 0;
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
                String normalized = normalizeKeyword(original);
                if (!caps.probeKeywords().contains(normalized)
                        || (focus != null && !focus.equals(normalized))) {
                    continue;
                }
                ForkCollector.HeldProbe held =
                        forks.probe(original, card, firstStrike);
                if (held != null) {
                    hold(held);
                    taken++;
                }
            }
        }
        return taken;
    }

    /** Is anything in this combat going to deal first-strike damage? */
    private boolean anyFirstStriker() {
        for (Card card : game.getCardsIn(ZoneType.Battlefield)) {
            if (inCombat(card) && (card.hasFirstStrike() || card.hasDoubleStrike())) {
                return true;
            }
        }
        return false;
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

    /**
     * Complete the branches this combat record is the counterfactual for.
     *
     * <p>The caller does not say which damage step it just wrote, so the oldest
     * held substep is taken to be the one: branches are held in the order the
     * steps happen, and a first-strike branch is therefore completed against the
     * first-strike record and not against the regular one that follows it.
     * Naming the substep outright is one parameter better and is what
     * {@link #writeHeldProbes(String, String)} is for.
     */
    public void writeHeldProbes(String combatRecordId) {
        if (forks == null || heldProbes.isEmpty()) {
            return;
        }
        writeHeldProbes(combatRecordId, heldProbes.get(0).substep());
    }

    /**
     * Complete only the branches taken at this damage step.
     *
     * <p>A branch and the record it mirrors have to describe the same step. In
     * the first corpus they did not: a branch forked at the first-strike step —
     * where nothing was ever assigned — outlived that step and was written
     * against the regular step's record, so gate 2's real-versus-fork
     * difference compared a step that did not happen with one that did.
     */
    public void writeHeldProbes(String combatRecordId, String substep) {
        if (forks == null || heldProbes.isEmpty()) {
            return;
        }
        List<ForkCollector.HeldProbe> remaining = new ArrayList<>();
        for (ForkCollector.HeldProbe held : heldProbes) {
            if (java.util.Objects.equals(held.substep(), substep)) {
                forks.writeHeldProbe(held, combatRecordId);
            } else {
                remaining.add(held);
            }
        }
        heldProbes.clear();
        heldProbes.addAll(remaining);
    }

    /**
     * Drop branches from a step that has been left behind.
     *
     * <p>A branch whose step produced no real record has nothing to mirror, and
     * writing it against the next record is worse than not writing it: the
     * fork's budget is already spent either way, and a mispaired counterfactual
     * is read as a difference the keyword caused.
     */
    public int discardStaleProbes(String substep) {
        int dropped = 0;
        List<ForkCollector.HeldProbe> remaining = new ArrayList<>();
        for (ForkCollector.HeldProbe held : heldProbes) {
            if (java.util.Objects.equals(held.substep(), substep)) {
                remaining.add(held);
            } else {
                dropped++;
            }
        }
        heldProbes.clear();
        heldProbes.addAll(remaining);
        return dropped;
    }

    /** Drop every held branch, for a combat that is over. */
    public int discardHeldProbes() {
        int dropped = heldProbes.size();
        heldProbes.clear();
        return dropped;
    }

    /** Branches taken and not yet completed. */
    public int heldProbeCount() {
        return heldProbes.size();
    }

    // ── clause events (an API's own parameters) ─────────────────────────

    /**
     * The events an effect API promises, taken from the clause that ran.
     *
     * <p>The memo is kept per thread and not per collector: a clause can resolve
     * another, and the engine runs games on more than one thread.
     */
    private final ThreadLocal<Deque<Object>> clauseMemos =
            ThreadLocal.withInitial(ArrayDeque::new);

    /**
     * Whether a clause, replacement, trigger or mana ability belongs to the
     * game this collector is installed for.
     *
     * <p>{@link #clauseHandler()} and {@link #outcomeHandler()} are wired to
     * Forge's clause and outcome hooks, which are plain JVM statics with no
     * {@code Game} reference of their own -- whichever resolution fires them
     * last wins. {@code ForkCollector.forceResolution} fires them too:
     * {@code GameSimulator.resolveStack} builds its own
     * {@code PlayerControllerAi} and unconditionally calls
     * {@code setUseSimulation(AIOption.USE_FULL_SIMULATION)}, regardless of
     * how this game's own two lobby seats were registered, so a forked
     * ability resolves through the exact same
     * {@code AbilityUtils.resolveApiAbility} these hooks are installed
     * against (ruling R31, final-fix-1.md F2 -- reversing an earlier ruling
     * of mine that treated this path as already inert). {@link ForkEventSink}
     * keeps a fork's own <em>bus</em> events out of this game's records by
     * subscribing to the fork's own {@code Game} instead of this one, but
     * these hooks are not bus subscriptions, so that separation never
     * reaches them -- without this check, a fork's {@code vote_taken} or
     * {@code coin_flipped} lands in the live game's bracket with nothing
     * marking it as hypothetical.
     *
     * <p>Game identity is the check because it needs no cooperation from
     * {@code ForkCollector} or {@code GameSimulator}, neither of which know
     * this class exists: {@code GameCopier} builds every copied {@code Card}
     * against a new {@code Game} object ({@code new Card(id, paperCard,
     * newGame)}), so a fork's clause's own host card already carries the
     * answer. Compared by reference, the same way
     * {@code SpellAbility.setActivatingPlayer} itself already compares
     * players ("don't use equals because player might be from simulation")
     * -- a copied {@code Game} is never {@code equals} to the live one by
     * accident, only by being the same object. A clause with no identifiable
     * host is treated as not belonging here rather than let through by
     * default: on the real resolution path a host card is always present
     * (the same assumption {@code AbilityUtils.resolveApiAbilityBody}
     * already makes, unconditionally dereferencing it), so this branch is
     * never live in production and only ever chooses the safer of two
     * defaults for a state that should not occur.
     *
     * <p>Checked in the handlers themselves, immediately before each one's
     * existing {@code bracket != null} (or, for {@link #rewriteHandler()} and
     * {@link #triggerFireHandler()}, sampling/dedup) delivery gate, rather
     * than inside {@link BusBracketCollector#recordClauseEvent}: that method
     * only ever receives the already-built {@code EffectEvent}, not the
     * {@code SpellAbility} that produced it, so checking there would mean
     * widening a documented, two-call-site contract just to re-derive
     * something both callers already have in scope. Both hooks' push/pop
     * bookkeeping in {@link #clauseMemos} is untouched by this: a fork's
     * clause still pushes and pops its own memo exactly as a live one does,
     * so this fix cannot desynchronize the deque F3 balances.
     *
     * <p>Typed {@link CardTraitBase} rather than {@link SpellAbility} (final-
     * fix-3.md item 2): {@code SpellAbility}, {@code ReplacementEffect} and
     * {@code Trigger} all extend it and all declare {@code getHostCard()} on
     * it, so {@link #rewriteHandler()} and {@link #triggerFireHandler()} --
     * whose acting argument is a {@code ReplacementEffect}/{@code Trigger},
     * never a {@code SpellAbility} -- reuse this same check rather than a
     * near-duplicate. Every existing caller passes a {@code SpellAbility},
     * which still satisfies the widened parameter unchanged.
     */
    private boolean belongsToLiveGame(CardTraitBase ability) {
        if (ability == null) {
            return false;
        }
        Card host = ability.getHostCard();
        return host != null && host.getGame() == game;
    }

    /**
     * As {@link #belongsToLiveGame(CardTraitBase)}, for {@link
     * #combatLegalityHandler()}: the attacker/defender Forge hands that hook
     * is a {@code Card} or {@code Player} directly, never a trait with a
     * host, so the game comes straight off the {@code GameEntity} itself.
     */
    private boolean belongsToLiveGame(GameEntity entity) {
        return entity != null && entity.getGame() == game;
    }

    /**
     * Whether a null-ability outcome belongs to the live game (final-fix-3.md
     * item 1; widened final-fix-4.md item 2).
     *
     * <p>{@link #outcomeHandler()}'s two production null-ability sources --
     * {@code GameAction.reveal}'s six-argument funnel (ruling R10) and
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}'s two
     * {@code damage_prevented} sites (rulings R11/R12) -- call
     * {@code EffectRecordOutcomes.note} with a literal {@code null} ability,
     * always, on the live game exactly as on a fork: {@link
     * #belongsToLiveGame(CardTraitBase)} has nothing to compare, and simply
     * keeping every null-ability outcome would readmit exactly what that
     * method exists to keep out.
     *
     * <p>Two signals, checked in this order:
     *
     * <ol>
     *   <li>{@link ForkCollector#isForkRunningOnThisThread()}. Set around the
     *   whole of {@code ForkCollector.intervene} and {@code
     *   ForkCollector.probe} -- both this class's only two fork mechanisms --
     *   so "a fork is running on this thread" is by construction true for
     *   every null-ability outcome a fork can produce today, whichever of the
     *   two produces it and whichever segment of either it fires from. Tried
     *   first because, unlike the pointer below, it does not depend on the
     *   call happening to be nested inside an ability resolution.
     *   <li>The resolving-clause pointer ({@code
     *   AbilityUtils.getEffectRecordSubAbility}, read via {@link
     *   PatchHooks#currentSubAbility()}), when the flag is false. Every
     *   ability resolves through {@code AbilityUtils.resolve}/{@code
     *   resolveApiAbility}, live or forked alike, and both set this pointer to
     *   the ability currently resolving before running its body. Kept as a
     *   second, independent signal instead of removed now that the flag alone
     *   is sufficient for both of this class's current fork mechanisms:
     *   insurance against a future one that resolves an ability without going
     *   through {@code intervene}/{@code probe}. When the flag is true this
     *   branch is redundant with it (both cover the resolving case), not a
     *   second opinion; it only does independent work when the flag is false.
     * </ol>
     *
     * <p>Neither signal fires for a real combat-damage prevention on the live
     * game -- {@code Combat.dealAssignedDamage} is a turn-based action, never
     * inside an ability resolution and never inside a fork -- which is the
     * intended "belongs to the live game" default this method falls through
     * to.
     *
     * <p><b>Why not also use {@code damage_prevented}'s own {@code "source"}
     * param</b> (a raw {@code Card}, held at {@link #outcomeRefOf} before it is
     * converted to a ref -- the reviewer's suggestion, final-fix-4.md item 2):
     * considered and not added. With the flag now covering the entirety of
     * both fork mechanisms, there is no remaining path where {@code source}
     * would catch a fork this method's two signals miss -- it would be a
     * third check for a gap that is already closed, not a narrower one. It is
     * also not general: {@code card_revealed}'s params carry no card at all
     * (just a count and a zone name), so a {@code source}-keyed check could
     * only ever cover one of the two known null-ability types, where the flag
     * and the pointer both cover either.
     */
    private boolean nullAbilityOutcomeBelongsToLiveGame() {
        if (ForkCollector.isForkRunningOnThisThread()) {
            return false;
        }
        if (PatchHooks.currentSubAbility() instanceof SpellAbility resolving) {
            return belongsToLiveGame(resolving);
        }
        return true;
    }

    /**
     * Dispatches by method name on a listener this side never compiles
     * against, so the argument order below is a runtime contract rather than
     * one the compiler enforces: {@code onClauseResolving} is called with
     * {@code args[0]} the clause; {@code onClauseResolved} with
     * {@code args[0]} the same clause and {@code args[1]} whether it threw.
     *
     * <p>Package-private rather than private, which none of the other
     * handlers but {@link #rewriteHandler} needs to be: that argument order
     * has no compiler behind it either, so {@code ClauseContractTest} calls
     * this with a synthesized argument array in the compiler's place.
     */
    InvocationHandler clauseHandler() {
        return (proxy, method, args) -> {
            if (args == null || args.length == 0
                    || !(args[0] instanceof SpellAbility clause)) {
                return null;
            }
            if ("onClauseResolving".equals(method.getName())) {
                clauseMemos.get().push(
                        java.util.Optional.ofNullable(ApiEvents.before(clause)));
                return null;
            }
            if (!"onClauseResolved".equals(method.getName())) {
                return null;
            }
            Deque<Object> memos = clauseMemos.get();
            Object memo = memos.isEmpty() ? null : memos.pop();
            boolean threw = args.length > 1 && Boolean.TRUE.equals(args[1]);
            if (threw) {
                // A clause that threw did not finish, and an event claiming it
                // did would be a fact the game never contained.
                return null;
            }
            Object value = memo instanceof java.util.Optional<?> opt
                    ? opt.orElse(null) : memo;
            if (bracket != null && belongsToLiveGame(clause)) {
                bracket.recordClauseEvent(ApiEvents.after(clause, value));
            }
            return null;
        };
    }

    // ── outcome records (what an effect computed and discarded) ─────────

    /**
     * The last event {@link #outcomeHandler()} built, whether or not a
     * bracket was wired to receive it.
     *
     * <p>{@link #outcomeHandler()} always forwards to {@link #bracket} the
     * way {@link #clauseHandler()} does, so this field is not part of the
     * production path -- it exists only so the argument-order contract can be
     * pinned by this hook's own test without also standing up a {@code Game}
     * and a {@code BusBracketCollector} the way {@code ClauseContractTest}
     * does for the clause hook.
     */
    private EffectEvent lastClauseEvent;

    /** The last event {@link #outcomeHandler()} built, for this hook's own test. */
    EffectEvent lastClauseEvent() {
        return lastClauseEvent;
    }

    /**
     * The outcome types an effect is allowed to name, built from the
     * {@code EffectEvent} constants themselves rather than listed in prose.
     *
     * <p>{@code test_event_schema_completeness.py}'s
     * {@code _emitted_event_types()} is a <em>textual</em> scan for
     * {@code EffectEvent.<NAME>} in this package's source, and a javadoc
     * listing satisfies that scan exactly as well as this set does — which
     * was the defect: before this set existed, all nine types this task
     * wired were referenced only inside a comment, so deleting an effect's
     * {@code note(...)} call, or misspelling the event name, would go
     * unnoticed by the guard built specifically to catch that, one level up
     * from where this whole plan started. This set is what
     * {@link #outcomeHandler()} actually validates incoming types against,
     * so the reference is load-bearing code, not prose a rename can silently
     * strand.
     *
     * <p>{@code CARD_REVEALED} and {@code DAMAGE_PREVENTED} come from the two
     * engine choke points ({@code GameAction.reveal},
     * {@code ReplacementHandler.runSingleReplaceDamageEffect}); the other
     * nine, from an effect directly.
     */
    private static final Set<String> KNOWN_OUTCOME_TYPES = Set.of(
            EffectEvent.COIN_FLIPPED, EffectEvent.CLASH_RESOLVED, EffectEvent.VOTE_TAKEN,
            EffectEvent.PILES_MADE, EffectEvent.DUNGEON_VENTURED, EffectEvent.SPELL_COPIED,
            EffectEvent.PERMANENT_COPIED, EffectEvent.CARD_MADE, EffectEvent.RESTRICTION_CHANGE,
            EffectEvent.CARD_REVEALED, EffectEvent.DAMAGE_PREVENTED);

    /** Set the first time {@link #outcomeHandler()} sees a type outside {@link #KNOWN_OUTCOME_TYPES}. */
    private static final AtomicBoolean UNKNOWN_OUTCOME_TYPE_REPORTED = new AtomicBoolean();

    /**
     * Prints one line to stderr the first time {@link #outcomeHandler()}
     * receives a type outside {@link #KNOWN_OUTCOME_TYPES}, then stays quiet
     * for every one after it — the same latched-once treatment
     * {@code ApiEvents.reportEmitterFailure} and
     * {@code AbilityUtils.reportClauseListenerFailure} give their own hooks'
     * failures, for the same reason: a deleted {@code note(...)} call or a
     * misspelled event name must not go dark silently, which is the exact
     * founding failure mode this whole plan exists to end, recreated one
     * level up.
     */
    private static void reportUnknownOutcomeType(String type) {
        if (UNKNOWN_OUTCOME_TYPE_REPORTED.compareAndSet(false, true)) {
            System.err.println("PatchedCollectors: outcomeHandler() received "
                    + "an outcome type not in KNOWN_OUTCOME_TYPES (\"" + type
                    + "\"); further unrecognized types will not be logged. A "
                    + "renamed or deleted EffectRecordOutcomes.note(...) call "
                    + "in Forge would produce exactly this.");
        }
    }

    /** Package-private seam for this hook's own test; production code never calls it. */
    static void resetUnknownOutcomeTypeLatchForTest() {
        UNKNOWN_OUTCOME_TYPE_REPORTED.set(false);
    }

    /**
     * An effect's own account of what it did.
     *
     * <p>Bound by argument position, because {@code PatchHooks} installs a
     * proxy rather than compiling against the interface: 0 is the ability, 1
     * the event type, 2 the parameters. The event type is a plain string the
     * effect names for itself rather than an enum this side switches on --
     * {@code EffectRecordOutcomes}'s own javadoc explains why: "the caller
     * names the event type, which keeps the vocabulary in the collector that
     * consumes it" -- checked against {@link #KNOWN_OUTCOME_TYPES}, the real
     * vocabulary.
     *
     * <p>Two params entries are not generic parameters:
     * <ul>
     *   <li>{@code "subjects"}, carrying the raw Card/Player list a
     *   restriction-change loop walked (Detain, Goad, MustBlock), is pulled
     *   out and converted to refs on {@link EffectEvent#subject} instead --
     *   the schema's {@code restriction_change} row normalizes only
     *   {@code restriction} and {@code value}, so a raw {@code subjects}
     *   entry left in {@code params} would fail the per-type allow-list in
     *   the Python {@code EventRecord}'s own construction. Its presence also
     *   means the acting ability's own host is <em>not</em> added as a
     *   default subject below: Detain/Goad/MustBlock's host is the spell or
     *   permanent that cast the restriction, not one of the cards it landed
     *   on, and stating otherwise would double it into a subject list a
     *   reader counts against.
     *   <li>{@code "source"} (currently only {@code damage_prevented}),
     *   carrying a raw Card, is converted to an entity ref the same way --
     *   every other producer of a {@code source} key ({@code damage_dealt},
     *   and the trigger-derived {@code damage_prevented}) renders one, and a
     *   display name here would silently fail a join against those refs and
     *   collide two sources sharing a name.
     * </ul>
     *
     * <p>Package-private rather than private, the same reason
     * {@code clauseHandler} is: that argument order has no compiler behind it
     * either, so this hook's own test calls it directly, the way
     * {@code ClauseContractTest} calls {@code clauseHandler}.
     */
    InvocationHandler outcomeHandler() {
        return (proxy, method, args) -> {
            if (!"onOutcome".equals(method.getName()) || args == null || args.length < 3) {
                return null;
            }
            if (!(args[1] instanceof String type)) {
                return null;
            }
            if (!KNOWN_OUTCOME_TYPES.contains(type)) {
                reportUnknownOutcomeType(type);
                return null;
            }
            // An effect that supplies its own "subjects" is authoritative
            // about who this outcome happened to; the host is a fallback
            // default for the effects that supply nothing, not an addition
            // on top of an explicit list (see the "subjects" bullet above).
            boolean explicitSubjects = args[2] instanceof Map<?, ?> paramsMap
                    && paramsMap.containsKey("subjects");
            EffectEvent event = new EffectEvent(type);
            if (!explicitSubjects && args[0] instanceof SpellAbility sa && sa.getHostCard() != null) {
                event.subject(SnapshotBuilder.entityId(sa.getHostCard()));
            }
            if (args[2] instanceof Map<?, ?> params) {
                for (Map.Entry<?, ?> entry : params.entrySet()) {
                    if ("subjects".equals(entry.getKey())) {
                        addOutcomeSubjects(event, entry.getValue());
                        continue;
                    }
                    if ("source".equals(entry.getKey())) {
                        event.param("source", outcomeRefOf(entry.getValue()));
                        continue;
                    }
                    event.param(String.valueOf(entry.getKey()), entry.getValue());
                }
            }
            lastClauseEvent = event;
            boolean liveGame = args[0] instanceof SpellAbility sa
                    ? belongsToLiveGame(sa)
                    : nullAbilityOutcomeBelongsToLiveGame();
            if (bracket != null && liveGame) {
                bracket.recordClauseEvent(event);
            }
            return null;
        };
    }

    /**
     * Converts a raw Card/Player list reported under the {@code "subjects"}
     * params key into the event's own subject refs -- see
     * {@link #outcomeHandler()} for why that key is not a generic param.
     */
    private static void addOutcomeSubjects(EffectEvent event, Object value) {
        if (!(value instanceof Iterable<?> subjects)) {
            return;
        }
        for (Object subject : subjects) {
            if (subject instanceof Card card) {
                event.subject(SnapshotBuilder.entityId(card));
            } else if (subject instanceof Player player) {
                event.subject(SnapshotBuilder.playerId(player));
            }
        }
    }

    /**
     * Converts a raw Card/Player reported under the {@code "source"} params
     * key into an entity/player ref -- see {@link #outcomeHandler()} for why
     * that key is not a generic param. Anything else passes through
     * unconverted, which is what lets a future producer of a differently
     * shaped {@code source} degrade to its own {@code String.valueOf} rather
     * than being silently dropped.
     */
    private static Object outcomeRefOf(Object value) {
        if (value instanceof Card card) {
            return SnapshotBuilder.entityId(card);
        }
        if (value instanceof Player player) {
            return SnapshotBuilder.playerId(player);
        }
        return value;
    }

    // ── mana records ────────────────────────────────────────────────────

    /**
     * One resolution record per mana ability that produced mana.
     *
     * <p>A mana ability never uses the stack, so the bracket collector sees no
     * cast and no resolution for it and the corpus has no effect half at all.
     * That is what leaves the role-polarity probe unrunnable: paying {@code R}
     * is observable from stage one, producing it is not.
     *
     * <p>Package-private rather than private (final-fix-3.md item 2), the same
     * reason {@link #rewriteHandler()} already is: this hook's own
     * game-identity test calls it with a synthesized argument array.
     */
    InvocationHandler manaHandler() {
        return (proxy, method, args) -> {
            if (!"onManaProduced".equals(method.getName()) || args == null
                    || args.length < 3) {
                return null;
            }
            SpellAbility ability = args[0] instanceof SpellAbility sa ? sa : null;
            String produced = String.valueOf(args[2]);
            // final-fix-3.md item 2: a fork's own mana abilities can resolve
            // through the inline path too (forceResolution's forced ability,
            // or costs it pays along the way), and this hook is a JVM static
            // like the others.
            if (ability == null || produced.isEmpty() || !belongsToLiveGame(ability)) {
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
                    .ability(resolveKey(ability))
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
     * not be handed a board that already contains it. That removal is the
     * record's whole premise, so a checkout that cannot do it collects nothing
     * here rather than a corpus whose inputs contain their own labels.
     */
    public void collectContinuous() {
        if (mode == AttributionMode.DEGRADED || !CAN_SUPPRESS) {
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
            nameContributions(card, byStatic, entity);
            for (Table.Cell<Long, Long, KeywordsChange> cell
                    : card.getChangedCardKeywords().cellSet()) {
                Contribution into = contribution(byStatic, cell.getColumnKey(), entity);
                for (KeywordInterface keyword : cell.getValue().getKeywords()) {
                    into.keyword(keyword.getOriginal());
                }
                // The same channel's other half. A static that takes a keyword
                // away had no path to keywords_lost, which is a head field, so
                // "loses flying" and "does nothing" recorded identically.
                for (String removed : cell.getValue().getRemoveKeywords()) {
                    into.keyword("-" + removed);
                }
            }
            for (Map.Entry<Long, List<Object>> changed
                    : changesByStatic(card, "getChangedCardTypesByStatic").entrySet()) {
                Contribution into = contribution(byStatic, changed.getKey(), entity);
                for (Object change : changed.getValue()) {
                    typeTokens(change, into);
                }
            }
            for (Map.Entry<Long, List<Object>> changed
                    : changesByStatic(card, "getChangedCardColorsByStatic").entrySet()) {
                Contribution into = contribution(byStatic, changed.getKey(), entity);
                for (Object change : changed.getValue()) {
                    colorTokens(change, into);
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
            ProvenanceKey.Resolved acting = ProvenanceKey.resolve(source);
            String state = snapshots.toJson(null, List.of(), entry.getKey());
            String payload = "{\"contributions\":" + contributions
                    + ",\"board_hash\":" + Json.string(boardHash) + "}";
            // A second gate behind the per-board one, which cannot see this:
            // two *distinct* static ids -- a second copy of the same anthem,
            // two copies of one aura -- resolve to the same key, the same
            // contributions and the same suppressed board, so they render
            // identically and the board hash says they are different statics.
            if (!allowDistinctRecord("continuous", payload, state)) {
                continue;
            }
            emit(new EffectRecord(
                    writer.nextRecordId(), writer.runId(),
                    RecordShardWriter.timestamp(), gameId,
                    EffectRecord.KIND_CONTINUOUS, mode)
                    .actor(SnapshotBuilder.playerId(source.getHostCard().getController()))
                    .ability(acting)
                    .state(state)
                    .payload(payload));
        }
    }

    private static Contribution contribution(
            Map<Long, Map<String, Contribution>> byStatic, Long staticId, String entity) {
        return byStatic
                .computeIfAbsent(staticId, id -> new LinkedHashMap<>())
                .computeIfAbsent(entity, Contribution::new);
    }

    /**
     * A patched layer accessor's answer, grouped by the static that wrote it.
     *
     * <p>Reflective because the accessor is the patch's, and the P/T and keyword
     * accessors beside it are not: those exist in stock Forge, which is what the
     * connector compiles against. An unpatched checkout answers an empty map,
     * though {@link #collectContinuous()} has already returned by then.
     */
    @SuppressWarnings("unchecked")
    private static Map<Long, List<Object>> changesByStatic(Card card, String accessor) {
        Object answer = PatchHooks.read(card, accessor);
        return answer instanceof Map ? (Map<Long, List<Object>>) answer : Map.of();
    }

    /**
     * A card's name contribution, filed under the static that wrote it.
     *
     * <p>Reflective because {@code getChangedCardNames} is the patch's own
     * accessor. Unlike the type and colour "ByStatic" readers above, that
     * table is not split across layers -- it is keyed (timestamp, static id)
     * exactly like {@code boostPT} -- so its shape here is the P/T loop's
     * rather than theirs: the raw table, filtered by column key directly
     * instead of pre-grouped on the Forge side.
     */
    static void nameContributions(
            Card card, Map<Long, Map<String, Contribution>> byStatic, String entity) {
        if (!(PatchHooks.read(card, "getChangedCardNames")
                instanceof Table<?, ?, ?> names)) {
            return;
        }
        for (Table.Cell<?, ?, ?> cell : names.cellSet()) {
            if (cell.getColumnKey() instanceof Long staticId) {
                nameTokens(cell.getValue(), contribution(byStatic, staticId, entity));
            }
        }
    }

    /**
     * What one type-layer entry does, as tokens.
     *
     * <p>Read off the change object rather than diffed out of the applied type
     * line, for the reason the P/T and keyword channels are: two statics that
     * both grant {@code Creature} are indistinguishable in the result, and the
     * record has to say which one this is.
     *
     * <p>A bare token is a type the static adds, {@code -x} one it removes, and
     * a lone {@code =} marks the rest as a type line the static sets outright
     * rather than adds to. Removing every type of a kind has no name to give, so
     * it gets {@code -all-creature-types} and its siblings.
     */
    static void typeTokens(Object change, Contribution into) {
        if (change instanceof CardChangedType typed) {
            addTypes(into, "", typed.addType());
            addTypes(into, "-", typed.removeType());
            if (typed.addAllCreatureTypes()) {
                into.type("all-creature-types");
            }
            for (RemoveType removed : typed.remove()) {
                into.type("-all-" + kebab(removed.name()));
            }
        } else if (change instanceof StateChangedType state) {
            into.type("=");
            addTypes(into, "", state.type());
        } else if (change instanceof WordChangedType word) {
            // A text change swaps one subtype word for another, which is a
            // removal and an addition in the same entry.
            into.type("-" + SnapshotBuilder.typeName(word.oldWord()));
            into.type(SnapshotBuilder.typeName(word.newWord()));
        }
    }

    private static void addTypes(Contribution into, String prefix, CardTypeView types) {
        if (types == null) {
            return;
        }
        for (var type : types.getCoreTypes()) {
            into.type(prefix + SnapshotBuilder.typeName(type));
        }
        for (var supertype : types.getSupertypes()) {
            into.type(prefix + SnapshotBuilder.typeName(supertype));
        }
        for (String subtype : types.getSubtypes()) {
            into.type(prefix + SnapshotBuilder.typeName(subtype));
        }
    }

    /** {@code CreatureTypes} as {@code creature-types}. */
    private static String kebab(String name) {
        return name.replaceAll("(?<=.)(?=\\p{Upper})", "-")
                .toLowerCase(java.util.Locale.ROOT);
    }

    /**
     * What one colour-layer entry does, as tokens.
     *
     * <p>The letters are the snapshot's, so a static's share of an entity's
     * colours reads the same as the entity's own. A lone {@code =} carries the
     * same meaning it does for types: the static sets the colour rather than
     * adding to it, and a {@code =} with nothing after it turns the entity
     * colourless.
     *
     * <p>Reflective on both components, because the record they belong to is the
     * patch's own nested type and this side cannot name it.
     */
    static void colorTokens(Object change, Contribution into) {
        // Both components or neither: a half-read entry would report a colour
        // change as a replacement, which is the one reading that is never safe
        // to guess at.
        if (!(PatchHooks.read(change, "additional") instanceof Boolean additional)
                || !(PatchHooks.read(change, "color") instanceof ColorSet colors)) {
            return;
        }
        List<String> letters = SnapshotBuilder.colorLetters(colors);
        if (!additional) {
            into.color("=");
            if (letters.isEmpty()) {
                // Setting the colour to nothing is what makes a permanent
                // colourless, which the vocabulary spells C rather than as an
                // empty list.
                into.color("C");
            }
        }
        for (String letter : letters) {
            into.color(letter);
        }
    }

    /**
     * What one name-layer entry does: the whole contribution is the name
     * itself, unlike a type or colour token, and a card either has a static
     * overwriting its name or it does not -- {@code newName} is null on
     * every entry that only ever set {@code addNonLegendaryCreatureNames}.
     *
     * <p>Reflective, because the record it belongs to is the patch's own
     * nested type and this side cannot name it.
     */
    static void nameTokens(Object change, Contribution into) {
        if (PatchHooks.read(change, "newName") instanceof String name) {
            into.name(name);
        }
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
    static final class Contribution {
        private final String entity;
        private int power;
        private int toughness;
        private final Set<String> keywords = new LinkedHashSet<>();
        // Sets rather than lists: a static that writes to two type layers can
        // name the same type twice, and the second says nothing the first did
        // not. Insertion-ordered so a "=" marker stays ahead of what it marks.
        private final Set<String> types = new LinkedHashSet<>();
        private final Set<String> colors = new LinkedHashSet<>();
        // Unlike the token sets above, a name contribution is a single value:
        // a static either overwrites the name or it does not, so the last
        // write standing is the answer rather than an accumulation. Null for
        // the statics that are not one of the handful that rename anything --
        // that is the correct, meaningful answer, not a missing one.
        private String name;

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

        void type(String token) {
            if (token != null && !token.isEmpty()) {
                types.add(token);
            }
        }

        void color(String token) {
            if (token != null && !token.isEmpty()) {
                colors.add(token);
            }
        }

        void name(String value) {
            if (value != null && !value.isEmpty()) {
                name = value;
            }
        }

        String toJson() {
            return "{\"entity\":" + Json.string(entity)
                    + ",\"pt_boost\":[" + power + "," + toughness + "]"
                    + ",\"keywords\":" + tokenJson(keywords)
                    + ",\"types\":" + tokenJson(types)
                    + ",\"colors\":" + tokenJson(colors)
                    + ",\"name\":" + Json.string(name) + "}";
        }

        private static String tokenJson(Set<String> tokens) {
            StringJoiner joiner = new StringJoiner(",", "[", "]");
            for (String token : tokens) {
                joiner.add(Json.string(token));
            }
            return joiner.toString();
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
     * <p>The map is Forge's own {@code AbilityKey} enum, which this side does
     * not link against, so it is read by key <b>name</b> — but by the name of
     * the key, never by the list of them, and what the record carries is the
     * <b>values</b>.
     *
     * <p>That distinction is the whole of the rewrite channel's first defect.
     * The previous reading built both halves of a rewrite record out of the
     * mode, one subject per map entry, and a {@code cause} that was the
     * comma-joined key names — so {@code incoming} and {@code outgoing} were
     * byte-identical on 100% of 62,546 records by construction, and
     * {@code cause} was a near-constant string per event type that named no
     * cause. Every value that matters already has a slot in the event
     * vocabulary, so filling those slots needs no change to the wire format.
     */
    private EffectEvent describeParams(Object trait, Object params) {
        String mode = modeOf(trait);
        Map<String, Object> byName = paramsByName(params);
        EffectEvent event = new EffectEvent(typeOf(mode, byName));
        // The engine's own name for what happened, kept whether or not the
        // vocabulary has a member for it. Forge has some two hundred trigger
        // modes and the vocabulary is a closed set of outcomes, so mapping
        // every one would be a table nobody could keep true; carrying the mode
        // loses nothing and lets a later stage map more of them.
        if (mode != null) {
            event.param("mode", mode);
        }
        if (byName.isEmpty()) {
            return event;
        }
        for (String subject : subjectsOf(byName)) {
            event.subject(subject);
        }
        normalizeParams(event, mode, byName);
        event.cause(causeOf(byName));
        return event;
    }

    /**
     * The run parameters keyed by the name Forge spells them with.
     *
     * <p>{@code AbilityKey.toString()} is the key's own name, so this is a
     * rename rather than a reading: what it buys is that everything below can
     * ask for {@code "Origin"} without linking against the enum.
     */
    private static Map<String, Object> paramsByName(Object params) {
        if (!(params instanceof Map<?, ?> map)) {
            return Map.of();
        }
        Map<String, Object> byName = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            byName.put(String.valueOf(entry.getKey()), entry.getValue());
        }
        return byName;
    }

    /**
     * Which outcome this is, before any parameter is read into it.
     *
     * <p>Almost always the mode's own mapping. The exception is prevention:
     * Forge reports it through the damage keys, and an event that says
     * {@code damage_dealt} with no amount is not what happened.
     */
    private static String typeOf(String mode, Map<String, Object> byName) {
        // Checked before the lookup rather than left to getOrDefault:
        // EVENT_BY_MODE is a Map.of, which throws on a null key instead of
        // answering the default. A trait whose getMode() is absent -- a hook
        // whose signature drifted, so the argument is not the trait we expect --
        // would take that NPE inside a dynamic proxy, and the engine would
        // surface it as an UndeclaredThrowableException in the middle of
        // canRunTrigger.
        String type = mode == null
                ? EffectEvent.STATE_FLAG_CHANGE
                : EVENT_BY_MODE.getOrDefault(mode, EffectEvent.STATE_FLAG_CHANGE);
        if (EffectEvent.DAMAGE_DEALT.equals(type)
                && !byName.containsKey("DamageAmount")
                && byName.containsKey("PreventedAmount")) {
            return EffectEvent.DAMAGE_PREVENTED;
        }
        return type;
    }

    /**
     * Keys that name what an event is <b>about</b>, most specific first.
     *
     * <p>A parameter map names one card several times over: {@code Card},
     * {@code CardLKI} and {@code Affected} routinely all point at it, which is
     * why 377 of 691 sampled trigger events carried a repeated subject. The
     * others in the map — {@code Cause}, {@code DamageSource},
     * {@code LastState*} — are causes and context, not subjects, and reading
     * every entry as one is what made the subject list meaningless.
     */
    private static final List<String> SUBJECT_KEYS = List.of(
            "Affected", "Card", "Player", "DamageTarget", "Attacker",
            // Combat's own names, in the order that keeps each event about the
            // creature it is about: an AttackerBlocked map holds Attacker and
            // Blockers and is about the attacker, a Blocks map holds Blocker
            // and Attackers and is about the blocker, and an AttackersDeclared
            // map holds only the collection. Without these three, every one of
            // them named no subject at all.
            "Blocker", "Attackers", "Attached",
            // Last, and only as a last resort. CardLKI is a copy of the card as
            // it was, so wherever the live object is also in the map one of the
            // keys above names it and this is never reached. Where it is not --
            // MagicStack puts CardLKI, Activator and SpellAbility and no Card at
            // all -- it is the only name the map has for the spell that was
            // cast, and without it all 1,917 SpellCast events of the smoke
            // corpus named no subject whatever.
            "CardLKI");

    /** How many subjects one event may name before the list is capped. */
    private static final int MAX_SUBJECTS = 8;

    /**
     * What this event is about: the first subject key the map carries.
     *
     * <p>A collection value contributes each of its members — a damage-all
     * event is about all of them — deduplicated, because the same card can
     * appear twice in one collection after a state change.
     */
    private static Set<String> subjectsOf(Map<String, Object> byName) {
        for (String key : SUBJECT_KEYS) {
            Object value = byName.get(key);
            if (value == null) {
                continue;
            }
            Set<String> subjects = new LinkedHashSet<>();
            if (value instanceof Iterable<?> members) {
                for (Object member : members) {
                    String id = subjectOf(member);
                    if (id != null && subjects.size() < MAX_SUBJECTS) {
                        subjects.add(id);
                    }
                }
            } else {
                String id = subjectOf(value);
                if (id != null) {
                    subjects.add(id);
                }
            }
            if (!subjects.isEmpty()) {
                return subjects;
            }
        }
        return Set.of();
    }

    /**
     * Keys that name the object a run-parameter map blames, most specific first.
     *
     * <p>Forge spells the causing object six ways and the reading was two of
     * them. Measured over the 28,343 trait-derived events of the smoke corpus:
     * {@code Cause} answers for {@code ChangesZone} (81%), {@code Sacrificed}
     * and {@code Exiled}; {@code SpellAbility} answers for {@code SpellCast}
     * (100%) and {@code LifeLost}. The four added here are where the rest of
     * the volume keeps it — {@code Source} on {@code CounterAdded}
     * ({@code Card.java:1790}) and on {@code LifeGained}
     * ({@code Player.java:487}), {@code SourceSA} on {@code BecomesTarget}
     * ({@code SpellAbilityStackInstance.java:170}), {@code Causer} on
     * {@code Destroyed} ({@code GameAction.java:2161}), {@code Activator} on
     * {@code TapsForMana} ({@code AbilityManaPart.java:251}).
     *
     * <p>Ordered rather than merged because a map often holds several: an
     * explicit {@code Cause} is the engine's own answer and outranks a source
     * card, which outranks the player who happened to be acting.
     *
     * <p>{@code DamageSource} is deliberately absent. The damage types
     * normalize it into their own {@code source} slot, and repeating it here
     * would make the cause channel a copy of a field the reader already has
     * rather than a second fact about the event.
     */
    private static final List<String> CAUSE_KEYS = List.of(
            "Cause", "SpellAbility", "SourceSA", "Source", "Causer", "Activator");

    /**
     * The object that caused this event, as a ref, or null when none is named.
     *
     * <p>{@code cause} is documented as an entity or player ref and was once
     * written as a comma-joined list of run-parameter key names — the one place
     * the schema contradicted itself, and the value nothing could learn from.
     * A cause that is a spell or ability is named by the card it is on, which
     * is the identity the snapshot carries.
     *
     * <p>Null is a real answer and stays one. {@code Phase} carries only the
     * turn's player ({@code PhaseHandler.java:438}, with the {@code Phase} key
     * commented out), {@code Untaps} carries only the card
     * ({@code Card.java:4864}), and {@code Attacks} and {@code Blocks} carry
     * combat's shape and no actor: nothing in the game caused them in the sense
     * this field means, and inventing one would be worse than the gap.
     */
    private static String causeOf(Map<String, Object> byName) {
        for (String key : CAUSE_KEYS) {
            String ref = subjectOf(byName.get(key));
            if (ref != null) {
                return ref;
            }
        }
        return null;
    }

    /**
     * Fill the slots this event type declares, from the values the map holds.
     *
     * <p>Per type and no wider: the vocabulary normalizes an outcome so that the
     * same thing from two different script APIs lands in the same slots, and a
     * key written outside the type's own set is one no reader will look for. A
     * type this has no reading for carries its mode and its cause and nothing
     * else, which is a smaller claim than the identity pair it used to make.
     */
    private static void normalizeParams(
            EffectEvent event, String mode, Map<String, Object> byName) {
        switch (event.type()) {
            case EffectEvent.ZONE_CHANGE -> {
                event.param("from_zone", zoneName(byName.get("Origin")));
                event.param("to_zone", zoneName(byName.get("Destination")));
            }
            case EffectEvent.DAMAGE_DEALT -> {
                event.param("amount", intOf(byName.get("DamageAmount")));
                Boolean combat = boolOf(byName.get("IsCombatDamage"));
                event.param("combat", combat != null
                        ? combat : boolOf(byName.get("IsCombat")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            case EffectEvent.DAMAGE_PREVENTED -> {
                event.param("amount", intOf(byName.get("PreventedAmount")));
                event.param("source", subjectOf(byName.get("DamageSource")));
            }
            case EffectEvent.DESTROYED ->
                    event.param("regenerable", boolOf(byName.get("Regeneration")));
            case EffectEvent.COUNTER_CHANGE -> counterParams(event, byName);
            // "Amount" last: it is how the replacement side spells the same
            // quantity the trigger side calls LifeAmount, and without it a
            // life-total replacement carries no number to differ in.
            case EffectEvent.LIFE_CHANGE -> event.param(
                    "delta",
                    firstInt(byName, "LifeAmount", "LifeGained", "Amount"));
            case EffectEvent.CARD_DRAWN, EffectEvent.CARD_MILLED,
                    EffectEvent.CARD_DISCARDED ->
                    event.param("count", firstInt(byName, "Number", "Num"));
            // Scry and Surveil each spell their own count, and neither is
            // Number: read under the general names alone this type carried no
            // count at all.
            case EffectEvent.CARD_LOOKED_AT -> {
                event.param("count", firstInt(
                        byName, "ScryNum", "SurveilNum", "Number", "Num"));
                event.param("from_zone", zoneName(byName.get("Origin")));
            }
            case EffectEvent.POISON_CHANGE -> event.param(
                    "delta", firstInt(byName, "Amount", "Num", "Number"));
            case EffectEvent.MANA_PRODUCED, EffectEvent.MANA_LOST ->
                    event.param("mana_by_color", manaOf(byName));
            case EffectEvent.PHASED ->
                    // Two modes, one type, and the direction is the whole of
                    // the difference between them.
                    event.param("out", !"PhaseIn".equals(mode));
            case EffectEvent.ATTACHED ->
                    event.param("attached_to", firstRef(
                            byName, "AttachTarget", "Attached", "Card"));
            case EffectEvent.UNATTACHED ->
                    event.param("detached_from", firstRef(
                            byName, "AttachTarget", "Attached", "Card"));
            case EffectEvent.ATTACKERS_DECLARED ->
                    // Attacked first: CombatUtil.checkDeclaredAttacker spells
                    // the entity this creature attacked that way and no other.
                    event.param("defender", firstRef(
                            byName, "Attacked", "Defender", "AttackedTarget",
                            "DefendingPlayer", "Defenders"));
            case EffectEvent.BLOCKERS_DECLARED ->
                    event.param("blocked", firstRef(
                            byName, "Attackers", "Attacker"));
            case EffectEvent.BECAME_BLOCKED ->
                    event.param("blockers", refList(byName, "Blockers", "Blocker"));
            case EffectEvent.FACE_CHANGE ->
                    event.param("to_state", faceState(mode));
            case EffectEvent.PLAYER_LOST ->
                    event.param("reason", textOf(byName.get("LoseReason")));
            case EffectEvent.SPELL_COPIED ->
                    event.param("count", firstInt(byName, "Number", "Num"));
            case EffectEvent.TOKEN_CREATED -> event.param(
                    "count", firstInt(byName, "TokenNum", "Number", "Num"));
            default -> {
                // Tapped, untapped, phased, spell_cast and the generic outcome
                // all carry their subject and their mode, which is everything
                // the vocabulary declares for them.
            }
        }
    }

    /**
     * Which counter moved and by how much.
     *
     * <p>Forge says it two ways. The plain one names the type and the number.
     * The other hands over {@code CounterMap}, a per-player multiset that a
     * counter-replacement rewrites <b>in place</b> — which is also why the
     * engine-side copy of the parameter map has to reach inside it, or both
     * halves of the record read the same however carefully this reads them.
     *
     * <p>A map naming several counter types at once has one delta and no single
     * type: the total is the honest answer and a joined type name would be a
     * value no reader could match.
     */
    private static void counterParams(EffectEvent event, Map<String, Object> byName) {
        Object type = byName.get("CounterType");
        Integer delta = firstInt(
                byName, "CounterNum", "CounterAmount", "NewCounterAmount");
        if (type != null) {
            event.param("counter_type", counterName(type));
        }
        if (delta != null) {
            event.param("delta", delta);
            return;
        }
        if (!(byName.get("CounterMap") instanceof Map<?, ?> counters)) {
            return;
        }
        Map<String, Integer> total = new LinkedHashMap<>();
        for (Object value : counters.values()) {
            if (!(value instanceof com.google.common.collect.Multiset<?> multiset)) {
                continue;
            }
            for (com.google.common.collect.Multiset.Entry<?> entry
                    : multiset.entrySet()) {
                total.merge(
                        counterName(entry.getElement()), entry.getCount(), Integer::sum);
            }
        }
        if (total.isEmpty()) {
            return;
        }
        if (total.size() == 1 && type == null) {
            event.param("counter_type", total.keySet().iterator().next());
        }
        int sum = 0;
        for (Integer count : total.values()) {
            sum += count;
        }
        event.param("delta", sum);
    }

    /**
     * The mana a mana event moved, in the slot the vocabulary declares.
     *
     * <p>Forge hands produced mana over as its own short text -- {@code "G"},
     * {@code "G G"}, {@code "1"} -- so the reading is the one
     * {@link #manaByColor(String)} already does for the activation channel,
     * which is what keeps a tapped land and a replaced mana production
     * comparable. A value that is not text yields nothing rather than an empty
     * object: absent is the honest answer for a mana structure this side does
     * not link against, and {@code {}} would claim no mana was produced.
     */
    private static Map<String, Integer> manaOf(Map<String, Object> byName) {
        Object produced = byName.get("Produced");
        if (!(produced instanceof String)) {
            produced = byName.get("Mana");
        }
        if (!(produced instanceof String text)) {
            return null;
        }
        Map<String, Integer> mana = manaByColor(text.trim());
        return mana.isEmpty() ? null : mana;
    }

    /**
     * Which face a card turned to.
     *
     * <p>Read from the mode rather than from a parameter, because the two modes
     * that reach this type each mean exactly one direction and neither map
     * carries a usable state name: {@code AbilityKey.CardState} exists but is a
     * {@code CardState} object put there by door unlocking alone
     * ({@code Card.java:8146}), and rendering it would fill the slot with an
     * object's {@code toString} instead of a face.
     */
    private static String faceState(String mode) {
        return "TurnFaceUp".equals(mode) ? "face_up" : "transformed";
    }

    /** The first of these keys that names an entity, single or collection. */
    private static String firstRef(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            String ref = subjectOf(byName.get(key));
            if (ref != null) {
                return ref;
            }
            if (byName.get(key) instanceof Iterable<?> members) {
                for (Object member : members) {
                    ref = subjectOf(member);
                    if (ref != null) {
                        return ref;
                    }
                }
            }
        }
        return null;
    }

    /** Every entity these keys name, deduplicated, or null for none. */
    private static List<String> refList(Map<String, Object> byName, String... keys) {
        Set<String> refs = new LinkedHashSet<>();
        for (String key : keys) {
            Object value = byName.get(key);
            if (value instanceof Iterable<?> members) {
                for (Object member : members) {
                    String ref = subjectOf(member);
                    if (ref != null && refs.size() < MAX_SUBJECTS) {
                        refs.add(ref);
                    }
                }
            } else {
                String ref = subjectOf(value);
                if (ref != null && refs.size() < MAX_SUBJECTS) {
                    refs.add(ref);
                }
            }
        }
        return refs.isEmpty() ? null : new ArrayList<>(refs);
    }

    /** A parameter's own text, or null where it holds none. */
    private static String textOf(Object value) {
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value).trim();
        return text.isEmpty() ? null : text;
    }

    private static String counterName(Object type) {
        return type instanceof forge.game.card.CounterType counter
                ? counter.getName().toUpperCase(java.util.Locale.ROOT)
                : String.valueOf(type).toUpperCase(java.util.Locale.ROOT);
    }

    /** A zone, however the parameter spells it. */
    private static String zoneName(Object value) {
        if (value instanceof ZoneType zone) {
            return zone.name().toLowerCase(java.util.Locale.ROOT);
        }
        return value == null
                ? null : String.valueOf(value).toLowerCase(java.util.Locale.ROOT);
    }

    private static Integer firstInt(Map<String, Object> byName, String... keys) {
        for (String key : keys) {
            Integer value = intOf(byName.get(key));
            if (value != null) {
                return value;
            }
        }
        return null;
    }

    private static Integer intOf(Object value) {
        if (value instanceof Number number) {
            return number.intValue();
        }
        try {
            return value == null ? null : Integer.valueOf(String.valueOf(value).trim());
        } catch (NumberFormatException e) {
            // A script variable Forge has not evaluated yet. Absent is the
            // right answer; zero would be a claim.
            return null;
        }
    }

    private static Boolean boolOf(Object value) {
        if (value instanceof Boolean flag) {
            return flag;
        }
        if (value == null) {
            return null;
        }
        String text = String.valueOf(value);
        return "true".equalsIgnoreCase(text) || "false".equalsIgnoreCase(text)
                ? Boolean.valueOf(text.equalsIgnoreCase("true")) : null;
    }

    /**
     * The entity a run-parameter value names, or null when it names none.
     *
     * <p>A replacement's and a trigger's parameter maps are the only place
     * those records say what they are about, and nothing read them: every
     * rewrite and trigger event in the corpus named no subjects at all, which
     * left the pending-event overlay with nothing to attach to.
     */
    private static String subjectOf(Object value) {
        if (value instanceof Card card) {
            return SnapshotBuilder.entityId(card);
        }
        if (value instanceof Player player) {
            return SnapshotBuilder.playerId(player);
        }
        if (value instanceof SpellAbility ability && ability.getHostCard() != null) {
            return SnapshotBuilder.entityId(ability.getHostCard());
        }
        return null;
    }

    /** A trigger's or replacement's mode, by whichever accessor it has. */
    private static String modeOf(Object trait) {
        Object mode = PatchHooks.read(trait, "getMode");
        return mode == null ? null : String.valueOf(mode);
    }

    /**
     * Forge modes that have a counterpart in the event vocabulary.
     *
     * <p>Deliberately partial. The vocabulary names observable outcomes and
     * Forge names engine hooks, so only some pairs are the same thing; the rest
     * carry their mode as a parameter and take the generic type.
     */
    private static final Map<String, String> EVENT_BY_MODE = Map.ofEntries(
            Map.entry("ChangesZone", EffectEvent.ZONE_CHANGE),
            Map.entry("Moved", EffectEvent.ZONE_CHANGE),
            Map.entry("Destroy", EffectEvent.DESTROYED),
            Map.entry("Destroyed", EffectEvent.DESTROYED),
            Map.entry("Sacrificed", EffectEvent.SACRIFICED),
            Map.entry("DamageDone", EffectEvent.DAMAGE_DEALT),
            Map.entry("DamageDealtOnce", EffectEvent.DAMAGE_DEALT),
            Map.entry("DamageAll", EffectEvent.DAMAGE_DEALT),
            Map.entry("Drawn", EffectEvent.CARD_DRAWN),
            Map.entry("Draw", EffectEvent.CARD_DRAWN),
            // The replacement side's spelling of the same event. This table was
            // written from the trigger vocabulary, so 31 of Forge's 42
            // ReplacementTypes fell through to the generic outcome carrying
            // nothing but their mode -- and an event with no value in it cannot
            // differ from itself, which is how a rewrite that halved a draw got
            // dropped as an identity pair. Only the five whose quantity a
            // reading above already picks up are added here; the rest are what
            // the identity tally is instrumented to answer.
            Map.entry("DrawCards", EffectEvent.CARD_DRAWN),
            Map.entry("Discarded", EffectEvent.CARD_DISCARDED),
            Map.entry("Milled", EffectEvent.CARD_MILLED),
            Map.entry("Mill", EffectEvent.CARD_MILLED),
            Map.entry("LifeGained", EffectEvent.LIFE_CHANGE),
            Map.entry("LifeLost", EffectEvent.LIFE_CHANGE),
            Map.entry("GainLife", EffectEvent.LIFE_CHANGE),
            Map.entry("LoseLife", EffectEvent.LIFE_CHANGE),
            Map.entry("LifeReduced", EffectEvent.LIFE_CHANGE),
            Map.entry("PayLife", EffectEvent.LIFE_CHANGE),
            Map.entry("Poisoned", EffectEvent.POISON_CHANGE),
            Map.entry("CounterAdded", EffectEvent.COUNTER_CHANGE),
            Map.entry("CounterAddedOnce", EffectEvent.COUNTER_CHANGE),
            Map.entry("CounterRemoved", EffectEvent.COUNTER_CHANGE),
            Map.entry("AddCounter", EffectEvent.COUNTER_CHANGE),
            Map.entry("RemoveCounter", EffectEvent.COUNTER_CHANGE),
            Map.entry("Taps", EffectEvent.TAPPED),
            Map.entry("TapsForMana", EffectEvent.MANA_PRODUCED),
            Map.entry("Untaps", EffectEvent.UNTAPPED),
            Map.entry("Attached", EffectEvent.ATTACHED),
            Map.entry("Unattach", EffectEvent.UNATTACHED),
            Map.entry("SpellCast", EffectEvent.SPELL_CAST),
            Map.entry("SpellAbilityCast", EffectEvent.SPELL_CAST),
            Map.entry("Countered", EffectEvent.SPELL_COUNTERED),
            Map.entry("TokenCreated", EffectEvent.TOKEN_CREATED),
            Map.entry("CreateToken", EffectEvent.TOKEN_CREATED),
            Map.entry("TokenCreatedOnce", EffectEvent.TOKEN_CREATED),
            Map.entry("Attackers", EffectEvent.ATTACKERS_DECLARED),
            Map.entry("AttackersDeclared", EffectEvent.ATTACKERS_DECLARED),
            // Attacks is one creature being declared as an attacker, which is
            // exactly what attackers_declared names -- and it is the third
            // heaviest trait-derived mode in the corpus (2,632 events), every
            // one of which took the generic type and carried nothing but its
            // own name. As attackers_declared it carries the defender.
            Map.entry("Attacks", EffectEvent.ATTACKERS_DECLARED),
            Map.entry("Blocks", EffectEvent.BLOCKERS_DECLARED),
            // Retargeted: the subject of an AttackerBlocked event is the
            // attacker, not the blocker, so blockers_declared put the wrong
            // creature in the slot named "blocked". became_blocked is the type
            // whose subject is the attacker and whose param is the blockers.
            Map.entry("AttackerBlocked", EffectEvent.BECAME_BLOCKED),
            Map.entry("AttackerBlockedByCreature", EffectEvent.BECAME_BLOCKED),
            Map.entry("TurnFaceUp", EffectEvent.FACE_CHANGE),
            Map.entry("Shuffled", EffectEvent.LIBRARY_SHUFFLED),
            Map.entry("Scry", EffectEvent.CARD_LOOKED_AT),
            Map.entry("Surveil", EffectEvent.CARD_LOOKED_AT),
            Map.entry("PhaseOut", EffectEvent.PHASED),
            Map.entry("PhaseIn", EffectEvent.PHASED),
            Map.entry("Regenerated", EffectEvent.REGENERATED),
            // ── the replacement side's own spellings ────────────────────
            // ReplacementType names its modes differently from TriggerType,
            // and this table was written from the trigger vocabulary. A
            // replacement mode with no entry here takes the generic type and
            // carries nothing but its mode -- and an event with no value in it
            // cannot differ from itself, so the pair is dropped as an identity
            // rewrite however much the replacement changed. Untap was two such
            // drops in the smoke run; the rest are the modes a random sealed
            // game reaches less often.
            Map.entry("Untap", EffectEvent.UNTAPPED),
            Map.entry("Tap", EffectEvent.TAPPED),
            Map.entry("DealtDamage", EffectEvent.DAMAGE_DEALT),
            Map.entry("Counter", EffectEvent.SPELL_COUNTERED),
            Map.entry("CopySpell", EffectEvent.SPELL_COPIED),
            Map.entry("ProduceMana", EffectEvent.MANA_PRODUCED),
            Map.entry("LoseMana", EffectEvent.MANA_LOST),
            Map.entry("Transform", EffectEvent.FACE_CHANGE),
            Map.entry("DeclareBlocker", EffectEvent.BLOCKERS_DECLARED),
            Map.entry("RollDice", EffectEvent.DICE_ROLLED),
            Map.entry("GameWin", EffectEvent.PLAYER_WON),
            Map.entry("GameLoss", EffectEvent.PLAYER_LOST));

    private static String keyListJson(List<ProvenanceKey> keys) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (ProvenanceKey key : keys) {
            joiner.add(key.toJson());
        }
        return joiner.toString();
    }

    /**
     * The printed line a runtime trait came from, as a one-element list.
     *
     * <p>Takes a {@link CardTraitBase} rather than a {@link SpellAbility}
     * because a trigger and a replacement effect are traits too, and both hooks
     * hand the collector the trait object itself — which is what the trigger and
     * rewrite handlers were dropping. A null key is not an error, so a record
     * whose line cannot be attributed still carries its host card and its
     * snapshot rather than being suppressed.
     */
    static List<ProvenanceKey> keysOf(CardTraitBase trait) {
        return resolveKey(trait).keys();
    }

    /**
     * The same lookup, keeping the reason there was no key.
     *
     * <p>What every record-envelope call site wants. {@link #keysOf} throws the
     * reason away, which is how every record with no acting line came to spell
     * it {@code ability: []} and say nothing more — leaving "this card has no
     * printed line anywhere" and "the resolver regressed" as the same row. The
     * list form survives only for the playability payload, where the keys go
     * into a candidate list that has no field to carry a reason.
     */
    static ProvenanceKey.Resolved resolveKey(CardTraitBase trait) {
        return ProvenanceKey.resolve(trait);
    }

    /**
     * The trait's host, as the snapshot's tier-1 referenced list.
     *
     * <p>Without it a dies-trigger's host sits in the graveyard, outside every
     * tier this run collects, and {@code refs.source} would name an entity id
     * that is in no {@code state.entities} — a join the reader cannot follow.
     */
    private static List<Card> referencedOf(Card source) {
        return source == null ? List.of() : List.of(source);
    }

    /**
     * The player whose line acted, falling back to the active player.
     *
     * <p>A triggered or replacement ability belongs to its host's controller,
     * which is often not whoever's turn it is: a death trigger fires on the
     * opponent's turn as often as on its own. Forge decides it the same way —
     * {@code TriggerHandler.runSingleTriggerBody} defaults the controller to
     * the host card's.
     */
    private String controllerOf(Card source) {
        if (source == null || source.getController() == null) {
            return activePlayerId();
        }
        return SnapshotBuilder.playerId(source.getController());
    }

    private String activePlayerId() {
        var phase = game == null ? null : game.getPhaseHandler();
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
