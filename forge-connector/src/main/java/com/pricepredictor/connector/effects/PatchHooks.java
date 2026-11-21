package com.pricepredictor.connector.effects;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * The inventory of the engine patch's hooks: each one by name, what it
 * unlocks, and which of them a worker needs ({@link #REQUIRED}).
 *
 * <p>{@code forge-connector} compiles against stock Forge, so every hook is
 * reached by name ({@link HookReflection}), and a listener through a dynamic
 * proxy ({@link Listener}). {@link #report()} and {@link #gaps()} name what
 * this checkout lacks.
 */
public final class PatchHooks {

    private PatchHooks() {
    }

    static final String TRIGGER_HANDLER = "forge.game.trigger.TriggerHandler";
    static final String REPLACEMENT_HANDLER =
            "forge.game.replacement.ReplacementHandler";
    static final String ABILITY_UTILS = "forge.game.ability.AbilityUtils";
    static final String EFFECT_RECORD_OUTCOMES = "forge.game.ability.EffectRecordOutcomes";
    static final String COMBAT_LEGALITY_OBSERVER =
            "forge.game.combat.CombatLegalityObserver";
    static final String ABILITY_MANA_PART =
            "forge.game.spellability.AbilityManaPart";
    static final String CANT_ATTACK_BLOCK =
            "forge.game.staticability.StaticAbilityCantAttackBlock";
    static final String CARD = "forge.game.card.Card";
    static final String COMBAT = "forge.game.combat.Combat";
    static final String PLAYER_CONTROLLER_AI = "forge.ai.PlayerControllerAi";
    static final String PHASE_HANDLER = "forge.game.phase.PhaseHandler";
    static final String MAGIC_STACK = "forge.game.zone.MagicStack";
    static final String COST_PAYMENT = "forge.game.cost.CostPayment";
    static final String EFFECT_RECORD_HOOK = "forge.game.ability.EffectRecordHook";
    /** The enum the clause hook's "resolved" call carries, read by constant name. */
    static final String CLAUSE_RUN = "forge.game.ability.ClauseRun";

    /** The resolving clause, and the hook mode detection probes. */
    static final StaticAccessor SUB_ABILITY = new StaticAccessor(
            new HookSite(ABILITY_UTILS, "getEffectRecordSubAbility"),
            "per-clause attribution: an event's attributed_to; this is "
            + "the hook mode detection probes");
    static final ListenerSetter TRIGGER_LISTENER = new ListenerSetter(
            new HookSite(TRIGGER_HANDLER, "setEffectRecordTriggerListener"),
            "trigger records, fired and not; a firing is confirmed by the trigger's run",
            List.of(Callback.ON_CONDITION_EVALUATED, Callback.ON_TRIGGER_RUN));
    static final ListenerSetter REPLACEMENT_LISTENER = new ListenerSetter(
            new HookSite(REPLACEMENT_HANDLER, "setEffectRecordListener"),
            "rewrite records: the event a replacement received, what it "
            + "rewrote in place, which of the five ReplacementResults "
            + "it returned and which ability stood in for the event; "
            + "their board, read before the replacement runs; a buffered "
            + "damage ability and the events of its later run",
            List.of(Callback.ON_REPLACING, Callback.ON_REPLACEMENT,
                    Callback.ON_BUFFERED_ABILITY_STARTING,
                    Callback.ON_BUFFERED_ABILITY_FINISHED));
    static final ListenerSetter CLAUSE_LISTENER = new ListenerSetter(
            new HookSite(ABILITY_UTILS, "setEffectRecordClauseListener"),
            "the events an effect API describes but the bus never announces",
            List.of(Callback.ON_CLAUSE_RESOLVING, Callback.ON_CLAUSE_RESOLVED));
    static final ListenerSetter COMBAT_LISTENER = new ListenerSetter(
            new HookSite(COMBAT_LEGALITY_OBSERVER, "setEffectRecordCombatListener"),
            "the playability record's attackers and blockers subkinds",
            List.of(Callback.ON_ATTACKERS_DECLARABLE, Callback.ON_BLOCKERS_DECLARABLE));
    static final StaticAccessor CANT_ATTACK_STATIC = new StaticAccessor(
            new HookSite(CANT_ATTACK_BLOCK, "cantAttackStatic"),
            "each forbidden attacker's responsible_static");
    static final StaticAccessor CANT_BLOCK_BY_STATIC = new StaticAccessor(
            new HookSite(CANT_ATTACK_BLOCK, "cantBlockByStatic"),
            "each forbidden blocker's responsible_static");
    static final StaticAccessor CANT_BLOCK_STATIC = new StaticAccessor(
            new HookSite(CANT_ATTACK_BLOCK, "cantBlockStatic"),
            "the responsible_static of a blocker no static lets block at all (Pacifism)");
    static final StaticAccessor FAILURE_COUNTS = new StaticAccessor(
            new HookSite(EFFECT_RECORD_HOOK, "failureCounts"),
            "the worker's report of the failures Forge contained in every hook slot,"
                    + " argument builds included");
    static final ListenerSetter MANA_LISTENER = new ListenerSetter(
            new HookSite(ABILITY_MANA_PART, "setEffectRecordManaListener"),
            "mana records, and with them the role-polarity probe",
            List.of(Callback.ON_MANA_PRODUCED));
    static final InstanceAccessor CHANGED_TYPES_BY_STATIC = new InstanceAccessor(
            new HookSite(CARD, "getChangedCardTypesByStatic"),
            "a continuous contribution's types");
    static final InstanceAccessor CHANGED_COLORS_BY_STATIC = new InstanceAccessor(
            new HookSite(CARD, "getChangedCardColorsByStatic"),
            "a continuous contribution's colors");
    static final InstanceAccessor TYPE_WITHOUT = new InstanceAccessor(
            new HookSite(CARD, "getTypeWithout"),
            "removing the acting static's types from a continuous snapshot");
    static final InstanceAccessor COLOR_WITHOUT = new InstanceAccessor(
            new HookSite(CARD, "getColorWithout"),
            "removing the acting static's colours from a continuous snapshot");
    static final InstanceAccessor KEYWORDS_WITHOUT = new InstanceAccessor(
            new HookSite(CARD, "getKeywordsWithout"),
            "removing the acting static's keywords from a continuous snapshot");
    static final InstanceAccessor PT_WITHOUT = new InstanceAccessor(
            new HookSite(CARD, "getNetPowerBreakdownWithout"),
            "removing the acting static's set and boosted power and toughness from a continuous snapshot");
    static final InstanceAccessor CONTROLLER_WITHOUT = new InstanceAccessor(
            new HookSite(CARD, "getControllerWithout"),
            "removing the acting static's control change from a continuous snapshot");
    static final InstanceAccessor TRAITS_WITHOUT = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "getSpellAbilitiesWithout"),
            "removing the lines the acting static granted, and restoring those it removed,"
                    + " in a continuous snapshot");
    static final InstanceAccessor CHANGED_NAMES = new InstanceAccessor(
            new HookSite(CARD, "getChangedCardNames"),
            "a continuous contribution's name");
    static final InstanceAccessor COPY_SOURCE = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordCopySource"),
            "the keyword keys and printed script of a token copy of a copy (Kiki-Jiki's token"
                    + " of a Clone): the card whose printed values its prototype took, read when it was made");
    static final InstanceAccessor COPIED_PERMANENT = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordCopiedPermanent"),
            "the keyword keys of a token copy of a copy whose copy effect added keywords (Kiki-Jiki's"
                    + " token of Dack's Duplicate): the permanent its prototype copied");
    static final InstanceAccessor CLONE_PRINTER = new InstanceAccessor(
            new HookSite("forge.game.card.CardCloneStates", "getEffectRecordPrinter"),
            "the keyword keys and printed script of a Clone whose original's copy effect has since ended"
                    + " (a Clone of a Cytoshaped creature after the turn): the card whose printed values it"
                    + " took, noted when it copied");
    static final InstanceAccessor PRIOR_CONTROLLER = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordPriorController"),
            "a continuous record's sickness of a permanent its static took: the controller it"
                    + " had before its most recent control change");
    static final InstanceAccessor PRIOR_SICKNESS = new InstanceAccessor(
            new HookSite(CARD, "isEffectRecordSickForPriorController"),
            "a continuous record's sickness of a permanent its static took: its summoning-sickness"
                    + " flag had that controller kept it");
    static final InstanceAccessor LEND_TYPE = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "lendEffectRecordType"),
            "a continuous record's static conditions and untapped_production on the types without"
                    + " its static: a card's type line lent for the read and given back");
    static final InstanceAccessor SET_ASIDE_COLORS = new InstanceAccessor(
            new HookSite(CARD, "setAsideEffectRecordColors"),
            "a continuous record's static conditions and untapped_production on the colours without"
                    + " its static: the static's colour changes set aside for the read and put back");
    static final InstanceAccessor PRIMARY_ABILITY = new InstanceAccessor(
            new HookSite(MAGIC_STACK, "getEffectRecordPrimaryAbility"),
            "which player action is in progress: a special action, resolved in place with no push and no"
                    + " cast event, is told from the state-based check that follows a cast");
    static final InstanceAccessor STATIC_EFFECTS_BY_STATIC = new InstanceAccessor(
            new HookSite("forge.game.StaticEffects", "getEffectRecordStaticEffects"),
            "the permanents a static affects, read without the entry the engine's own lookup creates");
    static final InstanceAccessor CHANGED_CARD_TYPES = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordChangedCardTypes"),
            "a resolved effect's type change, joined by its timestamp");
    static final InstanceAccessor CHANGED_CARD_COLORS = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordChangedCardColors"),
            "a resolved effect's colour change, joined by its timestamp");
    static final InstanceAccessor COLOR_LAYERS = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordColorLayers"),
            "the check that a lend of a continuous record's board puts every colour change back");
    static final InstanceAccessor CHANGED_CARD_MANA_COST = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordChangedCardManaCost"),
            "a continuous record's mana cost without its static");
    static final InstanceAccessor HIDDEN_EXTRINSIC_KEYWORDS = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordHiddenExtrinsicKeywords"),
            "a resolved effect's hidden keywords, and a static's, by timestamp and static id");
    static final InstanceAccessor CANT_HAVE_KEYWORDS = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordCantHaveKeywords"),
            "a resolved effect's can't-have keyword rows, joined by its timestamp");
    static final InstanceAccessor TEMP_CONTROLLERS = new InstanceAccessor(
            new HookSite(CARD, "getEffectRecordTempControllers"),
            "a resolved gain-control effect's row, joined by its timestamp");
    static final InstanceAccessor SPELL_ABILITIES_THROUGH = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "getEffectRecordSpellAbilitiesThrough"),
            "the printed slice of a card's lines, and a printed line's index among them");
    static final InstanceAccessor TRIGGERS_THROUGH = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "getEffectRecordTriggersThrough"),
            "the printed slice of a card's triggers, and a printed trigger's index among them");
    static final InstanceAccessor STATIC_ABILITIES_THROUGH = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "getEffectRecordStaticAbilitiesThrough"),
            "the printed slice of a card's statics, and a printed static's index among them");
    static final InstanceAccessor REPLACEMENT_EFFECTS_THROUGH = new InstanceAccessor(
            new HookSite("forge.game.card.CardState", "getEffectRecordReplacementEffectsThrough"),
            "the printed slice of a card's replacement effects, and a printed one's index among them");
    /**
     * The damage beyond lethal damage a creature's damage event carries,
     * measured before the damage was dealt. The same patch puts it in the
     * {@code DamageDone} trigger's run parameters as {@code DamageExcess},
     * which is read by key name and adds no method to probe for: this
     * component's absence names both.
     */
    static final InstanceAccessor DAMAGE_EXCESS = new InstanceAccessor(
            new HookSite("forge.game.event.GameEventCardDamaged", "excess"),
            "a damage_dealt's excess, the damage beyond lethal damage, on a creature's "
            + "damage and a DamageDone trigger's");
    static final InstanceAccessor ASSIGNED_DAMAGE = new InstanceAccessor(
            new HookSite(COMBAT, "getAssignedDamage"),
            "a combat record's assignment_choices");
    static final ListenerSetter CONFIRM_LISTENER = new ListenerSetter(
            new HookSite(PLAYER_CONTROLLER_AI, "setEffectRecordConfirmListener"),
            "the declined outcome: an optional effect the controller "
            + "was offered and turned down",
            List.of(Callback.ON_CONFIRM));
    /**
     * Private to Forge, and called by the directed forks rather than the
     * engine: an answer a fork forces is reported through it, the way the
     * AI's own answers are. On a jar without it, every fork that forces an
     * answer would throw from inside the engine's confirm call.
     */
    static final StaticAccessor NOTE_CONFIRM = new StaticAccessor(
            new HookSite(PLAYER_CONTROLLER_AI, "noteEffectRecordConfirm"),
            "a directed fork's forced answer to an optional effect, reported to the confirm "
            + "listener as an AI answer is; without it every fork that forces one fails",
            Reach.PRIVATE);
    static final ListenerSetter OUTCOME_LISTENER = new ListenerSetter(
            new HookSite(EFFECT_RECORD_OUTCOMES, "setEffectRecordOutcomeListener"),
            "the outcomes an effect computes and discards: a coin's "
            + "result, a clash's winner, a vote's tally, a pile's size",
            List.of(Callback.ON_OUTCOME));
    static final InstanceAccessor PENDING_DELAYED_TRIGGERS = new InstanceAccessor(
            new HookSite(TRIGGER_HANDLER, "getPendingDelayedTriggers"),
            "a snapshot's delayed_triggers and the links naming what each "
            + "is pending on; a fork copy's delayed triggers");
    static final InstanceAccessor SIMULTANEOUS_STACK_ENTRIES = new InstanceAccessor(
            new HookSite(MAGIC_STACK, "getSimultaneousStackEntries"),
            "the collector's own enumeration at a priority stop taking off the triggers it "
            + "queued, and none the game had queued before it");
    static final ListenerSetter STATE_BASED_LOOP_LISTENER = new ListenerSetter(
            new HookSite(PHASE_HANDLER, "setEffectRecordStateBasedLoopListener"),
            "the state-based loop's extent: a resolution's held effect half "
            + "spans it, and every death inside it is filed there",
            List.of(Callback.ON_STATE_BASED_LOOP_BEGIN, Callback.ON_STATE_BASED_LOOP_END));
    static final ListenerSetter PRIORITY_SETTLED_LISTENER = new ListenerSetter(
            new HookSite(PHASE_HANDLER, "setEffectRecordPrioritySettledListener"),
            "the moment both seats' decision candidates are enumerated, "
            + "before the priority holder chooses",
            List.of(Callback.ON_PRIORITY_SETTLED));
    static final ListenerSetter CAST_LISTENER = new ListenerSetter(
            new HookSite(MAGIC_STACK, "setEffectRecordCastListener"),
            "a cost half's board, read before the spell's card moves to the "
            + "stack or the activation's costs are paid (FR-042a)",
            List.of(Callback.ON_CAST_STARTING));
    static final ListenerSetter RESOLUTION_LISTENER = new ListenerSetter(
            new HookSite(MAGIC_STACK, "setEffectRecordResolutionListener"),
            "an effect half's board, read as the resolution starts (FR-042b)",
            List.of(Callback.ON_RESOLUTION_STARTING));
    static final ListenerSetter PAYMENT_LISTENER = new ListenerSetter(
            new HookSite(COST_PAYMENT, "setEffectRecordPaymentListener"),
            "a mana record's board, read before the mana ability's costs are "
            + "paid, and the permanents its cost tapped (FR-042c)",
            List.of(Callback.ON_PAYING_COSTS, Callback.ON_COSTS_PAID));

    /**
     * Every hook the collectors need, and the specification of the patch.
     *
     * <p>This list is the patch's description. The patch itself is a branch in
     * the sibling checkout — {@code effect-record-hooks} in {@code ../forge} —
     * and this repo keeps no exported copy of it, which could disagree with the
     * branch. What a branch cannot hold is *why* each hook exists and what
     * breaks without it, so that lives here, next to the code that reads them.
     *
     * <p>Reconstructing a lapsed patch starts here: each entry names the class,
     * the method, and the record channel it feeds.
     *
     * <p>One change in the patch is not listed, because it adds no method to
     * probe for: {@code GameAction.destroy} puts the causing ability into the
     * {@code Destroyed} trigger's run parameters, where it was in scope and
     * being dropped. Its absence shows up as a destroy record that cannot name
     * what destroyed the permanent, not as a missing hook.
     *
     * <p>Presence is not the whole contract for the {@code set*Listener} hooks.
     * A setter keeps its name when its listener loses a callback or gains
     * arguments, so {@link #report()} also checks each listener interface
     * against its hook's {@link Callback}s, by name and by the arguments the
     * handlers read. A jar whose {@code setEffectRecordListener} listener does
     * not pass the {@code ReplacementResult} is named there; its rewrite
     * records would carry a {@code result} of null on every rewrite. The
     * handlers still read the tail arguments through {@link Args#optional}
     * rather than assuming them, because a partly-current patch must degrade
     * the way an absent one does instead of throwing inside a dynamic proxy.
     */
    public static final List<Hook> REQUIRED = List.of(
            SUB_ABILITY, TRIGGER_LISTENER, REPLACEMENT_LISTENER, CLAUSE_LISTENER,
            COMBAT_LISTENER, CANT_ATTACK_STATIC,
            CANT_BLOCK_BY_STATIC, CANT_BLOCK_STATIC, FAILURE_COUNTS, MANA_LISTENER, CHANGED_TYPES_BY_STATIC,
            CHANGED_COLORS_BY_STATIC, TYPE_WITHOUT, COLOR_WITHOUT, KEYWORDS_WITHOUT,
            PT_WITHOUT, CONTROLLER_WITHOUT, TRAITS_WITHOUT,
            CHANGED_NAMES, COPY_SOURCE, COPIED_PERMANENT, CLONE_PRINTER, PRIOR_CONTROLLER, PRIOR_SICKNESS,
            LEND_TYPE, SET_ASIDE_COLORS, PRIMARY_ABILITY, STATIC_EFFECTS_BY_STATIC,
            CHANGED_CARD_TYPES, CHANGED_CARD_COLORS, COLOR_LAYERS, CHANGED_CARD_MANA_COST,
            HIDDEN_EXTRINSIC_KEYWORDS, CANT_HAVE_KEYWORDS, TEMP_CONTROLLERS, SPELL_ABILITIES_THROUGH,
            TRIGGERS_THROUGH, STATIC_ABILITIES_THROUGH, REPLACEMENT_EFFECTS_THROUGH, DAMAGE_EXCESS, ASSIGNED_DAMAGE, CONFIRM_LISTENER, NOTE_CONFIRM, OUTCOME_LISTENER,
            PENDING_DELAYED_TRIGGERS, SIMULTANEOUS_STACK_ENTRIES, STATE_BASED_LOOP_LISTENER,
            PRIORITY_SETTLED_LISTENER,
            CAST_LISTENER, RESOLUTION_LISTENER, PAYMENT_LISTENER);

    /**
     * The components of the three layer records {@link #CHANGED_COLORS_BY_STATIC},
     * {@link #CHANGED_NAMES} and {@link #CHANGED_CARD_MANA_COST} hand out.
     *
     * <p>Not in {@link #REQUIRED}, because a lookup cannot tell them from stock
     * Forge's: the records exist there too, private, and a record's accessors
     * are public whatever the record's own access, so {@link HookReflection#find} would
     * report them present on a checkout where reading them fails. What the
     * patch changes is the records' visibility, and the two accessors that
     * return them are what imply it.
     */
    static final InstanceAccessor COLOR_CHANGE_ADDITIONAL = new InstanceAccessor(
            new HookSite(CARD + "$CardColor", "additional"),
            "whether a continuous contribution adds its colours or sets them");
    static final InstanceAccessor COLOR_CHANGE_COLORS = new InstanceAccessor(
            new HookSite(CARD + "$CardColor", "color"),
            "a continuous contribution's colours");
    static final InstanceAccessor MANA_COST_CHANGE_MANA = new InstanceAccessor(
            new HookSite(CARD + "$CardManaCost", "mana"),
            "a mana cost change's cost");
    static final InstanceAccessor MANA_COST_CHANGE_ADDITIONAL = new InstanceAccessor(
            new HookSite(CARD + "$CardManaCost", "additional"),
            "whether a mana cost change adds to the cost before it or sets it");
    static final InstanceAccessor NAME_CHANGE_NAME = new InstanceAccessor(
            new HookSite(CARD + "$CardChangedName", "newName"),
            "a continuous contribution's name");

    /**
     * The components of the record {@link #PENDING_DELAYED_TRIGGERS} hands
     * out, the patch's own: one list of pending delayed triggers per kind.
     * Not in {@link #REQUIRED}: the accessor that returns the record is.
     */
    static final InstanceAccessor PENDING_DELAYED = new InstanceAccessor(
            new HookSite(TRIGGER_HANDLER + "$PendingDelayedTriggers", "delayed"),
            "a snapshot's delayed_triggers: every delayed trigger pending");
    static final InstanceAccessor PENDING_THIS_TURN = new InstanceAccessor(
            new HookSite(TRIGGER_HANDLER + "$PendingDelayedTriggers", "thisTurn"),
            "a snapshot's delayed_triggers: those registered for this turn only");
    static final InstanceAccessor PENDING_PLAYER_DEFINED = new InstanceAccessor(
            new HookSite(TRIGGER_HANDLER + "$PendingDelayedTriggers", "playerDefined"),
            "a snapshot's delayed_triggers: those keyed to a player, by player");

    /** The required hooks this checkout does not have. */
    public static List<Hook> missing() {
        List<Hook> absent = new ArrayList<>();
        for (Hook hook : REQUIRED) {
            if (!hook.present()) {
                absent.add(hook);
            }
        }
        return absent;
    }

    /**
     * One line per required hook this checkout lacks, or whose listener lacks
     * a callback or an argument the collectors read; empty on a current
     * patched checkout.
     */
    public static List<String> gaps() {
        return gaps(REQUIRED);
    }

    private static List<String> gaps(List<Hook> required) {
        List<String> gaps = new ArrayList<>();
        for (Hook hook : required) {
            hook.gap().ifPresent(gaps::add);
        }
        return gaps;
    }

    /**
     * One line per {@link #gaps() gap}, for a worker to print at startup.
     *
     * <p>A partly-applied patch is the state worth naming: the mode still reads
     * {@code patched} because that is one hook's answer, while a channel this
     * run was meant to collect is quietly empty. A stale installed jar is the
     * likely case, whose setters are all there and whose listeners are older.
     */
    public static String report() {
        return report(REQUIRED);
    }

    static String report(List<Hook> required) {
        List<String> gaps = gaps(required);
        if (gaps.isEmpty()) {
            return "effect-record hooks: all " + required.size() + " present, each listener whole";
        }
        StringBuilder out = new StringBuilder(
                "effect-record hooks: " + gaps.size() + " of " + required.size()
                        + " missing or short, so these channels stay empty or partial:");
        for (String gap : gaps) {
            out.append("\n  ").append(gap);
        }
        return out.toString();
    }
    /**
     * The constant name of the {@code ClauseRun} the clause hook's "resolved"
     * call carries, or null where the argument is absent or not an enum: a
     * jar whose listener predates it, or one that passed a boolean.
     */
    static String clauseRunName(Object run) {
        return run instanceof Enum<?> constant ? constant.name() : null;
    }

    /**
     * Forge's own count of the failures each hook slot contained in this JVM
     * ({@code EffectRecordHook.failureCounts()}), by slot name; empty on a
     * checkout without it.
     */
    public static Map<String, Long> forgeFailureCounts() {
        Object counts = FAILURE_COUNTS.read();
        Map<String, Long> read = new TreeMap<>();
        if (counts instanceof Map<?, ?> map) {
            map.forEach((slot, n) -> {
                if (n instanceof Number number) {
                    read.put(String.valueOf(slot), number.longValue());
                }
            });
        }
        return read;
    }

    /** The sub-ability currently resolving, when the pointer hook is present. */
    public static Object currentSubAbility() {
        return SUB_ABILITY.read();
    }
}
