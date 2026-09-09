package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.game.card.Card;
import forge.game.cost.CostTap;
import forge.game.mana.Mana;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringJoiner;


/**
 * One record line, assembled field by field.
 *
 * <p>The envelope is frozen before stage-one collection begins, so this builder
 * writes exactly the keys the schema names and nothing else. Later stages widen
 * the corpus — new kinds become reachable, new snapshot tiers appear — but never
 * redefine a field, which is what lets a stage-one corpus stay trainable
 * alongside stage-four records.
 *
 * <p>{@link #rewritePayload} is the one place that widening has been exercised,
 * and it obeys the rule rather than bending it: {@code incoming} keeps exactly
 * the meaning it had, {@code outgoing} gains {@code null} the way a field gains
 * a sentinel, and {@code result} and {@code replaced_by} are new keys. No
 * already-collected record is reinterpreted — an old line that carries no
 * {@code result} reads as "written before this contract", which is a third
 * state and not one of the five results.
 */
public final class EffectRecord {

    public static final String KIND_RESOLUTION = "resolution";
    public static final String KIND_REWRITE = "rewrite";
    public static final String KIND_CONTINUOUS = "continuous";
    public static final String KIND_COMBAT = "combat";
    public static final String KIND_TRIGGER = "trigger";
    public static final String KIND_PLAYABILITY = "playability";

    public static final String MOMENT_ACTIVATION = "activation";
    public static final String MOMENT_RESOLUTION = "resolution";

    public static final String OUTCOME_RESOLVED = "resolved";
    public static final String OUTCOME_FIZZLED = "fizzled";
    public static final String OUTCOME_PARTIALLY_FIZZLED = "partially_fizzled";
    public static final String OUTCOME_DECLINED = "declined";
    public static final String OUTCOME_COUNTERED = "countered";

    public static final String REWRITE_REPLACED = "replaced";
    public static final String REWRITE_NOT_REPLACED = "not_replaced";
    public static final String REWRITE_PREVENTED = "prevented";
    public static final String REWRITE_UPDATED = "updated";
    public static final String REWRITE_SKIPPED = "skipped";

    private final String recordId;
    private final String runId;
    private final String timestamp;
    private final String gameId;
    private final String kind;
    private final String mode;
    private String moment;
    private String subkind;
    private String linkId;
    private String mirrorOf;
    private String variantOf;
    private boolean interventional;
    private boolean fork;
    private boolean synthetic;
    private String actorPlayer;
    private List<ProvenanceKey> ability;
    private String abilityUnresolved;
    private String stateJson = "{}";
    private String payloadJson = "{}";

    public EffectRecord(
            String recordId, String runId, String timestamp, String gameId,
            String kind, AttributionMode mode) {
        this.recordId = recordId;
        this.runId = runId;
        this.timestamp = timestamp;
        this.gameId = gameId;
        this.kind = kind;
        this.mode = mode.wireValue();
    }

    public EffectRecord moment(String value) {
        this.moment = value;
        return this;
    }

    public EffectRecord subkind(String value) {
        this.subkind = value;
        return this;
    }

    /** Joins the two halves of a resolution pair; absent where there is no partner. */
    public EffectRecord linkId(String value) {
        this.linkId = value;
        return this;
    }

    public EffectRecord mirrorOf(String value) {
        this.mirrorOf = value;
        return this;
    }

    public EffectRecord variantOf(String value) {
        this.variantOf = value;
        return this;
    }

    public EffectRecord interventional(boolean value) {
        this.interventional = value;
        return this;
    }

    public EffectRecord fork(boolean value) {
        this.fork = value;
        return this;
    }

    public EffectRecord synthetic(boolean value) {
        this.synthetic = value;
        return this;
    }

    public EffectRecord actor(String playerId) {
        this.actorPlayer = playerId;
        return this;
    }

    /**
     * The acting line. Absent for {@code combat} and {@code playability}, where
     * no single line acts; several keys where the rendered line merged several
     * traits.
     *
     * <p>Prefer {@link #ability(ProvenanceKey.Resolved)} wherever a key was
     * actually looked for: this form cannot say why an empty list is empty.
     */
    public EffectRecord ability(List<ProvenanceKey> keys) {
        this.ability = keys == null ? null : new ArrayList<>(keys);
        return this;
    }

    /**
     * The acting line, or the reason there is none — both set from one answer.
     *
     * <p>The two fields are written together and only together, because they
     * are two halves of one statement and a record that got them out of step
     * would be unreadable: a reason beside a named line is a contradiction, and
     * an empty list with no reason is the ambiguity this field exists to
     * remove. An empty {@code ability} alone conflates "the Monarch has no
     * printed line in any tree, correctly" with "the resolver regressed", and
     * that is what hid a broken resolver for a whole collection run.
     *
     * <p>A null answer is treated as {@code unknown_kind} rather than as no
     * lookup at all: a caller reaching this method did look, so the record owes
     * a reason.
     */
    public EffectRecord ability(ProvenanceKey.Resolved resolved) {
        if (resolved != null && resolved.key() != null) {
            this.ability = List.of(resolved.key());
            this.abilityUnresolved = null;
            return this;
        }
        this.ability = List.of();
        String reason = resolved == null ? null : resolved.reason();
        this.abilityUnresolved = reason == null
                ? ProvenanceKey.UNRESOLVED_UNKNOWN_KIND : reason;
        return this;
    }

    public EffectRecord state(String json) {
        this.stateJson = json;
        return this;
    }

    public EffectRecord payload(String json) {
        this.payloadJson = json;
        return this;
    }

    public String recordId() {
        return recordId;
    }

    /**
     * The rendered state and payload, for a collector that coalesces.
     *
     * <p>A record is a duplicate of another when what it <b>says</b> repeats —
     * the same board, the same answer — and its id and timestamp always differ.
     * Reading the two rendered blocks is how a coalescer asks that question
     * without reparsing the line it is about to write.
     */
    String stateJson() {
        return stateJson;
    }

    String payloadJson() {
        return payloadJson;
    }

    public String toJson() {
        return "{\"record_id\":" + Json.string(recordId)
                + ",\"run_id\":" + Json.string(runId)
                + ",\"timestamp\":" + Json.string(timestamp)
                + ",\"game_id\":" + Json.string(gameId)
                + ",\"kind\":" + Json.string(kind)
                + ",\"moment\":" + Json.string(moment)
                + ",\"subkind\":" + Json.string(subkind)
                + ",\"link_id\":" + Json.string(linkId)
                + ",\"mirror_of\":" + Json.string(mirrorOf)
                + ",\"variant_of\":" + Json.string(variantOf)
                + ",\"mode\":" + Json.string(mode)
                + ",\"interventional\":" + interventional
                + ",\"fork\":" + fork
                + ",\"synthetic\":" + synthetic
                + ",\"actor_player\":" + Json.string(actorPlayer)
                + ",\"ability\":" + abilityJson()
                + ",\"ability_unresolved\":" + Json.string(abilityUnresolved)
                + ",\"state\":" + stateJson
                + ",\"payload\":" + payloadJson + "}";
    }

    /** Why {@link #abilityJson()} is empty, or null where it names a line. */
    String abilityUnresolved() {
        return abilityUnresolved;
    }

    String abilityJson() {
        if (ability == null) {
            return "null";
        }
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (ProvenanceKey key : ability) {
            joiner.add(key.toJson());
        }
        return joiner.toString();
    }

    /** ``{"events": [...]}`` — a resolution effect half or a combat step. */
    public static String eventsPayload(List<EffectEvent> events) {
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (EffectEvent event : events) {
            joiner.add(event.toJson());
        }
        return "{\"events\":" + joiner + "}";
    }

    /**
     * ``{"incoming": ..., "outgoing": ...|null, "result": ..., "replaced_by": []}``
     *
     * <p>The pair alone could not express what a replacement does. Forge hardly
     * ever edits an event's parameter map: {@code ReplacementHandler} substitutes
     * by running the {@code ReplaceWith$} ability and returning a
     * {@code ReplacementResult}, and its {@code Prevented}, {@code Skipped} and
     * {@code NotReplaced} branches return without touching the map at all. So
     * "enters tapped", "if it would die, exile it instead" and "prevent that
     * damage" all arrived here as two byte-identical halves, and the channel
     * held 34 records in 1.88M against a 7% share of the training mixture.
     *
     * <p>{@code outgoing} is {@code null} — never a copy of {@code incoming} —
     * when the map was not edited in place. A record whose two halves read the
     * same is what made this channel unreadable; null says "not rewritten"
     * outright, and the corpus validator judges an identity pair at zero rather
     * than letting it parse quietly.
     *
     * @param incoming    the event as the replacement received it
     * @param outgoing    the rewritten event, or null where nothing was rewritten
     * @param result      the wire spelling from {@link #rewriteResult}, or null
     *                    where the engine returned no answer
     * @param replacedBy  the ability that stood in for the event; empty when
     *                    none ran, and always a list rather than null
     */
    public static String rewritePayload(
            EffectEvent incoming, EffectEvent outgoing, String result,
            List<ProvenanceKey> replacedBy) {
        StringJoiner keys = new StringJoiner(",", "[", "]");
        if (replacedBy != null) {
            for (ProvenanceKey key : replacedBy) {
                keys.add(key.toJson());
            }
        }
        return "{\"incoming\":" + incoming.toJson()
                + ",\"outgoing\":" + (outgoing == null ? "null" : outgoing.toJson())
                + ",\"result\":" + Json.string(result)
                + ",\"replaced_by\":" + keys + "}";
    }

    /**
     * Forge's {@code ReplacementResult} in the spelling the corpus reads.
     *
     * <p>Lower snake case of the enum member, so {@code NotReplaced} becomes
     * {@code not_replaced}. Converted rather than looked up in a table of the
     * five known members on purpose: a sixth member added to Forge would take
     * the same rule and reach the Python reader, which refuses a value its own
     * closed enum does not have. That is a loud failure on the next shard, where
     * a table would have quietly written null and turned a new outcome into the
     * same "written before this contract" state an old shard reports.
     *
     * <p>Null in, null out. The engine has no answer to report in exactly one
     * case — the replacement threw, and the hook notifies anyway rather than
     * hiding it — and this side sees the same null when it is running against a
     * Forge jar whose listener predates the two extra arguments.
     */
    public static String rewriteResult(Object result) {
        if (result == null) {
            return null;
        }
        String name = String.valueOf(result);
        StringBuilder out = new StringBuilder(name.length() + 2);
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (Character.isUpperCase(c)) {
                if (i > 0) {
                    out.append('_');
                }
                out.append(Character.toLowerCase(c));
            } else {
                out.append(c);
            }
        }
        return out.toString();
    }

    /**
     * ``{"costs": {...}, "outcome": "..."}`` — a resolution cost half.
     *
     * <p>Convenience for the common case where both halves are read at the same
     * instant. A collector that defers the record until the outcome is known
     * reads {@link #costsJson} at the cast and calls the two-string form later.
     */
    public static String costPayload(SpellAbility ability, String outcome) {
        return costPayload(costsJson(ability), outcome);
    }

    /**
     * The same payload from a costs object read earlier.
     *
     * <p>The two halves become knowable at different moments: what was paid is
     * only readable while the cast event is being published, and whether the
     * spell resolved, fizzled or was countered is only readable later. Holding
     * the rendered costs string is what lets one record carry both without
     * keeping a reference to a {@link SpellAbility} whose paid lists the next
     * activation will overwrite.
     */
    public static String costPayload(String costsJson, String outcome) {
        return "{\"costs\":" + costsJson
                + ",\"outcome\":" + Json.string(outcome) + "}";
    }

    /** What a cost payload says when there is no ability to read. */
    private static final String NO_COSTS =
            "{\"mana_by_color\":{},\"tapped\":[],\"life\":0,"
                    + "\"sacrificed\":[],\"discarded\":[],\"exiled\":[]}";

    /**
     * What was actually paid, as the costs object alone.
     *
     * <p><b>Read at the cast event and nowhere later.</b> Everything here lives
     * on the {@link SpellAbility} object and is cleared by its next activation
     * ({@code PlaySpellAbility} resets the paid hash, {@code CostPartMana}
     * clears the paying mana), so a deferred record must hold this string
     * rather than the ability.
     *
     * <p>Deliberately not read off {@code getPayCosts()}, which is what the
     * first collected corpus did and why {@code tapped}, {@code sacrificed},
     * {@code discarded} and {@code exiled} were empty on all 940,973 activation
     * records. Two independent reasons, both in the engine:
     * {@code CostAdjustment.adjust} pays a {@code cost.copy()}, so the parts
     * hanging off the ability are never the ones that were paid; and
     * {@code CostPayment} calls {@code resetLists()} on every list part the
     * moment payment completes, which is before the cast event fires. The paid
     * hash and the paying-mana list are where the payment survives, and are
     * what the rest of Forge reads.
     *
     * <p>Mana is counted one entry per {@link Mana} actually spent, so a hybrid
     * paid as blue counts under {@code U} and under nothing else — where the
     * printed cost the corpus used to read counted it under both halves, and
     * ignored reductions and X entirely.
     */
    public static String costsJson(SpellAbility ability) {
        if (ability == null) {
            return NO_COSTS;
        }
        Map<Byte, Integer> mana = new LinkedHashMap<>();
        if (ability.getPayingMana() != null) {
            for (Mana spent : ability.getPayingMana()) {
                if (spent != null) {
                    mana.merge(spent.getColor(), 1, Integer::sum);
                }
            }
        }
        StringJoiner manaJson = new StringJoiner(",", "{", "}");
        // WUBRGC rather than the order the mana happened to be spent in: two
        // payments of the same cost must render the same bytes, because a
        // record that differs only in key order defeats every duplicate check
        // downstream.
        for (byte color : MagicColor.WUBRGC) {
            Integer count = mana.get(color);
            if (count != null && count > 0) {
                manaJson.add(Json.string(MagicColor.toShortString(color))
                        + ":" + count);
            }
        }
        return "{\"mana_by_color\":" + manaJson
                + ",\"tapped\":" + Json.stringArray(tappedFor(ability))
                + ",\"life\":" + ability.getAmountLifePaid()
                + ",\"sacrificed\":" + Json.stringArray(paid(ability, "Sacrificed"))
                + ",\"discarded\":" + Json.stringArray(paid(ability, "Discarded"))
                + ",\"exiled\":" + Json.stringArray(paid(ability, "Exiled")) + "}";
    }

    /**
     * The cards one cost channel took, by the engine's own hash key.
     *
     * <p>{@code CostSacrifice}, {@code CostDiscard}, {@code CostExile} and
     * {@code CostTapType} each report what they took under the name this looks
     * up ({@code getHashForLKIList}), and the entries are last-known-information
     * copies whose id is preserved — so the entity id still names the object the
     * snapshot describes.
     */
    private static List<String> paid(SpellAbility ability, String hashKey) {
        Set<String> ids = new LinkedHashSet<>();
        Iterable<Card> cards = ability.getPaidList(hashKey);
        if (cards != null) {
            for (Card card : cards) {
                if (card != null) {
                    ids.add(SnapshotBuilder.entityId(card));
                }
            }
        }
        return new ArrayList<>(ids);
    }

    /**
     * Everything this activation tapped, the host included.
     *
     * <p>{@code CostTap} — the bare {@code T} symbol, and much the commonest tap
     * cost there is — taps the host directly and is a plain {@code CostPart}
     * with no card list, so it could never have reached this field however the
     * paid lists were read. It is asked for by type rather than through
     * {@code hasTapCost()}, whose flag is only refreshed when a {@code Cost} is
     * copied.
     */
    private static List<String> tappedFor(SpellAbility ability) {
        Set<String> ids = new LinkedHashSet<>(paid(ability, "Tapped"));
        if (ability.getPayCosts() != null
                && ability.getPayCosts().hasSpecificCostType(CostTap.class)
                && ability.getHostCard() != null) {
            ids.add(SnapshotBuilder.entityId(ability.getHostCard()));
        }
        return new ArrayList<>(ids);
    }

}
