package com.pricepredictor.connector.effects;

import forge.card.MagicColor;
import forge.card.mana.ManaCost;
import forge.card.mana.ManaCostShard;
import forge.game.card.Card;
import forge.game.cost.Cost;
import forge.game.cost.CostDiscard;
import forge.game.cost.CostExile;
import forge.game.cost.CostPart;
import forge.game.cost.CostPartWithList;
import forge.game.cost.CostPayLife;
import forge.game.cost.CostSacrifice;
import forge.game.cost.CostTapType;
import forge.game.spellability.SpellAbility;

import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;


/**
 * One record line, assembled field by field.
 *
 * <p>The envelope is frozen before stage-one collection begins, so this builder
 * writes exactly the keys the schema names and nothing else. Later stages widen
 * the corpus — new kinds become reachable, new snapshot tiers appear — but never
 * redefine a field, which is what lets a stage-one corpus stay trainable
 * alongside stage-four records.
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
     */
    public EffectRecord ability(List<ProvenanceKey> keys) {
        this.ability = keys == null ? null : new ArrayList<>(keys);
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
                + ",\"state\":" + stateJson
                + ",\"payload\":" + payloadJson + "}";
    }

    private String abilityJson() {
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
     * ``{"costs": {...}, "outcome": "..."}`` — a resolution cost half.
     *
     * <p>Read after the cost has been paid, which is when a cast event fires,
     * so the card lists are what was actually spent rather than what the cost
     * asked for. A cost with no list part contributes nothing rather than an
     * empty entry.
     *
     * <p>Mana is counted per colour with a hybrid shard counted under each
     * colour it could have paid: the cost is what the card asks, and which half
     * of a hybrid was used is not in it. Generic and X land under {@code C}.
     */
    public static String costPayload(SpellAbility ability, String outcome) {
        Cost cost = ability == null ? null : ability.getPayCosts();
        StringJoiner mana = new StringJoiner(",", "{", "}");
        List<String> tapped = new ArrayList<>();
        List<String> sacrificed = new ArrayList<>();
        List<String> discarded = new ArrayList<>();
        List<String> exiled = new ArrayList<>();
        int life = 0;

        if (cost != null) {
            ManaCost total = cost.getTotalMana();
            for (char color : COST_COLORS) {
                int count = shardCount(total, color);
                if (count > 0) {
                    mana.add(Json.string(String.valueOf(color)) + ":" + count);
                }
            }
            int generic = total.getGenericCost()
                    + (ability.getXManaCostPaid() == null
                            ? 0 : ability.getXManaCostPaid() * total.countX());
            if (generic > 0) {
                mana.add("\"C\":" + generic);
            }
            for (CostPart part : cost.getCostParts()) {
                if (part instanceof CostPayLife payLife) {
                    life += amountOf(payLife, ability);
                } else if (part instanceof CostPartWithList listed) {
                    // Four different questions about one card list, told apart
                    // by which cost part produced it.
                    List<String> into =
                            part instanceof CostSacrifice ? sacrificed
                            : part instanceof CostDiscard ? discarded
                            : part instanceof CostExile ? exiled
                            : part instanceof CostTapType ? tapped
                            : null;
                    if (into != null) {
                        for (Card card : listed.getCardList()) {
                            into.add(SnapshotBuilder.entityId(card));
                        }
                    }
                }
            }
        }
        return "{\"costs\":{\"mana_by_color\":" + mana
                + ",\"tapped\":" + Json.stringArray(tapped)
                + ",\"life\":" + life
                + ",\"sacrificed\":" + Json.stringArray(sacrificed)
                + ",\"discarded\":" + Json.stringArray(discarded)
                + ",\"exiled\":" + Json.stringArray(exiled) + "},"
                + "\"outcome\":" + Json.string(outcome) + "}";

    }

    private static final char[] COST_COLORS = {'W', 'U', 'B', 'R', 'G'};

    /** How many shards of a cost could be paid with this colour. */
    private static int shardCount(ManaCost cost, char color) {
        byte mask = MagicColor.fromName(String.valueOf(color));
        int count = 0;
        for (ManaCostShard shard : cost) {
            if ((shard.getColorMask() & mask) != 0) {
                count++;
            }
        }
        return count;
    }

    private static int amountOf(CostPayLife part, SpellAbility ability) {
        try {
            return Integer.parseInt(part.getAmount());
        } catch (NumberFormatException | NullPointerException e) {
            // An X or a script variable; the announced value is the record's
            // business and the amount is not readable here without evaluating.
            return 0;
        }
    }

}
