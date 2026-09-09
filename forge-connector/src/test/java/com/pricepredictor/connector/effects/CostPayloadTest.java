package com.pricepredictor.connector.effects;

import com.pricepredictor.connector.ForgeExtension;
import forge.card.MagicColor;
import forge.game.ability.AbilityFactory;
import forge.game.card.Card;
import forge.game.mana.Mana;
import forge.game.spellability.SpellAbility;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * What an activation record says was paid.
 *
 * <p>Every one of these channels was empty on all 940,973 activation records of
 * the first corpus, and the reason was the same for all of them: the collector
 * read {@code getPayCosts()}, which is the cost the card <b>asks</b> for and not
 * the payment. Two engine facts make that unreadable — cost adjustment pays a
 * copy of the parts, and the payment clears the copies' lists before the cast
 * event fires — so these tests are written against the places the payment
 * actually survives: the paid hash, the paying-mana list and the life counter on
 * the {@link SpellAbility}.
 */
@ExtendWith(ForgeExtension.class)
class CostPayloadTest {

    private static SpellAbility ability(String script) {
        return AbilityFactory.getAbility(script, TestCards.build("Fountain of Youth"));
    }

    /** The costs object of a payload, without the outcome around it. */
    private static String costs(SpellAbility ability) {
        return EffectRecord.costsJson(ability);
    }

    // ── the four card channels ──────────────────────────────────────────

    /**
     * The bare {@code T} symbol taps the host, and nothing could ever say so.
     *
     * <p>{@code CostTap} is a plain cost part with no card list at all — unlike
     * {@code CostTapType}, which is the "tap another untapped creature you
     * control" form — so the commonest tap cost in the game had no path into
     * {@code costs.tapped} however the lists were read.
     */
    @Test
    void aTapCostNamesTheHostItTapped() {
        SpellAbility tapper = ability("AB$ GainLife | Cost$ T | Defined$ You | LifeAmount$ 1");
        String entity = SnapshotBuilder.entityId(tapper.getHostCard());

        assertTrue(costs(tapper).contains("\"tapped\":[\"" + entity + "\"]"),
                costs(tapper));
    }

    @Test
    void anAbilityWithNoTapCostTapsNothing() {
        SpellAbility free = ability("AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1");

        assertTrue(costs(free).contains("\"tapped\":[]"), costs(free));
    }

    /**
     * What was sacrificed comes from the paid hash, which is where it survives.
     *
     * <p>And it lands in that channel only: four different questions about one
     * card list, told apart by the engine's own name for the list.
     */
    @Test
    void eachCardChannelCarriesOnlyItsOwnPayment() {
        SpellAbility sa = ability("AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1");
        Card sacrificed = TestCards.build("Mountain");
        Card discarded = TestCards.build("Island");
        Card exiled = TestCards.build("Forest");
        sa.addCostToHashList(sacrificed, "Sacrificed", true);
        sa.addCostToHashList(discarded, "Discarded", true);
        sa.addCostToHashList(exiled, "Exiled", true);

        String json = costs(sa);

        assertTrue(json.contains(
                "\"sacrificed\":[\"" + SnapshotBuilder.entityId(sacrificed) + "\"]"), json);
        assertTrue(json.contains(
                "\"discarded\":[\"" + SnapshotBuilder.entityId(discarded) + "\"]"), json);
        assertTrue(json.contains(
                "\"exiled\":[\"" + SnapshotBuilder.entityId(exiled) + "\"]"), json);
        // And no channel borrowed another's card.
        assertFalse(json.contains(
                "\"tapped\":[\"" + SnapshotBuilder.entityId(sacrificed) + "\""), json);
    }

    /** A card the engine reported twice is named once. */
    @Test
    void oneCardPaidTwiceIsNamedOnce() {
        SpellAbility sa = ability("AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1");
        Card card = TestCards.build("Mountain");
        sa.addCostToHashList(card, "Sacrificed", true);
        sa.addCostToHashList(card, "Sacrificed", false);

        assertTrue(costs(sa).contains(
                "\"sacrificed\":[\"" + SnapshotBuilder.entityId(card) + "\"]"), costs(sa));
    }

    // ── life ────────────────────────────────────────────────────────────

    /**
     * Life is what was paid, not what the script asked for.
     *
     * <p>The two differ whenever the amount is announced or computed, and the
     * old reading parsed the script's literal — so an X or an SVar recorded
     * zero and a cost paid at a different value recorded the printed one.
     */
    @Test
    void lifePaidIsTheAmountPaidNotTheAmountPrinted() {
        SpellAbility sa = ability(
                "AB$ GainLife | Cost$ PayLife<3> | Defined$ You | LifeAmount$ 1");
        sa.setPaidLife(1);

        assertTrue(costs(sa).contains("\"life\":1"), costs(sa));
    }

    @Test
    void anAbilityThatPaidNoLifeSaysZero() {
        assertTrue(costs(ability("AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1"))
                .contains("\"life\":0"));
    }

    // ── mana ────────────────────────────────────────────────────────────

    /**
     * One entry per mana actually spent.
     *
     * <p>The printed cost cannot answer this: a hybrid symbol counted under both
     * of its colours, generic and X were guessed from the printed total, and a
     * cost reduction was invisible. Here a hybrid paid as blue is one blue mana
     * and nothing else.
     */
    @Test
    void aHybridPaidAsBlueCountsUnderBlueAlone() {
        SpellAbility sa = ability(
                "AB$ GainLife | Cost$ U/B | Defined$ You | LifeAmount$ 1");
        sa.getPayingMana().add(
                new Mana(MagicColor.BLUE, sa.getHostCard(), null, null));

        String json = costs(sa);

        assertTrue(json.contains("\"mana_by_color\":{\"U\":1}"), json);
        assertFalse(json.contains("\"B\""), json);
    }

    @Test
    void manaIsCountedPerManaSpentNotPerPrintedSymbol() {
        // A three-and-a-red spell cast for one red after a reduction: the
        // printed cost says four symbols, the payment says one mana.
        SpellAbility sa = ability(
                "AB$ GainLife | Cost$ 3 R | Defined$ You | LifeAmount$ 1");
        sa.getPayingMana().add(new Mana(MagicColor.RED, sa.getHostCard(), null, null));

        assertTrue(costs(sa).contains("\"mana_by_color\":{\"R\":1}"), costs(sa));
    }

    /**
     * Colours render in one order however they were spent.
     *
     * <p>Two payments of the same cost have to render the same bytes, or every
     * duplicate check downstream sees two records where there is one.
     */
    @Test
    void coloursRenderInWubrgcOrder() {
        SpellAbility sa = ability("AB$ GainLife | Cost$ 1 | Defined$ You | LifeAmount$ 1");
        for (byte colour : List.of(
                MagicColor.GREEN, MagicColor.COLORLESS, MagicColor.WHITE,
                MagicColor.BLUE, MagicColor.BLUE)) {
            sa.getPayingMana().add(new Mana(colour, sa.getHostCard(), null, null));
        }

        assertTrue(costs(sa).contains(
                "\"mana_by_color\":{\"W\":1,\"U\":2,\"G\":1,\"C\":1}"), costs(sa));
    }

    // ── the payload around them ─────────────────────────────────────────

    /**
     * The costs and the outcome become knowable at different moments.
     *
     * <p>What was paid is only readable while the cast is being published; how
     * the spell ended is only readable later. The two-string form is what lets
     * one record carry both without holding a {@link SpellAbility} whose paid
     * lists the next activation overwrites.
     */
    @Test
    void everyOutcomeRoundTripsThroughTheDeferredForm() {
        String costs = costs(null);
        for (String outcome : List.of(
                EffectRecord.OUTCOME_RESOLVED, EffectRecord.OUTCOME_FIZZLED,
                EffectRecord.OUTCOME_PARTIALLY_FIZZLED, EffectRecord.OUTCOME_DECLINED,
                EffectRecord.OUTCOME_COUNTERED)) {
            String payload = EffectRecord.costPayload(costs, outcome);
            assertTrue(payload.contains("\"outcome\":\"" + outcome + "\""), payload);
            assertTrue(payload.startsWith("{\"costs\":{"), payload);
        }
    }

    @Test
    void theTwoFormsAgreeWhenBothHalvesAreReadAtOnce() {
        SpellAbility sa = ability("AB$ GainLife | Cost$ T | Defined$ You | LifeAmount$ 1");

        assertEquals(
                EffectRecord.costPayload(
                        EffectRecord.costsJson(sa), EffectRecord.OUTCOME_RESOLVED),
                EffectRecord.costPayload(sa, EffectRecord.OUTCOME_RESOLVED));
    }

    /** No ability to read still renders every channel, empty. */
    @Test
    void anUnreadableCostRendersEveryChannelRatherThanNone() {
        assertEquals(
                "{\"mana_by_color\":{},\"tapped\":[],\"life\":0,"
                        + "\"sacrificed\":[],\"discarded\":[],\"exiled\":[]}",
                costs(null));
    }
}
