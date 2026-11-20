package com.pricepredictor.connector;

import com.pricepredictor.connector.effects.RandomChoices;
import forge.ai.AiCostDecision;
import forge.ai.AiPlayDecision;
import forge.ai.PlayerControllerAi;
import forge.game.Game;
import forge.game.GameActionUtil;
import forge.game.ability.AbilityUtils;
import forge.game.ability.ApiType;
import forge.game.ability.effects.CharmEffect;
import forge.game.card.Card;
import forge.game.card.CardCollection;
import forge.game.card.CardCollectionView;
import forge.game.card.CardLists;
import forge.game.card.CardPredicates;
import forge.game.cost.Cost;
import forge.game.cost.CostDecisionMakerBase;
import forge.game.cost.CostDiscard;
import forge.game.cost.CostExile;
import forge.game.cost.CostPayment;
import forge.game.cost.CostSacrifice;
import forge.game.cost.CostTapType;
import forge.game.cost.PaymentDecision;
import forge.game.player.Player;
import forge.game.spellability.SpellAbility;
import forge.game.zone.Zone;
import forge.game.zone.ZoneType;

import java.util.List;
import java.util.Random;
import java.util.function.Consumer;

/**
 * Casts or activates a play with a cost decision the caller supplies.
 *
 * <p><b>Why this is a copy of {@code ComputerUtil.handlePlayingSpellAbility}
 * rather than a call to it.</b> The random seat (FR-022c) draws its
 * additional-cost choices — what to sacrifice, discard, exile or tap —
 * uniformly from the legal objects, and the engine's method hard-codes
 * {@code new AiCostDecision(ai, sa, false)} at the one line that decides
 * them. Nothing injectable reaches that line: overriding a controller method
 * cannot, because {@code PlayerController.chooseCardsForCost} is a placeholder
 * no cost part consults, and FR-024 forbids adding a hook to the engine
 * (the {@code effect-record-hooks} branch does not change). So the fifty
 * lines are reproduced here, argument for argument, with the decision passed
 * in. Everything else the original does — splice, the move to the stack, the
 * extra keyword costs, the charm's mode choice, the freeze and the reveal —
 * is kept exactly, because a cast that differs from the engine's own in
 * anything but the cost decision would be a different game.
 *
 * <p>Mana payment stays the AI's: {@link RandomCostDecision} leaves
 * {@code CostPartMana} to its superclass, as FR-022c and FR-023 require.
 */
public final class RandomSpellPlayer {

    private RandomSpellPlayer() {
    }

    /**
     * {@code ComputerUtil.handlePlayingSpellAbility}, with {@code decision} in
     * place of the AI's own.
     *
     * <p>Returns false both before and after the card has moved to the stack,
     * exactly as the original does: a caller that wants to retry a draw must
     * make its draws before calling this, because a failure past the move
     * leaves the card where the engine's own failure path leaves it.
     *
     * @param chooseTargets the deferred targeting action, or null; the AI
     *                      passes one only for a {@code TargetingPlayer}
     *                      clause, which the random seat never draws for
     */
    public static boolean play(
            final Player ai, SpellAbility sa, CostDecisionMakerBase decision,
            Consumer<SpellAbility> chooseTargets) {
        final Card source = sa.getHostCard();
        final Game game = source.getGame();
        final Card host = sa.getHostCard();
        final Zone hz = host.isCopiedSpell() ? null : host.getZone();
        source.setSplitStateToPlayAbility(sa);

        if (castFromItsCard(sa, source)) {
            sa = AbilityUtils.addSpliceEffects(sa);
            if (!stillPlayedAfterSplice(ai, sa)) {
                return false;
            }
            sa.setHostCard(game.getAction().moveToStack(source, sa));
        }

        sa = withExtraKeywordCosts(sa);

        if (!modesAndTargetsChosen(sa, chooseTargets)) {
            return false;
        }
        // Spell Permanents inherit their cost from Mana Cost
        final Cost cost = sa.getPayCosts();

        game.getStack().freezeStack(sa);

        final CostPayment pay = new CostPayment(cost, sa);
        if (pay.payComputerCosts(decision)) {
            putOnStack(game, ai, sa);
            return true;
        }
        skipFailedPlay(game, sa, host, hz);
        return false;
    }

    /** Whether the play is a spell cast from its card, which splices and moves to the stack. */
    private static boolean castFromItsCard(SpellAbility sa, Card source) {
        return sa.isSpell() && !source.isCopiedSpell();
    }

    /**
     * Whether the AI still plays a spliced spell. The original's own comment:
     * after splice the SA must be retargeted, or the AI fails to add the card
     * to the stack and that knocks it out of the game. True for a spell
     * nothing was spliced onto, or one a non-AI controller casts.
     */
    private static boolean stillPlayedAfterSplice(Player ai, SpellAbility sa) {
        if (!hasSplicedCards(sa) || !ai.getController().isAI()) {
            return true;
        }
        sa.resetTargets();
        return ((PlayerControllerAi) ai.getController()).getAi().canPlaySa(sa)
                == AiPlayDecision.WillPlay;
    }

    private static boolean hasSplicedCards(SpellAbility sa) {
        return sa.getSplicedCards() != null && !sa.getSplicedCards().isEmpty();
    }

    /** The play with an original's paid record cleared and its extra keyword costs added. */
    private static SpellAbility withExtraKeywordCosts(SpellAbility sa) {
        if (!sa.isCopied()) {
            sa.resetPaidHash();
            sa.setPaidLife(0);
        }
        return GameActionUtil.addExtraKeywordCost(sa);
    }

    /**
     * Whether a charm's modes are chosen and the deferred targets valid.
     * 603.3c If no mode is chosen, the ability is removed from the stack.
     */
    private static boolean modesAndTargetsChosen(
            SpellAbility sa, Consumer<SpellAbility> chooseTargets) {
        if (sa.getApi() == ApiType.Charm && !CharmEffect.makeChoices(sa)) {
            return false;
        }
        if (chooseTargets == null) {
            return true;
        }
        chooseTargets.accept(sa);
        return sa.isTargetNumberValid();
    }

    /** A paid play onto the stack, its spliced cards revealed. */
    private static void putOnStack(Game game, Player ai, SpellAbility sa) {
        game.getStack().addAndUnfreeze(sa);
        if (hasSplicedCards(sa)) {
            game.getAction().reveal(
                    sa.getSplicedCards(), ai, true, "Computer reveals spliced cards from ");
        }
    }

    /**
     * The original's own recovery, kept verbatim: the card may be stuck on
     * the stack zone, so it is moved back and the play marked to skip.
     *
     * @param hz the zone the card was in before the play, or null for a copied spell
     */
    private static void skipFailedPlay(Game game, SpellAbility sa, Card host, Zone hz) {
        System.out.println("[" + sa.getActivatingPlayer() + "] random seat failed to play "
                + sa.getHostCard() + " [" + sa.getHostCard().getZone() + "]");
        sa.setSkip(true);
        if (!leftOnStack(host, hz)) {
            return;
        }
        Card c = game.getAction().moveTo(hz.getZoneType(), host, null, null);
        for (SpellAbility csa : c.getSpellAbilities()) {
            csa.setSkip(true);
        }
    }

    private static boolean leftOnStack(Card host, Zone hz) {
        return host != null && hz != null && hz.is(ZoneType.Stack);
    }

    /**
     * The AI's cost decision with its four object-choosing parts drawn at
     * random: that many distinct legal objects, uniformly (FR-022c).
     *
     * <p>Each override computes the legal list the way the cost part's own
     * {@code canPay} does and draws from it; a part whose type carries a rule
     * the draw cannot express — discard the last drawn card, exile with a
     * total mana value, tap creatures sharing a type — goes to the superclass,
     * which is the AI's choice and still a legal one. Mana is never touched.
     */
    static final class RandomCostDecision extends AiCostDecision {

        /** Discard types the draw cannot pick for: the whole hand, the last drawn, a random card. */
        private static final List<String> AI_DISCARD_TYPES = List.of("Hand", "LastDrawn", "Random");
        /** Discard type properties that relate the cards, or need a chosen colour or an X. */
        private static final List<String> AI_DISCARD_PROPERTIES =
                List.of("WithSameName", "WithDifferentNames", "ChosenColor", "X");
        /** Exile type properties the draw cannot express: the top of a graveyard, a total mana value. */
        private static final List<String> AI_EXILE_PROPERTIES =
                List.of("FromTopGrave", "+withTotalCMCGE");
        /** Tap type properties the draw cannot express: a shared creature type, a total power. */
        private static final List<String> AI_TAP_PROPERTIES =
                List.of("sharesCreatureTypeWith", "+withTotalPowerGE");

        private final Random random;

        RandomCostDecision(Player ai, SpellAbility sa, Random random) {
            super(ai, sa, false);
            this.random = random;
        }

        @Override
        public PaymentDecision visit(CostSacrifice cost) {
            if (leftToAi(cost)) {
                return super.visit(cost);
            }
            CardCollection valid = CardLists.getValidCards(
                    player.getCardsIn(ZoneType.Battlefield), cost.getType().split(";"),
                    player, source, ability);
            valid = CardLists.filter(valid, CardPredicates.canBeSacrificedBy(ability, isEffect()));
            return pick(valid, cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostDiscard cost) {
            String type = cost.getType();
            if (leftToAi(cost, type)) {
                return super.visit(cost);
            }
            CardCollectionView hand = player.canDiscardBy(ability, isEffect())
                    ? player.getCardsIn(ZoneType.Hand) : CardCollection.EMPTY;
            CardCollection valid = CardLists.getValidCards(
                    hand, type.split(";"), player, source, ability);
            valid = CardLists.filter(valid, c -> c.canBeDiscardedBy(ability, isEffect()));
            return pick(valid, cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostExile cost) {
            String type = cost.getType();
            if (leftToAi(cost, type)) {
                return super.visit(cost);
            }
            CardCollection valid = CardLists.getValidCards(
                    exileCandidates(cost), type.split(";"), player, source, ability);
            valid = CardLists.filter(valid, CardPredicates.canExiledBy(ability, isEffect()));
            return pick(valid, cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostTapType cost) {
            String type = cost.getType();
            if (containsAny(type, AI_TAP_PROPERTIES)) {
                return super.visit(cost);
            }
            CardCollection valid = CardLists.getValidCards(
                    player.getCardsIn(ZoneType.Battlefield), type.split(";"),
                    player, source, ability);
            valid = CardLists.filter(valid, CardPredicates.CAN_TAP);
            if (!cost.canTapSource) {
                valid.remove(source);
            }
            return pick(valid, cost.getAbilityAmount(ability));
        }

        /** Whether a sacrifice is the AI's: of the source, of the original host, or of all. */
        private static boolean leftToAi(CostSacrifice cost) {
            return cost.payCostFromSource() || "OriginalHost".equals(cost.getType())
                    || "All".equalsIgnoreCase(cost.getAmount());
        }

        /** Whether a discard is the AI's: of the source, or of a type the draw cannot express. */
        private static boolean leftToAi(CostDiscard cost, String type) {
            return cost.payCostFromSource() || AI_DISCARD_TYPES.contains(type)
                    || containsAny(type, AI_DISCARD_PROPERTIES);
        }

        /**
         * Whether an exile is the AI's: of the source, of all, of a type the
         * draw cannot express, from no restricted zone, or from the library alone.
         */
        private static boolean leftToAi(CostExile cost, String type) {
            return cost.payCostFromSource() || type.equals("All")
                    || containsAny(type, AI_EXILE_PROPERTIES) || cost.zoneRestriction == 0
                    || fromLibraryAlone(cost);
        }

        private static boolean fromLibraryAlone(CostExile cost) {
            return cost.getFrom().size() == 1 && cost.getFrom().get(0).equals(ZoneType.Library);
        }

        private static boolean containsAny(String type, List<String> properties) {
            return properties.stream().anyMatch(type::contains);
        }

        /** The cards an exile may take: in its zones across the game, or the payer's own. */
        private CardCollectionView exileCandidates(CostExile cost) {
            return cost.zoneRestriction != 1
                    ? player.getGame().getCardsIn(cost.getFrom())
                    : player.getCardsIn(cost.getFrom());
        }

        /** {@code count} distinct members of {@code valid}, or null when it has fewer. */
        private PaymentDecision pick(CardCollection valid, int count) {
            if (count < 0 || valid.size() < count) {
                return null;
            }
            return PaymentDecision.card(new CardCollection(
                    RandomChoices.drawDistinct(valid, count, random)));
        }
    }
}
