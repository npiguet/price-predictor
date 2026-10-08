package com.pricepredictor.connector;

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

import java.util.ArrayList;
import java.util.Collections;
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
final class RandomSpellPlayer {

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
    static boolean play(
            final Player ai, SpellAbility sa, CostDecisionMakerBase decision,
            Consumer<SpellAbility> chooseTargets) {
        final Card source = sa.getHostCard();
        final Game game = source.getGame();
        final Card host = sa.getHostCard();
        final Zone hz = host.isCopiedSpell() ? null : host.getZone();
        source.setSplitStateToPlayAbility(sa);

        if (sa.isSpell() && !source.isCopiedSpell()) {
            sa = AbilityUtils.addSpliceEffects(sa);
            if (sa.getSplicedCards() != null && !sa.getSplicedCards().isEmpty()
                    && ai.getController().isAI()) {
                // The original's own comment: after splice the SA must be
                // retargeted, or the AI fails to add the card to the stack
                // and that knocks it out of the game.
                sa.resetTargets();
                if (((PlayerControllerAi) ai.getController()).getAi().canPlaySa(sa)
                        != AiPlayDecision.WillPlay) {
                    return false;
                }
            }

            sa.setHostCard(game.getAction().moveToStack(source, sa));
        }

        if (!sa.isCopied()) {
            sa.resetPaidHash();
            sa.setPaidLife(0);
        }

        sa = GameActionUtil.addExtraKeywordCost(sa);

        if (sa.getApi() == ApiType.Charm && !CharmEffect.makeChoices(sa)) {
            // 603.3c If no mode is chosen, the ability is removed from the stack.
            return false;
        }
        if (chooseTargets != null) {
            chooseTargets.accept(sa);
            if (!sa.isTargetNumberValid()) {
                return false;
            }
        }
        // Spell Permanents inherit their cost from Mana Cost
        final Cost cost = sa.getPayCosts();

        game.getStack().freezeStack(sa);

        final CostPayment pay = new CostPayment(cost, sa);
        if (pay.payComputerCosts(decision)) {
            game.getStack().addAndUnfreeze(sa);
            if (sa.getSplicedCards() != null && !sa.getSplicedCards().isEmpty()) {
                game.getAction().reveal(
                        sa.getSplicedCards(), ai, true, "Computer reveals spliced cards from ");
            }
            return true;
        }
        // The original's own recovery, kept verbatim: the card may be stuck on
        // the stack zone, so it is moved back and the play marked to skip.
        System.out.println("[" + sa.getActivatingPlayer() + "] random seat failed to play "
                + sa.getHostCard() + " [" + sa.getHostCard().getZone() + "]");
        sa.setSkip(true);
        if (host != null && hz != null && hz.is(ZoneType.Stack)) {
            Card c = game.getAction().moveTo(hz.getZoneType(), host, null, null);
            for (SpellAbility csa : c.getSpellAbilities()) {
                csa.setSkip(true);
            }
        }
        return false;
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

        private final Random random;

        RandomCostDecision(Player ai, SpellAbility sa, Random random) {
            super(ai, sa, false);
            this.random = random;
        }

        @Override
        public PaymentDecision visit(CostSacrifice cost) {
            if (cost.payCostFromSource() || "OriginalHost".equals(cost.getType())
                    || "All".equalsIgnoreCase(cost.getAmount())) {
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
            if (cost.payCostFromSource() || type.equals("Hand") || type.equals("LastDrawn")
                    || type.equals("Random") || type.contains("WithSameName")
                    || type.contains("WithDifferentNames") || type.contains("ChosenColor")
                    || type.contains("X")) {
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
            if (cost.payCostFromSource() || type.equals("All") || type.contains("FromTopGrave")
                    || type.contains("+withTotalCMCGE") || cost.zoneRestriction == 0
                    || (cost.getFrom().size() == 1
                            && cost.getFrom().get(0).equals(ZoneType.Library))) {
                return super.visit(cost);
            }
            CardCollectionView from = cost.zoneRestriction != 1
                    ? player.getGame().getCardsIn(cost.getFrom())
                    : player.getCardsIn(cost.getFrom());
            CardCollection valid = CardLists.getValidCards(
                    from, type.split(";"), player, source, ability);
            valid = CardLists.filter(valid, CardPredicates.canExiledBy(ability, isEffect()));
            return pick(valid, cost.getAbilityAmount(ability));
        }

        @Override
        public PaymentDecision visit(CostTapType cost) {
            String type = cost.getType();
            if (type.contains("sharesCreatureTypeWith") || type.contains("+withTotalPowerGE")) {
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

        /** {@code count} distinct members of {@code valid}, or null when it has fewer. */
        private PaymentDecision pick(CardCollection valid, int count) {
            if (count < 0 || valid.size() < count) {
                return null;
            }
            List<Card> shuffled = new ArrayList<>(valid);
            Collections.shuffle(shuffled, random);
            return PaymentDecision.card(new CardCollection(shuffled.subList(0, count)));
        }
    }
}
