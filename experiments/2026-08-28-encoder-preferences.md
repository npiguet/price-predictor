# What the card encoder reads

## The short version

- The encoder's card ratings are faithful copies of its training labels, and the labels are only partly about the card. A card's winnability label mixes three things: which deck builders were willing to play the card, how often it got cast, and whether winners cast it more than losers. Only the last part is about what the card does in a game.
- Half of what the encoder appears to know is memorized card identity. On cards it never trained on, its accuracy drops by more than half, and on trained cards it fits the labels more tightly than their own sampling noise allows. The memory key is the card's text layout, not its words.
- What generalizes is mostly a bag of words. Sixty percent of the encoder's transferable winnability knowledge survives shuffling every word in every card. A thin compositional layer sits on top: the word "flying" earns its premium only on an ability line of a creature, and inverting a pump spell's sign or restricting removal to your own creatures moves the prediction the right way, weakly.
- The best keyword in Forge's eyes is flying, worth about 0.4 label standard deviations on its own, followed by deathtouch, haste, double strike, and lifelink. Hexproof, ward, and shroud are penalized even though the labels pay for them. Deathtouch plus trample is superadditive.
- The best spell text is direct damage, then fight, exile, and destroy. Lockdown auras top all noncreature text. Sweepers, tap effects, and counterspells sit at the bottom, and a spell whose whole text is lifegain is the worst text the encoder knows.
- Bodies beat effects. The same effect is worth about 0.6 standard deviations more stapled to a creature than printed on a sorcery. A mana dork is fine and a mana rock is bad for exactly this reason.
- The encoder's loudest internal axis is not card quality but "will this card leave Forge's hand": lands and equipment at one end, morph, fogs, sweepers, and counterspells at the other. Mana cost explains only a seventh of that axis.
- The embedding describes the card better than it judges it. Mana value, power, toughness, and card type are all decodable from it more accurately than any label it was trained on, and even the card's printing era and rarity are recoverable from wording alone.
- A hand-built table of 135 nameable features matches the encoder's winnability judgment on unseen cards almost exactly. The encoder's real advantage over a spreadsheet is in predicting cast frequency, not card quality.
- Two label heads turned out to be artifacts: the play/draw split is nearly all sampling noise, and the color-affinity heads mostly re-encode color identity plus a shrinkage artifact that punishes good cards. The cast-lift head, expected to be redundant, carries real independent signal and should stay.

## Method and subject

The subject is the gen-4 production sealed card encoder, `models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt`: d_model 512, 6 layers, 8 heads, an 8-query attention pool producing a 512-dim card vector, 21M parameters. It was trained 2026-05-17 from random init on nine per-card regression labels aggregated from 974,028 Bo1 Forge-AI self-play games over 27,983 cards, plus a masked-token auxiliary loss, per [`specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md). The sibling study [`2026-08-27-scorer-preferences.md`](2026-08-27-scorer-preferences.md) examined the deck scorer built on top of this encoder; it established that the scorer consumes mainly the encoder's winnability axis, secondarily its played-rate axis. This study asks what card text produces those axes.

The method had three phases. Fifty-eight hypotheses were brainstormed, critiqued by four independent Opus reviewers (an ML-methodology, an MTG-domain, and a statistics reviewer, plus one that ran label-level regressions against the real corpus), and consolidated into eighteen ranked study questions. A probe battery then tested them: several hundred thousand encoder forward passes and label-side analyses over the full game corpus, no training. Every probe lives in [`scripts/encoder_probes/`](../scripts/encoder_probes/README.md); staged outputs are under `output/encoder-probes/`.

Three instruments recur. The saved checkpoint contains no regression heads, so ridge probes refit from the cached card embeddings to the shrunk labels stand in for them. A "fidelity" probe is fit on all cards and read off for predictions; an "honest" probe is fit only on the encoder's own 22,387 training cards (the seed-42 split reconstructed exactly) and used for any generalization claim. Counterfactual edits change one thing in a card's converted text, re-encode with the production checkpoint, and read the predicted-label change; the harness reproduces the cached embeddings bit-exactly. Every edit contrast is layout-matched, because line-order placebo edits alone move predictions by about 0.3 label SD. The ruler throughout is the label standard deviation: SD(shrunk_score_play) = 0.0618, SD(shrunk_played_rate) = 0.1247. Effects quoted without a unit are in score_play SDs.

## A label has three parts, and only one is about what the card does in a game

A card's winnability label decomposes exactly into three channels, and only one of them is about the card's play. Writing w for the card's in-deck win rate, m for its average cast rate, and d for the cast-rate gap between won and lost games, the label satisfies score = m(2w−1) + d/2 identically. The w channel is set by which builders include the card: the four Forge build methods win between 26% and 60% of their games, a span worth about ±1.8 SD of label before any card property enters. The d channel, winners casting the card more often than losers, is the only within-deck evidence. The decomposition for the headline card classes:

![Channel decomposition of each card class's label premium](images/2026-08-28-encoder-channels.png)

*Source: `l_mediation.py` / `l_analyze.py`, WLS with MV and type controls, n ≥ 50 per class; rendered by `make_figures.py`.*

The channels separate builder taste from game evidence. Removal's premium is mostly that good builders pick it; a trick's premium is entirely that it gets cast in games its side was winning. Sweepers are the sharpest contradiction: builders include them and the games punish them. Recomputing every effect inside the forge-best-vs-forge-best mirror, where both sides' builder and opponent are held fixed, changes almost nothing. The counterspell penalty is the one effect that vanishes in the mirror: it is a builder-selection artifact, not a card fact. The expensive-is-good gradient survives the mirror and every game-length stratification at full strength, so it is not the mana-development artifact the reviewers suspected.

Two annotation channels leak into the labels from outside the games. Cards on Forge's hand-written `AI:RemoveDeck` blacklist carry a 0.64 SD label deficit of which 78% is the w channel, and Forge's bundled human draft rank correlates with the labels almost entirely through w. Both are inherited taste, not game evidence, and the encoder can only partly see them: on held-out cards it over-predicts blacklisted cards by about a fifth of a SD. The text explains a third of the blacklist deficit; the other two thirds is invisible to a text reader.

One label family is corrupted at the source. The Java match worker never logs a face-down card: a face-down cast resolves to an empty name and is dropped, and turning face up fires no event the collector subscribes to (`PlayedCardCollector.java`; the in-code comment claiming the cast branch covers it is wrong). A morph creature that spends the whole game face down is recorded as never played, which is why morph creatures carry the largest played-rate collapse in the corpus. The encoder learned this corpus fact faithfully; the reading "Forge cannot play morph" is false.

## Nine heads supervise about three real axes

The play/draw split supervises one axis twice. The two labels correlate at 0.74, their difference has split-half reliability near 0.10, and the encoder collapses them further: predicted score_play and score_draw correlate at 0.95. A trace of the split survives, correctly signed: defenders, sweepers, and kicker cards measurably prefer the draw in both the labels and the predictions, and haste leans to the play, all at about a twenty-fifth of the main axis. The split bought almost nothing.

The cast-lift head was expected to fall to the same argument and did not. At the label level cast_lift is nearly a re-expression of the d channel (r = 0.96 with d). No other head exposes d, though: predicting the cast-lift label on held-out cards from the score and played-rate heads plus mana value reaches R² 0.32, and adding the cast-lift head's own output lifts it to 0.48. Only a sixth of the head's probe direction lies in the span of the other heads. The head stays.

The five color-lift heads mostly teach color identity plus an artifact. Reading a card's color off its text is near-perfect (AUC 0.99), and the exact-zero diagonal pattern of the labels hands the heads a color-identity task that accounts for over half of their fit. The rest is a splash penalty whose magnitude tracks the card's quality and played rate rather than its pip count. That scaling is a shrinkage artifact. The with-color slice has a smaller denominator than the overall term at the same k, so a better card's off-color cells go negative in proportion to its own quality. A genuine allied-versus-enemy color structure exists in both labels and predictions, at about 1.5% of a color-lift SD. A gradient-boosted probe does no better than ridge on these heads, so the low ceiling reflects missing information rather than a linear probe's limits: the cross-color synergy the spec designed these heads to capture is not in them.

## Half of the encoder's knowledge is memorized card identity

The encoder fits its training cards beyond what any text reader could. A probe fit on half the training cards scores more than twice as high on the other training half as on the encoder's held-out cards, and the probe's own overfit is negligible, so the entire gap is the encoder's. The sharpest single statement needs no variance extrapolation: the training-card residual SD (0.027) is below the label's own irreducible noise floor, measured from the 171 groups of cards whose name-stripped text is bit-identical and whose labels still differ (within-group SD 0.029 at high observation counts, 0.038 overall). Fitting tighter than identical-text twins disagree is memorization by definition.

![R² ladder from the reliability ceiling down to shuffled text](images/2026-08-28-encoder-memorization.png)

*Source: `r2_*.py`, `s_r18*.py`; equivalence classes in `p0_build.py`; rendered by `make_figures.py`.*

The memory key is layout, not vocabulary. Meaning-preserving edits that move lines disturb trained cards' predictions 20–56% more than matched held-out cards'; a token substitution disturbs both equally. Under line permutation the training-card fit collapses toward the held-out fit, which says the memorized component lives in line order.

For the pipeline's stated purpose, scoring hypothetical cards, the operative numbers are the held-out ones: R² 0.37 for winnability and 0.61 for played rate. Paired counterfactual comparisons on one card cancel the memorized offset and remain safe. Absolute reads of a novel card do not, and a hypothetical card must be written in canonical converted-card layout, because non-canonical line order costs more than most real content edits.

## What transfers is mostly a bag of words, with thin composition on top

Destroying word order leaves most of the transferable winnability knowledge intact. Refitting probes on embeddings of shuffled text and evaluating on held-out cards:

![Held-out R² surviving each level of text destruction](images/2026-08-28-encoder-shuffle.png)

*Source: `r1a_shuffle.py`, two seeds averaged; rendered by `make_figures.py`.*

The compositional layer is real but thin, and keyed to slots. On creatures whose only text is a flying line, the line is worth +0.52; the same word moved into a granted-ability spell line keeps 13% of that, and under "can't block creatures with flying" it turns negative. Scope and negation flips all move predictions the correct way, and all weakly: restricting "destroy target creature" to "you control" and inverting +N/+N to −N/−N each clear their placebo null by about 0.14 SD. Turning a Pacifism into a self-lockdown moves the prediction a third as much as swapping its two static lines does. The encoder reads layout more loudly than meaning.

Wordiness itself is priced. Appending any clause to a spell's line pays about +0.09 whatever it says, a drawback clause included; only on a creature's own triggered line does the sign channel work, where a self-sacrifice rider costs −0.20 and a +1/+1-counter rider pays +0.20. A cantrip rider nets exactly zero against a matched control, confirming the label-side null causally.

## The embedding is a card description first and a judgment second

Every attribute printed on the card decodes from the embedding better than any label the encoder was trained on. Linear decoders on held-out cards:

![Decodability of visible attributes against the trained-label ceilings](images/2026-08-28-encoder-decode.png)

*Source: `q3_decode.py`; rendered by `make_figures.py`.*

Era and rarity are decodable even though nothing in the converted text names a set or a rarity. The encoder has learned design-language fingerprints strong enough to date a card within eight years and to separate rares from commons, without ever being asked to.

Number tokens carry ordinal meaning only where they are common. Decoded power tracks printed power nearly exactly through 4, compresses above 5, and goes flat past 8; the counterfactual sweep agrees, with a 12/12 statline scoring the same as a 0/0. The {X} symbol is priced as exactly one generic pip, everywhere: the encoder knows X-spells are castable early and does not know they scale.

![Counterfactual statline sweep and decoded power both collapse past 8](images/2026-08-28-encoder-integers.png)

*Source: `c2_statlines.py` (left) and `q3_decode.py` (right); rendered by `make_figures.py`.*

The 8-query attention pool did not specialize. Any single 64-dim query block recovers 95–96% of every label head's accuracy, all eight blocks' leading axes correlate above 0.97, and the attention profiles are near-uniform with no query attending to statlines or costs. The spec's intent of one query per card aspect did not materialize; the pool is a learned mean, matching what the scorer study found for the scorer's pooling one level up. The whole card cloud has an effective dimensionality near 3.

## Flying tops the keywords, direct damage tops the spells, and bodies beat effects

Flying is the most valuable keyword under causal edits, and two independent edit designs agree on the whole order (Spearman 0.94). The scale comes from substituting keywords inside one static line across 2,913 base creatures and fitting an additive value to every pairwise contrast; the independent deletion design, which removes the keyword line from real carriers, gives larger absolute premiums (flying +0.40) in the same order. Where the two estimates separate in the figure, the encoder's edit response disagrees with the labels: haste, double strike, reach, and menace sit above their label values, and the protection keywords invert.

![Keyword values: counterfactual edits against matched label regressions](images/2026-08-28-encoder-keywords.png)

*Source: `c1_keywords.py`; label values from the matched correlational regression in `c7_labelside.py`; rendered by `make_figures.py`.*

The flying premium is not monotone in size: it peaks on mid-size bodies (P+T 5–8) and falls back on the largest, in both edit designs. Deathtouch plus trample is superadditive: the pair is worth +0.14 over the sum of its parts. Only four real cards carry both keywords, so the claim is encoder-only, and the direction is clean: the model prices the classic combo as a combo.

The spell-effect ladder, all arms substituted into the same 200 base spells, spans three times the keyword scale. Direct damage to any target tops it, ahead of fight, exile, and destroy; conditional removal is discounted; bounce, card draw, and counterspells sit near zero or below; tap effects and sweepers are heavily negative, and a spell whose whole text is "you gain 4 life" is the single worst text measured. Two findings revise the scorer study's picture at the source: fight is not discounted in labels or encoder, so the builder's refusal of fight spells is a search-level behavior, and lockdown auras beat every removal template. Inside the aura family, "enchanted creature can't attack or block" and "can't block" differ by 0.77 SD on two words; that gap is the study's clearest piece of genuine composition.

![The spell-effect ladder from burn down to lifegain](images/2026-08-28-encoder-spells.png)

*Source: `c3_removal.py`, fifteen effect templates substituted into the same 200 base spells; rendered by `make_figures.py`.*

Bodies beat effects by construction, not by accident of the word "artifact". A matched-shell battery puts the same effect on a sorcery and on an ETB creature: the body is worth +0.60 on average, acting as a floor under weak effects (lifegain gains +1.5 by getting a body) and a cap over strong ones (destroy-a-creature loses a little). The mana-ability case reproduces the corpus's starkest class gap at identical cost and text: a creature that taps for mana prices 0.33 above the same ability on an artifact. Mana rocks are not punished for being artifacts; they are punished for not being creatures.

Casting cost is priced as a fee schedule the deck pays, mirroring the scorer study's color economics one level down. Swapping {1}{W} to {W}{W} raises predicted winnability and cuts predicted played rate; swapping to {2} does the reverse. Pips buy quality and cost castability, in both directions, on all five colors.

Creature-type nouns carry real but compressed value. Renaming a dragon to a lizard at identical statline and text costs −0.27. The noun scale (angel, wurm, hydra at the top; goblin, wall, lizard at the bottom) matches the label-side ordering at a third of its spread. About two-thirds of the corpus "dragon premium" is therefore statline and text, and one third is the word itself.

## The encoder disagrees with its labels in the response, not in the ranking

The encoder's outputs are distributed like its labels while its response to an edit is a different function. Across the keyword battery, correlational effects computed on the encoder's predictions match the label-side effects at r = 0.97, and the counterfactual effects match them at r = 0.52. The residual table shows the same agreement: of 246 feature-by-head cells tested for prediction-minus-label bias, ten clear the reporting bar, and seven of those are observation-count strata or the blacklist. The genuine content cells are planeswalkers (under-predicted by 0.14, loyalty text reads worse than it plays) and morph's played rate (over-predicted, the logging artifact resisting text explanation).

Where the edit response diverges from the labels, the divergences cluster into one pattern: features whose label premium arrives through the selection channel are mispriced causally, because the text never earned the premium in games. Haste, double strike, reach, and menace are overpriced by about +0.15 each; hexproof, ward, and shroud are penalized where labels pay; lifegain text is punished at −1.2 where labels are neutral; any appended clause pays the wordiness bonus. The encoder's generalization mode explains the rest: on held-out cards its prediction sits closer to the mean label of a card's templating neighborhood than to the card's own label (β 0.59 versus 0.26), so a card is priced as the average of cards worded like it.

## The scorecard: three falsifications, and most confirmations needed a corrected mechanism

The eighteen ranked questions resolve into three clean falsifications — cast-lift redundancy, pool-query specialization, and the game-length reading of the MV gradient — while the confirmed hypotheses mostly survived with a corrected magnitude or a different mechanism than the one proposed.

| ranked question | verdict | key evidence |
|---|---|---|
| R1 bag-of-words vs composition | both, quantified: 60% of transfer survives word shuffle; slot attribution index 0.13; negation read weakly, layout read loudly | `r1_*.py` |
| R2 memorization | verified, larger than hypothesized: half the apparent knowledge; key is layout | `r2_*.py` |
| R3 play/draw one axis | verified; a 1/25-size correctly-signed residue survives | `s_r3.py` |
| R4 cast_lift redundant | falsified: unique ΔR² +0.16 on held-out cards; the head stays | `s_r4.py` |
| R5 color heads = identity + artifact, no synergy | verified; allied/enemy structure exists at 1.5% of a SD | `s_r5.py` |
| R6 pool-query specialization | falsified: eight near-copies, attention ≈ mean pool | `q1`–`q2` |
| R7 visible attributes decodable | verified, beyond expectation (era ±7.7 yr, rarity AUC 0.87); integer tokens collapse past 8 | `q3_decode.py`, `c2` |
| R8 played_rate is an agency axis, not a cost axis | verified: cost is a seventh of the axis; five collapse classes are five orthogonal directions, not one | `s_r8*.py` |
| R9 MV gradient a game-length artifact | falsified: survives every stratification; the premium is real d-channel | `l_analyze.py` |
| R10 flying strongest keyword, premium grows with size | half verified: strongest yes; size interaction is an inverted U | `c1` |
| R10b deathtouch+trample superadditive | verified (encoder-level), +0.14 | `c1b` |
| R11 power over toughness | verified, smaller than the uncontrolled estimate: +0.04/point, concentrated on small bodies | `c2` |
| R12 removal ladder, lockdown on top | verified; fight/edict undiscounted, so the scorer's refusal is search-level | `c3` |
| R13 tricks as survivorship | reframed: the trick premium is entirely the d channel, "cast while winning" | `l_analyze.py` |
| R14 mana production worst text, body-vs-spell master axis | verified causally: dork +0.33 over rock at identical text | `c4` |
| R15 spell riders | cantrip rider exactly zero; drawbacks priced only on creature trigger lines; wordiness bonus +0.09 | `c6` |
| R16 tribal nouns, taplands, types | noun premium real at a third of label spread; removing "enters tapped" hurts (fixing beats speed); instant/sorcery type tokens interchangeable | `c5` |
| R17 divergence table | near-empty: the encoder distills its labels almost without distortion; divergence lives in the edit response | `s_r17*.py` |
| R18 nameability | 42% of the encoder's winnability taste reduces to 135 nameable features; a spreadsheet matches it on held-out winnability, and loses by 0.26 R² on played rate | `s_r18*.py` |

## Consequences for the next encoder generation

- The play/draw split spends half the score-family gradient on a distinction the labels barely contain and the encoder discards. A single winnability head plus the existing cast-lift head covers what the games can teach.
- The color-lift heads need a redesign before they can teach affinity: match the shrinkage denominators so the artifact term cancels, or drop the family and keep color in the deterministic pips, which is where the scorer reads it anyway.
- Fix the face-down logging hole in `PlayedCardCollector` before the next corpus is generated; morph-block sets are currently mislabeled at the source.
- Layout sensitivity is a liability. Canonicalizing line order at tokenization time, or augmenting training with line permutations, would convert memorized layout capacity into text generalization.
- The encoder's edge over hand-built features is played rate, not winnability. Work aimed at deck quality should either improve the winnability signal (more d-channel supervision, e.g. per-game cast records rather than per-card aggregates) or accept that a 135-feature table is currently an equal substitute on unseen cards.
- Two data-hygiene fixes landed during the study: the stale `village_watch` filename correction and a locator prefix-match bug that silently resolved missing cards ("Undercity") to unrelated files; both affected the gen-4 training inputs marginally.

## Limitations

Counterfactual deltas are read through refit linear probes, not the discarded training heads; the probes reproduce label-side effects to the third decimal where both exist, but absolute head scale is not recoverable. Edited cards are gated on nearest-neighbor distance to the real corpus, and conclusions were checked against the gated subset; multi-keyword ladders past three keywords and the synthetic body-vs-spell shells remain mechanism demonstrations on text the corpus never contains. The label-side channel decomposition controls builder and opponent identity but not deck context beyond it. Everything here is a property of this checkpoint trained on Forge-AI Bo1 self-play; none of it is a claim about human Magic. The morph finding is about the logger, not the AI, and any morph-related label in the corpus should be treated as unusable until the collector is fixed.
