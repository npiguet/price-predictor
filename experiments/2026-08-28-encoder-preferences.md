# What the card encoder reads

## The short version

- The encoder's card ratings are faithful copies of its training labels, and the labels are only partly about the card. A card's winnability label mixes three things: which deck builders were willing to play the card, how often it got cast, and whether winners cast it more than losers. Only the last part is about what the card does in a game.
- Half of what the encoder appears to know is memorized card identity. Its accuracy on validation cards is under half of its accuracy on the training set, and the memory is keyed to the card's text layout, not its words.
- The encoder reads words in context: "flying" is an upside on a creature's own ability line and a downside inside "can't block creatures with flying", and inverting a pump spell's sign or restricting removal to your own creatures moves the prediction the right way. But sixty percent of its transferable winnability knowledge survives shuffling every word, and layout changes that alter nothing move predictions as much as edits that invert meaning.
- The best keyword in Forge's eyes is flying, worth about 0.4 label standard deviations on its own, followed by deathtouch, haste, double strike, and lifelink. Against a vanilla creature the labels reward every combat keyword; the clear liabilities are defender and flash. Keyword pairs are priced for how their rules fit together: an indestructible defender beats the sum of its parts, while haste on a defender and reach on a flier are penalized as wasted lines.
- The best spell text is direct damage, then fight, exile, and destroy. Lockdown auras top all noncreature text. Sweepers, tap effects, and counterspells sit at the bottom, and a spell whose whole text is lifegain is the worst text the encoder knows.
- Bodies beat effects. The same effect is worth about 0.6 standard deviations more stapled to a creature than printed on a sorcery. A mana dork is fine and a mana rock is bad for exactly this reason.
- The encoder's loudest internal axis is not card quality but "will this card leave Forge's hand": lands and equipment at one end, morph, fogs, sweepers, and counterspells at the other. Mana cost explains only a seventh of that axis.
- The embedding describes the card better than it judges it. Mana value, power, toughness, and card type are all decodable from it more accurately than any label it was trained on, and even the card's printing era and rarity are recoverable from wording alone.
- A hand-built table of 135 nameable features matches the encoder's winnability judgment on validation cards almost exactly. The encoder's real advantage over a spreadsheet is in predicting cast frequency, not card quality.
- Two of the label groups turned out to be artifacts: the play/draw split is nearly all sampling noise, and the color-affinity labels mostly re-encode color identity plus a shrinkage artifact that punishes good cards. The cast-lift label carries real independent signal and should stay.

## Method and subject

The subject is the gen-4 production sealed card encoder, `models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt`: d_model 512, 6 layers, 8 attention heads, an 8-query attention pool producing a 512-dim card vector, 21M parameters. It was trained 2026-05-17 from random init on nine per-card regression labels aggregated from 974,028 Bo1 Forge-AI self-play games over 27,983 cards, plus a masked-token auxiliary loss, per [`specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md). A label is a training target: a per-card statistic measured from those game logs. The nine are the card's net win influence when cast, measured once on the play and once on the draw (the winnability labels — this document quotes the on-the-play one unless it says otherwise), the fraction of in-deck games it was cast at all (the played-rate label), the win-rate gap between games where it was cast and games where it stayed in hand (the cast-lift label), and five per-color affinity scores (the color-lift labels). The encoder learns to predict all nine from the card's rules text alone. The sibling study [`2026-08-27-scorer-preferences.md`](2026-08-27-scorer-preferences.md) examined the deck scorer built on top of this encoder; it established that the scorer consumes mainly the encoder's winnability axis, secondarily its played-rate axis. This study asks what card text produces those axes.

The method had three phases. Fifty-eight hypotheses were brainstormed, critiqued by four independent Opus reviewers, one of which ran label-level regressions against the real corpus, and consolidated into eighteen ranked study questions. A probe battery then tested them: several hundred thousand encoder forward passes and label-side analyses over the full game corpus, no training. Every probe lives in [`scripts/encoder_probes/`](../scripts/encoder_probes/README.md); staged outputs are under `output/encoder-probes/`.

Three instruments recur through the study.

1. Ridge probes stand in for the missing regression heads. The saved checkpoint contains no regression heads: the small output layers that turned the card vector into each label's prediction during training were not saved. A ridge regression refit from the cached card embeddings to the labels takes their place. The "fidelity" probe is fit on all cards and is used wherever a prediction just needs reading off. The "honest" probe is fit only on the encoder's own 22,387 training cards and evaluated on its 5,596-card validation set (the seed-42 split reconstructed exactly); every generalization claim uses it.

2. Counterfactual edits measure what a piece of text is worth to the encoder. An edit changes one thing in a card's converted text, re-encodes the card with the production checkpoint, and reads how the predicted label moved. The harness reproduces the cached embeddings bit-exactly, so a measured change comes from the edit and nothing else. Every edit is compared against a variant with the same line layout, because moving a line without changing its meaning already shifts predictions by about 0.3 label SD.

3. The label-SD ruler puts every effect on one scale. Raw label units are tiny and differ from label to label, so every effect in this document is divided by the standard deviation of its label across the corpus: SD(shrunk_score_play) = 0.0618, SD(shrunk_played_rate) = 0.1247. An effect quoted without a unit is in score_play SDs; a keyword worth "+0.27" moves predicted winnability by 27% of the card-to-card spread of the winnability label. One SD is a large step in card quality: the whole keyword ladder from best to worst spans about half of one, and the cards Forge's developers hand-blacklisted as unplayable sit about two-thirds of one below normal cards.

## A card can earn a strong winnability label without ever changing a game

A game's win cannot be cleanly attributed to any single card, so the winnability label shares the credit for a win, and the blame for a loss, among every card the side cast. The label counts the games the card's deck won with the card cast, minus the games it lost with the card cast, over all games the card sat in a deck. A card therefore gets credit for its deck's wins whether or not it caused them.

Three simpler quantities, called channels from here on, can be measured from the same game logs, and each one holds a different kind of fact:

- selection: how often decks containing the card win. Builder choice sets this: a card only the random builder plays is observed in decks that win 26% of their games, a card forge-best favors in decks that win 60%, and a deck's win rate flows into every card it contains. Which builders play a card therefore moves its winnability label by about ±1.8 SD before the card itself does anything.
- castability: how often the card gets cast.
- contribution: how much more often the card is cast in the games its decks win than in the games they lose. This is the only channel that measures the card changing games: it is positive when winners cast the card more often than losers did.

The winnability label equals `castability × (2·selection − 1) + contribution/2` exactly. The channels are therefore not new information: they are the same number, computed in a way that keeps its ingredients apart.

One detail of the castability definition carries the identity. Castability averages the card's cast rate in won games and its cast rate in lost games, weighting the two equally rather than by how common wins are; that weighting is what makes the identity exact and keeps the contribution term free of selection. It also means castability slightly understates a strong card's true cast rate, because strong cards are cast more in wins and their decks win more often. The understatement is the product of those two deviations, both of which are bounded; for the top 1% of cards by label it is about two percentage points of cast rate.

The rest of the document states its findings as premiums: a feature's premium is how much it moves the winnability label, measured against cards matched on cost and type, and a negative premium is a penalty. Every premium is read through this split, because a premium carried by the selection channel is inherited builder taste and only a premium carried by the contribution channel is evidence the card changed games. The decomposition for the headline card classes:

![Channel decomposition of each card class's winnability-label premium](images/2026-08-28-encoder-channels.png)

*Source: `l_mediation.py` / `l_analyze.py`, WLS with MV and type controls, n ≥ 50 per class; rendered by `make_figures.py`.*

Each bar in the chart is one channel's share of a class's premium, not the channel's own value: the class's deviation on that channel, multiplied by what that deviation is worth to the label at the corpus average. The black dot is the class's exact premium. The bars sum only approximately to the dot. The gap is the part of the premium earned by two channels deviating at once, which no single bar can carry, and it stays under 0.13 SD in every row.

Creatures earn most of their premium through the selection channel, before any game starts. Creature-heavy decks are the decks that win in this corpus, and the best builder in the pool fills its decks with creatures. The label shares each win's credit among every card the winner cast, so any creature in such a deck collects credit for the deck's wins, whether it was the best card in it or the 23rd. Creatures are also cast far more often than noncreatures: a bear is simply played on turn two, while a situational spell waits in hand. That difference earns them almost nothing. The rest of the premium is contribution: winning boards are boards with creatures on them. Unconditional removal has the same composition, with most of its premium in the selection channel.

A combat trick reaches nearly the same premium as a creature with essentially no selection share. Decks that run tricks win no more often than decks without them: a Giant Growth in the decklist says nothing about the deck's quality. Tricks are also cast slightly less often than the average card, because a trick sits in hand waiting for the right combat, and sometimes the right combat never comes. The whole premium is contribution: winners cast their Giant Growths, and losers die with them in hand. A pump spell is only cast into a fight worth winning (eating a blocker, forcing through lethal), so the cast causes the win and coincides with it at once, and the label cannot separate the two.

Sweepers set the selection and contribution channels in direct opposition. Good builders include sweepers, so a Wrath of God in a decklist marks a well-built deck, and the selection share is positive. The contribution share is negative and larger: when the sweeper is actually cast, it is disproportionately in games its side loses. A sweeper comes down when the board has gotten away from its caster, and the side that was ahead usually wins even after the board is cleared. The class nets slightly below average. The corpus's bad builders are not the cause: in games where the forge-best builder plays against itself, a cast sweeper still appears mostly in losses.

In short: a creature is good because of the company it keeps, a trick is good because of when it appears, and a sweeper is the card whose company vouches for it while its appearances testify against it.

The castability bar is near zero for every class, and its sign says nothing about card quality. Being cast a lot only means being present for whatever the deck was going to do anyway: each cast earns credit if the game is won and blame if it is lost. The average card's deck wins 46.5% of its games, because the corpus deliberately includes bad builders, so one cast is worth slightly less than nothing. Each class's bar multiplies its extra or missing casts by that value: creatures, cast more than average, get a small negative bar; sweepers, cast less, a small positive one. Morph is the extreme case: its recorded cast rate falls further than any class in the table and its label barely moves, because a card that is never cast is never blamed. The channel is still worth separating, because with the number of casts removed, the contribution bar measures only when a card is cast, not how often.

Recomputing every effect inside the forge-best-vs-forge-best mirror, where both sides' builder and opponent are held fixed, changes almost nothing. The counterspell penalty is the one effect that vanishes in the mirror: it is a builder-selection artifact, not a card fact. The expensive-is-good gradient survives the mirror and every game-length stratification at full strength: it is not an artifact of expensive cards being cast only in games that had already gone long.

Two annotation channels leak into the labels from outside the games. Cards on Forge's hand-written `AI:RemoveDeck` blacklist carry a 0.64 SD winnability deficit of which 78% is the selection channel, and Forge's bundled human draft rank correlates with the winnability label almost entirely through selection. Both are inherited taste, not game evidence, and the encoder can only partly see them: on validation cards it over-predicts blacklisted cards by about a fifth of a SD. The text explains a third of the blacklist deficit; the other two thirds is invisible to a text reader.

For one class of cards, every label is corrupted at the source. The Java match worker never logs a face-down card: a face-down cast resolves to an empty name and is dropped, and turning face up fires no event the collector subscribes to (`PlayedCardCollector.java`; the in-code comment claiming the cast branch covers it is wrong). A morph creature that spends the whole game face down is recorded as never played, which is why morph creatures carry the largest played-rate collapse in the corpus. The encoder learned this corpus fact faithfully; the reading "Forge cannot play morph" is false.

## The nine labels carry about three distinct signals

The three that survive are winnability, played rate, and cast lift; the other six labels are copies or artifacts. The second winnability label duplicates the first, and the five color-lift labels reduce to color identity plus a shrinkage artifact.

The play and draw labels are two copies of the same signal. In the raw labels the two correlate at 0.74, and their difference is almost entirely sampling noise (split-half reliability near 0.10). The encoder collapses them further: its predictions for the two correlate at 0.95. A trace of the split survives, correctly signed: defenders, sweepers, and kicker cards measurably prefer the draw in both the labels and the predictions, and haste leans to the play, all at about a twenty-fifth of the winnability signal itself. The split bought almost nothing.

The cast-lift label looks just as redundant and is not. In the raw labels, cast lift is nearly the same quantity as the contribution channel (the two correlate at 0.96). But contribution is exactly the valuable part of the winnability label, and cast lift is the only label that carries it on its own. The extra information is measurable on validation cards: the predicted winnability, the predicted played rate, and mana value together explain 32% of the cast-lift label, and adding the predicted cast lift raises that to 48%. Only a sixth of what the cast-lift probe reads from the embedding overlaps with what the other probes read. The label stays.

The five color-lift labels mostly restate the card's own colors. A card's color-lift value for a color is zero by construction whenever that color is one of its own, because a card's colors are always present in its own deck. Knowing which of the five values are zero is therefore the same as knowing the card's colors, which the mana cost gives away (the embedding decodes color at AUC 0.99), and that alone accounts for over half of what these labels ask the encoder to predict.

The nonzero values were meant to measure affinity: does the card win more when its deck also plays that color? What they mostly measure instead is the card's own quality, with the sign flipped. Each value subtracts the card's overall win score from its win score in decks that play the color. Both scores are pulled toward zero to tame small samples, and the with-color score rests on fewer games, so it is pulled harder. Subtracting a lightly-pulled score from a heavily-pulled one leaves a remainder proportional to the score itself, so the better the card, the more negative its color-lift values. A real color preference survives underneath, in both the labels and the encoder's predictions, but it is tiny: allied colors beat enemy colors by about 1.5% of a color-lift SD. The rest is genuinely absent rather than hidden, because a probe that can read nonlinear patterns finds no more of it than the linear one: the cross-color synergy these labels were designed to teach never made it into them.

## Half of the encoder's knowledge is memorized card identity

On the cards in its training set, the encoder explains 81% of the card-to-card variation in the winnability label; on the 5,596 cards in its validation set, 37%. The probe that reads these numbers out is not the cause: fit on one half of the training set, it does just as well on the other half, so the gap comes from the embeddings themselves. The encoder gives its training cards embeddings that carry their labels, and gives validation cards only what their text implies. In standard terms this is overfitting; the rest of the section measures how much of the fit is memorized and what the memory is keyed on.

The storage works because text can serve as a name as well as a description. With card names stripped, 99% of cards' converted text is unique in the corpus, so a card's exact wording can act as its identifier, and training pushes each unique wording's prediction toward that card's own measured label until the value is simply stored. The text supplies only the key; the stored value arrives through the other half of training, the measured label itself.

![The encoder's fit on training-set cards, on line-shuffled training-set cards, and on validation-set cards](images/2026-08-28-encoder-memorization.png)

*Source: `r2a_memorization.py`; the line-shuffled bar from `r1a_shuffle.py`; rendered by `make_figures.py`.*

The stored key is the line layout more than the words. Swapping two ability lines, which changes nothing about what a card does, disturbs training-set cards' predictions 20–56% more than it disturbs validation cards'; swapping a creature type for an equally common one disturbs both the same. And destroying line order collapses the training-set fit most of the way down to the validation fit, as the middle bar of the figure shows.

For the pipeline's stated purpose, scoring invented cards, the validation numbers are the honest ones: 37% of winnability, 61% of played rate. Comparing two versions of the same card stays safe, because the memorized part is the same on both sides and cancels in the difference. A single absolute score does not. An invented card must also be written in the converter's standard line order, because unusual layout moves a prediction more than most real content changes.

## The encoder reads words in context, but meaningless edits move predictions as much as real ones

Word order carries a substantial share of the encoder's transferable knowledge: destroying it costs 40% of the validation-card winnability accuracy and 62% of the cast-frequency accuracy. The measurement shuffles each card's words, re-encodes the result, refits the probes on the shuffled embeddings, and scores them on validation cards:

![Validation accuracy surviving each level of text destruction](images/2026-08-28-encoder-shuffle.png)

*Source: `r1a_shuffle.py`, two seeds averaged; rendered by `make_figures.py`.*

Context is read correctly wherever a direct test was run. The word "flying" is priced by the role it plays: worth +0.52 as a creature's own static line, 13% of that inside a spell that merely grants it, and negative under "can't block creatures with flying", where it marks a drawback. Restricting "destroy target creature" to "you control" lowers the prediction, and so does flipping +N/+N to −N/−N, each by about 0.14 SD beyond what a meaning-free control edit moves.

The response sizes are what is wrong: an edit that blanks a card moves the prediction no more than an edit that changes nothing. Rewriting the line "enchanted creature can't attack or block" into "CARDNAME can't attack or block" strips a pure Pacifism of its whole effect, because the aura itself could never attack or block anyway. On the ten auras whose lockdown is their whole text, the encoder reads the loss correctly: nine of ten predictions drop, by −0.35 SD on average. Swapping the same cards' two static lines, which changes nothing at all, moves predictions just as far.

Appended riders are read on some lines and merely counted on others. A clause appended to a spell's line pays about +0.09 whatever it says, a drawback included. The same kind of rider on a creature's own triggered line is actually read: a self-sacrifice rider costs −0.20 and a +1/+1-counter rider pays +0.20. A cantrip rider nets exactly zero against a matched control, and the labels price cantrip riders at nothing, so there the encoder agrees with them.

## The embedding is a card description first and a judgment second

Every attribute printed on the card decodes from the embedding better than any label the encoder was trained on. Linear decoders on validation cards:

![Decodability of visible attributes against the trained-label ceilings](images/2026-08-28-encoder-decode.png)

*Source: `q3_decode.py`; rendered by `make_figures.py`.*

Era and rarity are decodable even though nothing in the converted text names a set or a rarity. The encoder has learned design-language fingerprints strong enough to date a card within eight years and to separate rares from commons, without ever being asked to.

Number tokens keep their order only where they are common. Decoded power tracks printed power nearly exactly through 4, compresses above 5, and goes flat past 8. A statline sweep shows the same collapse in the predictions. The sweep takes thirty real creatures and rewrites each one's power-toughness line to N/N, once for every N from 0 to 12, leaving cost and abilities untouched. Re-encoding each variant and reading its predicted winnability gives a curve that rises through the middle sizes and then falls back, until the 12/12 version scores the same as the 0/0 version. The {X} symbol is priced as exactly one generic pip, everywhere: the encoder knows X-spells are castable early and does not know they scale.

![The statline sweep and decoded power both go flat past 8](images/2026-08-28-encoder-integers.png)

*Source: `c2_statlines.py` (left) and `q3_decode.py` (right); rendered by `make_figures.py`.*

The flat top of the sweep reflects missing number information, not a learned law of diminishing returns. The labels themselves do diminish: mean winnability plateaus at printed power 5–6 and declines past 8. Training therefore never rewarded telling the big numbers apart, and cards that big are also rare (85 creatures at power 8, 10 at power 12). The sweep still rules out the diminishing-returns reading. It changes only the statline, so within the sweep a bigger body is an upgrade in essentially every game. Genuine diminishing returns would keep the 12/12 at least as good as the 8/8. The sweep instead scores it half an SD lower, level with the 0/0. The token embeddings show where the information stops. The integers 1 through 7 line up along one direction in embedding space, evenly ordered by value. Every integer from 8 upward lands on that direction at roughly a four, in no consistent order.

*Source: `q4_token_geometry.py`.*

The 8-query attention pool did not specialize. Any single 64-dim query block recovers 95–96% of every label's prediction accuracy, all eight blocks' leading axes correlate above 0.97, and the attention profiles are near-uniform with no query attending to statlines or costs. The spec's intent of one query per card aspect did not materialize; the pool is a learned mean, matching what the scorer study found for the scorer's pooling one level up. The whole card cloud has an effective dimensionality near 3.

## Flying tops the keywords, direct damage tops the spells, and any effect is worth more on a creature

Flying is the most valuable keyword under counterfactual edits, and two independent edit designs agree on the whole keyword order (Spearman 0.94).

The first design swaps keywords on the same card. Each of 2,913 base creatures carries one keyword in a static line; the probe replaces that keyword with each of the others in turn and re-encodes. The change in predicted winnability measures how much better one keyword is than the one it replaced. Each keyword then gets a single value on a common scale, chosen so that the differences between the values match the measured swap effects.

The second design deletes keywords instead of swapping them. It takes the real cards that carry a keyword line, removes that line, and re-encodes; the drop in predicted winnability is the keyword's value. The deletion design gives larger values than the substitution design, in the same order.

The figure also shows each keyword's label value: what carrying the keyword is worth in the winnability labels of real creatures, measured against keywordless creatures of the same cost and statline, then centered on the same zero as the edit scale. Against a vanilla creature the labels reward every combat keyword; haste, vigilance, first strike, and trample are each worth about a fifth of an SD. The clear liabilities are defender and flash. Haste's reward is almost entirely the contribution channel: cast haste creatures show up in wins, while deckbuilders favor haste carriers no more than any other creature. The edit and label orders agree (Spearman 0.87). Where they part, the encoder prices a keyword by how it reads rather than how it plays: haste reads a tenth of an SD better than it plays, flash and indestructible read about a quarter better, double strike reads worse, and hexproof is punished about twice as hard as its label. An earlier version of this comparison controlled for keyword count, which benchmarked each keyword against carriers of the other keywords, mostly flying, and made nearly every keyword read as a liability; the vanilla baseline replaces it.

![Keyword values: counterfactual edits against matched label regressions](images/2026-08-28-encoder-keywords.png)

*Source: `c1_keywords.py`; label values from the vanilla-baseline joint regression in `c8_kw_vanilla.py`; rendered by `make_figures.py`.*

The flying premium peaks on mid-size bodies (power plus toughness 5–8) and falls back on the largest, in both edit designs.

Keyword pairs are priced for how their rules fit together. The pair sweep adds every pair of the sixteen keywords as two static lines to 150 mid-size keywordless creatures, reads the predicted winnability against single-keyword arms, and removes each keyword's average interaction so only the pair-specific part remains. The strongest synergies are pairs whose rules multiply: an indestructible, hexproofed, or shrouded defender is a wall that never dies, flash plus haste is an ambush that attacks the turn it appears, and double strike doubles what lifelink and deathtouch trigger on. Among the largest penalties are pairs where one rule idles the other: haste on a defender has nothing to speed up, and reach on a flier duplicates a block flying already makes. The single largest penalty, flash with ward, fits no such story. The pair-specific effects reach about half the size of a strong single keyword, and most pairs move the prediction beyond their confidence interval in one direction or the other. All of this is encoder-only: real cards carrying any given pair are too rare for a label-side check.

![Keyword-pair interactions: rules that multiply price above the sum, rules that idle price below](images/2026-08-28-encoder-pairs.png)

*Source: `c9_kw_pairs.py`, all 120 pairs, each keyword's mean interaction removed; the chart shows the eight largest in each direction.*

The deathtouch-plus-trample bonus reported by the original 2×2 does not survive the full sweep. The original design measured the pair against vigilance and reach in its control slots, and most of its +0.14 was the controls' own interaction, chiefly vigilance with trample. Centered against all pairs, the classic combo prices barely above the sum of its parts.

Direct damage to any target tops the spell-effect ladder, and a spell whose whole text is "you gain 4 life" is the single worst text measured. The ladder writes fifteen different effect texts into the same 200 base spells, and it spans three times the keyword scale. Its shape is removal over card advantage: fight, exile, and destroy sit high, bounce, card draw, and counterspells sit near zero or below, and tap effects and sweepers are heavily negative.

Two rungs of the ladder revise findings from the scorer study. Fight is not discounted, in the labels or in the encoder, so the builder's refusal of fight spells is a search-level behavior. Lockdown auras beat every removal template. Inside the aura family, "enchanted creature can't attack or block" and "can't block" differ by 0.77 SD on two words; that gap is the study's clearest piece of genuine composition.

![The spell-effect ladder from burn down to lifegain](images/2026-08-28-encoder-spells.png)

*Source: `c3_removal.py`, fifteen effect templates substituted into the same 200 base spells; rendered by `make_figures.py`.*

Any effect is worth more on a creature than on a spell. The probe writes the same effect twice: once as a sorcery, and once as a creature that performs the effect when it enters the battlefield. The creature version is worth +0.60 on average. The body works as a floor under weak effects and a cap over strong ones: lifegain gains +1.5 by getting a body, while destroy-a-creature loses a little. The same design reproduces the corpus's starkest class gap at identical cost and text: a creature that taps for mana prices 0.33 above an artifact with the same ability. Mana rocks are not punished for being artifacts; they are punished for not being creatures.

Colored pips buy predicted winnability and cost predicted played rate, in both directions and in all five colors. Swapping a cost of {1}{W} to {W}{W} raises predicted winnability and cuts predicted played rate. Swapping {1}{W} to {2} does the reverse. This is the scorer study's color economics, reproduced one level down at the single-card level.

Creature-type nouns carry real but compressed value. Renaming a dragon to a lizard at identical statline and text costs −0.27. The noun ordering (angel, wurm, hydra at the top; goblin, wall, lizard at the bottom) matches the label-side ordering at a third of its spread. About two-thirds of the corpus dragon premium is therefore statline and text; one third is the word itself.

## The encoder disagrees with its labels in the response, not in the ranking

The encoder's outputs are distributed like its labels while its response to an edit is a different function. Across the keyword battery, correlational effects computed on the encoder's predictions match the label-side effects at r = 0.97, and the counterfactual effects match them at r = 0.86 once both scales share the vanilla-creature baseline from `c8_kw_vanilla.py`. The residual table shows the same agreement: of 246 feature-label pairs tested for prediction-minus-label bias, ten clear the reporting bar, and seven of those are observation-count strata or the blacklist. The two genuine disagreements are planeswalkers (under-predicted by 0.14, loyalty text reads worse than it plays) and morph's played rate (over-predicted, the logging artifact resisting text explanation).

Where the edit response diverges from the labels, the encoder misprices in both directions, and its generalization mode explains most of it. Haste reads a tenth of an SD better than it plays, and flash and indestructible read about a quarter better, above labels the games set low. Double strike reads worse than it plays, and hexproof is punished about twice as hard as its label. Lifegain text is punished at −1.2 where labels are neutral, and any appended clause pays the wordiness bonus. On validation cards the encoder's prediction sits closer to the mean label of a card's templating neighborhood than to the card's own label (β 0.59 versus 0.26), so a card is priced as the average of cards worded like it, and a keyword that plays unusually well or badly for how it reads is pulled toward how it reads.

## The scorecard: three falsifications, and most confirmations needed a corrected mechanism

The eighteen ranked questions resolve into three clean falsifications — cast-lift redundancy, pool-query specialization, and the game-length reading of the MV gradient — while the confirmed hypotheses mostly survived with a corrected magnitude or a different mechanism than the one proposed.

| ranked question | verdict | key evidence |
|---|---|---|
| R1 bag-of-words vs composition | both, quantified: composition correctly signed (flying priced by role, negation lowers) and carries 40% of winnability transfer; but layout placebos move predictions as much as meaning flips | `r1_*.py` |
| R2 memorization | verified, larger than hypothesized: half the apparent knowledge; key is layout | `r2_*.py` |
| R3 play/draw one axis | verified; a 1/25-size correctly-signed residue survives | `s_r3.py` |
| R4 cast_lift redundant | falsified: unique ΔR² +0.16 on validation cards; the label stays | `s_r4.py` |
| R5 color labels = identity + artifact, no synergy | verified; allied/enemy structure exists at 1.5% of a SD | `s_r5.py` |
| R6 pool-query specialization | falsified: eight near-copies, attention ≈ mean pool | `q1`–`q2` |
| R7 visible attributes decodable | verified, beyond expectation (era ±7.7 yr, rarity AUC 0.87); integer tokens collapse past 8 | `q3_decode.py`, `c2` |
| R8 played_rate is an agency axis, not a cost axis | verified: cost is a seventh of the axis; five collapse classes are five orthogonal directions, not one | `s_r8*.py` |
| R9 MV gradient a game-length artifact | falsified: survives every stratification; the premium sits in the contribution channel | `l_analyze.py` |
| R10 flying strongest keyword, premium grows with size | half verified: strongest yes; size interaction is an inverted U | `c1` |
| R10b deathtouch+trample superadditive | overturned by the all-pairs sweep: the 2×2's +0.14 was mostly its controls' own interaction (vigilance with trample); centered, the pair is near zero. Pair pricing is real but lives elsewhere: defender+indestructible up, haste+defender and flying+reach down | `c1b`, `c9` |
| R11 power over toughness | verified, smaller than the uncontrolled estimate: +0.04/point, concentrated on small bodies | `c2` |
| R12 removal ladder, lockdown on top | verified; fight/edict undiscounted, so the scorer's refusal is search-level | `c3` |
| R13 tricks as survivorship | reframed: the trick premium is entirely the contribution channel, "cast while winning" | `l_analyze.py` |
| R14 mana production worst text, body-vs-spell master axis | verified causally: dork +0.33 over rock at identical text | `c4` |
| R15 spell riders | cantrip rider exactly zero; drawbacks priced only on creature trigger lines; wordiness bonus +0.09 | `c6` |
| R16 tribal nouns, taplands, types | noun premium real at a third of the label-side spread; removing "enters tapped" hurts (fixing beats speed); instant/sorcery type tokens interchangeable | `c5` |
| R17 divergence table | near-empty: the encoder distills its labels almost without distortion; divergence lives in the edit response | `s_r17*.py` |
| R18 nameability | 42% of the encoder's winnability taste reduces to 135 nameable features; a spreadsheet matches it on validation-card winnability, and loses by 0.26 R² on played rate | `s_r18*.py` |

## Consequences for the next encoder generation

- The play/draw split spends half of the winnability training signal on a distinction the labels barely contain and the encoder discards. A single winnability label plus the cast-lift label covers what the games can teach.
- The color-lift labels need a redesign before they can teach affinity: match the shrinkage denominators so the artifact term cancels, or drop the family and keep color in the deterministic pips, which is where the scorer reads it anyway.
- Fix the face-down logging hole in `PlayedCardCollector` before the next corpus is generated; morph-block sets are currently mislabeled at the source.
- The layout sensitivity and part of the memorization have a plausible common cause in the positional encoding. Positions are learned absolute indices over the flat token stream, so reordering two lines changes every token's embedding, and the exact word-and-position pattern is a high-precision fingerprint to store labels against; the converter also emits lines in near-fixed order, so line-order invariance is never learnable from the data. The principled fix is hierarchical: encode lines as an unordered bag and words as an ordered list within their line, with a line-type embedding (static, spell, triggered, activated) carrying the slot information the composition results show matters. Cheaper approximations are canonicalizing line order at tokenization time or augmenting training with line permutations.
- The encoder's edge over hand-built features is played rate, not winnability. Work aimed at deck quality should either improve the winnability signal (richer contribution-channel labels, e.g. per-game cast records rather than per-card aggregates) or accept that a 135-feature table is currently an equal substitute on validation cards.
- Two data-hygiene fixes landed during the study: the stale `village_watch` filename correction and a locator prefix-match bug that silently resolved missing cards ("Undercity") to unrelated files; both affected the gen-4 training inputs marginally.

## Limitations

Counterfactual deltas are read through refit linear probes, not the discarded training heads; the probes reproduce label-side effects to the third decimal where both exist, but absolute head scale is not recoverable. Edited cards are gated on nearest-neighbor distance to the real corpus, and conclusions were checked against the gated subset; multi-keyword ladders past three keywords and the synthetic body-vs-spell shells remain mechanism demonstrations on text the corpus never contains. The label-side channel decomposition controls builder and opponent identity but not deck context beyond it. Everything here is a property of this checkpoint trained on Forge-AI Bo1 self-play; none of it is a claim about human Magic. The morph finding is about the logger, not the AI, and any morph-related label in the corpus should be treated as unusable until the collector is fixed.
