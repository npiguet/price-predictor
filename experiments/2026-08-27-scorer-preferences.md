# What the deck scorer prefers

## The short version

- The scorer likes creature-heavy decks in two or three colors: 19 or 20 creatures, with an average mana cost just above three.
- A splash has to earn its place: a new color helps only when its cards are better than the cards they push out. A fourth color rarely clears that bar, and a fifth never does.
- Cards are ranked mostly by how often they won games in training. Creatures come first, removal second, card draw and do-nothing artifacts last. The BREAD rule puts removal right after bombs; the scorer puts it after every creature.
- A deck has about five noncreature slots, and half of them go to removal that kills what it points at. Fight spells and sweepers are passed over, and so are combat tricks, fogs and graveyard recursion. Which archetype fills the rest depends on the color: red and black spend nearly everything on removal, blue splits four ways across bounce, draw, auras and counters, and green leads with auras. How much removal a deck ends up with is a separate question, and one it has no opinion on.
- Flying is the ability it prizes most, and more so on big creatures. Expensive cards beat cheap ones, and only mythics get a rarity bonus.
- It does not see synergies. A tribal payoff is worth the same with or without its tribe, and a second copy of a card is worth the same as the first.
- Its taste comes from watching Forge's AI play itself. That AI attacks and blocks well, defends against fliers badly, and misplays instants; the scorer's likes and dislikes mirror those strengths and weaknesses.
- Scores only mean something within a single set. One point of score is worth about 18 percentage points of match win rate.
- Under the hood, the scorer boils every card down to a few learned numbers and averages them across the deck. Combos, exact counts, and which ability sits on which creature are invisible to it.

## Method and subject

This document records an interpretability study of the production sealed deck scorer: which deck properties raise its score, which lower it, and where those preferences come from. It is the evidence base for an article on what the scorer likes and dislikes when building from a sealed or draft pool.

The method had three phases. Fifty hypotheses were brainstormed and critiqued by four independent Opus reviewers, one of which ran label-level regressions against Forge's bundled human draft rankings. The surviving hypotheses were ranked. A battery of inference-only probes then tested them: roughly half a million scorer forward passes, no training. Every probe lives in [`scripts/scorer_probes/`](../scripts/scorer_probes/README.md); each figure and table below names its script, and the numbers regenerate from the staged outputs in `output/scorer-probes/`.

The subject is the gen-4 production checkpoint `models/sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt`. Its input is one 544-wide vector per card: 512 dims from the sealed text encoder and 32 deterministic features. Its body is a 6-layer Set Transformer with 4-seed attention pooling. It was trained 2026-05-18 on 70,134 Bo7 Forge-AI self-play matches with log margin weighting. Background: [`2026-05-15-gen4-initial-training.md`](2026-05-15-gen4-initial-training.md), [`specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md), [`specs/2026-03-28-sealed-deck-picker.md`](../specs/2026-03-28-sealed-deck-picker.md).

A probe outside the scorer's training distribution measures the network's arithmetic, not a preference. The scorer only ever saw realistic decks: Forge-built, scorer-built, and versions of those with cards replaced at random, always from realistic pools. A real deck with a few cards swapped stays inside that world. A deck of 23 copies of one card does not, and every such probe below is labeled a mechanism demonstration.

## The ruler: one score unit is worth about 18 winrate points

Score differences convert to win probability at a stable rate, measured on matches the scorer never trained on. The yardstick is 4,708 Bo7 matches: gen5-, gen4-, and Forge-built decks playing each other on same-set pools, Forge AI piloting, recorded in `match-outcomes-gen5-vs-gen4-forge.txt` eight days after the training cutoff. Every "held-out" number in this document is measured on these matches, unless the probe names a wider pool.

Binning them by score margin gives a monotone calibration across all ten deciles, and the fitted curve is in the figure. Its slope sits below 1.0, so the model's own training objective, `sigmoid(Δscore)`, is mildly overconfident.

![Calibration of score differences against held-out match outcomes](images/2026-08-27-scorer-calibration.png)

*Source: `t7_artifacts.py`, probe C.*

Two comparisons recur through the document. Held-out accuracy compares the scorer to the reality of played matches: the share of held-out matches in which the actual winner receives the higher score. Ranking agreement compares the scorer to another version of itself: both versions score the same pile of decks, and Spearman ρ measures how similar the two resulting orderings are, 1.0 for the identical order and 0 for an unrelated one. Played matches play no part in ranking agreement. In the ablations of the mechanism section, the other version is always the same checkpoint with part of its card input erased, judged against the full model.

Held-out accuracy for the full model is 71.9%, at the Bo7 oracle ceiling of 0.72–0.78 estimated in [`2026-04-26-gen2-initial-training.md`](2026-04-26-gen2-initial-training.md). Accuracy per matchup runs from ~99% on gen-vs-random pairs down to ~60% on gen4-vs-gen5 and mirror pairs. Every Δscore below converts at roughly 18 winrate points per unit.

## Shape: it wants 19 creatures, two or three colors, and a 3.2 curve

Shape probes swap cards inside real decks and read the score response. Contexts are 250–800 decks sampled from 10,000 aligned pool/deck pairs. One baseline number recurs: swapping any chosen card for a card the builder rejected costs about −0.40, because chosen cards are simply better. Ladder effects are read against that baseline and against each ladder's own rung-to-rung marginals.

![Score response to creature count, curve, and off-color additions](images/2026-08-27-scorer-shape-ladders.png)

*Source: `t3_ladders.py`; the add-a-card probe is `t1_meansum.py`.*

### Color count: the price is paid per color, and the first off-color card pays it

The first card of a new color costs three to four times what every further card of that color costs. The first card's marginal is −0.48; the second and later cost ordinary swap prices near −0.13 (right panel above). The single-pip splash ladder shows the same threshold. The scorer prices the presence of a third color, not the number of off-color pips. That is the economics of a manabase that must find slots for every color it plays.

Color fixing is priced too, at about a tenth of the splash cost. In a 2×2 probe (splash spell, on-color dual land, both, neither) the dual offsets the splash penalty by +0.04 (t = 14): real color-fixing logic, small magnitude.

The ladders compose into a decision rule. A new color enters the deck when the summed value gains of its cards, over the cards they displace, exceed the one-time color fee, less the fixing refund. The fee repeats for every additional color. The greedy search qualifies the rule: when several splash cards clear the fee only jointly, the first swap is score-negative on its own, and pure hill-climbing cannot take it. The simulated-annealing moves and color-pair restarts exist to cross exactly that valley.

The deployed builds show the preference. gen4-512 builds 2 colors 34% of the time and 3 colors 58%, and it adapts to set structure: 2.2 mean colors on artifact-heavy Mirrodin sets, 4.0 on all-gold Alara Reborn (`post_hoc_slices.py`, section `decks`).

The builds also carry the rule's signature: a color is added only for cards better than the ones already in the deck. In the gen4-512 builds, off-color cards hold higher win-rate labels than the main-color cards beside them, and the premium grows with color count.

![Win-rate labels of main-color and off-color cards by deck color count](images/2026-08-27-scorer-color-economics.png)

*Source: `post_hoc_slices.py`, section `colors`.*

Two readings follow from the chart. Third colors are deep, not lone bombs: a 3-color deck carries between three and four off-color cards, and a 4-color deck nearly six, because once one strong card has paid the color fee, backfilling the color with its ordinary playables costs only normal swap prices. And color creep happens in rich pools, not poor ones: 4-color decks hold higher main-color quality than 2-color decks, so the driver is surplus good material across colors rather than barren main colors. The comparison covers the cards the builder included; the best cards it left unused are not measured.

Five colors is a fee the builder never volunteers to pay. 28 of 10,000 builds reach five colors, and they concentrate in multicolor-themed sets: Dissension, New Capenna, the Alara and Invasion blocks. About a fifth of their cards are natively multicolor, so a deck reaches "five colors" by pip arithmetic when its gold cards jointly cover all five colors, not because cards from five colors were chosen one by one; hybrid pips count for both of their colors, inflating the tally further. The 4-color tier says the same — its most common set by far is all-gold Alara Reborn. Gen-2, whose encoder carried no castability signal, built 7.6% five-color decks and lost with them at 28.9%; the repeated fee is what removed them.

### Creature count: the optimum is 19–20, and too few is punished harder than too many

The creature ladder is an inverted U with its peak at 19–20 creatures out of 23 spells (left panel above). Removing four creatures costs −0.69; adding four costs −0.54. The optimum sits above Forge's own ~14.6 creatures and above even gen-4's built average of 18.2.

The asymmetry matches the training corpus, where creature-light decks lose badly under Forge piloting. Whether it matches human play is a question the corpus cannot answer; [`2026-05-13-gen3-initial-training.md`](2026-05-13-gen3-initial-training.md) documents the piloting bias.

### Curve: the optimum is near mean mana value 3.2–3.3, and cheap is punished harder than expensive

The curve ladder peaks at a mean spell mana value (MV) of 3.2–3.3 (middle panel above). Four swaps toward cheaper cards cost −1.09; four toward more expensive cost −0.80. The peak sits at the bottom edge of the 3.4–4.0 band the gen-2-era win-rate tables identified as strongest, so the deck-level curve taste is roughly right with a slight cheap-side lean.

The deck-level optimum coexists with a card-level preference for expensive cards (next section). The two are consistent: a good curve needs its cheap slots filled, but an individual cheap card is rarely the best card.

### Curve shape beyond the mean does not matter

A mean-preserving spread, swapping two 3-drops for a 2-drop and a 4-drop or the reverse, moves the score less than a quality-matched control swap at the same mean (|Δ| lower by 0.07, t = 3.7). The scorer holds no opinion on curve smoothness, gaps, or bimodality at fixed mean. The "no 2-drops" alarm a human builder raises has no analogue in the model.

### A 24th card is nearly always refused; nonbasic lands are refused least

Adding a 24th card to a built deck lowers the score in almost every case, and even the best on-color candidates are usually refused. Spell counts above 23 never occur in the training data, so the penalty is a count effect that card quality barely moderates. Lands escape the refusal most easily, and the land classes are ordered correctly inside it.

![Score change from adding one card of each class to a built deck](images/2026-08-27-scorer-land-adds.png)

*Source: `t1_meansum.py`, 400 contexts; on/off-color split in `post_hoc_slices.py`, section `t1color`.*

Lands are the least-refused addition, and on-color duals the least-refused lands, but most land adds still read as negative. That is why scorer-built decks carry 0.5 nonbasic lands on average where Forge's carry 1.1, and why 62% of gen4-512 decks run zero. The builder under-plays duals as a direct result: its 23-spell invariant forbids trading a spell for a land, and adding the land on top usually reads as dilution.

## Cards: winnability first, creatures over removal over do-nothing spells

Card-level values come from two probes that agree with each other (r = 0.78). The first is leave-one-out deltas inside 3,500 real decks. The second is a standardized swap-in value, `v_swap`: put the card into the median slot of up to eight fixed two-color Forge-built decks matching its colors. `v_swap` was measured for all 26,402 cards with 30+ game observations; it is on the same score scale as everything else, and 0 means "as good as the median card of a real deck".

*Source: `t2_marginal_values.py`; analysis in `t2_analyze.py`.*

### The card ranking is learned winnability first

The scorer's per-card value tracks the encoder's empirical win-rate labels more strongly than any visible card property: Spearman 0.68 against `shrunk_score_play`. A regression on visible properties (type, cost, stats, keywords, rarity) explains 40% of the value variance; adding the empirical label lifts it to 58% and dwarfs every other coefficient. Before anything else, the scorer reads "this card won games in the self-play corpus". The deck-shape terms above are layered on top.

### The category order is creatures, then removal, then everything else, not BREAD's

An average creature outranks an average removal spell by about a tenth of a unit, and removal outranks the rest of the noncreature pile by a further tenth:

| category | n | mean v_swap |
|---|---|---|
| creature | 14,891 | −0.01 |
| noncreature removal | 1,600 | −0.12 |
| card draw | 1,493 | −0.25 |
| other noncreature | 8,418 | −0.25 |

*Source: `t2_analyze.py`, section 3.*

The scorer demotes removal below generic bodies, where the human BREAD ordering puts Removal second only to Bombs. The label-level comparison against Forge's shipped human pick orders (`human_rank_probe.py`) locates the divergence, in units of the label's card-to-card spread: removal sits 0.2 below where humans rank it, card draw 0.45 below, artifacts 0.63 below, while flying creatures sit 0.57 above.

The mechanism is the Forge-AI meta the labels were earned in. The AI attacks and blocks competently but times instants and card advantage poorly, so bodies convert to wins and answers convert to card disadvantage. Instant-speed removal earns no premium over sorcery-speed (−0.10 both).

The extremes of the card ranking tell the same story. The bottom is uniformly do-nothing artifacts and engines, cards that spend mana without affecting the board: Cloudstone Curio, Witch's Oven, Krark's Thumb. The top is 26 creatures out of 30 cards, mostly 4–6 MV flying and lifelink rares.

### Expensive cards are preferred at the margin, and the flying premium grows with size

Marginal value rises monotonically with mana value, from −0.19 for 0–1 MV cards to −0.01 for 6+ MV. A flier outscores a ground creature of the same MV bucket everywhere, and the gap grows from +0.06 at low cost to +0.16 at the top end.

![Swap-in value by mana value and by card class](images/2026-08-27-scorer-card-values.png)

*Source: `t2_marginal_values.py` data; figure by `make_figures.py`. Classes in the right panel overlap where a card qualifies for more than one.*

Cheapness is not a per-card virtue anywhere in the model. Castability is priced at the deck level, through the splash thresholds and pip support above. The labels agree: `score_play` rises with MV while `played_rate` falls, and the scorer follows the winnability axis.

Statline composition matters more than efficiency. Conditioned on MV and total stats, the label regression (`label_probe.py`) prices +1 power at +0.004 and +1 toughness at −0.002. Stats-per-mana carries almost no signal (correlation with v_swap: 0.04). Creatures with a combat keyword average +0.08 over keyword-less ones; vanilla creatures score the same as creatures whose text has no combat keyword.

### Only mythics carry a rarity premium, and rarity is inferred from text alone

Commons, uncommons, and rares all average the same value (−0.11 to −0.12); mythics average +0.01. The encoder receives no rarity input, so the premium is whatever bomb-ness the text itself reveals. Controlling for the empirical label flips the rare coefficient negative: the average rare's text reads worse than its label warrants, consistent with sealed rares including many narrow or constructed-facing cards.

### Class by class: tricks at the bottom, planeswalkers priced like creatures

The right panel of the figure above ranks the classes; the exact values (`post_hoc_slices.py`, section `t2class`):

| class | n | mean v_swap | note |
|---|---|---|---|
| combat trick (pump instant) | 295 | −0.27 | worst class measured |
| X-cost spell | 416 | −0.21 | X encodes as 0 MV, so these read as bad cheap spells |
| vehicle | 139 | −0.20 | not fooled by the printed P/T in the deterministic features |
| counterspell | 405 | −0.19 | |
| hybrid-mana card | 545 | −0.16 | low-MV confounded |
| noncreature token-maker | 1,206 | −0.12 | |
| removal, instant or sorcery | 1,100 | −0.10 | excludes artifact/enchantment removal, which drags the category table's broader removal mean to −0.12 |
| planeswalker | 223 | +0.00 | at parity with MV-matched creatures |

Two rows contradict the hypotheses that motivated them. Combat tricks were predicted to be overvalued: their `cast_lift` label is the highest of any class. That label is a survivorship artifact, because a cheap card goes uncast only in games that were already lost. The scorer ranks tricks lowest anyway; the cheap-instant penalties dominate, and it lands near the human consensus on tricks by a different route. Planeswalkers were predicted to be depressed by the AI's poor piloting; they price at creature parity instead.

Two hypothesized artifacts have no measurable effect. Vehicles carry a printed P/T in the deterministic features despite doing nothing uncrewed, yet they price low: the text-embedding labels carry the truth. A wordiness bias (long text as a power proxy) measures at a standardized +0.003, effectively zero.

### Half of every noncreature slot goes to removal, and almost none of that removal is conditional

A gen4-512 deck holds 17.9 creatures and 5.1 noncreature spells, and removal fills half of those five slots. The probe reads the 10,000 aligned pool/deck pairs, sorts every noncreature nonland card in each pool into one archetype by its rules text, and compares what each deck could cast with what it took. On-color creatures are taken at 63.7%, on-color noncreature spells at 22.8%.

Two quantities separate a preference for an archetype from a preference for the cards that happen to carry it. The take rate is cards chosen over cards available, counted only over pool cards whose colors the deck plays. The lift divides that take rate by the rate predicted from the card's winnability label, mana value and color count alone. That prediction comes from strata measured over every eligible noncreature card, so lift 1.0 means the archetype is taken as often as any other noncreature card of the same quality and cost.

![Take-rate lift and slot share for each noncreature archetype](images/2026-08-27-scorer-noncreature-mix.png)

*Source: `t8_noncreature_mix.py`.*

Only four archetypes are taken more often than their quality labels predict: removal, bounce, token makers, and auras and equipment. Planeswalkers land on par. Every other archetype falls below it.

Combat tricks and fogs are the two the builder actively refuses. Tricks are taken at roughly a quarter of their predicted rate, despite being among the most abundant cards in the pools. Both measurements agree on tricks: `v_swap` prices them worst of any class in the table above, and only fogs are declined more often.

Inside removal the preference is for answers that need nothing else to work:

| removal subtype | available | take rate | lift |
|---|---|---|---|
| destroy or exile | 18,697 | 52.4% | 1.58 |
| lockdown aura | 6,699 | 47.9% | 1.57 |
| damage | 18,978 | 47.4% | 1.54 |
| shrink (−X/−X) | 5,300 | 42.3% | 1.44 |
| edict (opponent sacrifices) | 2,960 | 18.9% | 1.07 |
| sweeper | 6,366 | 10.6% | 0.55 |
| fight | 1,551 | 16.1% | 0.46 |

Removal that names its target unconditionally sits about half again above par. Removal that needs a board state drops to par or below. A fight spell needs a creature already in play, an edict needs the opponent to hold only the creature worth killing, and a sweeper needs a battlefield worth clearing. Sweepers have a second reason to be refused: the deck casting them runs nearly 18 creatures of its own.

Planeswalkers are the one archetype taken often without being preferred. They are chosen at 45.7%, a rate no archetype but token makers matches, and at par once their winnability labels are accounted for. The builder is taking the cards, not the card type. The class table above found parity too, against a different baseline: planeswalkers price level with mana-value-matched creatures.

Two checks say the ranking is not an artifact of one measurement or a handful of sets. Joining the same archetype labels onto the `v_swap` values reproduces the same broad shape, with planeswalkers and token makers near the top and fogs last. The two orderings agree at Spearman 0.64 (p = 0.010, n = 15 families). That is an independent measurement, because `v_swap` swaps cards into fixed Forge decks and never watches the builder choose. Per set, removal's lift exceeds 1.0 in 91% of the 180 sets, and combat tricks' in 1% of the 162 sets that carry them.

Removal loses to creatures and wins among noncreatures. `v_swap` says an average removal spell is worth less than an average creature, which is why creatures take more than three quarters of the spell slots. The take rates say that among the cards competing for the five slots creatures do not fill, removal wins by a wide margin. The BREAD divergence is therefore specific rather than general: the scorer ranks removal below bodies, not below every other noncreature card.

Removal's edge does not fade as a deck accumulates removal. Ranking each pool's eligible removal by label, the best one lifts at 1.37 and the sixth at 1.63, while non-removal cards fall from 0.93 to 0.61 over the same ranks. The edge is per-card rather than a quota the deck fills, which is why the removal-share ladder below finds no optimum in a deck's total removal count.

### Removal takes the plurality in every color but blue and green

Red and black spend three quarters or more of their noncreature slots on removal, and blue spends barely one slot in eight. White is a removal color too, and it takes both removal and token makers above par. White's removal lift of 1.82 is the highest of any color. Blue is the only color whose slots split near-evenly, across card draw, bounce, auras and counterspells, with removal fifth. Green leads with auras and equipment, and gives combat tricks a larger share than any other color.

![Composition of each color's noncreature deck slots](images/2026-08-27-scorer-noncreature-colors.png)

*Source: `t8_noncreature_mix.py`, mono-colored cards only.*

Colorless is where ramp earns its slot. Mana rocks lift at 1.51 and take a sixth of all colorless slots. Colored ramp is refused: green fixing lifts at 0.49, and red's own ramp was taken zero times out of 265 chances. The color fee measured in the color-count ladder applies to a ramp spell like any other card, so acceleration is worth a slot only when it commits no color.

Green's noncreature spells are taken less often than any other color's, and the label-and-cost control does not remove the gap. Green mono-colored noncreature cards are taken at 16.2% against 21.0% to 29.1% for the other four, and at lift 0.64 against 1.06 to 1.14. The deficit sits inside removal rather than in green's mix of archetypes:

| green removal | available | take rate | lift |
|---|---|---|---|
| damage | 1,943 | 36.4% | 0.90 |
| destroy or exile | 1,631 | 9.6% | 0.53 |
| fight | 1,303 | 18.3% | 0.49 |
| sweeper | 1,328 | 3.5% | 0.18 |
| edict | 302 | 1.7% | 0.13 |

Every kind of removal green offers carries a condition, and none of the five clears par. Green's destroy-or-exile row looks unconditional and is not: 86% of the cards in it can only target a creature with flying or reach. The same class lifts at 1.87 in white and 0.53 in green. Only the targeting restriction separates the two.

## Synergy is absent; only density effects survive

### Pair synergy is absent: enabler density does not raise a payoff's value

The scorer does not pay more for a synergy payoff as its enablers enter the deck. The probe used 57 curated payoff/enabler/control triples spanning tribal lords, sacrifice payoffs, heroic targets, devotion, and spells-matter, from 20 sets, commons and uncommons only. Every control matches its payoff on set, color, broad type, and mana value within 1. For each triple and eight real same-set decks, filler slots were swapped for 0–3 enablers, and the payoff's swap-in marginal was measured at each dose against the control's.

![Payoff and control marginals as enablers are added](images/2026-08-27-scorer-synergy-dose.png)

*Source: `t4_synergy.py`, probe P-A; dataset `synergy_pairs.json`.*

| statistic (mean over 57 entries) | value |
|---|---|
| Δdose = (payoff − control) gain from 0 → 3 enablers | +0.008 ± 0.010 (p = 0.43) |
| same, with mismatched enablers (another mechanism, same set) | +0.002 ± 0.009 |
| matched − mismatched, paired | +0.006 ± 0.010 (p = 0.55) |
| payoff standalone gap vs control at dose 0 | +0.011 ± 0.021 |

Both the aggregate and the mismatched-enabler control are null. Three on-mechanism enablers buy a payoff about a tenth of a winrate point over its control, indistinguishable from what off-mechanism cards of the same set buy. The mechanism section below predicts this: a model that does not track which card carries which text has no substrate for card-pair reasoning. What the scorer calls synergy is what the encoder labels carry: per-card color affinity and the deck-shape terms.

The apparent exceptions mostly dissolve under the mismatched-arm control. Gray Merchant of Asphodel posts the largest matched Δdose (+0.42), but its mismatched arm posts +0.36: any decent same-set cards raise its marginal, so the gain comes from the deck improving around it, not from devotion. One entry survives the control, Wingsteed Rider (matched +0.19, mismatched −0.15). Heroic's value is literally the density of spells that can target it, the kind of statistical property a plain average over cards can carry.

### Duplicates are priced like distinct cards of the same quality

The second and third copy of a card are worth what the first is worth. The probe replaces k filler slots with k copies of a decent creature, and separately with k distinct cards matched on single-swap marginal. The copies come out ahead by +0.01 to +0.02 per rung, a statistically significant bonus worth under one winrate point over three swaps, and there is no diminishing-returns penalty. Both arms' marginals shrink together as k rises, which is the mean-pooling dilution at work, not a duplicate effect.

*Source: `t4_synergy.py`, probe P-B.*

### The scorer has no opinion on how much removal a deck runs, only on which card fills a slot

The scorer barely distinguishes decks by how much removal they hold. Stripping two removal spells from an average deck costs −0.07 net of matched control swaps. Adding one to three more costs nothing. Across base removal counts from 2 to 8 the net deltas stay within ±0.17 with no consistent interior optimum. The scorer holds creature count and curve strongly, and removal count barely at all.

Removal count and removal preference are different questions, and each probe answers one of them. This ladder moves the count inside a finished deck, and counts every removal effect including the creatures that carry one, so its base count of 4.23 removal cards is wider than the 2.6 noncreature removal spells an average deck holds. The take rates above ask instead which noncreature card wins a slot, and find removal preferred by about half again over label-matched alternatives. Both probes control on the same winnability label, so the difference between them is not in what they hold fixed.

The preference is small per card and decisive in aggregate. A per-swap edge worth a few hundredths of a point barely moves the deck's score when stacked, which is the quantity this ladder reads. The same edge still wins most of the slot contests it enters, which is what the take rates read. Both agree with the card-level finding: removal is priced like a slightly-below-average body, and that is still better than the other noncreature cards competing for the slot.

*Source: `t4_synergy.py`, probe P-C.*

## The taste is the Forge-AI meta plus inherited human annotations

### The meta is Forge-AI self-play, and its piloting asymmetries are the largest single influence

Every preference above is a preference over decks as piloted by Forge's AI in Bo7 self-play. The piloting asymmetries — combat math handled well, aerial defense and instant timing badly — are what the category demotions and the flying premium of the Cards section trace back to. [`2026-05-13-gen3-initial-training.md`](2026-05-13-gen3-initial-training.md) established the asymmetry; every card-level probe here is consistent with it. The scorer optimizes the right objective for its deployment target and a measurably different objective from human Magic.

### Scores are only comparable within a set

Training pairs are always same-set, so the Bradley-Terry graph is disconnected across sets and per-set score offsets are unidentified. The offsets are large: Forge-built decks average −0.97 on Dissension and +2.13 on Double Masters, a spread of 3.1 units against a within-set deck spread of 0.93. Simple creature-dense sets land high; gold-heavy sets land low. A cross-set score comparison mixes set identity into deck quality roughly 3:1 and should never be used.

*Source: `post_hoc_slices.py`, section `t0`.*

### Part of the card ranking is inherited human taste, and part of the blacklist is inherited too

Forge's own builder picks by a bundled human pick-order file, and 40% of the original training decks were built that way. The labels therefore partially encode human taste rather than independently rediscovering it: human draft rank correlates with `shrunk_score_play` at Spearman 0.45.

Forge's hand-written `AI:RemoveDeck` blacklist (4.7K cards its builder refuses to play) leaves a direct corpus footprint. Blacklisted cards show a played-rate label 0.15 below matched controls, against a quality-label difference of only −0.03. The scorer's dislike of those cards is annotation inheritance, not game evidence.

*Source: `forge_hints.py` extraction; joins in `t7_artifacts.py`, probe D2.*

### The builder-family signature adds little once quality and shape are controlled

Builder families are trivially separable in the corpus: every Forge deck has 22 spells, every learned-builder deck exactly 23, and shape alone identifies the family at AUC 0.91. A scorer could in principle score that fingerprint instead of the deck. The measured residual is modest. After controlling mean card quality plus creature count and color count, the score's partial correlation with the shape fingerprint is 0.11. Most of the preference for gen-family decks routes through card quality and the shape preferences above, which are themselves win-correlated.

*Source: `t7_artifacts.py`, probe A.*

### Single-swap decisions are partly checkpoint noise

Three sibling gen-4 checkpoints agree on whole-deck ranking (Spearman 0.86–0.92) and on the direction of a swap (96–98%). They pick the same best swap out of twenty candidates only 48–66% of the time. The greedy builder's exact top choice is therefore partly model noise, even where the ranking it climbs is stable. Per-set held-out accuracy spread is likewise dominated by sampling noise (sd 0.087 observed vs 0.077 binomial): no robust per-set blind spot at ~30 matches per set.

*Source: `t7_artifacts.py`, probes D1 and C.*

### Encoding faults exist and mostly wash out

Three input-encoding faults were verified in code and then measured. X costs contribute zero to the mana-value feature, so Fireball-likes read as ~1-drops; their class value is low (−0.21), plausibly double-punished as bad cheap spells. Hybrid pips are counted for both colors, overstating the cost constraint; hybrid cards measure −0.16 with the low-MV confound and no clean isolation. Phantom P/T on vehicles and zero P/T on token-makers do not mislead, because the text-embedding labels dominate. Multi-face cards, whose deterministic features come from the front face only, show no measurable valuation distortion (n = 71 on-color adds: −0.16 vs −0.22 for single-face). No scorer-built deck contains snow basics or Wastes (0 of 10,000).

*Source: `post_hoc_slices.py`, sections `t2class`, `t1color`, `decks`.*

## The scorer is a mean pool over a 2–4 number summary of each card

### The pooling layer is a plain average, not learned attention

The scorer's learned pooling seeds attend uniformly. Measured over 400 real decks, every seed and head spreads its attention within 0.6% of exactly uniform, and no seed specializes on lands or any card class. The deck vector is therefore the arithmetic mean of the card representations.

Three invariances confirm the mean-pooling reading directly:

| test | result |
|---|---|
| replicate the deck (every card ×2, ×3) | score changes by 0.0000 |
| k identical copies of one card, k = 1…23 | score constant in k (mechanism demonstration) |
| scale the pooled representation ×0.25 … ×4 | scores unchanged (ρ = 1.0000); LayerNorm strips magnitude |

*Source: `t6_mechanism.py`, probes P2–P4.*

The consequence is that the model reads proportions, never counts. Creature share, color mix, and average card quality are visible to it; "how many cards" is not, except through the fixed deck sizes of its training data.

Inside the stack, card representations converge layer by layer: mean pairwise cosine rises from 0.19 at the input to 0.73 after layer 6. The stack computes deck-level averages, and only deck-level averages.

Uniform attention is better explained by the task than by a failed optimization. Three measurements support this reading. When the gen-2 sweep added explicit max-pool and mean-pool readouts alongside the learned pooling ([`2026-04-26-gen2-initial-training.md`](2026-04-26-gen2-initial-training.md)), validation accuracy did not change, so a non-uniform readout was available and carried no extra signal. The scorer reaches the Bo7 oracle ceiling with uniform attention, so a sharper mechanism had no headroom left. The quantities that predict a Bo7 outcome in this corpus are proportions: mean card winnability, creature share, color mix, curve mean. A uniform average is the correct operator for a proportion, not a degenerate one.

A training-side contribution cannot be ruled out, because three properties of the setup favor uniform attention from the start. The pooling seeds initialize near zero, so attention is uniform at the first step. The over-smoothed card representations give the pooling layer near-identical keys, so the gradient toward sharper attention is near zero whatever the queries learn. Bo7 label noise makes any small benefit of selective pooling hard to detect. Two discriminating experiments are cheap and have not been run: training an otherwise-identical scorer with a fixed mean in place of the pooling layer, and running the same attention capture on the gen-1 through gen-3 checkpoints.

Selective attention would matter only where deck value depends on extremes or pairings: a single bomb, a single unplayable card, a hole in the curve, a synergy pair. The corpus removes those signals before the scorer sees them. Bo7 aggregation averages out single-card extremes, and the Forge AI neither builds around nor pilots synergies. Attention still does necessary work in this model: the uniform mixing in the self-attention stack carries the deck context that the off-color penalty and splash thresholds above depend on. What never differentiated is selective attention, because a uniform broadcast was all the task required.

### The card-text embedding carries the signal; the 32 deterministic features are nearly redundant

Erasing per-card text costs nine points of held-out accuracy. Erasing the 32 deterministic features costs two. Each ablation replaces one block of every card's vector with its corpus mean and rescores all 4,708 held-out matches.

![Held-out accuracy under representation ablations](images/2026-08-27-scorer-ablation.png)

*Source: `t5_ablation.py`.*

The ablation reverses the gen-2-era diagnosis. [`2026-05-02-deterministic-feature-reliance.md`](2026-05-02-deterministic-feature-reliance.md) hypothesized the scorer leaned almost entirely on the deterministic features, and its planned ablations were never run. On gen-4 the text embedding is the primary channel. Ranking agreement shows the same asymmetry. Erasing text scrambles the model's deck ordering (ρ 0.51 against the full model); erasing the deterministic features leaves it largely intact (ρ 0.85). The two metrics can come apart, because an edit can reorder decks that sit close together in quality without flipping any pair a match actually compares.

Inside the deterministic features, the color pips are the one group the scorer clearly needs. The probe erases one feature group at a time, with the same mean-substitution design as the block-level ablation above. Two changes make the small per-group effects resolvable. The evaluation pool widens to 21,564 held-out matches, because the gen-4 round-robin corpus was generated 2026-05-19 through 2026-05-21, after the training cutoff, and joins the gen-5 file. Significance comes from the paired per-match difference against the full model, whose standard error is set by the prediction flip rate rather than the base accuracy.

![Held-out accuracy change from erasing each deterministic-feature group](images/2026-08-27-scorer-det-groups.png)

*Source: `t5b_det_groups.py`.*

Erasing the six pip counts flips 14% of match predictions, the largest effect of any single group. The full mana-cost group accounts for two thirds of what all 32 features contribute. The color flags and the power/toughness/loyalty slots have small effects, distinguishable from zero (z ≈ −2.5 each). Mana value alone is marginal. Mana production and is_land contribute nothing, because the text embedding already knows what a card produces and whether it is a land. The single-group effects sum to approximately the all-32 effect, so the contribution decomposes additively across the groups, with nothing left to interactions.

Which card carries which text vector barely matters. Permuting the text blocks across a deck's cards, so the deck keeps the same collection of vectors, costs under half an accuracy point. Any preference that requires knowing that this body carries that ability cannot survive this test.

Telling a Forge deck from a scorer-built deck is entirely a text-embedding judgment. With text erased, forge-vs-gen pairs drop to coin-flip accuracy while every other pair type degrades far less.

The same dependency explains the generation history: the aggregation barely changed from gen-2 to gen-4, and the gains came from the inputs. Gen-2's text dims came from the price-predictor encoder, whose training signal is card prices, a quantity dominated by collector value. Vectors like that carry no sealed card quality, so a mean over them can express deck shape and nothing else. The consequence was measured at the time: at matched deck shape, forge-best beat gen-2 by 8–17 winrate points ([`2026-05-02-deterministic-feature-reliance.md`](2026-05-02-deterministic-feature-reliance.md)). Gen-2's four null aggregation experiments — depth, dropout, multi-view pooling, hand-computed deck stats — all located the bottleneck upstream of the pooling.

Replacing the encoder is what moved match play. Gen-3 swapped the price encoder for one trained from scratch on per-card winnability labels, distilled from about a million self-play games. The swap took gen-3 from losing the matched-shape comparisons to beating forge-best on 47 of 48 pools ([`2026-05-13-gen3-initial-training.md`](2026-05-13-gen3-initial-training.md)). Gen-4 widened the encoder from 256 to 512 dims, worth ~3.3σ over the 256-wide build in match play ([`2026-05-15-gen4-initial-training.md`](2026-05-15-gen4-initial-training.md)).

The form of supervision mattered as much as the encoder. Phase B in the gen-2 era fine-tuned the price encoder against match outcomes and did not improve on the frozen baseline ([`2026-04-30-gen2-unfrozen-embeddings.md`](2026-04-30-gen2-unfrozen-embeddings.md)); a Bo7 match outcome is one noisy bit, too little signal to teach per-card quality through the scorer. Dense per-card labels from game logs are what succeeded. Across the whole lineage, the scorer was always a mean over per-card summaries, and every real gain came from improving what those summaries contain.

### Two numbers per card explain almost everything the scorer reads from text

Keeping only the top two principal components of every card's text vector reproduces the scorer's held-out accuracy. Four components saturate it. Ranking agreement with the full model keeps improving out to 256 components, so the remaining directions still shift scores; they just stop changing which deck of a held-out pair ranks higher.

![Held-out accuracy and ranking agreement with the full model as the text block is truncated to k principal components](images/2026-08-27-scorer-pc-truncation.png)

*Source: `make_text_pca.py` for the PCA; `t6_mechanism.py`, probe P1, for the truncation.*

The encoder's card cloud is this compressible because it is low-rank to begin with: PC1 carries 55% of card-to-card variance, the top 8 carry 75%, and the participation ratio is 3.2. The gen-3 encoders measured the same way in [`2026-05-11-sealed-encoder-hparam-sweep.md`](2026-05-11-sealed-encoder-hparam-sweep.md) had the same shape.

The two leading axes have measured meanings. Regressing the encoder's own training labels on the PC coordinates, over the 25,441 cards that carry every label, identifies PC1 as the played-rate axis and PC2 as the winnability axis. PC1 correlates +0.84 with the played-rate label and near zero with the quality labels; adding PC2 lifts the quality labels from near nothing to most of their variance. The score_draw curve coincides with the score_play curve: winning on the play and winning on the draw are one axis to the encoder.

![R² of each encoder label on the top-k text principal components](images/2026-08-27-scorer-pc-labels.png)

*Source: `make_text_pca.py`, label-regression section.*

Color affinity is missing from the leading components: the color-lift labels stay low across the whole charted range. The color information the scorer needs arrives through the deterministic pips instead, which is why the pips are the one deterministic group whose erasure hurts (the per-group ablation above).

Each card therefore reaches the scorer as a short list of meaningful numbers: castability and winnability from the text block, and its color pips from the deterministic features. The deck score is a shape-aware average of those summaries.

### The scorer pulls hardest on the winnability axis

Knowing which labels reach the scorer does not say how hard each one is used. Two measurements answer that, one associational and one causal. The associational measurement regresses the scorer's per-card values (`v_swap`, defined under Cards above) on the three label axes at once, winnability being the mean of the near-identical score_play and score_draw. In that regression, winnability carries about three times the weight of played-rate. Cast_lift adds nothing once the other two are held fixed (unique R² 0.003).

The causal measurement perturbs one card inside a real deck and reads the score response. The perturbation step is the average change in a card's text vector that accompanies a one-standard-deviation increase of one label. Each step is applied to every card of 300 held-out decks, one card at a time, in both directions.

![Associational and causal weight of each label axis in the scorer](images/2026-08-27-scorer-label-weights.png)

*Source: `t5c_label_weights.py`.*

The two measurements agree: improving one card by a standard deviation along the winnability axis moves the score about six times as much as along the played-rate axis. The tall causal cast_lift bar is overlap, not an independent weight. The labels correlate at 0.69, and the causal steps are not orthogonalized. The cast_lift step therefore largely retraces the winnability direction. The regression does hold the other axes fixed, and it puts cast_lift's own contribution near zero.

The gray PC bars repeat the axis identities from the chart above. A PC2 step moves the score as much as the winnability step. A PC1 step moves it a quarter as much, although PC1 carries most of the embedding's variance. The encoder's loudest axis is not the axis the scorer uses most.

On the ruler, a one-standard-deviation winnability improvement on a single card is worth roughly two winrate points.

### Three quarters of the score range separates incoherent decks from coherent ones

Of the 6.1 score units between the mean random-pile deck and the mean gen-5 deck, 4.6 lie below the forge-best baseline. Realistic candidate decks for one pool live on the remaining quarter of the range. Within that quarter the ordering is right: mean score by builder is strictly monotone in the builders' real match-play strength.

![Mean score by deck builder, with the coherence and quality spans marked](images/2026-08-27-scorer-builder-scores.png)

*Source: `t0_landscape.py`; cuts in `post_hoc_slices.py`, section `t0`.*

## Limitations

Card-level values are context-relative: `v_swap` comes from two-color Forge-built contexts, so a card's value in an archetype the corpus never builds is not measured. The archetype take rates are the builder's revealed choice under greedy hill-climbing, so a preference the search cannot reach does not appear in them. Their archetype labels come from regexes over card text, and 5.9% of noncreature slots land in a residual class that audits as land auras, prison cards and do-nothing artifacts rather than as a missed archetype. Shape ladders perturb decks built by a sibling scorer, so rung 0 sits at a local optimum and raw rung deltas include a chosen-vs-rejected quality baseline. Conclusions therefore rest on rung-to-rung marginals and controls, not raw deltas. Everything is a preference of this checkpoint under Forge-AI-piloted Bo7. None of it is a claim about human Magic except where explicitly compared to human pick orders. A few hypotheses from the study's ranked list remain untested: whether swap sensitivity is flat across deck slots, whether narrow situational cards are penalized through quality or through played-rate, and whether frequently-observed cards are memorized beyond what their text supports.
