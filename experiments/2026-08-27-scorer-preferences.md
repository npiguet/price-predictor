# What the deck scorer prefers

An interpretability study of the production sealed deck scorer: which deck properties raise its score, which lower it, and where those preferences come from. The method was three phases: a fifty-hypothesis brainstorm critiqued by four independent Opus reviewers (one of which ran label-level regressions against Forge's human draft rankings), a ranking pass, and then a battery of inference-only probes — about 400,000 scorer forward passes, no training. This document records the hypotheses, the tests, and the verdicts. It is the evidence base for an article on what the scorer likes and dislikes when building from a sealed or draft pool.

The subject is the gen-4 production checkpoint `models/sealed/scorer/512-best_l6_h4_s4_ff2176_mlp512_lr1e-05_mwlog.pt`: the 512d attention-pool sealed encoder plus 32 deterministic features per card, a 6-layer Set Transformer (4 heads, 4 pooling seeds, ff 2176, mlp 512), log margin weighting, trained 2026-05-18 on the 70,134-match Bo7 corpus. Card embeddings come from `output/cardsfolder-512/`. Held-out evaluation uses `match-outcomes-gen5-vs-gen4-forge.txt` (4,708 Bo7 matches generated 2026-05-26, after the training cutoff). Background: [`2026-05-15-gen4-initial-training.md`](2026-05-15-gen4-initial-training.md), [`specs/2026-05-03-card-winnability-pretraining.md`](../specs/2026-05-03-card-winnability-pretraining.md), [`specs/2026-03-28-sealed-deck-picker.md`](../specs/2026-03-28-sealed-deck-picker.md).

Probes far outside the training distribution measure only the arithmetic of the network, never preferences, and are labeled as mechanism demonstrations throughout. The scorer was trained exclusively on realistic decks — Forge-built, scorer-built, and versions of those with 3 to 23 cards replaced at random — drawn from realistic pools. A real deck with one to four cards swapped stays inside that envelope; a deck of 23 copies of one card or a 5-card deck does not, and its score means nothing about deck-building taste.

## The ruler: one score unit is worth about 18 winrate points

Score differences convert to win probability at a stable, measurable rate. On the held-out matches, binning by score margin (`t7_artifacts.py`, probe C) gives a monotone calibration across all ten deciles, fitted by `P(A wins) = sigmoid(−0.07 + 0.77·Δscore)` — about +18 winrate points per score unit near the midpoint, flattening toward the tails. The fitted slope below 1.0 means the raw Bradley-Terry reading `sigmoid(Δscore)` is mildly overconfident. Every Δscore below can be multiplied by roughly 18 pp/unit to get a felt size.

| Δscore decile mean | −4.6 | −2.1 | −1.1 | −0.55 | −0.17 | +0.16 | +0.51 | +1.02 | +1.98 | +4.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| empirical P(A wins) | .040 | .153 | .295 | .387 | .454 | .499 | .585 | .675 | .803 | .949 |

Held-out prediction accuracy is 71.9% overall — at the estimated Bo7 oracle ceiling of 0.72–0.78 from [`2026-04-26-gen2-initial-training.md`](2026-04-26-gen2-initial-training.md) — ranging from ~99% on gen-vs-random pairs down to ~60% on gen4-vs-gen5 and mirror pairs.

## How the scorer reads a deck

### The pooling layer is a plain average, not learned attention

The PMA layer's learned seed vectors attend uniformly, so the deck vector is the arithmetic mean of card representations. Measured over 400 real decks (`t6_mechanism.py`), every seed×head attention distribution is uniform to within a coefficient of variation of 0.6% (max weight 1.01× uniform), and no seed specializes on lands or any card class. Three consequences were verified directly. Replicating a deck (every card twice or three times) changes the score by exactly zero. A deck of k identical copies of one card scores the same for k = 1 and k = 23 — the score of the "deck" equals the score the single card's vector produces (mechanism demonstration; such inputs are far outside training). And scaling the whole representation entering the pool by 0.25× to 4× leaves scores unchanged (Spearman 1.0000), because the pool's LayerNorm strips magnitude. The model therefore reads proportions — creature share, color mix, average quality — and cannot read absolute counts except through the fixed deck-size conventions of its training data.

Inside the stack, card representations converge layer by layer (mean pairwise cosine 0.19 at input → 0.73 after layer 6), the classic over-smoothing pattern. Over-smoothing also explains why multi-view pooling and hand-computed deck stats added nothing in the gen-2 sweeps ([`2026-04-26-gen2-initial-training.md`](2026-04-26-gen2-initial-training.md)): the architecture was already computing deck-level averages, and only deck-level averages.

### The card-text embedding carries the signal; the 32 hand features are nearly redundant

Erasing per-card text identity costs nine points of held-out accuracy; erasing the deterministic features costs two. Each ablation replaces one block of every card's vector with its corpus mean and rescores all 4,708 held-out matches (`t5_ablation.py`):

| condition | held-out acc | Spearman vs full scores |
|---|---|---|
| full model | .719 | 1.000 |
| text block → corpus mean | .630 | .514 |
| text blocks shuffled within deck | .715 | .969 |
| det features → corpus mean | .699 | .848 |
| both → corpus mean | .494 | −.075 |

The ablation reverses the gen-2-era diagnosis. [`2026-05-02-deterministic-feature-reliance.md`](2026-05-02-deterministic-feature-reliance.md) hypothesized the scorer leaned almost entirely on the 32 deterministic features; its planned ablations were never run. Run now on gen-4, they show the opposite: the sealed-trained text embedding is the primary channel, and the deterministic block is mostly a redundant copy of information the text also carries. The per-pair breakdown makes the point sharper: with text erased, forge-vs-gen pairs drop to coin-flip accuracy — telling a Forge deck from a scorer deck is entirely a text-embedding judgment.

The shuffle row answers a different question: permuting which card carries which text vector, while keeping the deck-level bag of vectors intact, costs almost nothing (71.5% vs 71.9%). The scorer scores the bag, not the binding. A preference that requires knowing that this body carries that ability cannot survive this test, which constrains how much card-pair reasoning the model can be doing.

### Two numbers per card explain almost everything the scorer reads from text

Projecting every card's 512-dim text vector onto its top two principal components reproduces held-out accuracy; four components saturate it. The encoder's card cloud is extremely low-rank (PC1 carries 55% of variance, top-8 75%, participation ratio 3.2 — same shape as the gen-3 encoders measured in [`2026-05-11-sealed-encoder-hparam-sweep.md`](2026-05-11-sealed-encoder-hparam-sweep.md)), and the scorer reads only the head of it (`make_text_pca.py`; truncation in `t6_mechanism.py`, P1):

| text dims kept (top-k PCs) | 0 | 1 | 2 | 4 | 16 | 64 | 512 |
|---|---|---|---|---|---|---|---|
| held-out acc | .628 | .650 | .709 | .717 | .717 | .708 | .714 |
| Spearman vs full scores | .24 | .33 | .88 | .92 | .95 | .98 | 1.00 |

The picture of the whole system that emerges: each card is compressed to a 2–4 number summary (roughly: a learned winnability scalar, a castability/played-rate axis, and color affinity), and the deck score is a shape-aware average of those summaries. Score fidelity keeps improving out to k=256, so the remaining directions do shift scores — they just no longer change which deck of a pair ranks higher.

### Three quarters of the score range separates incoherent decks from coherent ones

Of the 6.1-unit span between the mean random-pile deck and the mean gen-5 deck (`t0_landscape.py`, cuts in `post_hoc_slices.py`), 4.6 units lie below the Forge-builder baseline and only 1.5 above it. Same-set decks from one builder spread with a standard deviation of 0.93. Realistic candidate decks for one pool therefore live on roughly a quarter of the scorer's dynamic range; the rest is spent recognizing that 23 random cards are not a deck. Mean score by builder is strictly monotone in the builders' real match-play strength — random −3.7, forge-best +0.9, gen3-256 +1.9, gen4-512 +2.4, gen5 +2.4 — so within its range the ordering is right.

## Deck-shape preferences

Shape probes swap cards inside real decks (250–800 contexts sampled from 10,000 aligned pool/deck pairs) and read the score response; the ladders are `t3_ladders.py`, the add-a-card probe `t1_meansum.py`, the spell-count probe `t7_artifacts.py` (probe B). One baseline number recurs: swapping any chosen card for a card the builder rejected costs about −0.40 on average, because chosen cards are simply better. Ladder effects are read against that baseline and against each ladder's own rung-to-rung marginals.

### Color count: the price is paid per color, and the first off-color card pays it

The first card of a new color costs about three times what every further card of that color costs. Swapping in 1, 2, 3, 4 cards of a new color moves the score by −0.48, −0.64, −0.75, −0.87 — marginals of −0.48, then −0.16, −0.11, −0.12, where the later marginals are ordinary quality-trade swap costs. The single-pip splash ladder shows the same threshold (−0.39, −0.16, −0.13). The scorer prices the presence of a third color, not the number of off-color pips — the same economics as a manabase that must find slots for every color it plays. A 2×2 probe (add a single-pip splash spell, an on-color-for-the-splash dual land, both, or neither) shows the dual genuinely offsets the splash penalty, by about a tenth of it (interaction +0.041, t=14): real color-fixing logic, small magnitude. In deployment this preference is visible as gen-4-512 building 2 colors 34% and 3 colors 58% of the time, adapting to set structure (2.2 mean colors on artifact-heavy Mirrodin sets, 4.0 on all-gold Alara Reborn; `post_hoc_slices.py`, section `decks`).

### Creature count: the optimum is 19–20, and too few is punished harder than too many

Swapping creatures out hurts more than swapping creatures in at every rung. Four creature-for-spell swaps (matched on mana value and color) cost −0.69; four spell-for-creature swaps cost −0.54; the score peak across realized creature counts sits at 19–20 creatures in a 23-spell deck. That is above the ~14.6 creatures Forge's own builder plays and above even gen-4's observed 18.2 average. The asymmetry matches the empirical corpus, where creature-light decks lose badly under Forge piloting; whether it matches human play is a separate question the corpus cannot answer ([`2026-05-13-gen3-initial-training.md`](2026-05-13-gen3-initial-training.md) documents the piloting bias).

### Curve: the optimum is near 3.2–3.3 average mana value, and cheap is punished harder than expensive

Shifting a deck's curve down costs more than shifting it up by the same number of swaps. Four swaps toward cheaper same-color cards (realized mean MV 2.66) cost −1.09; four toward more expensive (MV 3.72) cost −0.80; the peak sits at mean MV ≈ 3.2–3.3. That is at the bottom edge of the 3.4–4.0 band the win-rate tables identified as strongest in the gen-2 era, so the scorer's curve taste is roughly right with a slight cheap-side lean at the deck level — while at the card level it prefers expensive cards (next section). The two are consistent: a good curve needs the cheap slots filled, but an individual cheap card is rarely the best card.

### Curve shape beyond the mean does not matter

Mean-preserving spread — replacing two 3-drops with a 2-drop and a 4-drop, or the reverse — moves the score less than a quality-matched control swap at the same mean (|Δ| lower by 0.07, t=3.7). The scorer holds no opinion about curve smoothness, gaps, or bimodality at fixed mean; the "no 2-drops" alarm a human builder raises has no analogue in the model.

### The scorer wants 22 spells and 18 lands, and the deployed builder cannot give them to it

Offered the same deck at 22, 23, or 24 spells (`t7_artifacts.py`, probe B), the scorer prefers 22 in 88% of decks and 24 in under 2%. Dropping the deck's worst-label spell (making room for an 18th land) gains +0.08; dropping a random spell is exactly neutral, so the gain is card quality, not a count preference downward. Adding a 24th spell costs −0.44 regardless of the card's quality — a hard prior against spell counts that never occur in its training data (Forge builds 22 spells, every learned builder exactly 23). The preference is not curve-conditional (correlation with deck mean MV: 0.01). The deployment mismatch is direct: `GreedyDeckBuilder` pins 23 spells by construction, so the scorer's 18-land opinion is unexpressible at build time.

The same size prior governs nonbasic lands (`t1_meansum.py`; on/off-color split in `post_hoc_slices.py`). Adding a land to a built deck is the least-penalized addition (lands producing the deck's colors −0.07 with 19% of adds positive, off-color lands −0.12, colorless-only −0.12, versus −0.22 for on-color spells and −0.52 for off-color spells), and land classes are ordered correctly — but most land adds still read as negative, which is why scorer-built decks carry 0.5 nonbasic lands on average where Forge's carry 1.1, and 62% of gen-4-512 decks run zero. The builder under-plays duals as a direct result: its 23-spell invariant forbids trading a spell for a land, and adding the land on top usually reads to the scorer as dilution.

## Card preferences

Card-level values come from two probes that agree (r = 0.78; `t2_marginal_values.py`, analysis in `t2_analyze.py`): leave-one-out deltas inside 3,500 real decks, and a standardized swap-in value — put the card into the median slot of up to eight fixed two-color Forge-built decks matching its colors — measured for all 26,402 cards with 30+ game observations. The swap-in value `v_swap` is the number quoted below; its scale is the same score scale as everything else (≈18 pp/unit), and 0 means "as good as the median card of a real deck".

### The card ranking is learned winnability first

The scorer's per-card value tracks the encoder's empirical win-rate labels more strongly than any visible card property: Spearman +0.68 against `shrunk_score_play`, +0.53 against `cast_lift`, +0.41 against `played_rate`. A regression on visible properties (type, cost, stats, keywords, rarity) explains 40% of value variance; adding the empirical label lifts it to 58%, and the label's coefficient dwarfs everything else. The scorer is, before anything else, a reader of "this card won games in the self-play corpus" — as intended, with the deck-shape terms above layered on top.

### The category order is creatures, then removal, then everything else — BREAD it is not

An average creature outranks an average removal spell by about a tenth of a unit, and removal outranks the rest of the non-creature pile by a further tenth:

| category | n | mean v_swap |
|---|---|---|
| creature | 14,891 | −0.01 |
| noncreature removal | 1,600 | −0.12 |
| card draw | 1,493 | −0.25 |
| other noncreature | 8,418 | −0.25 |

Against the human BREAD ordering (Bombs, Removal, Evasion, Aggro, Duds), removal is demoted below generic bodies, and the label-level comparison against Forge's shipped human pick orders (`human_rank_probe.py`) puts removal 0.2 standard deviations below where humans rank it, card draw 0.45 below, and artifacts 0.63 below. The mechanism is the Forge-AI meta the labels were earned in: the AI attacks and blocks competently but times instants and card advantage poorly, so bodies convert to wins and answers convert to card disadvantage. Instant-speed removal earns no premium over sorcery-speed (−0.10 vs −0.10). The extreme bottom of the card ranking is uniformly durdle artifacts and engines (Cloudstone Curio, Witch's Oven, Krark's Thumb), not bad creatures; the extreme top is 26 creatures out of 30 cards, mostly 4–6 MV flying/lifelink rares.

### Expensive cards are preferred at the margin; cheap cards are the worst cards

Marginal value rises monotonically with mana value, from −0.19 for 0–1 MV cards to −0.01 for 6+ MV. The scorer pays for top-end and treats one-drops as the weakest card class — while simultaneously holding the deck-level curve optimum at 3.2–3.3. Castability is priced at the deck level (splash thresholds, pip support), not as a per-card cheapness bonus; the label-level analysis shows the same thing (`score_play` rises ~0.005 per MV while `played_rate` falls, and the winnability axis wins).

### Flying is the biggest text-derived premium and grows with size

A flier outscores a ground creature of the same MV bucket by +0.06 at 1–2 MV and +0.16 at 6+. At the label level flying sits 0.57 standard deviations above human pick order — the largest positive divergence measured — because the Forge AI handles ground combat well and aerial defense badly. Creatures with any combat keyword average +0.08 above keyword-less creatures; vanilla creatures score the same as creatures whose text has no combat keyword. Power beats toughness: conditioned on MV and stats total, the label regression (`label_probe.py`) prices +1 power at +0.004 and +1 toughness at −0.002.

### Only mythics carry a rarity premium, and rarity is inferred from text alone

Commons, uncommons, and rares all average the same value (−0.11 to −0.12); mythics average +0.01. The encoder receives no rarity input — the premium is whatever bomb-ness the text itself reveals. Controlling for the empirical label flips the rare coefficient negative: the average rare's text reads worse to the scorer than its label warrants, consistent with sealed rares including many narrow or constructed-facing cards.

### Class by class: tricks at the bottom, planeswalkers priced like creatures

The class slices come from `post_hoc_slices.py` over the T2 card-value table:

| class | n | mean v_swap | note |
|---|---|---|---|
| combat trick (pump instant) | 295 | −0.27 | worst class measured |
| X-cost spell | 416 | −0.21 | X encodes as 0 MV, so these read as bad cheap spells |
| vehicle | 139 | −0.20 | not fooled by the printed P/T in the det block |
| counterspell | 405 | −0.19 | |
| hybrid-mana card | 545 | −0.16 | low-MV confounded |
| noncreature token-maker | 1,206 | −0.12 | |
| removal, instant or sorcery | 1,100 | −0.10 | |
| planeswalker | 223 | +0.00 | at parity with MV-matched creatures |

Two of these contradict the hypotheses that motivated them. Combat tricks were predicted to be overvalued (their `cast_lift` label is the highest of any class, inflated by a survivorship artifact — cheap cards are only ever uncast in disaster games), yet the scorer ranks them lowest of all: the cheap-instant penalties dominate, and the scorer lands near the human consensus on tricks by a different route. Planeswalkers were predicted to be depressed by the AI's poor piloting; they price at creature parity instead. Vehicles show the deterministic block's phantom P/T (a Vehicle "is" a 4/4 in the features even though it does nothing uncrewed) does not mislead the model — the text embedding's learned labels carry the truth. A hypothesized wordiness bias (long text as a power proxy) measures at a standardized +0.003, effectively zero.

## Synergy and interactions

### Pair synergy is absent: enabler density does not raise a payoff's value

The scorer does not pay more for a synergy payoff as its enablers enter the deck. The probe (`t4_synergy.py`, dataset `synergy_pairs.json`) used 57 curated payoff/enabler/control triples (tribal lords, sacrifice payoffs, heroic targets, devotion, spells-matter — 20 sets, commons and uncommons only, every control matched to its payoff on set, color, type, and mana value within 1). For each triple and eight real same-set context decks, filler slots were swapped for 0–3 enablers, and the payoff's swap-in marginal was measured at each dose against the control card's marginal in the same slot:

| statistic (mean over 57 entries) | value |
|---|---|
| Δdose = (payoff − control) gain from 0 → 3 enablers | +0.008 ± 0.010 (p = 0.43) |
| same, with mismatched enablers (another mechanism, same set) | +0.002 ± 0.009 |
| matched − mismatched, paired | +0.006 ± 0.010 (p = 0.55) |
| payoff standalone gap vs control at dose 0 | +0.011 ± 0.021 |

Both the aggregate and the leakage control are null: three on-mechanism enablers buy a payoff about a tenth of a winrate point over its control, indistinguishable from what off-mechanism cards of the same set buy. The bag-of-text and mean-pooling mechanism results predicted exactly this — a model that does not track which card carries which text has no substrate for card-pair reasoning. What the scorer calls synergy is what the encoder labels carry: per-card color affinity (`color_lift`) and the deck-shape terms.

The apparent exceptions mostly dissolve under the mismatched-arm control. Gray Merchant of Asphodel posts the largest matched Δdose (+0.42), but its mismatched arm posts +0.36 — any decent same-set cards raise its marginal, which is deck drift, not devotion. The one entry that survives the control is Wingsteed Rider (matched +0.19, mismatched −0.15); heroic's value is literally the density of spells that can target it, the kind of statistical property a bag-of-cards average can carry. Tribal lords, sacrifice packages, equipment packages, and every other family sit within noise of zero.

### Duplicates are priced like distinct cards of the same quality

The second and third copy of a card are worth what the first is worth (`t4_synergy.py`, P-B). Replacing k filler slots with k copies of a decent common versus k distinct cards matched on single-swap marginal gives copy-vs-distinct differences of +0.01 to +0.02 per rung — a small but statistically significant bonus for copies (+0.05 over three swaps, under one winrate point) and no diminishing-returns penalty. Both arms' marginals shrink together as k rises (+0.28 → +0.19), which is the mean-pooling dilution at work, not a duplicate effect.

### The removal-share opinion is weak: cutting below three hurts a little, stacking is free

Stripping two removal spells from an average deck costs −0.07 net of matched control swaps; adding one to three more removal spells costs nothing (`t4_synergy.py`, P-C). Across base removal counts from 2 to 8 the net deltas stay within ±0.16 with no consistent interior optimum. The scorer holds creature count and curve strongly and removal share barely at all — consistent with the card-level finding that removal is priced like a slightly-below-average body rather than a scarce role to fill.

## Where the taste comes from

### The meta is Forge-AI self-play, and its piloting asymmetries are the largest single influence

Everything above is a preference over decks as piloted by Forge's AI in Bo7 self-play. The AI's known strengths (combat math) and weaknesses (instant timing, card-advantage conversion, aerial blocking) explain the creature tilt, the removal demotion, the card-draw penalty, and the flying premium — this was established in [`2026-05-13-gen3-initial-training.md`](2026-05-13-gen3-initial-training.md) and every card-level probe here is consistent with it. The scorer optimizes the right objective for its deployment target and a measurably different objective from human Magic.

### Part of the card ranking is inherited human taste, and part of the blacklist is inherited too

Forge's own builder picks by a bundled human pick-order file, and 40% of the original training decks were built that way, so the labels partially encode human taste rather than independently rediscovering it: human draft rank correlates with `shrunk_score_play` at Spearman 0.45 (`forge_hints.py` extraction, joined in `t7_artifacts.py`, probe D2). Forge's hand-written `AI:RemoveDeck` blacklist (4.7K cards the builder refuses to play) leaves a direct corpus footprint: blacklisted cards show a played-rate label 0.15 below matched controls, with only a −0.03 quality-label difference. The scorer's dislike of those cards is annotation inheritance, not game evidence.

### Scores are only comparable within a set

Training pairs are always same-set, so the Bradley-Terry graph is disconnected across sets and per-set score offsets are unidentified. Observed (`post_hoc_slices.py`, section `t0`): Forge-built decks average −0.97 on Dissension and +2.13 on Double Masters, a spread of 3.1 units against a within-set deck spread of 0.93. Simple creature-dense sets (Portal, core sets) land high; gold-heavy sets land low. A cross-set score comparison mixes set identity into deck quality roughly 3:1 and should never be used.

### The builder-family signature adds little once quality and shape are controlled

Builder families are trivially separable in the corpus (`t7_artifacts.py`, probe A) — every Forge deck has 22 spells, every learned-builder deck exactly 23, and shape alone identifies the family at AUC 0.91 — so a scorer could in principle score the fingerprint instead of the deck. The measured residual is modest: after controlling mean card quality plus creature count and color count, the score's partial correlation with the shape fingerprint is 0.11. Most of the scorer's preference for gen-family decks routes through card quality and the shape preferences above, which are themselves win-correlated; a small fingerprint residual remains.

### Single-swap decisions are partly checkpoint noise

Three sibling gen-4 checkpoints (production log-margin, unweighted small, unweighted big; `t7_artifacts.py`, probe D1) agree on whole-deck ranking (Spearman 0.86–0.92) and on the direction of a swap (96–98%), but pick the same best swap out of twenty candidates only 48–66% of the time. The greedy builder's exact top choice is therefore partly model noise even where the ranking it climbs is stable. Per-set held-out accuracy spread, for the same reason, is dominated by sampling noise (sd 0.087 observed vs 0.077 binomial) — no robust per-set blind spot was detectable at ~30 matches per set.

### Encoding faults exist and mostly wash out

Three input-encoding faults were verified in code and then measured for effect (`post_hoc_slices.py`, sections `t2class`, `t1color`, `decks`). X costs contribute zero to the mana-value feature, so Fireball-likes read as ~1-drops; their class value is low (−0.21), plausibly double-punished as "bad cheap spells". Hybrid pips are counted for both colors, overstating their cost constraint; hybrid cards measure −0.16 with the low-MV confound and no clean isolation. Phantom P/T on vehicles and zero P/T on token-makers do not mislead (vehicles priced −0.20, token-makers mid-pack) because the text-embedding labels dominate. Multi-face cards (front-face-only det features) show no measurable valuation distortion (n=71 on-color adds, −0.16 vs −0.22 single-face). No scorer-built deck contains snow basics or Wastes (0 of 10,000).

## Hypothesis verdicts

The fifty ranked hypotheses, with verdicts. "Confirmed"/"falsified" mean the probe result matched/contradicted the hypothesis as sharpened after critique; "partial" means the mechanism held with a materially different magnitude or route; pointers name the probe.

| # | hypothesis (short) | verdict | evidence |
|---|---|---|---|
| R1 | score ≈ text-carried per-card quality sum; det features minor | confirmed | T5 ablation; T2 correlations; T6 PC-truncation |
| R2 | 2–3 colors preferred; 4–5 penalized; castability-floor mechanism | confirmed | T3-A/E/F ladders |
| R3 | creature-dense optimum (~17–18) | confirmed, optimum higher (19–20) | T3-B |
| R4 | flying premium above human norm | confirmed | T2 flags; label E5 |
| R5 | scores set-relative, cross-set incomparable | confirmed | T0 set offsets |
| R6 | board-centric midrange, big-over-cheap, not aggro | confirmed | T2 MV/category; T3-C |
| R7 | mean-like pooling; below-average additions lower score | confirmed (mechanism exact) | T6 P2/P4 |
| R8 | scorer under-plays nonbasic lands vs Forge | confirmed | T0 counts; T1 land adds |
| R9 | counterspells undervalued | confirmed | T2 class slice |
| R10 | card-draw/durdle penalized; lifegain/scry neutral | confirmed | T2 categories; labels E1/E5 |
| R11 | removal priced at/below median creature | confirmed | T2 category means |
| R12 | size over efficiency; stats-per-mana unrewarded | confirmed | T2 ((P+T)/MV corr ≈ 0.04) |
| R13 | power > toughness at fixed stats | confirmed (label level) | E1 regression |
| R14 | creature bombs ≥ human, noncreature bombs ≪ human | partial — top-30 all creatures; walkers at parity | T2 extremes, class slice |
| R15 | rarity gradient without rarity input | confirmed, mythics only | T2 rarity means |
| R16 | keyword story ≈ flying + small rest | confirmed | T2 per-keyword |
| R17 | 6+MV marginal declines with count | untested as stated; deck-level curve optimum confirms diminishing top-end | T3-C |
| R18 | score-optimal curve ≤ win-optimal band | confirmed (3.2–3.3 vs 3.4–4.0) | T3-C |
| R19 | on-color castability priced; cheapness not a per-card bonus | confirmed | T1 on/off-color; T2 MV |
| R20 | on-color gold ≥ mono | supported (n_colors +0.25 std in context-matched swaps) | T2 model A |
| R21 | artifacts penalized, not bonused | confirmed | T2 categories; bottom-30 |
| R22 | single-swap resolution near noise | partial — top-1 noisy, ranking stable | T7-D1 |
| R23 | builder-fingerprint detection | partial — 0.11 residual after controls | T7-A |
| R24 | vehicles overrated by phantom P/T | falsified — priced low | T2 class slice |
| R25 | X/hybrid encoding faults distort | confirmed for X (direction: undervalue); hybrid unresolved | T2 class slice |
| R26 | systematic per-set blind spots | not detectable at n≈30/set | T7-C |
| R27 | human-taste inheritance | confirmed (ρ 0.45) | T7-D2 |
| R28 | wordiness bias | falsified (β +0.003) | T2 post-hoc |
| R29 | AI:RemoveDeck blacklist inheritance | confirmed via played_rate | T7-D2 |
| R30 | tricks overvalued (label artifact) | falsified — scorer ranks tricks lowest | T2 class slice |
| R31 | pair-synergy detection | falsified (Δdose +0.008, p=0.43); one entry (Wingsteed Rider, heroic) survives the mismatched-arm control | T4 P-A |
| R32 | set-archetype awareness beyond shape | no evidence — matched vs mismatched enablers indistinguishable | T4 P-A control |
| R33 | duplicate penalty/bonus | falsified — copies priced like matched distinct cards | T4 P-B; T6 P4 |
| R34 | role-balance concavity (removal share) | weak — small cost below ~3 removal, stacking free | T4 P-C |
| R35 | curve-dispersion preference at fixed mean | falsified | T3-D |
| R36 | splash penalty per-color threshold | confirmed | T3-E |
| R37 | fixing×splash interaction | confirmed, ~10% offset | T3-F |
| R38 | land classes priced (dual > utility > off-color) | confirmed | T1 land subset |
| R39 | consistency (played_rate at fixed quality) | weakly supported — played_rate correlates (+0.47 raw); partial effect untested | T2 correlations |
| R40 | flat swap-sensitivity across slots (no replacement-level concept) | untested | — |
| R41 | narrow cards penalized via score not played_rate | untested (no tag built) | — |
| R42 | dynamic range spent on coherence | confirmed (75%) | T0 |
| R43 | mode-seeking miscalibration OOD | reframed: OOD scores arbitrary, per user note | T6 P5 |
| R44 | det-feature reliance (gen-2 hypothesis) | falsified for gen-4 — text dominates | T5 |
| R45 | multi-face mis-encoding distorts value | mechanism real, effect not measurable | T1 slice |
| R46 | mean-not-sum pooling | confirmed | T6 |
| R47 | OOD degeneracy | confirmed as mechanism demo only | T6 P5 |
| R48 | deck-size gradient | confirmed — anti-24th-card prior, pro-22-spell | T7-B; T1 |
| R49 | high-n card memorization | untested (cost) | — |
| R50 | snow-basics quirk | absent from builds | T0 check |

## Probe inventory

| probe | script (`scripts/scorer_probes/`) | what it did | scale |
|---|---|---|---|
| T0 landscape | `t0_landscape.py` | scored every builder's decks from the match corpora, joined deck features | 42,525 decks |
| T5 ablation | `t5_ablation.py` | block-ablated rescoring of all held-out matches | 8,658 decks × 5 conditions |
| T1 add-a-card | `t1_meansum.py` | scored every remaining pool card added to built decks | 400 contexts, 16K forwards |
| T2 marginal values | `t2_marginal_values.py`, `t2_analyze.py` | leave-one-out + fixed-context swap-in value for every observed card | 26,402 cards, ~235K forwards |
| T3 ladders | `t3_ladders.py` | six controlled swap ladders (color, creature, curve, spread, splash, fixing) | 250 contexts, 7,127 decks |
| T6 mechanism | `t6_mechanism.py` (+ `make_text_pca.py`) | PC-truncation, attention/over-smoothing capture, invariance checks, OOD envelope | 400 decks |
| T7 artifacts | `t7_artifacts.py` (+ `forge_hints.py`) | builder fingerprint, spell-count preference, calibration, sibling agreement, Forge-annotation joins | 800 pairs + 4,708 matches × 3 models |
| T4 synergy | `t4_synergy.py` (+ `synergy_pairs.json`, `verify.py`) | dose-response super-additivity (57 curated triples × 8 contexts, matched + mismatched arms), duplicates ladder, removal-share ladder | 9,397 decks |

The probe scripts live in [`scripts/scorer_probes/`](../scripts/scorer_probes/README.md) (see its README for the run order and the full script-to-section map); their outputs — the `t*_report.md` / `t*_results.json` / CSV files this document's numbers come from — are kept in `output/scorer-probes/` and regenerate on rerun. Prose numbers with no dedicated probe script (within-set correlations, set offsets, nonbasic-land counts, card-class slices, the add-a-card robustness check) come from `post_hoc_slices.py`; the label-level numbers ("sd vs human pick order", the power/toughness and rarity label regressions) come from `label_probe.py` → `human_rank_probe.py` / `rarity_probe.py`. Every probe is inference-only against the production checkpoint plus the `cardsfolder-512` embedding cache, the Y: corpus files, and `cards-win-rates.txt`.

Limitations. All card-level values are context-relative (real 2-color decks; a card's value in an archetype the corpus never builds is not measured). Shape ladders perturb decks built by a sibling scorer, so rung-0 sits at a local optimum and rung deltas include a chosen-vs-rejected quality baseline; conclusions rest on rung-to-rung marginals and controls, not raw deltas. Everything is a preference of this checkpoint under Forge-AI-piloted Bo7; none of it is a claim about human Magic except where explicitly compared to human pick orders.
