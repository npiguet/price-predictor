# Ability effect model — gen-2 improvements

Design record for the second generation of the ability effect model. The model has two parts. The
encoder reads an ability's script text and outputs a fixed-length vector `e`, the ability's
embedding. The effect head, a transformer, reads `e` together with the board and predicts what the
ability does to each entity on it. Gen-1 is the model trained on the corpus of 2026-09-17 (run
`09.17f`, filed under `models/effects/runs/2026-09-17-full-textless-corpus/`) and its four
baselines. The design rationale is
[`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md), called
the design doc below.

Gen-1 is validated on two strata of games kept out of training. The game-disjoint stratum holds
whole games whose ability texts also occur in training games. The card-disjoint stratum holds games
that contain a held-out card, whose texts never trained. Three gates judge a checkpoint. Gate 1
compares the encoder with the `identity` baseline on texts the model never trained on. Gate 2 checks
that predictions move the way the rules say when a damage-step keyword is removed from a creature.
Gate 3 checks that the variance of the embeddings is not concentrated in a few directions.

Each baseline trains on the identical pipeline and differs from gen-1 in one input, so each answers
one question. `identity` gives every distinct ability text a free vector in place of the encoder's
output, and asks whether the encoder reads text at all. `taxonomy` replaces that vector with a hash
of the script's API type and parameter keys, and measures what reading a script's parameters adds
over knowing its effect category. `state-only` zeroes every ability vector and gives each record
kind its floor: a kind the full model scores at that floor is one where the text changed nothing
the model could find. `no-state` was meant to remove the board as well, so that the model could
learn only each text's average effect, and to be compared on the embeddings. Its run also zeroed
the acting ability's vector, so it answered nothing (section on baselines).

Each of the first ten sections records one gap found while gen-1 trained, the evidence for it, and
the change gen-2 makes. The three after them set how gen-2 is collected, evaluated and trained. Its
corpus is collected anew. It trains no baselines, and is judged instead with the knowledge probes
of [`2026-09-19-effect-knowledge-probes-design.md`](2026-09-19-effect-knowledge-probes-design.md),
called the probes record below. A probe is a small model fitted to read a known property from a
model's internal vectors. And gen-2 is a sweep: several models, the arms, each trained at a
different width of `e` or size of encoder. Gen-2's gate figures are not compared with gen-1's,
because the corpus, the split and the encoding all change. The run plan at the end orders all of
it.

## The encoder reads only the first script line of an ability, so a trigger's effect never reaches it

On the script surface the encoder is fed one Forge script line per ability: the parameters of the
runtime trait the sidecar keys the line by. An ability that Forge writes across several lines loses
every line but the first. For a triggered ability the first line is the trigger condition, and the
effect is a separate `SVar` line the trigger names through `Execute$`. Ajani's Pridemate encodes as
the trigger line alone:

```
Execute$ TrigPutCounter | Mode$ LifeGained | TriggerDescription$ Whenever you gain life, put a +1/+1 counter on CARDNAME. | TriggerZones$ Battlefield | ValidPlayer$ You
```

The `DB$ PutCounter | CounterType$ P1P1 | CounterNum$ 1` line that does the work is not in the
text. The model knows that this is a life-gain trigger which does something the description calls
"put a +1/+1 counter"; it does not see the counter type or the count as parameters.

The omission is near-universal for triggers and common for the rest. The counts
below are over every sidecar line in the converted card corpus that carries a script surface,
excluding keyword lines.

| lines with a script surface | 41,914 |
| triggered lines | 16,135 |
| triggered lines whose `Execute$` effect is absent from the encoded text | 16,098 |
| lines with a sub-ability chain below the first effect | 8,820 (21%) |
| chain length 1 / 2 / 3 / 4 or more sub-abilities | 5,595 / 2,247 / 698 / 280 |
| longest chain | 13 sub-abilities |
| median ratio of full-chain text to root text, chained lines only | 1.4× |
| 90th percentile of that ratio | 2.1× |

Modal abilities lose every mode. A charm's root line names its modes only by label, as in
`Choices$ DBDmgC,DBDmgP`, and where it has a description, the description ends in the placeholder
`ABILITY` that Forge fills from the modes at runtime. The encoder sees that the ability offers a
choice and not what any choice does.

| charm lines in the converted card corpus | 783, on 781 cards |
| charm lines whose encoded text names no mode | 742, on 740 cards |

The mechanism is in the converter. `TraitScript.of` renders `script_text` from the trait's own
parameter map, and a trait's map holds only the line the trait was parsed from. For a trigger that
is the `T:` line. The effect the trigger executes is a separate `SpellAbility` reachable through
`getOverridingAbility()`, and for any ability the sub-abilities are reachable through
`getSubAbility()`. The converter already walks both chains: the sub-abilities become the
`sub_ability_links` index paths the record attribution uses, and the prose renderer walks the same
chain to find the description. Only the script text stops at the root.

The design doc's argument for the script surface assumed the whole mechanism was present. It gives
`Cost$`, `ValidTgts$`, `Mode$` and `NumDmg$` as the predicate structure every record supervises,
and it counts on paraphrases of one mechanism collapsing to one script. Neither holds for the
effect half of a trigger, because that half is never encoded. Gen-1 still sees the effect through
the description parameter: Forge attaches the rules text to the root line, so the prose is present for
all but about one line in eighty. The encoder is reading prose plus the trigger condition or the
first effect's parameters, and the rest of the mechanism reaches it only as English.

The description is not a full substitute. The design doc's third argument for the script surface
is synthetic script variants, which perturb a parameter and play the variant for engine ground truth
of a text that never existed. A variant that changes `CounterNum$ 1` to `CounterNum$ 2` on a trigger
produces a record whose encoded text is identical to the original, because the changed line is not
in the text. The variant records then teach the model that one text has two outcomes.

### Gen-2 encodes the whole chain in resolution order

`script_text` becomes the concatenation of every script line the ability owns: the root line, then
for a trigger its executed ability, then each sub-ability in chain order, including a
`RepeatSubAbility` where one exists, and for a charm each mode its `Choices$` names, since a charm
reaches its modes through that list rather than through `Execute$` or `SubAbility$`. Segments are separated by a dedicated token and each segment
keeps the SVar label the parent referenced it by, so `SubAbility$ DBChange` in one segment and
`DBChange:` opening a later one give the model the link. Replacement effects get the same treatment
through `ReplaceWith$`.

Four things follow from the change, and the third is the one that decides the order of work.

1. **The converter rewrites the sidecars but not the converted text.** `script_text` is a sidecar
   field. The rendered prose does not change, so the byte-identity guarantee the sidecar contract
   makes holds.
2. **The script vocabulary is rebuilt.** `build-vocab --surface script` scans the sidecars' script
   lines; after the change those lines carry the chained segments and the separator token. Most
   parameter keys already appear on root lines, so the vocabulary grows by the segment labels and
   the rarer sub-ability parameters.
3. **The holdout is chosen on the chained text.** A text is held out by the hash of its
   normalised script text, and the corpus's rarity table is keyed on the same string. Changing
   `script_text` changes every key. The holdout is therefore computed on the chained text before
   the new corpus is collected, which is the only point at which changing it costs no training
   games (section on collecting from scratch).
4. **Sequence length grows but stays well under the cap.** The encoder truncates at 512 tokens. The median
   chained line grows by less than half and the 90th percentile roughly doubles; the longest chains
   run past a dozen segments. The distribution of chain lengths in tokens should be measured after the rebuild,
   and a chain that exceeds the cap should be logged rather than silently cut from the tail, because
   the tail of a chain is its last effect.

Prose stays as it is: the fallback for a line with no script, which is a keyword-derived line or a
synthetic land mana line, and nothing else.

### The paired-prose loss the design doc describes is not built, and gen-2 decides on it with the noise

The design doc has the script surface as primary and the converted prose as a paired secondary,
with an asymmetric loss pulling the two `e` vectors of a line together. The trainer has no pairing
term. The prose reaches the encoder only inside the root line's description parameter, or as the
fallback above. Whether to build the pairing loss is a separate gen-2 decision; it is recorded here
because the chain change above is what makes the script surface carry the mechanism the pairing
was meant to anchor prose to, and the two should be weighed together.

## Every outcome in the corpus is one Forge chose, so the model can learn abilities without learning targets

Every record comes from games in which both seats follow Forge's AI. The corpus is therefore
observational under one decision rule, one policy in reinforcement-learning terms. A resolution record exists because Forge decided to
take that action, and Forge takes an action only when it already predicts the outcome it wants. In
the records, "Lightning Bolt targets a creature" and "that creature dies" are therefore almost the
same event, and "Murder targets a creature" is never paired with an indestructible target. A model
reaches a good loss on that corpus by learning that Bolt means death. It never has to learn that the
target's toughness decides it, because no record shows Bolt on a creature that survived. Combat
records do show damage that failed to kill, and the shared per-entity head carries some of that
over, but the ability-conditioned half of the mapping is learned only on the support the policy
chose.

A corpus like this understates any measure of whether the encoder reads text. A lookup table learns
"Bolt means death" as well as the encoder does, so the encoder's margin over gen-1's identity
baseline is smaller than it would be on a corpus where the outcome depended on the target.

The fork probes already in the corpus, games copied at the damage step and replayed with one
keyword stripped from one creature, are the interventional answer for one case, the damage-step
keywords. Each further case needs its own fork machinery. Off-policy play, in which one seat takes
actions Forge's AI would not have chosen, is the cheap general answer. Forge's rules engine computes
every record's outcome, so an action Forge's AI would never take still produces a correct record
when it resolves. The records that appear are the ones the corpus lacks:

- Bolt on an 8/8, Murder on an indestructible creature, a player exiling their own creature. The
  ability resolves and the entity stays, or the wrong side loses something. These teach the effect head's
  affected-or-unaffected output and its zone-outcome field to read the target's state, which is what the factored design is for.
- Targeting through a ward cost the caster cannot pay. The cost record's outcome field already
  carries `countered`; the class is simply empty today.
- A 1/1 attacking into a 4/4, and blocks Forge would never assign. Combat records are the
  observational complement, and random declarations reach the region Forge's attack logic never
  enters.
- "Tap target creature" on a creature that is already tapped. A resolution that changes nothing,
  which the corpus has never recorded.

The combat case is measured for deathtouch. Forge declines the attacks and blocks in which
deathtouch would decide who dies, which is part of why gate 2 lacks deathtouch records; the tables
are in the design doc's section
[on why deathtouch and indestructible combats are scarce](2026-09-04-ability-effect-model-design.md#deathtouch-and-indestructible-combats-are-scarce-because-few-creatures-carry-the-keywords-and-forges-combat-ai-avoids-the-combats-deathtouch-decides).

### Gen-2 collects from games where one seat plays legal but random actions against a normal Forge seat

One seat stays the standard Forge AI, so the game keeps a shape Forge would produce and ends. Two
random seats never play each other. The other seat is Forge's controller with its choices
overridden at the decision points that matter: the spell or ability to play, its targets, the
attackers to declare, and the blocks to assign. At each of those points the seat acts at random
with one probability, the same at every decision point. When it does, it draws uniformly from the
legal options; otherwise it takes Forge's ranked choice. Land drops, mana payment and mulligans
stay with Forge, because a random land drop produces no record the corpus lacks and only makes the board less like one the model will be asked about.

The override lives in the connector, not in Forge. The match worker already builds both seats as
`LobbyPlayerAi` in `GamePlayer`, and Forge routes spell choice, target choice and combat
declarations through `PlayerControllerAi`, so the random seat is a controller subclass the worker
installs on one seat. Nothing on the `effect-record-hooks` branch changes.

Four things are settled here rather than at collection time.

1. **Records from the random seat carry a flag.** The flag lets the evaluation score the model
   separately on off-policy records, which is the number that says whether this section worked.
2. **Playability decision records are unaffected.** The `decision` subkind records the rules'
   verdict on every candidate, not the action the seat took, so a random seat produces the same
   decision records a Forge seat does. Nothing in that class needs the flag.
3. **One game in eight has a random seat.** The random seat produces rare outcomes for common
   texts, and on-policy games supply everything else, so the two share one corpus.
4. **Random-seat games are played in both collections.** They appear in the depleted training
   collection and in the full-strength card-disjoint collection, so the off-policy records of
   training have held-out counterparts to be scored on.

Two limits are accepted. Random targets on a wide board mostly produce uninformative no-ops, so the
share of useful records per game is lower than under Forge's play. And random attacks and blocks
cost the random seat life and creatures, so its games run shorter and its boards are thinner than
Forge's. The probability of acting at random sets how far both go. The pilot measures how many
unusual outcomes it yields per family, and the probability is raised only if the corpus falls short
of the outcomes curation balances on.

## The rarity weight is flat for 97% of texts, so a text seen once trains no harder than one seen in two hundred games

The rarity weight the trainer applies within a class does nothing for the tail of the curated corpus.
A record is weighted by the inverse square root of the games its text was seen in, capped at twenty
times the weight of the most-observed text. The most-observed texts are the five basic-land mana
abilities, each seen in about ninety thousand games. Measured against that reference, the ceiling
binds for every text seen in fewer than about 228 games, which is 97% of the rarity table. A text
recorded in one game and a text recorded in two hundred receive the same weight. Only the mana
abilities, the enters-tapped replacement and the eight hundred or so texts above that line weigh
less. The figures below are from the manifest of the corpus gen-1 trained on.

| texts in the rarity table | 31,561 |
| most-observed text (`Add {B}`), games | 91,133 |
| 99th / 95th / 50th percentile, games | 447 / 153 / 11 |
| texts seen in one game | 2,639 (8%) |
| texts seen in four games or fewer | 8,681 (28%) |
| texts at the weight ceiling | 30,740 (97%) |

Under the flat weight, a rare text reaches the encoder about a hundred times less often than a
common one. A curated shard holds 2,000 records. An epoch, one pass of the training schedule, visits
256 shards in 5,000 training steps of 32 records each, so it consumes about a third of each shard it
visits. The weighted shuffle decides which third, heaviest first. With a flat table that draw is
uniform, so a text's share of an epoch is its share of the records that survived the per-text cap.
The cap is 200 records, and a text seen in one game contributes one to three. The exponent meant to
take the square root of that ratio has no effect.

The per-text cap is the only setting that changes the mix of texts in the curated corpus, and it
acts only on the most common texts. The cap drops about three of every five resolution-effect
records read and about half or more in every other ability-keyed class. A cap is a ceiling and
never a floor, so curation cannot add tail records. What it can do is stop the common texts from
crowding them out, and today it does that only down to two hundred records per text.

| class | records read | kept | dropped by the cap |
|---|---|---|---|
| resolution-effect | 2,226,674 | 233,317 | 1,311,253 |
| resolution-cost | 1,832,641 | 61,719 | 943,357 |
| trigger | 1,864,406 | 62,447 | 936,613 |
| continuous | 1,354,894 | 92,007 | 875,225 |
| rewrite | 187,434 | 54,130 | 101,220 |

### Gen-2 sets the weight ceiling against the 99th-percentile text and logs what the tail receives

The ceiling is expressed against the text at the 99th percentile of games rather than the single
most-observed one. With the reference at about 450 games, a one-game text weighs about twenty times
the reference and the cap barely binds, which is what the cap was written to do: stop a single text
from dominating a batch without flattening everything below it. The mana abilities and the few
hundred texts above the reference weigh less than it, which is where they belong. The rarity table
in the manifest does not change; only the trainer's reading of it does. The weight balances the
texts inside one rule family; the balance between families is set when the corpus is built, as the
next two subsections describe.

Nothing reports today whether training reaches the tail. The manifest's unique texts per output
show only whether curation kept it. The trainer's epoch line therefore gains the share of records
trained by rarity bucket of their text: one game, two to four, five to nineteen, twenty or more.

### A per-text weight shows a mechanism once per variant, so a keyword with few variants stays rare

Equalising texts does not equalise mechanisms. Every "deal N damage to target X" script is its own
text, so the damage mechanism reaches the encoder once for each of its variants, and a keyword
written on a handful of cards reaches it a handful of times however heavily each of its texts is
weighted. The weight fix above removes the bias between texts and leaves this one untouched. The
table groups the rarity table's texts by their Forge API type, trigger mode or keyword.

| mechanism family | distinct texts | games |
|---|---|---|
| ChangesZone triggers | 5,393 | 312,514 |
| Pump | 1,989 | 73,949 |
| DealDamage | 1,224 | 66,343 |
| Flying | 608 | 20,153 |
| Suspend | 52 | 3,095 |
| Ninjutsu | 22 | 738 |
| Cascade | 11 | 1,052 |
| Dredge | 7 | 602 |
| Cypher | 0 | 0 |

Under equal per-text weights the damage family gets well over a hundred times the attention of
dredge.

### Gen-2 balances the curated corpus across rule families, as close to uniform as the records allow

The embedding should carry a comparable understanding of every rule of the game, not a picture of
how often each rule comes up in Forge's games. The ideal is the same number of examples for every
rule. No mapping from records to the numbered rules of the comprehensive rules exists, but Forge
implements roughly one class per rule mechanism. Each record kind therefore takes its family from
the mechanism vocabulary Forge already uses, and a keyword line's family is the keyword whatever
the record kind.

| Record kind | Family |
|---|---|
| resolution, effect and cost halves | the acting ability's API type |
| trigger | the trigger mode |
| continuous | the static mode |
| rewrite | the replacement type |
| combat | the damage-step keywords present in the fight, or none |
| playability | the restriction or rejection reason the verdict turns on |

`build-corpus` gives every family within a class the same target number of records. A family with
more records than the target is cut down to it. A family with fewer is written whole, and each of
its records is repeated up to a reuse cap. Each family therefore receives the target or its records
times the reuse cap, whichever is smaller.

The reuse cap is what stops uniformity from turning into memorization. Dredge has seven texts in
about six hundred games. A budget equal to that of the ChangesZone triggers would replay those few
cards and boards hundreds of times, and the model would learn the cards rather than the rule.

A family short of its target is a collection problem rather than a sampling one. Cypher has no
record at all, and no weight reaches it. The manifest reports each family's available records, its
target, the records written, the repeats and the shortfall. The shortfall list is the target list
for `collect-coverage` and `collect-variants`.

Inside a family, the text weight and the per-text cap of the previous subsections balance the texts.
The epoch line reports the share of records trained per family beside the share per rarity bucket.
The validation results are reported per family and averaged over families, so the evaluation
weighs the rules the way training does.

### Within a family, curation balances the outcomes as well as the texts

A family's records are balanced over a coarse outcome signature: for each affected entity, its zone
outcome and whether anything about it changed. Lightning Bolt killing its target and Lightning Bolt
leaving it standing then appear in comparable numbers. On-policy play records almost only the first.

The balance does not distort what the head learns. Most outcomes follow from the board under the
rules: Bolt on a creature with three toughness always kills it. Balancing the outcomes changes which
boards the model sees, and leaves the right answer on every board unchanged. It shifts the head only
where the outcome depends on a player's choice, such as which creature an opponent sacrifices, and
that choice is Forge's policy rather than a rule.

The minority outcomes come mostly from the random seat, so this balance is what turns the random
seat's games into training signal. The random-seat flag is not itself a curation key: the signature
selects the unusual outcomes directly, from whichever seat produced them. The manifest records the
on-policy and off-policy record counts per class beside the existing per-class counts.

## The training noise on the ability vector is under 1% of its length, so the small-noisy-`e` lever does nothing

Gen-1 implements only the noise half of the design doc's first memorization lever, and its noise
does not scale with `e`. Memorization here means predicting from which text an ability is, rather
than from what the text says. The lever keeps `e` small and noisy in training, so that the effect
head cannot use `e` as a precise per-text lookup key ("Coverage and memorization are separate tail
problems, and both get levers" in the design doc). The encoder adds Gaussian noise with a fixed
standard deviation of 0.05 to each of the 64 coordinates (`e_noise` in `ability_encoder.py`).
Nothing keeps `e` small. `e` is a linear projection of the encoder's output, restrained only by the
optimizer's weight decay, a pull of every weight toward zero by a factor of 0.01 of the learning
rate per step. A larger `e` makes the same absolute noise relatively smaller, and the effect head
reads `e` through a linear layer, which absorbs any scale.

In the trained gen-1 encoder the vectors are large enough that the noise no longer matters. The
figures are over every row of the shipping cache under `output/effects/abilities/cardsfolder/`, with the taxonomy baseline's files
left out. The noise moves a vector by less than a hundredth of its length. Along the ninth principal
direction, the weakest the encoder uses, the spread of the vectors is almost ninety times the noise.

| shipping ability cache | |
|---|---|
| vectors | 66,142 |
| length, mean | 53.4 |
| length, 10th / 50th / 90th percentile | 40.5 / 54.3 / 64.4 |
| length, shortest | 23.6 |
| standard deviation of one coordinate, over all vectors | 6.8 |
| standard deviation along the ninth principal direction (design doc Outcome) | 4.4 |
| noise, standard deviation per coordinate | 0.05 |
| noise, expected length in 64 dimensions | 0.40 |

The effect head can therefore tell texts apart at a precision far finer than the typical distance
between two of them, which is the lookup the lever was meant to make expensive. In gen-1 only the design doc's other four
levers acted against memorization.

The pairing-loss argument in the design doc rests on the same noise. A symmetric pull between a
line's script and prose vectors adds pressure to collapse `e`, and the doc expects the noisy-`e`
lever to amplify that pressure. With negligible noise that interaction is absent today. It appears
once the noise is fixed, so the pairing decision left open in the chain section has to be made against the
fixed noise.

### Gen-2 shapes the noise like the cloud of `e` vectors, measured over a running average

Five fixes are open, and they differ in whether the encoder can still escape the noise.

- **Normalizing each `e` to a fixed length before the noise.** The fixed noise then has a scale it
  cannot escape. It also discards length, which ranges over a factor of nearly three in the shipping
  cache and may carry information. It changes the vectors the cache stores, so every consumer reads
  a different `e`.
- **A penalty on the length of `e` in the loss.** It adds a second weight, and the balance between
  that weight and the noise is what sets the noise's effective size.
- **One noise level for every direction, proportional to the average spread of `e`.** Rescaling
  `e` as a whole then changes nothing. The encoder can still widen a few directions, though. The
  noise is sized to the average, so along those directions it is small against the spread, and
  they can carry a precise per-text key. The same reward for piling variance into a few directions
  is what gate 3's concentration check flags.
- **Noise in each coordinate proportional to that coordinate's spread.** Widening one coordinate
  no longer helps. The coordinates are arbitrary axes, however. A signal laid along a diagonal
  across many coordinates meets noise up to √d times smaller relative to its spread, where d is the
  width of `e`.
- **Noise with the covariance of the `e` vectors.** The noise is wide along the directions in which
  the vectors spread widely and narrow along the others, diagonals included. The ratio of noise to
  spread is then the same along every direction, and no reshaping of `e` changes it.

Gen-2 takes the last. Five details settle it.

1. **The covariance is measured around the mean, with the gradient stopped.** A common offset added
   to every vector separates no two texts, so it must not raise the noise. With the gradient
   stopped, the covariance is treated as a constant when the loss is differentiated, so the noise
   cannot become a path through which the loss changes `e`.
2. **The covariance is a running average over recent batches.** One batch spans at most one
   direction fewer than it has distinct texts. A batch with fewer distinct texts than `e` has
   dimensions would leave some directions without noise, and those directions could carry a key.
   One batch's estimate also varies with the batch size, which differs between sweep arms that fit
   the 8 GB card differently. The average starts from the first batch's covariance.
3. **The ratio ramps up over the first epoch.** It rises linearly from zero to its value, the same
   way in every arm. Early in training `e` grows quickly and the running average lags behind it,
   which makes the noise smaller than intended rather than larger. The ramp makes the start of
   training independent of that lag.
4. **The noise moves from the encoder to the batcher.** It is applied to the vectors the batch
   carries, which is where the running average is kept.
5. **The ratio comes from a pilot on the gen-1 corpus.** Gen-1's effective ratio per coordinate was
   under a hundredth. The pilot trains at 0.05, 0.1 and 0.2 with the auxiliary head of the next
   section switched on. It reads each run's results on unseen texts and the concentration of its
   `e`. Every sweep arm then uses the chosen ratio.

The cache and its consumers do not change. The noise is applied in training only, and the encoded
`e` keeps its form and its unconstrained scale.

Gate 3 is where the change should show. Gate 3 caps the share of the embedding's variance on its top
principal component, and that share is a ratio of variances, so a pure rescale of `e` leaves it
exactly where it is. Noise shaped like the cloud changes what the encoder is rewarded for. Every
direction the encoder uses has the same ratio of spread to noise, so each further direction adds
the same amount of separation between texts. Widening one direction adds none. The encoder
therefore separates the most texts by spreading them over as many directions as it can. The change
is expected to lower the top component's share, and gate 3 reports whether it does.

## The ability vector keeps too little of an ability's amounts and costs, so gen-2 trains a head that reads them from `e`

Gen-1's `e` holds less about amounts and costs than a hash that records only which parameters a
script sets. The figures are linear probes from the design doc's Outcome. The taxonomy hash
holds every parameter key's presence and no value.

| Property | Metric | gen-1 `e` | taxonomy hash |
|---|---|---:|---:|
| damage amount | R² | 0.34 | 0.55 |
| mana in the activation cost | R² | 0.06 | 0.33 |

Lightning Bolt's nearest neighbours show the same loss. They are the same sentence with 1, 2, 4, 5,
6, 7, 10 and X damage, all at a cosine similarity of at least 0.997.

The loss rarely depends on the number. Forge casts Bolt mostly at creatures it kills, so on the
records the corpus holds, three damage and four have the same outcome. Costs are supervised only
through the cost halves of resolutions and the playability verdicts, which few fields score. The
outcome balancing above makes the amount matter more often, but nothing in the loss asks `e` for a
cost at all.

### Gen-2 adds a training-only head from `e` to the values the script states

The head reads `e` alone and predicts numbers parsed from the script:

- damage dealt, power and toughness change, counters placed and cards drawn;
- the mana cost by colour, plus its generic part;
- whether the cost taps or sacrifices.

The labels come from the sidecar's script parameters, for every segment of the chain, at no cost.
A label is masked where the script states no fixed number, as with X. The head follows the
script-API head's pattern: a small loss weight, used in training only, and filtered out at save
time.

The weight is the one setting to watch. Gen-1's `e` predicted observed outcomes better than the
parsed script features did, because it groups abilities by what they do rather than by how their
scripts are written. A large weight would pull `e` back toward the parser's grouping.

The game agent reads `e` through the cache, and the amounts and costs are what it needs to weigh one
card against another. The knowledge probes that read amounts and costs from `e` become a check that
the head did its job. The probes that ask whether the model uses an amount against a toughness stay
independent of it.

## Gate 1 scores fewer than half of the held-out texts, and a third of those have a numeric twin in training

Held-out texts are chosen by hash over the converted texts, not over the recorded ones, and nearly
half of them were never recorded. Sealed self-play never cast the cards that carry them, and
`collect-coverage` keeps held-out cards out of its decks by design, so nothing fills the gap. Of
the texts that were recorded, a third were seen in four games or fewer. The eligibility rule of at
most eight carriers excludes almost nothing, because nine texts in ten are on exactly one card.

| held-out texts | 735 |
| with any record | 393 |
| with a gate-one resolution record | 342 |
| recorded, seen in one game | 43 |
| recorded, seen in two to four games | 82 |
| recorded, seen in five to nineteen games | 142 |
| recorded, seen in twenty games or more | 126 |
| converted texts carried by one card | 33,456 of 36,981 |

The validation sample that selects checkpoints is dominated by the well-recorded held-out texts.
The sample is drawn by smallest record hash, which is uniform over records, and the gate-one slice
holds about eighteen thousand records over its 342 texts. The resolution slots of the 2,048-record
sample therefore go to the texts with the most records, and a text seen in one game holds a few at
most. Gate 1 itself averages over entities and records, not over texts, so the same texts decide its three
margins. The number gate 1 reports is a well-recorded-holdout number. How the model reads a text it
was shown in one game is measured nowhere.

A third of the recorded held-out texts have a near-twin in training. A twin is a training text that
is identical once numbers, `CARDNAME` and the description parameters are masked, such as
`NumDmg$ 2` held out beside `NumDmg$ 3` in training. This masked form is the text's template.
Corpus-wide, two recorded texts in five share a template with another. The identity baseline cannot
exploit a twin, because it keys on the exact text. The encoder can, and reading `NumDmg$` is what it
should do. A gate averaged over held-out texts of which a third are one number away from a training
text measures interpolation between siblings more than reading. The design doc's four-way
stratified report does not isolate these twins: its numeric-extrapolation and novel-combination
strata are different cases.

| recorded held-out texts | 393 |
| with a training text equal up to numbers, names and descriptions | 137 |
| recorded texts corpus-wide | 31,561 |
| sharing such a template with another recorded text | 13,320 |

### Gen-2 holds out whole templates and reports unseen texts per text and per rule family

Four changes. The first two change how the card-disjoint sample is drawn and how its results are
reported. The third changes the holdout key, which acts at collection time. The fourth is a
coverage round that gives the held-out texts enough records to score.

1. **The card-disjoint sample is drawn per held-out text.** Its resolution slots fill round-robin
   over the held-out texts, a fixed number of records per text by smallest hash within the text,
   until the class quota is met. Every recorded held-out text is then in the sample, and the loss
   that selects checkpoints weighs a one-game text comparably to a hundred-game one. The
   card-disjoint stratum already caps games per held-out text for the same reason; this is the
   same rule one level down.
2. **Results on unseen texts are reported per text and by bucket.** A per-text mean sits beside
   the per-record numbers. The buckets are the rarity bucket of the held-out text's games and the
   rule family of the text, and the family results are also averaged over families. The family
   view is what shows a keyword the model reads badly, which a mean over the damage spells would
   hide. The memorization measure is the gap between a family's results on the game-disjoint
   stratum, whose texts trained, and on the card-disjoint stratum, whose texts did not. Gate 1
   itself is not computed, because gen-2 trains no identity baseline (section on baselines).
3. **The holdout key is the masked template.** A text is held out when the hash of its template
   falls under the permille, so a text's numeric siblings go with it and every held-out text is
   unseen in training, numbers included. The template is taken from the chained script text, and
   the new corpus is depleted against it (section on collecting from scratch). The eligibility cap counts the cards
   that carry any text of the template, not the cards of one text: a template such as
   `Pump +N/+N until end of turn` spans thousands of cards, and holding it out would deplete them
   all. The permille is set against the share of cards the holdout depletes, which `holdout-cards`
   reports before collection starts. The build lists the held-out texts with no gate-one record and
   those under a floor of five games, which is the target list for the next point.
4. **A coverage round over the held-out cards alone.** `collect-coverage` gains the inverse of
   `--exclude-cards`: a run restricted to the held-out cards, writing full-strength records. Every
   game it produces names a held-out card, so the split rule routes it to the card-disjoint stratum
   on its own. The round runs until every held-out text with a castable carrier reaches the floor,
   and the residue it reports is the list of held-out texts no result can ever cover.

## Keyword expansion builds the cache from definitions gates 1 and 2 never score, and skips every keyword whose name is not one word

Keyword expansion replaces a keyword token with its definition, the reminder text Forge prints for
the keyword ("New keywords are handled by expansion dropout over their definitions" in the design
doc). It has four defects. The cache and the gates read different encodings of every keyword line.
Multi-word and hyphenated keywords never find their definition. The definition inserted is a
damaged template. And `build-vocab` scans Forge's generated keyword scripts into the script
vocabulary, though no expansion reads them. The evidence comes from
`scripts/effect_embedding_probes/keyword_expansion.py`, with its outputs in
`output/effects/reports/keyword-expansion-20260919/`. What the analysis found about the encoder
itself is in the design doc's Outcome, in "A keyword token encodes only loosely like its own
definition" and "An unknown keyword expands only when its name is one word".

The shipping cache encodes every known keyword as its definition, while the effect head trained on
the keyword token three times in four and gates 1 and 2 score the token every time. The expansion
probability is the chance that a known keyword is replaced, and each caller sets its own. Training
uses 0.25. Validation and gates 1 and 2 go through `SurfaceBatcher` with 0. `encode-abilities`,
which builds the cache, uses 1.0, so that two runs of the command write the same file. An unknown
keyword is expanded at every probability. Gate 3, the ward canary (the distance between ward and
the card lines that spell out its effect), the embedding probes and the
deployed head all read the cache. For a keyword line, each of them therefore reads an input the head
saw only a quarter of the time in training, and never the input gates 1 and 2 scored.

Expansion also fires on a keyword word inside other lines, such as a grant's description, so the
cache moves more than the keyword lines. Keyword lines move furthest. Cosine similarity below is
between a text's unexpanded and fully expanded encodings, 1 for the same direction.

| unique script texts in the corpus | 37,106 |
|---|---:|
| changed by full expansion | 7,658 (20.6%) |
| of which keyword lines | 1,925 |
| of which other lines naming a keyword | 5,733 |
| cosine over changed texts, median / 10th percentile / minimum | 0.968 / 0.748 / −0.662 |
| cosine over changed keyword lines, median / 10th percentile | 0.893 / 0.371 |
| cosine over changed other lines, median / 10th percentile | 0.979 / 0.868 |
| share of a text's 10 nearest neighbours that the other encoding keeps, 500 changed texts | 0.33 |

The cache is the fully expanded one. Forty keyword rows of the shipping cache match the full
expansion at a mean cosine of 1.0000 and the unexpanded encoding at 0.77.

Expansion finds a definition through the token the tokenizer produced, so a keyword whose name is
not a single word expands only if the vocabulary already knows it. A multi-word name becomes one
token, `first_strike`, only by matching a merged vocabulary entry, which exists only for known
keywords. A hyphenated name never expands: the tokenizer cuts `Jump-start` at the hyphen and the
definition is filed under `jump-start`. Jump-start, web-slinging and beam me up never expand, on 24
lines. Starting intensity, on 20 more, has no reminder text to expand to. The same mismatch between
a display name and its token disables `HOST_BODIED_KEYWORDS`, the list of keywords whose body is
printed on the host card and which are never to expand. It spells `level up` and `read ahead` with a
space, the tokens are `level_up` and `read_ahead`, and both expand, on 26 and 10 lines.

The definition inserted is damaged in four ways.

- `_instantiate` strips only `%s`. The 38 templates written with `%d` or `%1$s` keep the specifier
  as tokens: toxic's reads "also get % d poison counters".
- No caller passes the line's own values. `expand_keywords` accepts `instance_values` and nothing
  supplies them, so a template loses its value and the line's parameter is left dangling after the
  sentence.
- 16 definitions contain `[UNK]`. `build-vocab` scans the definitions but keeps only the 5,000
  most frequent tokens, and 21 words of reminder text fall below that cut:
  `encoded`, `promise`, `specified` and `teammate` among them.
- Five templates (enlist, increment, read ahead, station, web-slinging) use the typographic
  apostrophe `’`, which becomes a token of its own rather than the `'` every other text uses.

| keyword line | what the encoder reads under expansion |
|---|---|
| `Toxic:1` | players dealt combat damage by this creature also get % d poison counters . : 1 |
| `Ward:2` | whenever this permanent becomes the target of a spell or ability an opponent controls , counter it unless that player . : 2 |
| `Enchant:Creature` | target a % 1 $ s as you cast this . this card enters attached to that % 1 $ s . : creature |
| `Start your engines` | if you have no speed , it [UNK] at 1 . it [UNK] once on each of your turns when an opponent loses life . max speed is 4 . |

Both surfaces expand to the reminder template: `_definition_text` returns it, and no code
substitutes the generated script. `generated_script` is still loaded and scanned into the script
vocabulary, and nothing else reads it.

### Gen-2 builds the cache with known keywords left as tokens

`encode-abilities` expands at probability 0, the setting validation and gates 1 and 2 use. The
choice keeps what 1.0 was chosen for: an unknown keyword still expands at every probability, so the
cache stays deterministic and a new set still reads through its definitions. It also makes the
cache the input the head saw three times in four in training and the one gates 1 and 2 score.
Expanding at 1.0 in training as well would make the two agree the other way, but it removes the
keyword token the design keeps, and the design doc's Outcome shows the encoder has learned that token
as a symbol of its own rather than as its definition.

1. **One setting, read by both callers.** The probability for scoring and for the cache is one
   constant that `SurfaceBatcher`'s scoring path and `AbilityEncoderRunner` both read, just as
   `SurfaceBatcher.text_of` is the one definition of the encoding text. Each caller setting its
   own value is how the cache and the gates came to read different inputs.
2. **The geometry is measured again.** Gate 3, the ward canary and the embedding probes are re-run
   on the new cache. Their keyword lines then describe the keyword token, which is what the head
   reads.

### Gen-2 recognises a keyword line by its display name, not by a vocabulary token

The definition is looked up by the display name, the part of a keyword line's script text before
its first colon, matched case-insensitively against the definitions table. The whole name is
replaced, however many tokens it split into, and the text after the colon becomes the line's
instance values. A keyword word inside another line keeps the token path it has now.

1. **The match is checked against the sidecar.** The batcher sees only the encoding text, so the
   match is made on the text. The corpus build asserts that every text it matches belongs to sidecar
   lines whose `script_api_type` is `Keyword`. The sidecar's `line_kind` is not the key: it is the
   converter's category, `alternate cost` for jump-start.
2. **The host-bodied list is checked on the same path.** It compares display names, so level up and
   read ahead are excluded as the list intends.
3. **The corpus's never-expanding keywords are the test.** Jump-start, web-slinging and beam me up
   expand, each with the vocabulary token deleted and restored.

### Gen-2 fills each template with its line's values, strips every format specifier and reserves the definition words

1. **Instance values are filled.** The values after the colon fill the placeholders in order, and
   `%1$s` repeats its value at each use. Forge formats some values before filling them: ward's `%s`
   stands for a clause such as "pays {2}", not the bare `2`. Those formatted strings come from Forge,
   recorded per keyword by `extract-keyword-definitions`, rather than being rebuilt in Python.
2. **Every specifier is stripped where no value fills it.** The pattern covers `%s`, `%d` and the
   positional forms, and no expansion leaves a `%` token.
3. **The apostrophe is normalised.** `’` becomes `'` when the definitions are loaded.
4. **The definition words are reserved in the vocabulary.** `build-vocab` seeds every word of every
   template after stripping, as it already seeds `[PAD]` and `[CLS]`, so `--target-size` cannot cut
   them. The reservation lands with the script-vocabulary rebuild the chain encoding and the
   camel-case split already require, in the chain and tokenizer sections, so it costs no rebuild of
   its own.
   The build asserts that no definition expands to `[UNK]`.

### Gen-2 keeps the reminder template on both surfaces and stops scanning generated scripts

The generated script is not yet worth expanding to. It is the root line of the keyword's trait
alone. Prowess's is its `SpellCast` trigger condition without the +1/+1 it grants, and no trigger
among the 75 scripts carries the effect it executes: the same first-line gap the chain section
describes. Its description parameter holds the reminder text anyway. And only 75 of the 202
keywords have one, so expanding to it would read three keywords in five in one language and the rest
in another.

1. **The template stays the definition on both surfaces.** The generated script is revisited once
   `extract-keyword-definitions` renders the whole chain, as the converter will for card lines.
2. **`build-vocab` stops scanning generated scripts**, so the capped vocabulary counts only text the
   encoder reads.

### Gen-2 withholds one keyword, so the zero-shot check has something to measure

The zero-shot keyword check is the only test that a keyword the model never trained on is read
through its definition, which is how a new set's keyword reaches the model. Gen-1 withheld no
keyword, so the check reported nothing. The design doc's Outcome adds a reason to run it: an
unknown keyword's expansion lands far from the vector of the trained keyword it defines.

1. **Every sweep arm withholds the same keyword.** `train-effect-model --withhold-keyword` keeps
   its token out of training and expands every occurrence to its definition. The keyword is
   chosen before the sweep, among the script-generated keywords new sets introduce, with enough
   records in the corpus for its fields to be scored.
2. **The check measures.** The evaluator today names the withheld keyword and stops. Gen-2 adds
   the measurement the design doc specifies: the effect head's per-field results on records
   where the withheld keyword acts or sits on the board, beside the same fields for keywords that
   trained.

## The script tokenizer is never called, and the prose grammar that runs in its place splits selectors better

`AbilityTokenizer.tokenize_script` is the tokenizer written for the script surface, and no code calls
it. Encoding for the cache (`ability_encoder_runner.py`) and encoding during training
(`surface_batching.py`) both pass the script text to `tokenize`, the prose grammar. The vocabulary
is built with the prose grammar too. `build-vocab --surface script` hands the script lines to the
shared price-predictor vocabulary builder, which splits them with `MtgTokenizer`. The design doc
counts a script tokenizer that splits compound selectors such as `Creature.nonDragon+OppCtrl` among
the costs of the script surface, in "The script is the primary surface; prose is the paired
secondary" of [`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md).

The prose grammar already makes that split. It ends a word at every character that is not a letter or
an underscore, so `.` and `+` end a word and are kept as tokens of their own. `tokenize_script` makes the same cut
at `.` and `+` but drops the separators, and before cutting it keeps letters, digits and `-`
together in one run. A sign then either vanishes, so `NumAtt$ +1` reads as `1`, or fuses with its
number into `-1`. A threshold fuses with its digit, so `powerGE4` becomes one token and the 4 never
reaches the numeric embedding, the encoder's separate input path that reads a number's value. The vocabulary was built with the prose grammar, so every fused token
is unknown to it. Unknown tokens (`[UNK]`) are the placeholder the encoder reads for any word
missing from the vocabulary.

| script text | `tokenize` (runs) | `tokenize_script` (never called) |
|---|---|---|
| `ValidTgts$ Creature.nonDragon+OppCtrl` | validtgts $ creature . nondragon + oppctrl | validtgts $ creature nondragon oppctrl |
| `NumAtt$ +1` | numatt $ + 1 | numatt $ 1 |
| `NumDef$ -1` | numdef $ - 1 | numdef $ -1→[UNK] |
| `CounterType$ P1P1` | countertype $ p 1 p 1 | countertype $ p1p1→[UNK] |
| `ValidTgts$ Creature.powerGE4` | validtgts $ creature . powerge 4 | validtgts $ creature powerge4→[UNK] |

Measured against the shipping vocabulary `models/effects/vocab-script.txt`, the unused tokenizer
produces more than twice the unknown tokens of the one that runs. The sample is 5,000 script lines
from randomly drawn cards. "Parameters only" is the same lines with the text of every
`*Description$` parameter removed, which leaves the part of a line the script tokenizer was written
for.

| on 5,000 script lines | `tokenize` | `tokenize_script` |
|---|---|---|
| unknown-token rate, whole line | 0.34% | 0.85% |
| unknown-token rate, parameters only | 0.47% | 1.12% |
| lines with at least one unknown token | 413 (8%) | 824 (16%) |
| tokens per line, mean (median), whole line | 32.5 (34) | 30.4 (31) |
| tokens per line, mean (median), parameters only | 18.4 (20) | 17.6 (19) |

Neither grammar splits compound words, and compound words are where the unknown tokens come from.
Forge writes restrictions and SVar labels in camel case: `nonDragon`, `YouCtrl`,
`TrigDestroyYourLand`. Both grammars cut only at characters that are not letters, so each compound
is one token. `YouCtrl` and `OppCtrl` share nothing, and `nonDragon` shares nothing with the
`dragon` token the prose surface knows. In the parameters-only sample, camel-case compounds are
about one token in six and three quarters of the unknown tokens. For nearly two thirds of those
unknown compounds, every part is already in the vocabulary.

| parameters only, `tokenize`, 5,000 script lines | |
|---|---|
| tokens | 92,191 |
| tokens from a camel-case compound | 15,612 |
| distinct compound tokens | 1,174 |
| unknown tokens | 433 |
| unknown tokens that are compounds | 327 |
| unknown compounds whose parts are all in the vocabulary | 205 |

### Gen-2 deletes the script tokenizer and splits camel-case compounds on the script surface

`tokenize_script` is removed rather than wired in, because the grammar that runs keeps the signs,
separators and digits it loses. The prose grammar gains one rule on the script surface: a word is
also cut where a lowercase letter is followed by an uppercase one, so `nonDragon` reads as
`non dragon` and `YouCtrl` as `you ctrl`.

1. **The vocabulary scan and the tokenizer apply the same rule.** `build-vocab --surface script`
   already stages the script lines as a text file for the shared builder, so it applies the split to
   that staged text. The price predictor's `MtgTokenizer` does not change. `tokenize` applies the
   rule when the loaded vocabulary is a script vocabulary, which `surface_of` already determines
   from the vocabulary path.
2. **The rule lands with the chain encoding's vocabulary rebuild.** The chain change rebuilds the
   script vocabulary anyway. The holdout hash reads the normalised script text rather than its
   tokens, so this rule does not move the held-out set.
3. **The measurement above is repeated on the rebuilt vocabulary.** The build log reports the
   unknown-token rate on the parameters and the number of distinct compound parts, so the rebuild
   shows whether the unknown compounds went away.
4. **SVar labels stay whole.** The chain encoding links a segment to its parent by SVar label
   (chain section). Split, `DBChange` becomes `db change`, which recurs across the segments of one
   card and no longer names one of them. The split therefore skips the values of `Execute$`, `SubAbility$`,
   `Choices$` and `ReplaceWith$`, and the label that opens each segment.
5. **A counter type is one token.** The prose grammar reads `CounterType$ P1P1` as `p 1 p 1`, so the
   digits of a counter type reach the numeric embedding as if they were amounts. On the script
   surface a counter type's value stays whole, `p1p1` or `m1m1`, in the tokenizer and in the
   vocabulary scan alike.

## Two build settings leave gate 2 and the trainer short of rare-keyword combats

Gate 2 cannot score six of its eight keywords for want of records, and two `build-corpus` settings
decide how many records it and the trainer get. The game-disjoint stratum that gate 2 reads is
1,000 games drawn uniformly. Curation keeps a combat record for training at one rate whatever the
fight, so a deathtouch combat is kept no more often than a fight between two vanilla creatures.
The design doc's section
[on why deathtouch and indestructible combats are scarce](2026-09-04-ability-effect-model-design.md#deathtouch-and-indestructible-combats-are-scarce-because-few-creatures-carry-the-keywords-and-forges-combat-ai-avoids-the-combats-deathtouch-decides)
gives two causes: few creatures carry the keywords, and Forge avoids the combats deathtouch decides.
Both settings act on the first cause only. The random seat acts on the second.

### Gen-2 draws the game-disjoint stratum toward games with rare-keyword combats

A uniform draw of 1,000 games holds too few of the rare keywords' combats for gate 2. The counts
below are gate 2's qualifying records on gen-1's stratum, against a floor of 200.

| Keyword | Qualifying records, 1,000 uniform games |
|---|---:|
| first strike | 47 |
| deathtouch | 69 |
| trample | 179 |
| indestructible | 19 |
| wither | 11 |
| infect | 73 |

Drawing the stratum toward the games that hold these combats gives gate 2 enough records without
enlarging the stratum. A game enters the stratum when a hash of its `game_id` falls under a
threshold, and the threshold is higher for a game that holds a qualifying combat for a keyword
short of the floor. The cost falls on training. Wither qualifies about once per hundred games in
the stratum above and about twice per hundred over both of gen-1's validation strata. A corpus of
gen-1's size therefore holds between one and three thousand wither combats, and reaching 200 takes
between a tenth and a sixth of them out of training. The other keywords lose less.

The rule depends only on the game, so a rebuild over a grown corpus keeps every game it already
placed in the stratum and only adds new ones. The seeded sample `build-corpus` draws today redraws
the whole stratum whenever the corpus grows. That would move the games the knowledge probes read
into training, and the probes would score later models on boards they trained on.

The stratum is not a typical sample of games, and it is not meant to be one. Its results are
reported per rule family and averaged over families, the balance training is built to. The
manifest records which games entered for a keyword, so a figure over the stratum can also be read
without them.

### Rule-family balancing gives the rare-keyword combats their own budget

Gen-1's curation cannot favour a deathtouch combat over an ordinary one. It keeps 158,348 of the
1,066,917 combat records it reads, 15%, and chooses them by a hash of the record id, so the choice
is uniform over fights. The per-text cap and the rarity weight never see a combat record, because
a combat record carries no acting ability text.

On the 1,000 game-disjoint games, a damage-step keyword appears in about a quarter of combat
records, and the five rarest in about one in twelve.

| Combat records, 1,000 game-disjoint games | Records | Share |
|---|---:|---:|
| all | 8,762 | 100% |
| with any of the eight damage-step keywords | 2,051 | 23.4% |
| with deathtouch, double strike, indestructible, infect or wither | 740 | 8.4% |

Gen-2 balances combat records by the family table of the rarity section. A combat's family is the
set of damage-step keywords present in the fight, so a wither fight and a fight between two vanilla
creatures receive the same target. The manifest's per-family counts show how far each keyword's
combats fall short of it.

## Most `blockers` records are the AI's what-if queries, and nothing in a record says which kind it is

The collector writes a legality record every time Forge's AI computes a legal attacker or blocker
set, not only when a player declares. Forge computes those sets while planning in its main phases, at
upkeep, and while the attacking AI simulates how the defender would block each candidate attacker.
Only one `blockers` record in twenty-two is taken at a real blocking decision. Two in three
`attackers` records are.

A record counts as a real decision here by the same test as the design doc's avoidance analysis.
An `attackers` record is real when its snapshot phase is `combat_declare_attackers` and its actor is
the active player. A `blockers` record is real when its phase is `combat_declare_blockers` and its
anchor attacker is attacking. The counts are over both validation strata, 6,848 whole games.

| Snapshot phase | `attackers` records | `blockers` records |
|---|---:|---:|
| upkeep | 267 | 9,017 |
| draw | 26 | 1,819 |
| main 1 | 987 | 15,231 |
| beginning of combat | 1,250 | 31,040 |
| declare attackers | 4,813 | 55,317 |
| declare blockers | 0 | 6,050 |
| combat damage, first-strike damage, end of combat | 0 | 810 |
| main 2 | 0 | 7,977 |
| end of turn, untap | 0 | 761 |
| total | 7,343 | 128,022 |
| real decisions | 4,813 (65.5%) | 5,895 (4.6%) |

The legality answer in a what-if record is still a rules fact about the board in its snapshot. The
query that produced it is hypothetical, but the verdict of which creatures may attack or block is
not. No `attackers` record in either stratum lists a tapped or summoning-sick creature as a legal
attacker, and no `blockers` record lists a tapped creature as a legal blocker. The legality head
therefore learns nothing false from them. What it learns from is a class dominated by planning
queries about combats Forge was only considering, and a record gives no way to tell the two kinds
apart.

Two further collector rules shrink the real decisions the corpus holds.

- De-duplication keys on the subkind and the payload, the list of legal ids, and ignores the board.
  The collector's comment says only byte-identical records collapse, but a record from a later turn
  with the same legal creatures on a different board is dropped as a duplicate. A what-if query
  earlier in the same turn can consume the key, so the real decision that follows it is dropped too.
- After de-duplication, `--legality-rate` keeps one record in ten. The two validation strata hold
  4,813 real declare-attackers decisions over 6,848 games, fewer than one a game.

### Gen-2 marks each legality record as a real decision or a what-if, as collection metadata

The collector applies the test above as it writes each record. It knows the phase, the active
player and whether the anchor attacker is attacking in the game's combat, and it writes an envelope
flag naming a what-if query.

### Gen-2 includes the board in the legality de-duplication key

The key becomes the subkind, the payload and the snapshot, the same three parts
`allowDistinctRecord` already hashes for the other record kinds. The repeats the de-duplication was
written for, the AI re-asking during one combat evaluation, share a board and still collapse. A
later turn with the same creatures on a different board is kept.

That the re-asks share a board is an assumption, and the pilot collection tests it before the full
run. If the snapshot carries the attack declarations the AI is simulating, each re-ask has a board
of its own. The legality class then grows back toward the four fifths of the corpus it held before the collector began
merging repeated records of an unchanged board.

### Gen-2 records every real decision and fills the legality class from them first

The real decisions are exempt from `--legality-rate`, and the rate applies to what-if queries only.
Real decisions happen only at the two declare steps of each turn, so exempting them adds little
volume. They are also the only legality records that join to the combat that follows, which is
what a measurement of Forge's attack and block choices needs.

Curation keeps what-if records rather than dropping them, because their verdicts are true and they
show the legality rules on boards and phases the real decisions never reach. It fills the legality
class from real decisions first, up to half its share, and the remainder from what-ifs. The manifest
records the real and what-if counts per subkind, and the evaluation reports the legality fields on
real decisions separately.

## Gen-2 collects its corpus from scratch, because the holdout and the random seat both act at collection time

A holdout cannot change without new games, because depletion freezes it into the games when they
are played. The training games are drawn from pools depleted of every card that carries a held-out
text, which is what lets a text-keyed holdout cost no training games. Gen-2 changes both inputs to
the holdout. The encoded text becomes the whole script chain, and the holdout unit becomes the
masked template. A new holdout applied to games depleted against the old one would move most of
them out of training, by the arithmetic in the design doc's section
[on depleting the pools](2026-09-04-ability-effect-model-design.md#the-held-out-set-is-built-by-depleting-the-pools-not-by-discarding-games).
The off-policy random seat also needs games played with it.

Everything collection fixes is therefore settled before the first game:

- the holdout: masked templates of the chained script text, with the eligibility cap counted per
  template (section on gate 1);
- the two new envelope fields, the random-seat flag and the what-if legality flag. Each is
  collection metadata, like `mode`, `fork` and `synthetic`, and never reaches the model. Each is
  additive, so the frozen-schema contract test gains one line per field, and both are in the
  schema before the first record;
- the random seat's share of games and its probability of acting at random (off-policy section);
- the rule that places a game in the game-disjoint stratum, which depends only on the game itself
  (section on rare-keyword combats).

A pilot of a few hundred games runs before the full collection, and the run plan says what it
checks.

## Gen-2 trains no baselines, because the knowledge probes measure what the baselines measured

The four baselines cost about two days of GPU time against the gen-1 corpus, and they answer
questions about the design rather than about one model. Gen-1 settled three of them. Reading the
text lowers the card-disjoint loss by about a third from the board-alone floor. The encoder's
margins over memorization clear every gate-1 threshold, the narrowest by nearly a factor of two.
Knowing an ability's API type and parameter keys recovers about two fifths of the encoder's gain.
None of gen-2's changes is expected to reverse these findings.

Each baseline's measurement has a stand-in that costs no training run and is computed on every
sweep arm.

| Baseline | What it measured | Gen-2 stand-in |
|---|---|---|
| `identity` | whether the encoder reads text rather than recalling it | the gap between each family's game-disjoint and card-disjoint results |
| `taxonomy` | what reading a script's parameters adds over its effect category | probes from `e` to the API type and the parsed parameters, beside the same probes from the parsed features themselves |
| `state-only` | what the board alone predicts | the probes record's board-only probe, which reads the board features with every `e` zeroed |
| `no-state` | whether conditioning on the board yields a better `e` | the state-dependence probe, which asks whether `e` predicts how much a text's outcome varies across boards |

The `no-state` question was never answered. Gen-1's `no-state` run zeroed the acting ability's
vector along with the board, so no gradient reached its encoder. The state-dependence probe is the
first measurement of the claim the design rests on.

Without an identity checkpoint, `evaluate-effect-model` reports gate 1 as skipped and runs gates 2
and 3 and the reported checks. It already behaves that way when no `--variant-checkpoint identity`
is given.

## Gen-2 is a sweep over the width of `e` and the size of the encoder, compared by hand

Gen-2's training time goes to a sweep rather than to baselines. The main axis is the width of `e`.
The encoder's depth and width are a second axis where time allows. Arms are compared by hand after
the runs, and no selection rule is fixed in advance.

Three things keep the arms comparable.

- **One corpus and one split.** Every arm trains on the same `build-corpus` output, so a difference
  between arms is a difference between models. The build step exists to make this possible, as the
  design record's section on built training sets says.
- **The same settings outside the swept axis.** The noise ratio and its ramp, the auxiliary head's
  weight, the withheld keyword and the curriculum, the schedule on which the sparse output fields
  join the loss, are the same in every arm. The running-average
  covariance keeps the noise level independent of the batch size an arm needs to fit 8 GB.
- **Measures that do not grow with the width of `e`.** A probe that reads a wider vector has more
  inputs and scores higher for that alone, so every probe on `e` is reported against a control of
  the same width, a fixed random vector per text (probes record). Concentration is compared by participation ratio, the number of
  equally weighted directions that would give the same spread. The top component's share is not
  comparable across widths.

A wider `e` is also a cheaper per-text key for the head to memorize, so each arm's memorization gap
is read beside every gain it shows. The comparison draws on:

- each arm's knowledge-probe scorecard;
- the card-disjoint results per rule family, and the memorization gap;
- gates 2 and 3, the ward canary and the zero-shot keyword check;
- the two downstream checks of the design doc: the decodability battery against the sealed
  pipeline's per-card win rates, and the pooled-`e` scorer smoke test.

## Run plan: a pilot before the full collection, and the gen-1 work while collection runs

The plan has six stages. Stage 0 reads only gen-1's artifacts and can overlap stages 1 to 3. Every
later stage needs the one before it.

### Stage 0 — on the gen-1 corpus

Both runs read the gen-1 corpus, whose manifest keys texts on the root line alone. They therefore
run against gen-1's sidecars: before `convert` is re-run in stage 1, or against a copy of
`output/cardsfolder/` and `output/tokenscripts/` kept aside for them.

1. **The knowledge probes on the gen-1 checkpoint**
   (`models/effects/runs/2026-09-17-full-textless-corpus/latest.pt`), following the probes record's
   run plan. This is the harness's first run, and it finishes before any gen-2 arm needs it.
2. **The noise pilot.** Three short runs at ratios 0.05, 0.1 and 0.2, with the ramp and the
   auxiliary head on. The auxiliary head reads root-line labels only, since gen-1's sidecars carry
   no chain. The chosen ratio is fixed for every sweep arm.

### Stage 1 — the text and the holdout

1. `python -m effects extract-keyword-definitions`, recording Forge's formatted value strings per
   keyword.
2. `python -m price_predictor convert`, writing sidecars whose `script_text` is the whole chain.
3. `python -m effects build-vocab --surface script`. The build log's unknown-token rate on the
   parameters is the check on the camel-case split, and the build asserts that no definition
   expands to `[UNK]`.
4. `python -m effects holdout-cards` with the template key. The share of cards it depletes sets
   the permille before anything is collected.
5. `python -m sealed generate-pools --exclude-cards` with that list.

### Stage 2 — the pilot collection

1. A few hundred games of `python -m sealed match-outcomes --effect-records`, with the random seat
   in one game of eight and both new envelope fields.
2. `python -m effects validate-corpus` on the first minutes, and `python -m effects field-coverage`
   on the pilot, which shows whether both new fields vary.
3. `python -m effects build-corpus` over the pilot shards into a scratch directory. Its manifest
   answers three questions:
   - Does each family hold enough records of the minority outcomes to balance on?
   - Does the legality class stay near its size with repeats merged, under the board-keyed de-duplication?
   - Do the on-policy and off-policy counts and the real and what-if counts look as intended?
4. The random seat's probability is raised only if the minority outcomes are too few.

### Stage 3 — the full collection

1. Depleted sealed self-play with the random seat, for training.
2. Full-strength collection with the random seat, for the card-disjoint stratum.
3. `python -m effects collect-coverage --training-corpus` over the cards self-play never reaches.
4. The coverage round restricted to the held-out cards.
5. `python -m effects collect-variants`.

Each run gets `validate-corpus` over its first minutes.

### Stage 4 — the corpus build

1. `python -m effects build-corpus` with rule-family balancing, outcome balancing and the
   game-disjoint stratum drawn toward rare-keyword combats.
2. The manifest's shortfall list goes back to `collect-coverage` and `collect-variants`, and its
   list of held-out texts under the floor goes back to the coverage round. The corpus is then
   rebuilt. The game-disjoint stratum keeps its games across the rebuild.
3. The final build is the one every sweep arm reads.

### Stage 5 — the sweep, and the checks on each arm

For each arm, in order:

1. `python -m effects train-effect-model --corpus <build> --withhold-keyword <keyword>`, at the
   arm's width of `e` and size of encoder. The epoch line's shares per family and per rarity bucket
   show whether the tail is reached.
2. `python -m effects encode-abilities`, with known keywords left as tokens.
3. `python -m effects evaluate-effect-model`: gates 2 and 3, the ward canary, the nearest
   neighbours, the zero-shot keyword check and the decodability battery. The battery reads
   `output/sealed/cards-win-rates.txt`, copied from `Y:\Nicolas\mtg\mtg-models-data\sealed\` on
   the NAS. The role-polarity check and the matched real-versus-fork check run if the corpus holds
   mana and fork records. Gate 1 reports as skipped.
4. The gen-1 embedding probes in `scripts/effect_embedding_probes/`, against the arm's cache:
   `build_texts.py` first, then `effect_profiles.py`, `pca_directions.py`, `lexical_probes.py`,
   `linear_probes.py`, `type_merging.py`, `neighbours.py` and `keyword_expansion.py`. Their
   `taxonomy` columns have no gen-2 counterpart, and the parsed-script features stand in for them.
   `pca_directions.py` adds the participation ratio.
5. The knowledge probes, following the probes record's run plan.
6. The pooled-`e` scorer smoke test: `train-scorer` Phase A on the sealed corpus, with each card's
   pooled `e` concatenated to its sealed vector.

The arms are then compared by hand, and the Outcome section records the comparison.

### Four pieces of tooling the run plan assumes do not exist yet

Four pieces of tooling are assumed by the plan and absent today.

- **The zero-shot measurement.** `evaluate-effect-model` names the withheld keyword and measures
  nothing on it.
- **The embedding probes take a checkpoint and a cache.** The scripts read gen-1's paths and the
  taxonomy baseline's cache. Each arm needs its own.
- **The participation ratio in `pca_directions.py`.**
- **A command for the scorer smoke test.** `write_scorer_smoke_cache` in `geometry_checks.py`
  writes the concatenated vectors, and no command runs Phase A on them.

## Outcome

To be filled in after the gen-2 sweep.
