# Ability effect model — gen-2 improvements

Design record for the second generation of the ability effect model. The model has two parts. The
encoder reads an ability's script text and outputs a fixed-length vector `e`, the ability's
embedding. The script text is the encoder's script surface. The rendered rules prose, its prose
surface, is the fallback for a line with no script. The effect head reads `e` together with the
board and predicts what the ability does to each entity on it. A head is a network trained on top of
a shared representation to predict one kind of output. The effect head is a transformer, a network
that reads a set of input vectors, here `e` and the board's entities, and lets each output depend on
all of them. Gen-1 is the model trained on the corpus of 2026-09-17 (run `09.17f`, filed under
`models/effects/runs/2026-09-17-full-textless-corpus/`), together with its four baselines. The
design rationale is
[`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md), called the
design doc below.

Each baseline trains on the identical pipeline and differs from gen-1 in one input, so each answers
one question. `identity` gives every distinct ability text a free vector in place of the encoder's
output, and asks whether the encoder reads text at all. `taxonomy` replaces that vector with a hash
of the script's API type and parameter keys, and measures what reading a script's parameters adds
over knowing its effect category. `state-only` zeroes every ability vector and gives each record
kind its floor: a kind the full model scores at that floor is one where the text changed nothing the
model could find. `no-state` also removes the board, so the model can learn only each text's average
effect, and is compared on the embeddings. Gen-1's run of it answered nothing (section on
baselines).

Gen-1 is validated on two strata of games kept out of training. The game-disjoint stratum holds
whole games whose ability texts also occur in training games. The card-disjoint stratum holds games
that contain a held-out card, whose texts never trained. Three gates judge a checkpoint, a saved
copy of a trained model's weights. Gate 1 compares the encoder with the `identity` baseline on texts
the model never trained on. Gate 2 checks that predictions move the way the rules say when a
damage-step keyword is removed from a creature. Gate 3 checks that the variance of the embeddings is
not concentrated in a few directions.

Gen-2 collects its corpus anew, trains no baselines, and trains several models, the arms of a sweep,
each at a different width of `e` or size of encoder. The arms are judged by the knowledge probes of
[`2026-09-19-effect-knowledge-probes-design.md`](2026-09-19-effect-knowledge-probes-design.md),
called the probes record below. A probe is a small model fitted to read a known property from a
model's internal vectors. Gen-2's gate figures are not compared with gen-1's, because the corpus,
the split and the encoding all change.

Each of the first ten sections records one gap found while gen-1 trained, the evidence for it, and
the change gen-2 makes. The three after them set how gen-2 is collected, evaluated and trained, and
the run plan at the end orders the work.

## The encoder reads only the first script line of an ability, so a trigger's effect never reaches it

On the script surface the encoder is fed one Forge script line per ability: the line the ability's
runtime trait was parsed from. An ability that Forge writes across several lines loses every line
but the first. For a triggered ability the first line is the trigger condition, and the effect is a
separate `SVar` line the trigger names through `Execute$`. Ajani's Pridemate encodes as the trigger
line alone:

```
Execute$ TrigPutCounter | Mode$ LifeGained | TriggerDescription$ Whenever you gain life, put a +1/+1 counter on CARDNAME. | TriggerZones$ Battlefield | ValidPlayer$ You
```

The `DB$ PutCounter | CounterType$ P1P1 | CounterNum$ 1` line that does the work is not in the text.
The model knows that this is a life-gain trigger which does something the description calls "put a
+1/+1 counter"; it does not see the counter type or the count as parameters.

The omission is near-universal for triggers and common for the rest.

| converted card corpus, excluding keyword lines | count |
|---|---:|
| lines with a script surface | 41,914 |
| triggered lines | 16,135 |
| triggered lines whose `Execute$` effect is absent from the encoded text | 16,098 |
| lines with a sub-ability chain below the first effect | 8,820 (21%) |
| chain length 1 / 2 / 3 / 4 or more sub-abilities | 5,595 / 2,247 / 698 / 280 |
| longest chain | 13 sub-abilities |
| median ratio of full-chain text to root text, chained lines only | 1.4× |
| 90th percentile of that ratio | 2.1× |

Modal abilities lose every mode. A charm's root line names its modes only by label, as in `Choices$
DBDmgC,DBDmgP`, and where it has a description, the description ends in the placeholder `ABILITY`
that Forge fills from the modes at runtime. The encoder sees that the ability offers a choice and
not what any choice does.

| charm lines | count |
|---|---:|
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
`Cost$`, `ValidTgts$`, `Mode$` and `NumDmg$` as the predicate structure every record supervises, and
it counts on paraphrases of one mechanism collapsing to one script. Neither holds for the effect
half of a trigger, because that half is never encoded. Gen-1 still sees the effect through the
description parameter: Forge attaches the rules text to the root line, so the prose is present for
all but about one line in eighty. The encoder is reading prose plus the trigger condition or the
first effect's parameters, and the rest of the mechanism reaches it only as English.

The description is not a full substitute. The design doc's third argument for the script surface is
synthetic script variants, which perturb a parameter and play the variant for engine ground truth of
a text that never existed. A variant that changes `CounterNum$ 1` to `CounterNum$ 2` on a trigger
produces a record whose encoded text is identical to the original, because the changed line is not
in the text. The variant records then teach the model that one text has two outcomes.

### Gen-2 encodes the whole chain in resolution order

`script_text` becomes the concatenation of every script line the ability owns, in this order:

- the root line;
- for a trigger, the ability it executes through `Execute$`;
- each sub-ability in chain order, including a `RepeatSubAbility` where one exists.

Replacement effects get the same treatment through `ReplaceWith$`. A dedicated token separates the
segments. Each segment after the root opens with the label its parent referenced it by, so
`SubAbility$ SV2` in one segment and `SV2:` opening a later one give the model the link. The labels
are renumbered by position within the line, so the link survives and the author's name for the SVar
does not (next subsection). A charm's modes are not inlined into its root line. Each mode is encoded
on a line of its own (subsection on charms).

Four things follow from the change, and the third is the one that decides the order of work.

1. **The converter rewrites the sidecars but not the converted text.** `script_text` is a field of
   the sidecar, the `.provenance.json` file `convert` writes beside each converted card to map each
   rendered line to the Forge script it came from. The rendered prose does not change, so the
   byte-identity guarantee the sidecar contract makes holds.
2. **The script vocabulary is rebuilt.** `build-vocab --surface script` scans the sidecars' script
   lines; after the change those lines carry the chained segments and the separator token. Most
   parameter keys already appear on root lines, so the vocabulary grows by the numbered labels,
   one per position in the longest chain, and the rarer sub-ability parameters.
3. **The holdout is chosen on the chained text.** A text is held out by the hash of its masked
   template, the normalised script text with its numbers, `CARDNAME` and descriptions masked
   (section on gate 1), and the corpus's rarity table is keyed on the script text. Changing
   `script_text` changes every key. The holdout is therefore computed on the chained text before the
   new corpus is collected, which is the only point at which changing it costs no training games
   (section on collecting from scratch).
4. **Sequence length grows, and most chains stay well under the cap.** The encoder truncates at 512
   tokens, and the table above gives the growth on chained lines. The distribution of chain lengths
   in tokens is measured after the rebuild. A chain that exceeds the cap is logged rather than
   silently cut from the tail, because the tail of a chain is its last effect.

Prose stays as it is: the fallback for a line with no script, which is a keyword-derived line or a
synthetic land mana line, and nothing else.

Gen-2 does not build the design doc's paired-prose loss. That loss would pull a line's prose
encoding toward its script encoding, so that a card with prose but no Forge script would still get a
usable `e`. Gen-2 serves cards implemented in Forge, and every one of them has a script.

### Gen-2 renumbers SVar labels by position, because the names authors give them carry no mechanism

Forge's card scripts name their SVars freely, and most names are used once. The table counts the
labels that `Execute$`, `SubAbility$` and `ReplaceWith$` reference across Forge's card scripts.

| SVar labels referenced in Forge's card scripts | count |
|---|---:|
| distinct labels | 3,488 |
| references | 36,501 |
| labels referenced once | 2,355 |
| labels referenced fewer than five times | 3,009 |

Labels kept as written damage the vocabulary. A label is one whole token, so a name used once
either takes a vocabulary slot of its own or reads as `[UNK]`. A label read as `[UNK]` no longer
links a reference to the segment it names, which is the only reason the label is in the text.

Labels kept as written also damage the holdout. Two texts that do the same thing under different
names, `SubAbility$ DBDraw` and `SubAbility$ DBDrawCard`, are different texts. The rarity table
counts them apart, and the holdout hashes them apart, so one twin can train while the other sits in
the card-disjoint stratum as an unseen text.

Gen-2 renames every chain label in a line's `script_text` to `SV1`, `SV2`, … in order of first
appearance. A label keeps its number everywhere in the line, so a reference and the segment it names
still match. Numbering restarts on every line. The rename covers the values of `Execute$`,
`SubAbility$`, `RepeatSubAbility$` and `ReplaceWith$`, each item of `Choices$`, and the label that
opens each segment. Amount SVars, such as the `X` of `NumDmg$ X`, are not chain links and keep their
names. What a label loses is redundant: `DBDraw` says "draw", and the segment it opens says
`DB$ Draw`.

### Gen-2 encodes and records each mode of a charm as its own ability

A charm's outcome depends on which modes its caster chose, and no single text can show that choice.
The converter already emits one `option` line per mode beside the charm's root line. Gen-1's
sidecars give those lines no provenance key and no script text, and the provenance join resolves
every chosen mode to the root line. A charm resolution therefore acts through one text whose
outcome is damage on one record and a drawn card on another, depending on a choice the model never
sees.

Gen-2 makes each mode a separate ability, in four steps.

1. **Each `option` line carries its mode's chain.** Its sidecar entry holds the mode's provenance
   key and a `script_text` holding the mode's script line, opened by its label, followed by its
   sub-abilities. The root line keeps only its own script line: the choices, how many modes to
   choose, and whether a mode may repeat.
2. **Each mode gets its own `e`.** The effect model adds a learned option-kind embedding to every
   ability row that comes from an `option` line. Option rows follow their root row within the
   card's block, so the flag and the order link a mode to its charm.
3. **A patched worker records each chosen mode as its own effect.** A modal resolution becomes one
   cost record acting through the root line and one effect half per chosen mode, acting through
   that mode's `option` line. Each effect half carries only its mode's events and a snapshot taken
   just before that mode resolved. All of them share one `link_id`, which therefore joins a cost
   record with its effect halves rather than only a pair. A mode chosen twice, as a Pawprint charm
   allows, produces two effect halves.
4. **A degraded worker keeps the root line.** Without the engine hooks the collector cannot tell
   where one mode's events end and the next begin. A degraded worker writes one effect half
   through the root line carrying every chosen mode's events.

## Every outcome in the corpus is one Forge chose, so the model can learn abilities without learning targets

Every record comes from games in which both seats follow Forge's AI. The corpus is therefore
observational under one decision rule, one policy in reinforcement-learning terms. A resolution
record exists because Forge decided to take that action, and Forge takes an action only when it
already predicts the outcome it wants. In the records, "Lightning Bolt targets a creature" and "that
creature dies" are therefore almost the same event, and "Murder targets a creature" is never paired
with an indestructible target. A model reaches a good loss on that corpus by learning that Bolt
means death. It never has to learn that the target's toughness decides it, because no record shows
Bolt on a creature that survived. Combat records do show damage that failed to kill, and the effect
head carries some of what it learns there across record kinds. The part of its mapping that depends
on the ability is still learned only on the boards and targets Forge chose.

A corpus like this understates any measure of whether the encoder reads text. A lookup table learns
"Bolt means death" as well as the encoder does, so the encoder's margin over gen-1's identity
baseline is smaller than it would be on a corpus where the outcome depended on the target.

Off-policy play, in which one seat takes actions Forge's AI would not have chosen, is the cheap
general fix. Forge's rules engine computes every record's outcome, so an action Forge's AI would
never take still produces a correct record when it resolves. The fork probes already in the corpus,
games copied at the damage step and replayed with one keyword stripped from one creature, fix only
the damage-step keywords, and each further case would need its own fork machinery. The records
off-policy play adds are the ones the corpus lacks:

- Bolt on an 8/8, Murder on an indestructible creature, a player exiling their own creature. The
  ability resolves and the entity stays, or the wrong side loses something. These teach the effect
  head's affected-or-unaffected output and its zone-outcome field to read the target's state.
- Targeting through a ward cost the caster cannot pay. The cost record's outcome field already
  carries `countered`; the class is simply empty today.
- A 1/1 attacking into a 4/4, and blocks Forge would never assign, including the combats deathtouch
  decides, which Forge's combat AI declines (section on rare-keyword combats). Combat records are
  the observational complement, and random declarations reach the region Forge's attack logic never
  enters.
- "Tap target creature" on a creature that is already tapped. A resolution that changes nothing,
  which the corpus has never recorded.

### Gen-2 collects one match in four with a seat that takes a random legal action at half its decisions

The random seat is Forge's AI with its choices overridden at the decision points that matter: the
spell or ability to play, its targets, the attackers to declare, and the blocks to assign. At each
of those points it acts at random with one probability, the same at every point, and then draws
from the legal options. Otherwise it takes Forge's ranked choice. Land drops, mana payment
and mulligans stay with Forge, because a random land drop produces no record the corpus lacks and
only makes the board less like one the model will be asked about. The other seat stays the standard
Forge AI, so the game keeps a shape Forge would produce and ends. Two random seats never play each
other.

The override lives in the connector, not in Forge. The match worker already builds both seats as
`LobbyPlayerAi` in `GamePlayer`, and Forge routes spell choice, target choice and combat
declarations through `PlayerControllerAi`, so the random seat is a controller subclass the worker
installs on one seat. Nothing on the `effect-record-hooks` branch changes.

A random draw picks one element at a time, because the legal attacks, blocks and target sets are
combinations too numerous to list and draw from.

| Decision | Random draw |
|---|---|
| spell or ability | uniform over the playable spells and abilities other than mana abilities |
| attackers | each creature that can attack does so with probability ½ |
| blockers | each creature that can block picks uniformly from no block and the attackers it may legally block |
| targets | a count uniform between the ability's minimum and maximum, then that many distinct legal targets |

A declaration Forge rejects as illegal is redrawn. After a fixed number of failed draws the seat
takes Forge's choice for that decision. The play draw never passes while something is playable,
because a pass produces no resolution record, and resolutions Forge would not choose are what the
random seat is for. Mana abilities are left out of the play draw because mana payment stays with
Forge. A mana ability drawn at random would tap a land for mana that nothing spends.

Four details are fixed in this design rather than left to the collection run.

1. **Records from the random seat carry a flag.** The flag lets the evaluation score the model
   separately on off-policy records, which is the number that says whether off-policy collection
   worked.
2. **Playability decision records are unaffected.** The `decision` subkind records the rules'
   verdict on every candidate, not the action the seat took, so a random seat produces the same
   decision records a Forge seat does. Nothing in that class needs the flag.
3. **One match in four has a random seat, and it acts at random at half its decision points.** The
   random seat produces rare outcomes for common texts, and the attacks Forge's AI would not make,
   which are the only source of block decisions against such attacks once what-if queries are not
   collected (section on what-if queries). On-policy games supply everything else, so the two share
   one corpus. The two settings multiply to one eighth, the expected share of one seat's decisions
   taken at random per match played. A seat that acted at random at every point would lose most games early, and its boards would say
   little about the boards the model is asked about.
4. **Random-seat games are played in both collections.** They appear in the training collection,
   played from pools depleted of every card that carries a held-out text, and in the full-strength
   collection, played from complete pools, that supplies the card-disjoint stratum. The off-policy
   records of training therefore have held-out counterparts to be scored on.

Two limits are accepted. Random targets on a wide board mostly produce uninformative no-ops, so the
share of useful records per game is lower than under Forge's play. And random attacks and blocks
cost the random seat life and creatures, so its games run shorter and its boards are thinner than
Forge's. The probability of acting at random sets how far both go. The stage 2 pilot, at one match
in eight and a quarter of the decisions, yielded too few block decisions against random attacks
to replace the what-if queries, which is what sets the values above (Outcome, stage 2).

## The rarity weight is flat for 97% of texts and nothing balances rule families, so rare texts and rare rules train least

The rarity weight the trainer applies within a class does nothing for the tail of the curated
corpus. Curation is the `build-corpus` step that selects which collected records enter the training
corpus. It fills each class, a record category such as the effect halves of resolution records or
the legality records, to a fixed share of the corpus. A record is weighted by the inverse square
root of the games its text was seen in, with a ceiling of twenty times the weight of the
most-observed text. The most-observed texts are the five basic-land mana abilities, each seen in
about ninety thousand games. Measured against that reference, the ceiling binds for every text seen
in fewer than about 228 games, which is 97% of the rarity table. A text recorded in one game and a
text recorded in two hundred receive the same weight. Only the mana abilities, the enters-tapped
replacement and the eight hundred or so texts above that line weigh less. The figures below are from
the manifest of the corpus gen-1 trained on.

| gen-1 corpus manifest | |
|---|---:|
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
The cap is 200 records, and a text seen in one game contributes one to three.

The per-text cap is the only setting that changes the mix of texts in the curated corpus, and it
acts only on the most common texts. The cap drops about three of every five resolution-effect
records read and about half or more in every other ability-keyed class. A cap is a ceiling and never
a floor, so curation cannot add tail records. What it can do is stop the common texts from crowding
them out, and today it does that only down to two hundred records per text.

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
the reference and the ceiling barely binds, which is what the ceiling was written to do: stop a
single text from dominating a batch without flattening everything below it. The mana abilities and
the few hundred texts above the reference weigh less than it, which is where they belong. The rarity
table in the manifest does not change; only the trainer's reading of it does.

Nothing reports today whether training reaches the tail. The manifest's unique texts per output show
only whether curation kept it. The trainer's epoch line therefore gains the share of records trained
by rarity bucket of their text: one game, two to four, five to nineteen, twenty or more.

### A per-text weight shows a mechanism once per variant, so a keyword with few variants stays rare

Equalising texts does not equalise mechanisms. Every "deal N damage to target X" script is its own
text, so the damage mechanism reaches the encoder once for each of its variants, and a keyword
written on a handful of cards reaches it a handful of times however heavily each of its texts is
weighted. The weight fix above removes the bias between texts and leaves this one untouched. The
table groups the rarity table's texts into rule families: by Forge API type, trigger mode or
keyword.

| rule family | distinct texts | games |
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
the mechanism vocabulary Forge already uses, and a keyword line's family is the keyword whatever the
record kind.

| Record kind | Family |
|---|---|
| resolution, effect and cost halves | the acting ability's API type |
| trigger | the trigger mode |
| continuous | the static mode |
| rewrite | the replacement type |
| combat | the damage-step keywords present in the fight, or none |
| playability | the mode of the static that forbids it, or else the verdict that failed |

A playability verdict that fails on mana has no static behind it, and neither does one that fails
for want of a target. Where no static is responsible, the first verdict that failed names the
family: cannot be played, unaffordable, or no legal target. Without that rule, every unaffordable
and every untargetable candidate would share one family with the candidates that pass.

`build-corpus` gives every family within a class the same target number of records. A family with
more records than the target is cut down to it. A family with fewer is written whole, and each of
its records is repeated up to a reuse cap. Each family therefore receives the target or its records
times the reuse cap, whichever is smaller.

The reuse cap is what stops uniformity from turning into memorization, in which the model predicts
from the identity of a text rather than from what the text says. Dredge has seven texts in about six
hundred games. A budget equal to that of the ChangesZone triggers would replay those few cards and
boards hundreds of times, and the model would learn the cards rather than the rule.

The per-text cap counts copies as well as distinct records. A record repeated four times counts four
against its text's cap of two hundred, so no text is replayed past the cap. A family made of a few
texts therefore stays short rather than filling its target with them, and its shortfall goes on the
list below.

A family short of its target is a collection problem rather than a sampling one. Cypher has no
record at all, and no weight reaches it. The manifest reports each family's available records, its
target, the records written, the repeats and the shortfall. The shortfall list is the target list
for `collect-coverage` and `collect-variants`.

Inside a family, the text weight and the per-text cap of the previous subsections balance the texts.
The epoch line reports the share of records trained per family beside the share per rarity bucket.
The validation results are reported per family and averaged over families, so the evaluation weighs
the rules the way training does.

### Within a family, curation balances the outcomes as well as the texts

A family's records are balanced over a coarse outcome signature: which pairs of zone outcome and
changed-or-not occur among the affected entities. Lightning Bolt killing its target and Lightning
Bolt leaving it standing then appear in comparable numbers. On-policy play records almost only the
first.

The signature ignores how many entities share a pair, so that a mass effect is not split by board
size. A wrath that kills three creatures and one that kills seven produce the same outcome on boards
of different sizes. If counts entered the signature, each board size would become a cell with its
own share, and the rare large boards would be repeated to fill it.

Balancing the outcomes changes which boards the model sees and not the right answer on any of them,
because most outcomes follow from the board under the rules: Bolt on a creature with three toughness
always kills it. It shifts the head only where the outcome depends on a player's choice, such as
which creature an opponent sacrifices, and that choice is Forge's policy rather than a rule.

The minority outcomes come mostly from the random seat, so this balance is what turns the random
seat's games into training signal. The random-seat flag is not itself a curation key: the signature
selects the minority outcomes directly, from whichever seat produced them. The manifest records the
on-policy and off-policy record counts per class beside the existing per-class counts.

## The training noise on the ability vector is under 1% of its length, so it does not stop the effect head memorizing texts

Gen-1's noise on `e` is too small to stop the effect head memorizing texts, because nothing keeps
`e` small and the noise does not scale with it. The design doc's first memorization lever keeps `e`
small and noisy in training, so that the effect head cannot use `e` as a precise per-text lookup key
("Coverage and memorization are separate tail problems, and both get levers" in the design doc).
Gen-1 implements only the noise half. The encoder adds Gaussian noise with a fixed standard
deviation of 0.05 to each of the 64 coordinates (`e_noise` in `ability_encoder.py`). `e` is a linear
projection of the encoder's output, restrained only by the optimizer's weight decay, a pull of every
weight toward zero by a factor of 0.01 of the learning rate per step. A larger `e` makes the same
absolute noise relatively smaller, and the effect head reads `e` through a linear layer, which
absorbs any scale.

In the trained gen-1 encoder the vectors are large enough that the noise no longer matters. The
figures are over every row of the shipping cache under `output/effects/abilities/cardsfolder/`, with
the taxonomy baseline's files left out. The noise moves a vector by less than a hundredth of its
length. Along the ninth principal direction, the weakest the encoder uses, the spread of the vectors
is almost ninety times the noise.

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
between two of them, which is the lookup the lever was meant to make expensive. In gen-1 only the
design doc's other four levers acted against memorization.

### Gen-2 draws the noise with the covariance of the `e` vectors, estimated as a running average

Five fixes are open, and they differ in whether the encoder can still escape the noise.

- **Normalizing each `e` to a fixed length before the noise.** The fixed noise then has a scale it
  cannot escape. It also discards length, which ranges over a factor of nearly three in the shipping
  cache and may carry information. It changes the vectors the cache stores, so every consumer reads
  a different `e`.
- **A penalty on the length of `e` in the loss.** It adds a second weight, and the balance between
  that weight and the noise is what sets the noise's effective size.
- **One noise level for every direction, proportional to the average spread of `e`.** Rescaling `e`
  as a whole then changes nothing. The encoder can still widen a few directions, though. The noise
  is sized to the average, so along those directions it is small against the spread, and they can
  carry a precise per-text key. The same reward for piling variance into a few directions is what
  gate 3's concentration check flags.
- **Noise in each coordinate proportional to that coordinate's spread.** Widening one coordinate no
  longer helps. The coordinates are arbitrary axes, however. A signal laid along a diagonal across
  many coordinates meets noise smaller relative to its spread by up to the square root of the width
  of `e`.
- **Noise with the covariance of the `e` vectors.** The noise is wide along the directions in which
  the vectors spread widely and narrow along the others, diagonals included. The ratio of noise to
  spread, the noise ratio, is then the same along every direction, and no reshaping of `e` changes
  it.

Gen-2 takes the last. Five details settle it.

1. **The covariance is measured around the mean, with the gradient stopped.** A common offset added
   to every vector separates no two texts, so it must not raise the noise. With the gradient
   stopped, the covariance is treated as a constant when the loss is differentiated, so the noise
   cannot become a path through which the loss changes `e`.
2. **The covariance is a running average over recent batches.** One batch spans at most one
   direction fewer than it has distinct texts. A batch with fewer distinct texts than `e` has
   dimensions would leave some directions without noise, and those directions could carry a key. One
   batch's estimate also varies with the batch size, which differs between sweep arms that fit the 8
   GB card differently. The average starts from the first batch's covariance.
3. **The noise ratio ramps up over the first epoch.** It rises linearly from zero to its value, the
   same way in every arm. Early in training `e` grows quickly and the running average lags behind
   it, which makes the noise smaller than intended rather than larger. The ramp makes the start of
   training independent of that lag.
4. **The noise moves from the encoder to the batcher.** It is applied to the vectors the batch
   carries, which is where the running average is kept.
5. **The noise ratio comes from a pilot on the gen-1 corpus.** Gen-1's effective ratio per
   coordinate was under a hundredth, from an absolute noise of 0.05 against a spread of 6.8. The
   pilot trains at ratios of 0.05, 0.1 and 0.2 with the auxiliary head of the next section switched
   on. It reads each run's results on unseen texts and the concentration of its `e`. Every sweep arm
   then uses the chosen noise ratio.

The cache and its consumers do not change. The noise is applied in training only, and the encoded
`e` keeps its form and its unconstrained scale.

Gate 3 is where the change should show. Gate 3 caps the share of the embedding's variance on its top
principal component. Under noise with the covariance of the `e` vectors, the noise ratio is the same
along every direction, so widening one direction separates no more texts. The encoder separates the
most texts by spreading them over as many directions as it can. The change is therefore expected to
lower the top component's share, and gate 3 reports whether it does.

## The ability vector keeps less of an ability's amounts and costs than a hash of its parameter keys

Gen-1's `e` holds less about amounts and costs than a hash that records only which parameters a
script sets. The figures are linear probes from the design doc's Outcome. The taxonomy hash holds
every parameter key's presence and no value.

| Property | Metric | gen-1 `e` | taxonomy hash |
|---|---|---:|---:|
| damage amount | R² | 0.34 | 0.55 |
| mana in the activation cost | R² | 0.06 | 0.33 |

Lightning Bolt's nearest neighbours show that the amount is lost. They are the same sentence with 1,
2, 4, 5, 6, 7, 10 and X damage, all at a cosine similarity of at least 0.997.

The effect head's loss rarely depends on the number. Forge casts Bolt mostly at creatures it kills,
so on the records the corpus holds, three damage and four have the same outcome. Costs are
supervised only through the cost halves of resolution records and the playability verdicts, and few
of the effect head's output fields score either. The outcome balancing above makes the amount matter
more often, but nothing in the loss asks `e` for a cost at all.

### Gen-2 adds an auxiliary head, used in training only, that reads from `e` the values the script states

The auxiliary head reads `e` alone and predicts numbers parsed from the script:

- damage dealt, power and toughness change, counters placed and cards drawn;
- the mana cost by colour, plus its generic part;
- whether the cost taps or sacrifices.

The labels come from the sidecar's script parameters, for every segment of the chain, at no cost. A
label is masked where the script states no fixed number, as with X. The head follows the pattern of
the script-API head, gen-1's training-only head that predicts each script's API type from `e`: a
small loss weight, used in training only, and filtered out at save time.

Costs come from the line's `Cost$` alone. A spell's mana cost is printed on the card rather than on
any line, and it pays for the whole card, so no line is labelled with it. A spell line's mana-cost
labels are masked even when an additional cost puts mana in its `Cost$`, as in Bone Splinters'
`Cost$ B Sac<1/Creature>`. The effect head reads the card's mana value and colours from the board
snapshot and relates them to the card's abilities itself.

The weight is the one setting to watch. Gen-1's `e` predicted observed outcomes better than the
parsed script features did, because it groups abilities by what they do rather than by how their
scripts are written. A large weight would pull `e` back toward the parser's grouping.

The game agent reads `e` through the cache, and the amounts and costs are what it needs to weigh one
card against another. The knowledge probes that read amounts and costs from `e` become a check that
the head did its job. The probes that ask whether the model uses an amount against a toughness stay
independent of it.

## Gate 1 scores fewer than half of the held-out texts, and a third of the recorded ones have a numeric twin in training

Held-out texts are chosen by hash over the converted texts, not over the recorded ones, and nearly
half of them were never recorded. Sealed self-play never cast the cards that carry them, and
`collect-coverage` keeps held-out cards out of its decks by design, so nothing fills the gap. Of the
texts that were recorded, a third were seen in four games or fewer. The eligibility rule, which
holds out a text only if at most eight cards carry it, excludes almost nothing, because nine texts
in ten are on exactly one card.

| held-out texts, gen-1 | count |
|---|---:|
| held-out texts | 735 |
| with any record | 393 |
| with a gate-one resolution record | 342 |
| recorded, seen in one game | 43 |
| recorded, seen in two to four games | 82 |
| recorded, seen in five to nineteen games | 142 |
| recorded, seen in twenty games or more | 126 |
| converted texts carried by one card | 33,456 of 36,981 |

The validation sample that selects checkpoints is dominated by the well-recorded held-out texts. The
sample is drawn by smallest record hash, which is uniform over records, and the gate-one slice holds
about eighteen thousand records over its 342 texts. The resolution slots of the 2,048-record sample
therefore go to the texts with the most records, and a text seen in one game holds a few at most.
Gate 1 itself averages over entities and records, not over texts. The same well-recorded texts
therefore decide all three of its margins over the identity baseline: affected-entity F1,
zone-outcome accuracy and count deviance. Gate 1 therefore measures how the model reads the
well-recorded held-out texts. How it reads a held-out text recorded in a single game is measured
nowhere.

A third of the recorded held-out texts have a twin in training. A twin is a training text that is
identical once numbers, `CARDNAME` and the description parameters are masked, such as `NumDmg$ 2`
held out beside `NumDmg$ 3` in training. This masked form is the text's template. Corpus-wide, two
recorded texts in five share a template with another. The identity baseline cannot exploit a twin,
because it keys on the exact text. The encoder can, and reading `NumDmg$` is what it should do. A
gate averaged over held-out texts of which a third are one number away from a training text measures
interpolation between twins more than reading. The design doc's four-way stratified report does not
isolate these twins: its numeric-extrapolation and novel-combination strata are different cases.

| numeric twins | count |
|---|---:|
| recorded held-out texts | 393 |
| with a training text equal up to numbers, names and descriptions | 137 |
| recorded texts corpus-wide | 31,561 |
| sharing such a template with another recorded text | 13,320 |

### Gen-2 holds out whole templates and reports unseen texts per text and per rule family

Four changes. The first two change how the card-disjoint sample is drawn and how its results are
reported. The third changes the holdout key, which acts at collection time. The fourth is a coverage
round that gives the held-out texts enough records to score.

1. **The card-disjoint sample is drawn per held-out text.** Its resolution slots fill round-robin
   over the held-out texts, a fixed number of records per text by smallest hash within the text,
   until the class quota is met. Every recorded held-out text is then in the sample, and the loss
   that selects checkpoints weighs a one-game text comparably to a hundred-game one. The
   card-disjoint stratum already caps games per held-out text for the same reason; this is the same
   rule one level down.
2. **Results on unseen texts are reported per text and by bucket.** A per-text mean sits beside the
   per-record numbers. The buckets are the rarity bucket of the held-out text's games and the rule
   family of the text, and the family results are also averaged over families. The family view is
   what shows a keyword the model reads badly, which a mean over the damage spells would hide. The
   memorization gap is the gap between a family's results on the game-disjoint stratum, whose texts
   trained, and on the card-disjoint stratum, whose texts did not. Gate 1 itself is not computed,
   because gen-2 trains no identity baseline (section on baselines).
3. **The holdout key is the masked template.** A text is held out when the hash of its template
   falls below the holdout fraction, a number of templates per thousand (`--holdout-permille`), so a
   text's numeric twins go with it and every held-out text is unseen in training, numbers included.
   The template is taken from the chained script text, and the new corpus is depleted against it
   (section on collecting from scratch). Chain labels need no mask, because the encoding text has
   already renumbered them (subsection on SVar labels). The eligibility rule counts the cards that
   carry any text of the template, not the cards of one text: a template such as `Pump +N/+N until
   end of turn` spans thousands of cards, and holding it out would deplete them all. The holdout
   fraction is set against the share of cards the holdout depletes, which `holdout-cards` reports
   before collection starts. The build lists the held-out texts with no gate-one record and those under a floor of
   five games, which is the target list for the next point.
4. **A coverage round over the held-out cards alone.** `collect-coverage` gains the inverse of
   `--exclude-cards`: a run restricted to the held-out cards, writing full-strength records. Every
   game it produces names a held-out card, so the split rule routes it to the card-disjoint stratum
   on its own. The round runs until every held-out text with a castable carrier reaches the floor.
   The held-out texts it reports as still under the floor are the ones no game can cover.

## Keyword expansion builds the cache from definitions gates 1 and 2 never score, and misses every keyword whose name is not a single token

Keyword expansion replaces a keyword token with its definition, the reminder text Forge prints for
the keyword ("New keywords are handled by expansion dropout over their definitions" in the design
doc). It has four defects. The cache and the gates read different encodings of every keyword line. A
keyword whose name does not become a single token never finds its definition: every hyphenated
keyword, and every multi-word keyword the vocabulary does not already know. The definition inserted
is a damaged template. And `build-vocab` scans Forge's generated keyword scripts into the script
vocabulary, though no expansion reads them. A fifth gap is in the evaluation: gen-1 withheld no
keyword, so the check that a keyword is read through its definition measured nothing. The evidence
comes from `scripts/effect_embedding_probes/keyword_expansion.py`, with its outputs in
`output/effects/reports/keyword-expansion-20260919/`. What the analysis found about the encoder
itself is in the design doc's Outcome, in "A keyword token encodes only loosely like its own
definition" and "An unknown keyword expands only when its name is one word".

The shipping cache encodes every known keyword as its definition, while the effect head trained on
the keyword token three times in four and gates 1 and 2 score the token every time. The expansion
probability is the chance that a known keyword is replaced by its definition, and each caller sets
its own. Training uses 0.25. Validation and gates 1 and 2 go through `SurfaceBatcher` with 0.
`encode-abilities`, which builds the cache, uses 1.0, so that two runs of the command write the same
file. An unknown keyword is expanded at every probability. Forty keyword rows of the shipping cache
match the full expansion at a mean cosine of 1.0000 and the unexpanded encoding at 0.77. Gate 3, the
ward canary (the distance between ward and the card lines that spell out its effect), the embedding
probes and the deployed head all read the cache. For a keyword line, each of them therefore reads an
input the head saw only a quarter of the time in training, and never the input gates 1 and 2 scored.

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
- 16 definitions contain `[UNK]`, the placeholder token the encoder reads for any word missing from
  the vocabulary. `build-vocab` scans the definitions but keeps only the 5,000 most frequent tokens,
  and 21 words of reminder text fall below that cut: `encoded`, `promise`, `specified` and
  `teammate` among them.
- Five templates (enlist, increment, read ahead, station, web-slinging) use the typographic
  apostrophe `’`, which becomes a token of its own rather than the `'` every other text uses.

| keyword line | what the encoder reads under expansion |
|---|---|
| `Toxic:1` | players dealt combat damage by this creature also get % d poison counters . : 1 |
| `Ward:2` | whenever this permanent becomes the target of a spell or ability an opponent controls , counter it unless that player . : 2 |
| `Enchant:Creature` | target a % 1 $ s as you cast this . this card enters attached to that % 1 $ s . : creature |
| `Start your engines` | if you have no speed , it [UNK] at 1 . it [UNK] once on each of your turns when an opponent loses life . max speed is 4 . |

No expansion reads Forge's generated keyword scripts, yet `build-vocab` scans them into the script
vocabulary. Both surfaces expand to the reminder template: `_definition_text` returns it, and no
code substitutes the generated script.

### Gen-2 builds the cache with known keywords left as tokens

`encode-abilities` expands at probability 0, the setting validation and gate 2 use. The choice keeps
what 1.0 was chosen for: an unknown keyword still expands at every probability, so the cache stays
deterministic and a new set still reads through its definitions. It also makes the cache the input
the head saw three times in four in training and the one validation and gate 2 score. Expanding at
1.0 in training as well would make the two agree the other way, but it removes the keyword token the
design keeps, and the design doc's Outcome shows the encoder has learned that token as a symbol of
its own rather than as its definition.

1. **One setting, read by both callers.** The probability for scoring and for the cache is one
   constant that `SurfaceBatcher`'s scoring path and `AbilityEncoderRunner` both read, just as
   `SurfaceBatcher.text_of` is the one definition of the encoding text. Each caller setting its own
   value is how the cache and the gates came to read different inputs.
2. **The geometry is measured again.** Gate 3, the ward canary and the embedding probes are re-run
   on the new cache. Their keyword lines then describe the keyword token, which is what the head
   reads.

### Gen-2 recognises a keyword line by its display name, not by a vocabulary token

The definition is looked up by the display name, the part of a keyword line's script text before its
first colon, matched case-insensitively against the definitions table. The whole name is replaced,
however many tokens it split into, and the text after the colon becomes the line's instance values.
A keyword word inside another line keeps the token path it has now.

1. **The match is checked against the sidecar.** The batcher sees only the encoding text, so the
   match is made on the text. The corpus build asserts that every text it matches belongs to sidecar
   lines whose `script_api_type` is `Keyword`. The sidecar's `line_kind` is not the key: it is the
   converter's category, `alternate cost` for jump-start.
2. **The host-bodied list is checked on the same path.** It compares display names, so level up and
   read ahead are excluded as the list intends.
3. **The keywords that never expand today are the test.** Jump-start, web-slinging and beam me up
   must each expand both with and without a merged vocabulary token for their name.

### Gen-2 fills each template with its line's values, strips every format specifier and reserves the definition words

1. **Instance values are filled.** The values after the colon fill the placeholders in order, and
   `%1$s` repeats its value at each use. Forge formats some values before filling them: ward's `%s`
   stands for a clause such as "pays {2}", not the bare `2`. Those formatted strings come from
   Forge, recorded per keyword by `extract-keyword-definitions`, rather than being rebuilt in
   Python.
2. **Every specifier is stripped where no value fills it.** The pattern covers `%s`, `%d` and the
   positional forms, and no expansion leaves a `%` token.
3. **The apostrophe is normalised.** `’` becomes `'` when the definitions are loaded.
4. **The definition words are reserved in the vocabulary.** `build-vocab` seeds every word of every
   template after stripping, as it already seeds `[PAD]` and `[CLS]`, so `--target-size` cannot cut
   them. The build asserts that no definition expands to `[UNK]`. The reservation lands in the
   script-vocabulary rebuild that the chain encoding and the camel-case split (tokenizer section)
   already require, so it costs no rebuild of its own.

### Gen-2 keeps the reminder template on both surfaces and stops scanning generated scripts

The generated script is not yet worth expanding to. It is the root line of the keyword's trait
alone. Prowess's is its `SpellCast` trigger condition without the +1/+1 it grants, and no trigger
among the 75 scripts carries the effect it executes: the same first-line gap the chain section
describes. Its description parameter holds the reminder text anyway. And only 75 of the 202 keywords
have one, so expanding to it would read three keywords in five in one language and the rest in
another.

1. **The template stays the definition on both surfaces.** The generated script is revisited once
   `extract-keyword-definitions` renders the whole chain, as the converter will for card lines.
2. **`build-vocab` stops scanning generated scripts**, so the capped vocabulary counts only text the
   encoder reads.

### Gen-2 withholds one keyword, so the zero-shot check has something to measure

The zero-shot keyword check is the only test that a keyword the model never trained on is read
through its definition, which is how a new set's keyword reaches the model. Gen-1 withheld no
keyword, so the check reported nothing. The design doc's Outcome adds a reason to run it: an unknown
keyword's expansion lands far from the vector of the trained keyword it defines.

1. **Every sweep arm withholds the same keyword.** `train-effect-model --withhold-keyword` keeps its
   token out of training and expands every occurrence to its definition. The keyword is chosen
   before the sweep, among the script-generated keywords new sets introduce, with enough records in
   the corpus for its fields to be scored.
2. **The check measures.** The evaluator today names the withheld keyword and stops. Gen-2 adds the
   measurement the design doc specifies: the effect head's per-field results on records where the
   withheld keyword acts or sits on the board, beside the same fields for keywords that trained.

## The script tokenizer is never called, and the prose grammar that runs in its place splits selectors better

`AbilityTokenizer.tokenize_script` is the tokenizer written for the script surface, and no code
calls it. Encoding for the cache (`ability_encoder_runner.py`) and encoding during training
(`surface_batching.py`) both pass the script text to `tokenize`, the prose grammar. The vocabulary
is built with the prose grammar too. `build-vocab --surface script` hands the script lines to the
shared price-predictor vocabulary builder, which splits them with `MtgTokenizer`. The design doc
counts a script tokenizer that splits compound selectors such as `Creature.nonDragon+OppCtrl` among
the costs of the script surface, in the design doc's section "The script is the primary surface;
prose is the paired secondary".

The prose grammar already makes that split. It ends a word at every character that is not a letter
or an underscore, so `.` and `+` end a word and are kept as tokens of their own. `tokenize_script`
makes the same cut at `.` and `+` but drops the separators, and before cutting it keeps letters,
digits and `-` together in one run. A sign then either vanishes, so `NumAtt$ +1` reads as `1`, or
fuses with its number into `-1`. A threshold fuses with its digit, so `powerGE4` becomes one token
and the 4 never reaches the numeric embedding, the encoder's separate input path that reads a
number's value. The vocabulary was built with the prose grammar, so every fused token is unknown to
it.

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
also cut where a lowercase letter is followed by an uppercase one, so `nonDragon` reads as `non
dragon` and `YouCtrl` as `you ctrl`.

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
4. **Chain labels stay whole.** The chain encoding links a segment to its parent by its numbered
   label (subsection on SVar labels). Read by the prose grammar, `SV2` becomes `sv 2`, and the digit
   reaches the numeric embedding as if it were an amount. The tokenizer therefore keeps whole the
   values of `Execute$`, `SubAbility$`, `RepeatSubAbility$`, `Choices$` and `ReplaceWith$`, and the
   label that opens each segment. `build-vocab` reserves `sv1` up to the longest chain's label
   count, so the vocabulary size setting cannot drop one.
5. **A counter type is one token.** The prose grammar reads `CounterType$ P1P1` as `p 1 p 1`, so the
   digits of a counter type reach the numeric embedding as if they were amounts. On the script
   surface a counter type's value stays whole, `p1p1` or `m1m1`, in the tokenizer and in the
   vocabulary scan alike.

## Two build settings leave gate 2 and the trainer short of rare-keyword combats

Gate 2 cannot score six of its eight keywords for want of records, and two `build-corpus` settings
decide how many records it and the trainer get. The game-disjoint stratum that gate 2 reads is 1,000
games drawn uniformly. Curation keeps combat records for training at one rate whatever the fight.
The design doc's section [on why deathtouch and indestructible combats are
scarce](2026-09-04-ability-effect-model-design.md#deathtouch-and-indestructible-combats-are-scarce-because-few-creatures-carry-the-keywords-and-forges-combat-ai-avoids-the-combats-deathtouch-decides)
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
threshold, and the threshold is higher for a game that holds a qualifying combat for a keyword short
of the floor. The cost falls on training. Wither qualifies about once per hundred games in the
stratum above and about twice per hundred over both of gen-1's validation strata. A corpus of
gen-1's size therefore holds roughly 1,300 to 2,600 wither combats, and reaching 200 takes between a
thirteenth and a seventh of them out of training. The other keywords lose less.

The rule depends only on the game, so a rebuild over a grown corpus keeps every game it already
placed in the stratum and only adds new ones. The seeded sample `build-corpus` draws today redraws
the whole stratum whenever the corpus grows. That would move the games the knowledge probes read
into training, and the probes would score later models on boards they trained on.

The stratum over-represents rare-keyword games, so its results are reported per rule family and
averaged over families, the balance training is built to. The manifest records which games entered
for a keyword, so a figure over the stratum can also be read without them.

### Rule-family balancing gives the rare-keyword combats their own budget

Gen-1's curation cannot favour a deathtouch combat over an ordinary one. It keeps 158,348 of the
1,066,917 combat records it reads, 15%, and chooses them by a hash of the record id, so the choice
is uniform over fights. The per-text cap and the rarity weight never see a combat record, because a
combat record carries no acting ability text.

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
set, not only when a player declares. Forge computes those sets while planning in its main phases,
at upkeep, and while the attacking AI simulates how the defender would block each candidate
attacker. Only one `blockers` record in twenty-two is taken at a real blocking decision. Two in
three `attackers` records are.

A record counts as a real decision here by the same test as the design doc's avoidance analysis. An
`attackers` record is real when its snapshot phase is `combat_declare_attackers` and its actor is
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
  fewer than one real declare-attackers decision a game.

### Gen-2 marks each legality record as a real decision or a what-if, as collection metadata

The collector applies the test above as it writes each record. It knows the phase, the active player
and whether the anchor attacker is attacking in the game's combat, and it writes an envelope flag
naming a what-if query.

### Gen-2 includes the board in the legality de-duplication key

The key becomes the subkind, the payload and the snapshot, the same three parts
`allowDistinctRecord` already hashes for the other record kinds. The repeats the de-duplication was
written for, the AI re-asking during one combat evaluation, share a board and still collapse. A
later turn with the same creatures on a different board is kept.

That the re-asks share a board is an assumption, and the pilot collection tests it before the full
run. If the snapshot carries the attack declarations the AI is simulating, each re-ask has a board
of its own. The legality class then grows back toward the four fifths of the corpus it held before
the collector de-duplicated legality records.

### Gen-2 records every real decision and collects no what-if queries

The real decisions are exempt from `--legality-rate`, and the rate applies to what-if queries only.
Real decisions happen only at the two declare steps of each turn, so exempting them adds little
volume. They are also the only legality records that join to the combat that follows, which is what
a measurement of Forge's attack and block choices needs.

The rate defaults to 0, so no what-if query is collected. A what-if's verdict is true, but which
queries reach the corpus is not representative of play. The AI asks about one combat many times
while it plans an attack. The rate is drawn for each query before de-duplication, so a query asked
`K` times survives with probability `1 − (1 − rate)^K`. The kept queries are therefore weighted by
how long the AI deliberated over a board, which favours the wide boards it plans longest. The one
thing they hold that real decisions lack is blocks against attacks the AI considered and rejected.
Those blocks reach the corpus instead as real block decisions against the random seat's attacks, on boards the
game actually reached.

Curation splits the legality class equally over the halves the corpus holds, so a corpus without
what-ifs fills the class from real decisions. The manifest records the real and what-if counts per
subkind, and the evaluation reports the legality fields on real decisions separately.

## Gen-2 collects its corpus from scratch, because the holdout and the random seat both act at collection time

A holdout cannot change without new games, because depletion freezes it into the games when they are
played. The training games are drawn from pools depleted of every card that carries a held-out text,
which is what lets a text-keyed holdout cost no training games. Gen-2 changes both inputs to the
holdout. The encoded text becomes the whole script chain, and the holdout unit becomes the masked
template. A new holdout applied to games depleted against the old one would move most of them out of
training, by the arithmetic in the design doc's section [on depleting the
pools](2026-09-04-ability-effect-model-design.md#the-held-out-set-is-built-by-depleting-the-pools-not-by-discarding-games).
The off-policy random seat also needs games played with it.

Everything collection fixes is therefore settled before the first game:

- the holdout: masked templates of the chained script text with its labels renumbered, with the
  eligibility rule counted per template (section on gate 1);
- per-mode modal records: each chosen mode of a charm recorded through its own `option` line
  (subsection on charms);
- the two new envelope fields, the random-seat flag and the what-if legality flag. Each is
  collection metadata, like `mode`, `fork` and `synthetic`, and never reaches the model. Each is
  additive, so the frozen-schema contract test gains one line per field, and both are in the schema
  before the first record;
- the random seat's share of games (off-policy section), and its probability of acting at random as
  the pilot leaves it;
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

Each baseline's measurement has a stand-in that costs no training run and is computed on every sweep
arm.

| Baseline | What it measured | Gen-2 stand-in |
|---|---|---|
| `identity` | whether the encoder reads text rather than recalling it | the memorization gap, per rule family |
| `taxonomy` | what reading a script's parameters adds over its effect category | probes from `e` to the API type and the parsed parameters, beside the same probes from the parsed features themselves |
| `state-only` | what the board alone predicts | the probes record's board-only probe, which reads the board features with every `e` zeroed |
| `no-state` | whether conditioning on the board yields a better `e` | the state-dependence probe, which asks whether `e` predicts how much a text's outcome varies across boards |

The `no-state` question was never answered. Gen-1's `no-state` run zeroed the acting ability's
vector along with the board, so no gradient reached its encoder. The state-dependence probe is the
first measurement of the claim the design rests on: that training `e` against the board yields a
better `e`.

Without an identity checkpoint, `evaluate-effect-model` reports gate 1 as skipped and runs gates 2
and 3 and the checks it reports without a threshold, such as the ward canary and the decodability
battery (section on the sweep). It already behaves that way when no `--variant-checkpoint identity`
is given.

## Gen-2 is a sweep over the width of `e` and the size of the encoder, compared by hand

Gen-2's training time goes to a sweep rather than to baselines. The main axis is the width of `e`.
The encoder's depth and width are a second axis where time allows. Arms are compared by hand after
the runs, and no selection rule is fixed in advance.

Three things keep the arms comparable.

- **One corpus and one split.** Every arm trains on the same `build-corpus` output, so a difference
  between arms is a difference between models.
- **The same settings outside the swept axis.** The noise ratio and its ramp, the auxiliary head's
  weight, the withheld keyword and the curriculum are the same in every arm. The curriculum is the
  schedule on which the sparse output fields join the loss. The running-average covariance keeps the
  noise level independent of the batch size an arm needs to fit 8 GB.
- **Measures that do not grow with the width of `e`.** A probe that reads a wider vector has more
  inputs and scores higher for that alone, so every probe on `e` is reported against a control of
  the same width, a fixed random vector per text (probes record). Concentration is compared by
  participation ratio, the number of equally weighted directions that would give the same spread.
  The top component's share is not comparable across widths.

A wider `e` is also a cheaper per-text key for the head to memorize, so each arm's memorization gap
is read beside every gain it shows. The comparison draws on:

- each arm's knowledge-probe scorecard;
- the card-disjoint results per rule family, and the memorization gap;
- gates 2 and 3, the ward canary and the zero-shot keyword check;
- the two downstream checks of the design doc: the decodability battery, linear probes that read the
  sealed pipeline's per-card win rates from each card's pooled `e`, and the pooled-`e` scorer smoke
  test, which adds the mean and the maximum of each card's `e` vectors to the sealed scorer's input
  and retrains Phase A of `train-scorer`, the stage that trains the deck scorer on cached card
  vectors with the card encoder frozen.

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
2. `python -m price_predictor convert`, writing sidecars whose `script_text` is the whole chain
   with its labels renumbered, and whose `option` lines carry their modes.
3. `python -m effects build-vocab --surface script`. The build log's unknown-token rate on the
   parameters is the check on the camel-case split, and the build asserts that no definition expands
   to `[UNK]`.
4. `python -m effects holdout-cards` with the template key. The share of cards it depletes sets the
   holdout fraction before anything is collected.
5. `python -m sealed generate-pools --exclude-cards` with that list.

### Stage 2 — the pilot collection

1. A few hundred games of `python -m sealed match-outcomes --effect-records`, with the random seat
   in one game of eight and both new envelope fields.
2. `python -m effects validate-corpus` on the first minutes, and `python -m effects field-coverage`
   on the pilot, which shows whether both new fields vary.
3. `python -m effects build-corpus` over the pilot shards into a scratch directory. Its manifest
   answers three questions:
   - Does each family hold enough records of the minority outcomes to balance on?
   - Does the legality class stay near its gen-1 size under the board-keyed de-duplication, or grow
     back toward four fifths of the corpus?
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
   `output/sealed/cards-win-rates.txt`, copied from `Y:\Nicolas\mtg\mtg-models-data\sealed\` on the
   NAS. The role-polarity check and the matched real-versus-fork check run if the corpus holds mana
   and fork records. Gate 1 reports as skipped.
4. The gen-1 embedding probes in `scripts/effect_embedding_probes/`, against the arm's cache:
   `build_texts.py` first, then `effect_profiles.py`, `pca_directions.py`, `lexical_probes.py`,
   `linear_probes.py`, `type_merging.py`, `neighbours.py` and `keyword_expansion.py`. Their
   `taxonomy` columns have no gen-2 counterpart, and the parsed-script features stand in for them.
   `pca_directions.py` adds the participation ratio.
5. The knowledge probes, following the probes record's run plan.
6. The pooled-`e` scorer smoke test: `train-scorer` Phase A on the sealed corpus, with each card's
   pooled `e` concatenated to the card vector the sealed scorer already reads.

The arms are then compared by hand, and the Outcome section records the comparison.

### Four pieces of tooling the run plan assumes do not exist yet

- **The zero-shot measurement.** `evaluate-effect-model` names the withheld keyword and measures
  nothing on it.
- **The embedding probes take a checkpoint and a cache.** The scripts read gen-1's paths and the
  taxonomy baseline's cache. Each arm needs its own.
- **The participation ratio in `pca_directions.py`.**
- **A command for the scorer smoke test.** `write_scorer_smoke_cache` in `geometry_checks.py` writes
  the concatenated vectors, and no command runs Phase A on them.

## Outcome

### Stage 0 (2026-10-08): the sweep trains with a noise ratio of 0.1

The noise ratio for every sweep arm is 0.1. The three pilots cannot be told apart on validation loss, but their ability caches can: at 0.1 the top principal component takes the smallest share of the variance and `e` spreads over the most dimensions, while at 0.2 one direction dominates and the cache fails gate 3's top-component canary.

Each pilot trained for three epochs on gen-1's curated corpus with the same seed (7), so the three runs saw the same batches in the same order and differed only in the ratio. Every gen-2 training term was on: the value head at weight 0.05, and the verdict, created-objects, MLM and script-API losses. The script vocabulary was rebuilt with gen-2's tokenization rules over gen-1's sidecars, whose texts are the root line alone.

#### Validation loss moves by about 2% across the three ratios

At the end of epoch 3, the first epoch that trains the sparse fields, card-disjoint loss falls as the ratio rises and game-disjoint loss rises slightly. That is the direction noise is meant to push, from texts seen in training towards held-out ones, but the spread is small for a single seed and the ordering of the first two epochs was not the same. The value head, which reads `e` alone, gets worse as the noise grows.

| epoch 3 | 0.05 | 0.1 | 0.2 |
|---|---:|---:|---:|
| card-disjoint loss | 4.566 | 4.509 | 4.459 |
| game-disjoint loss | 4.479 | 4.510 | 4.511 |
| gate F1 | 0.774 | 0.775 | 0.769 |
| zone accuracy | 0.938 | 0.947 | 0.949 |
| value-head loss (training) | 1.71 | 1.82 | 2.06 |

#### The ability caches separate the ratios, and 0.1 uses `e` best

Each pilot's cache was encoded over gen-1's sidecars and measured with gate 3's two collapse canaries and the participation ratio, the number of dimensions the variance effectively spreads over. At 0.1 the top component's share is the lowest and the participation ratio the highest. At 0.2 the mean pairwise cosine is lowest, yet a single direction carries more of the variance than gate 3 allows. One reading is that under heavy noise the encoder moves its signal onto a single high-variance axis the noise cannot swamp; the pilots do not test it.

| cache | mean pairwise cosine | top component share | participation ratio |
|---|---:|---:|---:|
| ratio 0.05 | 0.475 | 28.5% | 6.57 |
| ratio 0.1 | 0.482 | 25.7% | 7.07 |
| ratio 0.2 | 0.333 | 31.5% | 5.72 |
| gen-1, 40 epochs | 0.152 | 40.0% | 3.93 |

Gate 3 requires a mean pairwise cosine of at most 0.5 and a top component of at most 30%. The gen-1 row is the shipped checkpoint after forty epochs, not a three-epoch run, so it shows where training this long ends rather than a fourth pilot.

The knowledge probes' first run, on the gen-1 checkpoint, is recorded in [`2026-09-19-effect-knowledge-probes-design.md`](2026-09-19-effect-knowledge-probes-design.md#gen-1-2026-10-08).

### Stage 2 (2026-10-09): the pilot collection writes every gen-2 field, and one fork bug surfaced

The pilot shows each gen-2 collection mechanism at work, and `validate-corpus` holds all twenty of
its judged invariants. The pilot played 133 matches, 833 games and 321,792 records, with a random
seat in one match in eight acting at random at a quarter of its decision points.

| what the records show | pilot |
|---|---:|
| games holding a random-seat record | 80 of 833 |
| real attack declarations | 10,950 |
| real block declarations | 15,296 |
| what-if block queries | 70,010 |
| charm mode halves acting through an `option` key, spells / triggers | 158 / 79 |
| real resolutions that dealt damage | 1,249 |
| of those, the damaged creature dies in the same record | 684 |

Gen-1 wrote none of those deaths: on a sample of its shards, none of 565 lethal-looking resolution
damage events carried the death (knowledge probes record). Gen-2 holds a resolution's record
through the state-based check that follows it, so the record carries the deaths its own damage
causes.

#### A fork split a creature spell by its own enters trigger's modes

A fork that forces a creature spell also resolves the creature's modal enters trigger, on the same
card. The fork counted the trigger's modes as the spell's and wrote keys naming `spell[0]` with an
option, a line no sidecar carries. `build-corpus` refuses such a key, and the pilot holds four. A
fork now counts a mode only when its charm resolves to the forced ability's own provenance key. The
trial build below read the pilot with those four records removed.

#### The trial corpus delivers its class mix, and the rule families are short as a pilot's must be

`build-corpus` over the pilot wrote 15,243 training records, with every class within 3% of its
requested share. 199 cells of the family split came up short of their share, which is the volume a
full collection supplies. One game names a held-out card, though the decks excluded every held-out
card: Brazen Boarding, an Alchemy card, creates Staunch Crewmate during play. The build places any
game naming a held-out card in the card-disjoint stratum, so the card reaches no training record.

#### Block decisions against random attacks were too few to replace the what-if queries

The random seat attacks at random at only a quarter of its decisions, so few of the AI's real block
decisions face an attack the AI would not have made. The legality class therefore stops collecting
what-if queries and the random seat runs at one match in four and half its decisions instead, which
quadruples the random decisions per match played.

| block legality records | pilot |
|---|---:|
| real, the AI blocking the random seat's attacks | 361 |
| of those, against an attack drawn at random (expected) | about 90 |
| real, both seats Forge's AI | 14,194 |
| what-if queries, all games | 70,010 |

### The sweep

To be filled in after the gen-2 sweep.
