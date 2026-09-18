# Ability effect model — gen-2 improvements

Design record for the second generation of the ability effect model. Gen-1 is the model trained on
the textless-abilities corpus of 2026-09-17 (run `09.17f`, filed under
`models/effects/runs/2026-09-17-full-textless-corpus/`) and its four baselines. Each section below
records one gap found while gen-1 trained, the evidence for it, and the change gen-2 makes. The
design rationale the gaps sit against is
[`2026-09-04-ability-effect-model-design.md`](2026-09-04-ability-effect-model-design.md).

Each baseline trains on the identical pipeline and differs from gen-1 in one input, so each answers
one question. `identity` gives every distinct ability text a free vector in place of the encoder's
output, and asks whether the encoder reads text at all. `taxonomy` replaces that vector with a hash
of the script's API type and parameter keys, and measures what reading a script's parameters adds
over knowing its effect category. `state-only` zeroes every ability vector and gives each record
kind its floor: a kind the full model scores at that floor is one where the text changed nothing
the model could find. `no-state` removes the board as well, so the model can only learn each text's
average effect, and it is compared on the embeddings rather than on the predictions.

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
effect half of a trigger, because that half is never encoded. What holds gen-1 together is the
description parameter: Forge attaches the rules text to the root line, so the prose is present for
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
`RepeatSubAbility` where one exists. Segments are separated by a dedicated token and each segment
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
3. **The holdout split moves.** A text is held out by the hash of its normalised script text, and
   the corpus's rarity table and the identity baseline's lookup table are keyed on the same string.
   Changing `script_text` changes every key, so gen-2 needs a new corpus build with a new split, and
   its card-disjoint numbers are not comparable to gen-1's on the same held-out cards. If
   comparability across generations matters more than the cleaner key, the holdout hash can keep
   reading the root line alone while the encoder reads the chain; that is a one-line choice in
   `build-corpus` and it should be made before the rebuild, not after.
4. **Sequence length grows, and the cap is far.** The encoder truncates at 512 tokens. The median
   chained line grows by less than half and the 90th percentile roughly doubles; the longest chains
   run past a dozen segments. The distribution of chain lengths in tokens should be measured after the rebuild,
   and a chain that exceeds the cap should be logged rather than silently cut from the tail, because
   the tail of a chain is its last effect.

Prose stays as it is: the fallback for a line with no script, which is a keyword-derived line or a
synthetic land mana line, and nothing else.

### Also noted: the paired-prose loss the design doc describes is not built

The design doc has the script surface as primary and the converted prose as a paired secondary,
with an asymmetric loss pulling the two `e` vectors of a line together. The trainer has no pairing
term. The prose reaches the encoder only inside the root line's description parameter, or as the
fallback above. Whether to build the pairing loss is a separate gen-2 decision; it is recorded here
because the chain change above is what makes the script surface carry the mechanism the pairing
was meant to anchor prose to, and the two should be weighed together.

## Every outcome in the corpus is one Forge chose, so the model can learn abilities without learning targets

The corpus is observational under one policy. A resolution record exists because Forge decided to
take that action, and Forge takes an action only when it already predicts the outcome it wants. In
the records, "Lightning Bolt targets a creature" and "that creature dies" are therefore almost the
same event, and "Murder targets a creature" is never paired with an indestructible target. A model
reaches a good loss on that corpus by learning that Bolt means death. It never has to learn that the
target's toughness decides it, because no record shows Bolt on a creature that survived. Combat
records do show damage that failed to kill, and the shared per-entity head carries some of that
over, but the ability-conditioned half of the mapping is learned only on the support the policy
chose.

The identity baseline is where this will show first. A lookup table learns "Bolt means death" as
well as the encoder does, so the encoder's margin over it on the unique-text stratum will be smaller
than it would be on a corpus where the outcome depended on the target. Gate 1 measures the encoder against
that baseline, and a corpus that lets both succeed by the same shortcut understates the encoder.

The fork probes already in the corpus are the interventional answer for one case, the damage-step
keywords, and each further case needs its own probe machinery. Off-policy play is the cheap
general answer. The engine is the label, so an action Forge would never take still produces a
correct record when it resolves. The records that appear are the ones the corpus lacks:

- Bolt on an 8/8, Murder on an indestructible creature, a player exiling their own creature. The
  ability resolves and the entity stays, or the wrong side loses something. These teach the gate and
  the zone-outcome field to read the target's state, which is what the factored design is for.
- Targeting through a ward cost the caster cannot pay. The cost record's outcome field already
  carries `countered`; the class is simply empty today.
- A 1/1 attacking into a 4/4, and blocks Forge would never assign. Combat records are the
  observational complement, and random declarations reach the region Forge's attack logic never
  enters.
- "Tap target creature" on a creature that is already tapped. A resolution that changes nothing,
  which the corpus has never recorded.

### Gen-2 collects from games where one seat plays legal but random actions against a normal Forge seat

One seat stays the standard Forge AI, so the game keeps a shape Forge would produce and ends. The
other seat is Forge's controller with its choices overridden at the decision points that matter: the
spell or ability to play, its targets, the attackers to declare, and the blocks to assign. At each of
those points the override draws uniformly from the legal options instead of taking Forge's ranked
choice. Land drops, mana payment and mulligans stay with Forge, because a random land drop produces
no record the corpus lacks and only makes the board less like one the model will be asked about.

The override lives in the connector, not in Forge. The match worker already builds both seats as
`LobbyPlayerAi` in `GamePlayer`, and Forge routes spell choice, target choice and combat
declarations through `PlayerControllerAi`, so the random seat is a controller subclass the worker
installs on one seat. Nothing on the `effect-record-hooks` branch changes.

Three things are settled here rather than at collection time.

1. **Records from the random seat carry a flag.** A per-record field naming the actor as the random
   seat is collection metadata in the sense the schema already has for `mode`, `fork` and
   `synthetic`: it never reaches the model. It is what lets the evaluation score the model separately
   on off-policy records, which is the number that says whether this section worked. It is an
   additive envelope field, so the frozen-schema contract test gains one line and every existing
   record reads as on-policy.
2. **Playability decision records are unaffected.** The `decision` subkind records the rules'
   verdict on every candidate, not the action the seat took, so a random seat produces the same
   decision records a Forge seat does. Nothing in that class needs the flag.
3. **On-policy collection continues alongside.** The random seat produces rare outcomes for common
   texts; the rarity-weighted sampler leans on rare texts. Both belong in one corpus, and the share
   of random-seat games is a collection parameter to be chosen, not a replacement for the existing
   runs.

Two limits are accepted. Random targets on a wide board mostly produce uninformative no-ops, so the
share of useful records per game is lower than under Forge's play and the run has to be longer to
cover the interesting cases. And the state distribution stays Forge-shaped, because what gets cast
still follows Forge's mana and hand on both seats; that is the right distribution for a model whose
every consumer runs inside Forge.

## The rarity weight is flat over all but the mana abilities, so a text seen once trains no harder than one seen in two hundred games

The rarity weight the trainer applies within a class does nothing for the tail of the curated corpus.
A record is weighted by the inverse square root of the games its text was seen in, capped at twenty
times the weight of the most-observed text. The most-observed texts are the five basic-land mana
abilities, each seen in about ninety thousand games. Measured against that reference, the ceiling
binds for every text seen in fewer than about 228 games, which is 97% of the rarity table. A text
recorded in one game and a text recorded in two hundred receive the same weight. Only the mana
abilities, the enters-tapped replacement and the eight hundred or so texts above that line weigh less. The figures
below are from the manifest of the corpus gen-1 trained on.

| texts in the rarity table | 31,561 |
| most-observed text (`Add {B}`), games | 91,133 |
| 99th / 95th / 50th percentile, games | 447 / 153 / 11 |
| texts seen in one game | 2,639 (8%) |
| texts seen in four games or fewer | 8,681 (28%) |
| texts at the weight ceiling | 30,740 (97%) |

The weight does select, which is why its flatness matters. A curated shard holds 2,000 records, and
an epoch visits 256 shards over 5,000 steps of 32 records, so one visit consumes about a third of
its shard. The weighted shuffle decides which third, heaviest first. With a flat table that draw is
uniform, and the share of an epoch a text gets is its share of the records that survived the
per-text cap. The cap is 200 records. A text seen in one game contributes one to three. A rare text
therefore reaches the encoder about a hundred times less often than a common one, and the exponent
that was meant to take the square root of that ratio has no effect.

The per-text cap is the only thing shaping the corpus, and it shapes the head alone. The cap drops
about three of every five resolution-effect records read and about half or more in every other
ability-keyed class. A cap is a ceiling and never a floor, so curation cannot add tail records;
what it can do is stop the head from drowning them, and today it does that only down to two hundred
records per text.

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
in the manifest does not change; only the trainer's reading of it does.

Whether the tail is being seen is otherwise invisible, so the trainer's epoch line gains the share of
records trained by rarity bucket of their text: one game, two to four, five to nineteen, twenty or
more. The manifest records unique texts per output, which says whether curation kept the tail, and
nothing says whether training reached it.

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
dredge. Cypher has no record at all, and no weight reaches a family with nothing in it; that family
is a coverage residue and belongs to collection.

Gen-2 samples in two levels. The family is drawn first, with the inverse-square-root weight over the
family's games and the same percentile ceiling as the text weight; then a text inside the family is
drawn with the per-text weight. A mechanism with hundreds of variants shares one family budget, and
a keyword with three variants gets a comparable budget of its own. The family key already exists:
the `taxonomy` baseline hashes an ability's API type and parameter keys, and a keyword line's key
is the keyword. The sampler reads the same key. The exponent is what keeps a broad family from being
starved: ChangesZone triggers have about five hundred times the games of dredge, so dredge's
family weight is about twenty times theirs rather than five hundred. Whether that is enough is what the per-family share in the epoch log is for,
and it is reported beside the rarity-bucket share above.

### The cap has to keep the off-policy records the random seat produces

The uniform drop under the per-text cap will remove the off-policy records at the same rate as the
rest. A common text will have many on-policy records and a small minority from the random seat,
and a cap that drops at random keeps that minority at the same small share. Bolt on an 8/8 is one
record in hundreds of Bolt on a 2/2, and the cap would keep the ratio. The random-seat flag is
already in the envelope for evaluation; the cap reads it as well. A capped text fills up to half its
cap from flagged records first and the rest from the remainder, so a text with few off-policy
records keeps every one and a text with many keeps a balanced set. The manifest records the
on-policy and off-policy record counts per class beside the existing per-class counts.

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
most. Gate 1
itself averages over entities and records, not over texts, so the same texts decide its three
margins. The number gate 1 reports is a well-recorded-holdout number. How the model reads a text it
was shown in one game is measured nowhere.

The holdout is also less unique than its name. A third of the recorded held-out texts have a
training text that is identical once numbers, `CARDNAME` and the description parameters are masked:
`NumDmg$ 2` held out beside `NumDmg$ 3` in training. Corpus-wide, two recorded texts in five share
such a template with another. The identity baseline cannot exploit a twin, because it keys on the
exact text. The encoder can, and reading `NumDmg$` is exactly what it should do. But a gate averaged
over a stratum where a third of the texts are one digit away from a training text measures
interpolation between siblings more than it measures reading. The four-way stratified report
separates numeric extrapolation and novel combinations of known API types, and neither of those is
this.

| recorded held-out texts | 393 |
| with a training text equal up to numbers, names and descriptions | 137 |
| recorded texts corpus-wide | 31,561 |
| sharing such a template with another recorded text | 13,320 |

### Gen-2 stratifies the validation sample by held-out text and reports gate 1 per rarity bucket

Four changes, of which the first three are curation and the fourth is the one-flag collection step
that makes them worth having.

1. **The card-disjoint sample is drawn per held-out text.** Its resolution slots fill round-robin
   over the held-out texts, a fixed number of records per text by smallest hash within the text,
   until the class quota is met. Every recorded held-out text is then in the sample, and the loss
   that selects checkpoints weighs a one-game text comparably to a hundred-game one. The
   card-disjoint stratum already caps games per held-out text for the same reason; this is the
   same rule one level down.
2. **Gate 1 reports a per-text mean beside its per-record numbers, and both by bucket.** The
   buckets are the rarity bucket of the held-out text's games, whether the text has a training
   twin, and the mechanism family of the text. The family view is what shows a keyword the model
   reads badly, which a mean over the damage spells would hide. The pass thresholds stay on the
   per-record numbers so gen-1 and gen-2 compare; the one-game, no-twin and small-family buckets
   are the rare-ability signal, and they are what the Outcome section below has to quote.
3. **The holdout key becomes the masked template.** A text is held out when the hash of its
   template falls under the permille, so a text's numeric siblings go with it and the unique-text
   stratum is unique. The chain change moves the split anyway, so gen-2 is the one moment this
   costs nothing extra. The holdout grows by the sibling families; the manifest reports the held-out
   text count before and after so the permille can be lowered to keep the held-out share of games
   where it is. The build also lists the held-out texts with no gate-one record and those under a
   floor of five games, which is the target list for the next point.
4. **A coverage round over the held-out cards alone.** `collect-coverage` gains the inverse of
   `--exclude-cards`: a run restricted to the held-out cards, writing full-strength records. Every
   game it produces names a held-out card, so the split rule routes it to the card-disjoint stratum
   on its own. The round runs until every held-out text with a castable carrier reaches the floor,
   and the residue it reports is the list of held-out texts gate 1 can never score.

## Only the identity baseline is retrained with the model; the other three are trained once

Training the four baselines costs about two days of GPU time against the gen-1 corpus, and only one
of them has to be repeated. The gates are what a shipped checkpoint must pass, and only gate 1 reads
a baseline. The other three baselines test claims about the design, and a claim once established
holds until the thing it is about changes.

`identity` is retrained on every corpus rebuild. A rebuild moves the split, and the evaluator refuses
a baseline that records a different split or vocabulary from the model it is compared with. It is
not retrained for a change confined to the encoder, because the baseline has no encoder: the existing
checkpoint remains a valid gate 1 opponent for a run that changes the encoder's depth, width or
learning rate. A change to the effect head or to the training schedule calls for a matched identity
run, because the design defines the baseline as the model with everything but its input unchanged,
and a deeper head could gain on board state alone.

`taxonomy`, `state-only` and `no-state` are trained once and repeated only when a change is aimed at
what each measures. Two planned gen-2 changes are aimed that way. Encoding the whole script chain
puts the parameters of every sub-ability in front of the encoder, and the taxonomy comparison
measures what reading parameters adds, so taxonomy is retrained once after that change. Off-policy
games are meant to make an outcome depend on the target's state rather than on the ability alone.
The state-only floor per record kind is the direct reading of whether that worked, so state-only is
retrained once after those games enter the corpus. A hyperparameter sweep repeats none of the three.

| Baseline | What it measures | Retrain when |
|---|---|---|
| `identity` | gate 1: whether the encoder reads text at all | every corpus rebuild; a change to the head or the schedule |
| `taxonomy` | what reading a script's parameters adds over knowing its effect category | once; again after the chain encoding |
| `state-only` | each record kind's floor, what the board alone predicts | once; again after off-policy games enter the corpus |
| `no-state` | whether state-conditioning produces a better embedding | once |

The recurring cost is therefore one identity run per corpus rebuild. At gen-1 pace that is about
half a day, and it can share the card with another head-only run, as identity and taxonomy did on
2026-09-18. The driver script that queued all four after the gen-1 run queues identity alone from the
next cycle.

## Outcome

To be filled in after the gen-2 run.
