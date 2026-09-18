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
