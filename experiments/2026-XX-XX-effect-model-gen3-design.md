# Ability effect model — gen-3 (draft)

Draft design record for the third generation of the ability effect model. It collects the gaps in
collection and in the curated corpus that gen-2 leaves open, so that gen-3's design starts from
them. Gen-2 is described in
[`2026-09-18-effect-model-gen2-improvements-design.md`](2026-09-18-effect-model-gen2-improvements-design.md),
called the gen-2 record below. Nothing here is decided yet.

A few terms recur. A record is one observed game event: an ability resolving, a combat, a trigger, a
legality decision. A resolution record comes in two halves joined by a link, the cost half and the
effect half. A record names the ability that acted by its provenance key, the script file, face,
trait kind and index of the Forge trait. A sidecar is the file beside each converted card that maps
those keys to the card's rendered ability lines. The random seat is a player that makes a random
legal choice at a share of its decisions instead of the Forge AI's. A probe fork is a copy of the
game, taken at the combat damage step, in which one keyword is stripped from one creature to see
what changes.

## Collection can lose or misplace events without leaving a trace

**Events dropped by the collectors are never counted.** An engine event that arrives while no
record is open is discarded, and nothing records that it happened. A state-based death was lost
this way for a whole gen-1 corpus before anyone noticed. Gen-3 should count every discarded event
by event type and engine phase, behind a debug flag, and run a short pilot with the count on.

**A death can be written into a trigger that it happened before.** A dies-trigger put on the stack
during the state-based check that follows a resolution ends the open effect half. A death in the
next pass of the same check then lands in the trigger's bracket. It is written into the trigger's
effect half when the trigger resolves, although the death came first.

**Damage events on a probe fork lose their source.** On a probe fork, every `damage_dealt` event
lacks `params.cause`, and damage to a creature reports `combat: false`. Observed combats carry both
correctly. The source can be recovered from the payload's `assignment_choices`. Because the source
is missing, several attackers' identical hits produce byte-identical events, which inflates the
validator's "repeats an event" check. Gen-1 and gen-2 shards both carry the gap.

## Some records cannot be joined to the text of the ability that acted

**A charm cast through Flashback is keyed to no line.** The collector keys the chosen mode as the
Flashback keyword trait plus the mode's `option`, as in `wreck_and_rebuild.txt keyword[0]
option=0`. The converter writes `option` keys only for a charm's spell root and for saga chapter
keywords, so that key matches no line in the sidecar. Gen-2 holds one such record. Its
`build-corpus` refuses records like it as `unjoinable-key`, which is a workaround rather than a fix.

**An ability a clone gains from the card it copies cannot be keyed.** A Gigantoplasm copying
"Winter Soldier, Bucky Barnes" gains that card's "{X}: base power and toughness X/X" ability through
`AddAbilities$`. The collector's key resolution does not reach abilities granted that way, so the
records come out with an empty ability and `build-corpus` drops them. Traits a clone copies are
likewise recorded as `granted_trait`. Gen-3 should key these abilities, because the plays they
record are ones the model should learn from, as the random-seat section below explains.

**Some variant records are keyed under the card tree.** About a hundred keys from the gen-2 variant
run name files like `cardsfolder/a/a_cosmos_charger_variant_5001.txt` instead of
`variant-scripts/`. They appear to be variants of Alchemy "A-" cards. No sidecar exists at that
path, so the key resolves to no text, and the record is dropped or kept without text rather than
failing the build. The cause has not been looked into.

## The random seat can repeat a free activation that changes nothing

**The random seat never passes while an ability is still activatable.** In one gen-2 coverage game,
the random seat activated the Gigantoplasm's granted ability about 2,190 times in a single turn, at
X = 0. Each activation cost nothing and resolved to an empty change. The ability sets base power and
toughness, which sits below the creature's +2/+2 boost, so the creature stayed a 2/2 rather than
dying as a 0/0. Once such records are keyed, the corpus's per-text cap trims the repeats to a
handful. The play itself is wanted: a pointless activation, and an activation that kills the
creature, are both decisions the model should learn the outcome of. What gen-3 needs is for the
random seat to stop looping on a free, repeatable ability.

**The snapshot may hide a creature's dependence on a boost.** After the base became 0/0 under the
+2/+2 boost, the snapshot still reported `base [2,2], boosts [0,0]`. It appears to report the
printed base and fold set-base effects into the boosts, so a creature that survives only because of
a boost looks like an ordinary 2/2. This needs checking in general, not only on this card.

## Two sampling rules over-represent some questions

**What-if legality queries are sampled before they are de-duplicated.** A what-if query is the AI
asking whether an attack or block would be legal without committing to it. `--legality-rate` draws
each query before identical snapshots are merged, so a question the AI asks K times survives with
probability 1 − (1 − r)^K. Blocker what-ifs outnumber real block decisions about five to one.

**The held-out coverage round covers more than the held-out texts.** `collect-coverage
--only-cards` targets every text carried only by held-out cards. That set is larger than the
held-out texts themselves.

## Outcome

To be filled in once gen-3 is designed and run.
