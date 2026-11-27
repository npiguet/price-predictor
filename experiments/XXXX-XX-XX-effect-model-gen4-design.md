# Ability effect model — gen-4 (draft)

## TL;DR

Some of the effect model's output fields stay weak while the others improve, and training cannot
respond: each record's share of training is fixed when the corpus is built. Gen-4 plans training
that draws more of the records a weak field needs, from the whole corpus, with the mechanism and the
controller still to choose. It also takes up what gen-3 leaves to a later generation: the model sees
both players' hidden information, no output says which library cards a player knows, three outcome
fields learn only from the times they happened, and keyword reminder text comes from a hand-written
copy of Forge's formatting. Nothing has run.

## Background

Gen-4 is the fourth generation of the ability effect model. It follows gen-3, described in
[`2026-10-30-effect-model-gen3-design.md`](2026-10-30-effect-model-gen3-design.md) and called the
gen-3 record below. Gen-3 rebalances its training by rule of the game when its corpus is built, and
leaves training that reacts to a weak field to gen-4.

### Some output fields stay weak while the others improve

**A few fields with a real amount to learn barely moved between the first two epochs that scored
them.** Each output field is scored by its explained deviance: the share of the loss of the best
constant prediction that the model removes, where 0% is no better than always guessing the field's
average. The table reads gen-2's first arm, which predicts 46 output fields per entity
([gen-3 record](2026-10-30-effect-model-gen3-design.md#some-output-fields-stay-weak-while-the-others-improve)).

| field | loss of the constant prediction | epoch 3 | epoch 4 |
|---|---:|---:|---:|
| `keywords_gained` | 0.565 | 21% | 21% |
| `poison_delta` | 0.138 | −11% | −14% |
| `life_delta`, for comparison | 2.402 | 80% | 81% |

**Training cannot respond to a weak field.** Each record's share of training is fixed when
`build-corpus` writes the dataset, and the trainer only shuffles within the one shard it holds.

**Which fields are weak after gen-3 is not known until gen-3 trains.** Gen-3 adds outputs, reads
every line of a permanent, ties each line to its card, rebalances by rule and puts every field on
one scale in the loss. Gen-4 starts from gen-3's per-field scores rather than gen-2's.

### The model sees both players' hidden information, though it is meant to play from one seat

**Gen-3's input shows the model every card a player could not see.** Collection is omniscient by
design: a record holds both hands, the real face of every face-down card and the order of each
library. Gen-3's reader puts all of it in the input. It gives both hands' cards a slot each, and it
reads a face-down card's real face and the ability lines printed on it whoever controls the card. A model meant
to advise one player therefore learns from information that player never has.

### No output says how many cards from the top of a library a player knows

**After a scry, a look or a reveal, a player knows the top cards of a library, and gen-3 neither
records nor predicts it.** Gen-3's collector records what happens to cards, not what each player
knows about them. No output field can therefore say how many cards from the top of a library a
player has seen.

### Three outcome fields learn only from the times they happened

**`door_unlocked`, `entered_in_combat` and `became_blocked` take no value when no event names them,
so the model sees them only where they are 1.** An output field takes a silent value, 0 when no
event names it, only where the collector writes its event for every occurrence. Gen-3's records miss
some occurrences of these three:
- A Room spell's entry unlocks the half that was cast and writes no `door_unlocked`.
- Forge puts a creature into combat through `Dig`, `DigUntil`, `Meld`, `ChangeCombatants`, `Block`
  or a sneak cast, and writes no `entered_attacking` or `entered_blocking`.
- An attacker blocked by a creature put onto the battlefield blocking, or through `SwitchBlock` or
  `Camouflage`, gets no `became_blocked`.

A silent 0 on these fields would teach the model that the outcome did not happen in records where it
did. The collector fix is planned for the end of gen-3's implementation, but the gen-3 records
collected before it keep the gaps.

### Gen-3 leaves problems whose fix needs the cards converted again

**A keyword line's reminder text is filled by a Python copy of Forge's formatting, which can drift
from Forge.** The model reads a keyword line with values, such as `Ward:2` or
`TypeCycling:Basic:1 B`, through its reminder text with the values filled in. Forge fills that text
with one Java class per kind of keyword, about fifteen in all, and with its own cost formatting.
Gen-3's `src/effects/domain/keyword_formatting.py` re-implements those classes by hand. When Forge
changes one of them, the copy can fill a different text than Forge does, and nothing reports the
difference.

**Gen-3 keeps the copy because removing it changes every converted card.** Removing it means the
converter writes each printed keyword line's filled text into the card's provenance sidecar, the
file beside each converted card that lists its lines and the keys records name them by. That
changes the sidecars' digest, a hash over every sidecar of the conversion. A collection refuses to
add to a records directory collected against another digest, and `build-corpus` refuses such
records too. Gen-3's records collected before the change would therefore no longer build into a
corpus with the new conversion.

## Description of the experiment

### Training draws more of the records a weak output field needs, from the whole corpus

**When a field stays weak, training shows more of the records that supervise it.** The idea is
established as adaptive data mixing and adaptive loss weighting. Large language-model pretraining
stores its data as separate sources and draws from each by a mixing weight. Methods such as DoReMi
(Xie et al., [DoReMi: Optimizing Data Mixtures Speeds Up Language Model
Pretraining](https://arxiv.org/abs/2305.10429), NeurIPS 2023) raise a source's weight while its loss
stays high against a reference model. Multi-task training, of which the output fields are a case,
balances the tasks' loss weights instead, with methods such as GradNorm (Chen et al.,
[GradNorm](https://arxiv.org/abs/1711.02257), ICML 2018) and uncertainty weighting (Kendall, Gal and
Cipolla, [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115),
CVPR 2018).

**Four mechanisms are possible, from cheapest to most flexible.**

1. **A weight per field in the loss.** The trainer already computes every field's loss separately,
   so nothing on disk changes.
2. **A field factor in the weighted shuffle within a shard.** An epoch uses about a third of each
   shard's records, so this can raise a field's share about threefold and no further.
3. **Pools with mixing weights.** `build-corpus` writes the training stratum as several shard
   directories, one per pool, and files each record in the pool of the rarest field it supervises.
   Gen-2 already gives a legality record the rarest single mode on its board. The trainer draws each
   epoch's shards from the pools by weights a controller adjusts at each epoch boundary. A pool of a
   few hundred records can stay in memory and be mixed into every step.
4. **An index from field to records, possibly in a database.** A record would then count for every
   field it supervises, not only its rarest. The cost is reading scattered records across the whole
   corpus, where the trainer reads whole shards in sequence.

**Gen-3's rules index could supply the pools.** Gen-3's per-rule detectors give each record the
rules it exercises, which is the same index that sampling by field needs. The two would share the
choice between pools, an index and a database.

**Five constraints apply whichever mechanism is chosen.**

- **Calibration.** A field drawn at five times its natural share teaches the model that its outcome
  is five times as common. Each record's loss then needs an importance weight, its natural share
  divided by its drawn share. Loss weighting does not shift the base rates in the same way.
- **Memorization.** A rare field may rest on few distinct texts; poison comes from a handful of
  infect and toxic abilities. Raising its share repeats those texts. The ceiling has to be on how
  often each distinct text is seen per epoch, as gen-2's reuse cap is.
- **Reproducibility.** A gen-2 run repeats bit for bit from its seed. A controller driven by losses
  does not, because GPU arithmetic varies between runs. Changing the weights only at epoch
  boundaries, and logging them, lets a rerun replay them.
- **Weak against unlearnable.** A field can be weak because the snapshot lacks what decides it. The
  controller should boost a field only while its training loss keeps falling, and pass a field that
  stays flat to collection.
- **Validation stays out of the loop.** The controller reads training loss. If the card-disjoint
  validation steered training, it would stop being an unbiased measure.

**The controller computes each field's explained deviance as gen-3's control score does, on
training records.** Gen-3 scores each field that way on the card-disjoint stratum to judge
checkpoints, so the training side and checkpoint selection share their bookkeeping and keep their
data apart.

### The model sees the board from one seat

**Gen-4 shows each record from one player's seat and masks what that player cannot see.** The seat
is the point-of-view player, chosen independently of the player who acts in the record. The reader
replaces the other player's hand by a count of its cards, and drops the real faces of that player's
face-down cards with the lines printed on them. It may also mask that player's library top and the cards it
exiled face down. The records stay omniscient, so the change is in the reader alone and needs no
recollection.

**Three questions are open.**
- Which seat a record is shown from: fixed for a whole game, drawn for each example, or both.
- What an outcome the seat cannot predict does to the loss, such as a random discard from the other
  player's hand or the card a tutor finds.
- Which of the rules catalogue's 68 hidden-zone rules become unlearnable from one seat.

### An output says how many cards from the top of a library are known

**The collector records what each player knows of each library's top, and an output field predicts
how many cards from the top a player knows.** A scry, a look or a reveal adds to that count, and a
shuffle clears it.

### The three outcome fields take their silent 0 once every record comes from the fixed collector

**`door_unlocked`, `entered_in_combat` and `became_blocked` take a silent 0 on a resolution's effect
half in a corpus whose records all come from the collector that writes every occurrence of their
events.** A corpus that still holds records from before that fix keeps the three fields without a
silent value.

### The converter writes each keyword line's filled reminder text

**Gen-4's conversion writes each printed keyword line's reminder text as Forge fills it, into the
line's sidecar entry.** The Python copy of Forge's formatting then serves only the keywords granted
while a game runs, which no sidecar line describes. The change goes in before the cards are
converted for gen-4's first collection, so no collected record has to be given up for it.

## Result

Gen-4 has not run.

## Future plans

None yet.
