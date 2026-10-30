# Ability effect model — gen-4 (draft)

## TL;DR

Some of the effect model's output fields stay weak while the others improve, and training cannot
respond: each record's share of training is fixed when the corpus is built. Gen-4 plans training
that draws more of the records a weak field needs, from the whole corpus, with the mechanism and the
controller still to choose. Nothing has run.

## Background

Gen-4 is the fourth generation of the ability effect model. It follows gen-3, described in
[`2026-XX-XX-effect-model-gen3-design.md`](2026-XX-XX-effect-model-gen3-design.md) and called the
gen-3 record below. Gen-3 rebalances its training by rule of the game when its corpus is built, and
leaves training that reacts to a weak field to gen-4.

### Some output fields stay weak while the others improve

**A few fields with a real amount to learn barely moved between the first two epochs that scored
them.** Each output field is scored by its explained deviance: the share of the loss of the best
constant prediction that the model removes, where 0% is no better than always guessing the field's
average. The table reads gen-2's first arm, which predicts 46 output fields per entity
([gen-3 record](2026-XX-XX-effect-model-gen3-design.md#some-output-fields-stay-weak-while-the-others-improve)).

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

## Result

Gen-4 has not run.

## Future plans

None yet.
