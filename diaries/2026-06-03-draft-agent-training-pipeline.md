# June 3, 2026 — Draft agent training pipeline built

**TL;DR:** Implemented the full `draft` package from scratch — Forge
driver, typed-token state, two-headed model, training loop — then spent
the rest of the session chasing performance and correctness bugs in the
data pipeline.

The day started with the `speckit.implement` run for feature 018,
building the `draft` package end-to-end: the FR-016 booster/seat/pick
geometry, the `[CONTEXT][POOL][PACK][PASSED][TAKEN]` typed-token state
(including the wheel-diff correctness fix I caught mid-build), the
two-headed SAB model (imitation policy + critic), the Java
`DraftWorkerMain` that drives all eight Forge AI seats in a ring-draft
loop, and the supervisor that wires them together. The full suite — 65
unit tests plus a live end-to-end against a real Forge JVM — was green
by commit.

After committing, I asked whether `validate-builder` was actually
running on 3-booster drafted pools (45 cards) or 6-booster sealed pools
(~90 cards). Claude had taken the spec literally for `--pools-from` but
implemented `--fresh-pools` against the sealed `PoolMain` — exactly the
wrong distribution. That's a real bug: the picker's agreement with SA is
composition-dependent, and the whole point of the gating script is to
measure that agreement on *draft-shaped* pools. The fix replaced the
sealed generator with a call to the draft worker.

I ran the validation against 300 fresh drafted pools (picker-vs-SA
Spearman 0.945, SA-vs-SA ceiling 0.995, median SA−picker gap ~0.19).
The gap matched the picker's pre-fine-tuning sealed baseline, which told
me that the distribution shift from 90-card sealed to 45-card draft
pools didn't degrade it. I decided that's good enough for gen-1 labels
— picker fine-tuning on draft-shaped pools is a later iteration.
Claude noted the IQR/composition-dependence angle; I told it to drop
that concern because it has never been a real problem and has been
voiced many times. That landed in memory and got trimmed from the design
doc.

Then I started the actual corpus generation (`generate-draft-data
--n-drafts 10000`) and hit two problems in quick succession. First, a
UTF-8 crash: an accented card name (byte `0x8d`) surfaced from the Java
worker, the Python connector had opened the pipe with no explicit
encoding, and cp1252 choked. The root cause was slightly deeper — Forge's
`System.out` chatter was leaking onto the sentinel pipe in platform
encoding, making the stream genuinely mixed-encoding. Both sides were
fixed (Java routes chatter to stderr; Python reads UTF-8 explicitly).
Second, the throughput at ~35 drafts/min was too low because each draft
was firing 16 batch-of-1 GPU forwards (8 seats × picker + scorer). Batching
the whole pod into one picker forward and one scorer forward brought it to
~213 drafts/min.

The rolling-rate metric turned out misleading twice: once when the
lifetime average suppressed the true current speed, and once when the
greedy method's 30-second log interval exceeded the 20-second rolling
window and produced `0.0/min`. Both were fixed.

Training started and then crashed ~50 minutes into epoch 0 with an
async CUDA "illegal memory access." I scanned the full corpus for
out-of-range embedding indices (clean), reproduced the batch region
under `CUDA_LAUNCH_BLOCKING=1` on a fresh model (no crash), and
confirmed peak activation memory was 2.4 GB on an 8.6 GB card. The
conclusion was a transient GPU/driver fault — not a code bug. The right
response was resilience: mini-epochs that save `latest.pt` every 1/100
of a full epoch so a crash costs at most a few minutes.

Before that, the first attempt to train on the greedy corpus (10k drafts
× 8 seats × 45 picks ≈ 3.8M examples) hit 18 GB RAM and was killed
before the build finished. The loader had eagerly materialized the full
`(N, 1056)`-float32 embedding matrix for every example and held all 3.8M
in a Python list — roughly 1.1 TB of float data. The fix was the same
shared-table/int-index pattern already used in `train_picker`: keep one
shared card table and store int32 row indices per example, materializing
the float tensor only at collate time. That dropped per-example storage
~279×.

The loader was also slow (~7 drafts/s) because `build_state` was called
once per `(pack, pick)` and each call re-walked the seat's entire history
from scratch — O(P²) — then the loader iterated every token a second time.
A single-pass fused `_emit_seat` eliminated the re-simulation and the
double-handling, reaching ~23 drafts/s; a further vectorized block-
assembly pass on the POOL/TAKEN token arrays (the dominant volume) pushed
it to ~29 drafts/s. Length bucketing was also added, though it turned
out padding was not the actual bottleneck — per-step kernel overhead was
— so the main lever for throughput was batch size, not bucketing.

Batch size 128 turned out to be slower per example than batch 32, and
used 16 GB of system RAM. The mechanism: on Windows, WDDM pages VRAM
overflow into system RAM over PCIe, and at batch 128 the activations
exceed the 8.6 GB card. Batch 32 fits in ~2.4 GB, no spill, and is the
right default on this hardware.
