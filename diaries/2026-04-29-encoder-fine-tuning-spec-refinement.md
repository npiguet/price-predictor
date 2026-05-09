# April 29, 2026 — Encoder fine-tuning spec refinement

**TL;DR:** A full day of spec work on feature 015 (encoder fine-tuning),
running the speckit pipeline end-to-end from clarify through analyze.
One notable course correction: Claude had the checkpoint resume semantics
backwards, and I caught it.

The day started with clarifying the spec for 015 — the feature that will
let `train-scorer` fine-tune the sealed encoder in Phase B instead of
keeping it frozen. Most of the questions Claude surfaced were about edge
cases in the CLI flag interactions: what happens if you pass
`--scorer-checkpoint` without `--clean` to `encode-cards` (I said don't
worry about it — a fully mixed cache isn't likely), whether to bother
migrating pre-feature Phase A checkpoints after the Adam-to-AdamW switch
(I said no, I'm retraining from scratch), and what validation cadence
to use for `embedding_drift` logging (end-of-epoch, which I agreed with).

After two separate clarify sessions and a planning pass, I ran
`speckit.analyze`. It found 11 issues across the artifacts. Two were
HIGH: a missing task to flip the `lr` default from `1e-3` to `1e-5`
(C1), and an ambiguity about where Phase B `--resume` gets the encoder
architecture config from (I1). For I1 I chose Option A — save
`encoder_config` in the Phase B checkpoint itself, making it
self-contained, rather than relying on the price-predictor's `latest.pt`
being stable across the run.

The more interesting correction came from Claude's proposed fix for I3,
the resume-precedence issue. Claude wrote that "the resumed `train_config`
is informational only — CLI-resolved values always drive the current run
unconditionally." I pushed back: the whole point of resume is that if I
restart training without specifying parameters, it should pick up where
it left off with the same settings. So the correct precedence is explicit
CLI flag > resumed `train_config` > argparse default — not the other way
around. That required distinguishing which flags the user explicitly
passed from which ones argparse filled in with defaults, which in turn
required registering resumable flags with `default=None` sentinel values
and doing a late resolution step.

The third clarify session, run after the analyze pass, surfaced the most
consequential design change of the day. Originally the spec allowed
`--resume` to cross phase boundaries — a user could resume a Phase A
checkpoint with `--embedding-lr` to bootstrap Phase B. I decided to
reject this entirely: resume stays within the same phase, no override
flag. The consequence was that Phase B needed a new `--scorer-checkpoint`
flag (analogous to `--encoder-checkpoint`) specifically for weight
transfer across the Phase A → Phase B boundary. Claude pointed out that
starting Phase B with a randomly-initialized scorer and immediately
fine-tuning the encoder against that noise would push the encoder in
arbitrary directions during early epochs, and I agreed the phase-lock
was cleaner.

In between the spec work, I also improved the `/review` command. I asked
Claude to add dead-code detection, low-value test detection, and simple
performance problems. Claude noted that the new sections were much more
detailed than the original bullets, and asked whether that imbalance
would cause the agent to spend disproportionate effort on the new areas.
I agreed that was a real risk and asked it to expand the original bullets
with the same pattern of concrete examples. I also decided the review
should run in plan mode — the agent cannot make changes until the user
approves — which is structurally stronger than any prompt-level emphasis.
