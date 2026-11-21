# Training runs

Rules to respect when running or changing the three trainers, reading an anchor margin, or pairing a corpus with a checkpoint. This file does not describe the trainers: flags are in `--help`, losses and schedules in the specs and design records listed in `src/draft/CLAUDE.md`.

## Shared by the trainers

- The critic's target standardization mean and std are stored in the checkpoint and reused: inference de-standardizes with them, and the RL trainer keeps the critic in the reference checkpoint's reward space. They are never recomputed downstream.

The two offline trainers (`train-draft-agent`, `train-draft-agent-rl`) only; the online trainer has no `--resume`, no architecture flags and no validation split, and its patience rules are in its own section:

- Architecture flags together with `--resume` or `--checkpoint` fail fast; the architecture comes from the checkpoint.
- Train and validation splits are draft-disjoint.
- `--lr-decay-patience` and `--patience` share one strict-best counter. `--lr-decay-patience` must be smaller than `--patience`, and early stop only fires once the LR floor is reached. The decay position is checkpointed so `--resume` continues annealing; an explicit `--lr` override restarts it.

## `train-draft-agent` (gen-1)

- The policy imitates only the seats whitelisted by `--imitation-agents` (default `forge-full`). The critic learns from every non-failed seat.

## `train-draft-agent-rl` (gen-2)

- `--checkpoint` warm-starts actor and critic and is the KL anchor. It is exclusive with `--resume`.
- `--drafts-path` is the on-policy corpus and the sole source of the policy gradient. `--critic-corpus` adds off-policy corpora for the critic only.
- `--learner-agents` is required: the labels whose seats feed the policy gradient. All non-failed seats feed the critic.
- `--rollout-temperature` is required and has no default. It must match the `generate-draft-data --temperature` that produced the corpus, because every policy distribution in the loss is evaluated at it.
- Pairing a corpus with the checkpoint that generated it is the operator's responsibility. No provenance is stored. The only safeguard is a one-time behaviour-anomaly summary that warns and never aborts.
- Best-checkpoint and early stop use the held-out RL objective as an in-run guard only. Strength across generations is judged by the yardstick below.
- `rl_metadata` makes a checkpoint self-describing for the next cycle. No encoder weights are saved.

## The yardstick and the self-play loop

These are operator runbooks over existing commands, not code:

1. Generate a corpus with the current checkpoint and `generate-draft-data --pick-mode sample`.
2. Fine-tune on it with `train-draft-agent-rl`.
3. Evaluate with one greedy `generate-draft-data --pick-mode argmax` run on a fixed mix with the agents co-seated, then `analyze-generated-decks --agent <each>` for per-agent mean `deck_score`.
4. Promotion is a manual judgment.

Gen-3 uses the same yardstick and promotion.

## `train-draft-agent-online` (gen-3)

Shape of the loop: each round one resident Forge worker drafts `--drafts-per-round` pods with the learner's seats piloted by the live policy, every deck is built and scored with the frozen scorer, one pass of `−mean(A·logπ_T(a|s))` is taken over the learner's picks, and the batch is discarded. There is no critic term, GAE, KL anchor, entropy bonus, validation split or `--resume`. The operator tunes `--lr`, `--agent-temp` and `--drafts-per-round`.

Agent wiring:

- Exactly one `--learner LABEL=PATH`; a bare `PATH` is rejected. `--frozen LABEL=PATH` adds untrained references. `--anchor` names the frozen label the margin is measured against and is fixed for the run.
- `--agent-temp` must name the learner with a positive value. Sampling is the learner's sole exploration mechanism, and the same value is the temperature the loss evaluates the policy at, so one map keeps rollout and loss in lockstep.
- Frozen labels default to argmax. Sampling the frozen field would not merely lower its scores: a draft allocates one fixed pool of cards, so a card a frozen agent wrongly declines flows to its podmates and hands the learner cards a properly playing agent would have kept.
- A Forge built-in cannot be given a temperature; naming one is a startup error.
- The trainer builds all pick services itself and passes them through `AgentRegistry.build(..., BuildOptions(preloaded=...))` with `agent_checkpoints` empty, so `BuildOptions.pick_mode` never applies.
- Every generated pod carries at least one learner seat, through `-Ddraft.required.agent`. The fat JAR must be rebuilt before the first gen-3 run.

Reading the margin:

- The margin opens negative. The learner carries a sampling handicap the anchor does not, and round 0 measures that offset exactly.
- The margin is field-relative. An improving learner also takes cards its podmates would have had; denial is draft strategy and the reward is itself pod-relative, so the margin is never discounted for it. It is reported with the field it was measured against.
- Margins do not compare across a change of field: a different `--mix` or `--agent-temp`.
- The two KLs answer different questions. A per-round `KL(π_k‖π_{k+1})` near zero does not show the policy is stationary, because small steps in a consistent direction accumulate; only the distance from `π_0` shows that.
- `π_0` is the learner's own warm start, never the anchor.
- A degenerate round (fewer than two surviving learner rewards, or reward std below `1e-8`) takes no optimizer step and logs `skipped (no signal)`.

Checkpoints and run control:

- `latest.pt` every round, `best_{timestamp}.pt` on each new best margin, `{timestamp}.pt` every `--snapshot-every` rounds and at run end. The critic head is carried through untrained.
- No best is recorded until the sliding window holds a full `--anchor-window` drafts, so a partly filled window cannot pin the run's best.
- The recorded best is a maximum over a correlated series and reads optimistically. It nominates a checkpoint; the yardstick measures it. The margin lags the policy by about half the window, so the better weights may precede it, which is why periodic snapshots stay useful.
- Optimizer state and the LR schedule are continuous across rounds.
- `--patience` and `--lr-decay-patience` are off by default and share one stall counter that starts when the window fills. Startup rejects either when it does not exceed `ceil(anchor_window / drafts_per_round)`, and rejects `lr_decay_patience >= patience`.
- `lr_decay_count` in `rl_metadata` is provenance only and never restored.
