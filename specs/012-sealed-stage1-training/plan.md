# Implementation Plan: Stage 1 Training — Legal Pick Gate

**Branch**: `012-sealed-stage1-training` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/012-sealed-stage1-training/spec.md`

## Summary

Implement Stage 1 of the sealed deck builder training curriculum: a PPO training loop that teaches the pool-level transformer to make 40 consecutive legal picks from a 96-card sealed pool. New domain components (pool transformer, episode runner, replay buffer, PPO trainer) are added to the existing `sealed` Python module, and two new CLI subcommands (`train` and `sample`) are wired into the existing `sealed` CLI.

## Technical Context

**Language/Version**: Python 3.14+
**Primary Dependencies**: PyTorch (existing), numpy (existing), pytest (existing)
**Storage**: `.npz` embedding files (existing), `pools.txt` flat text files (existing), `.pt` checkpoint files (new)
**Testing**: pytest — unit tests with miniaturized tensors (d_model=8, n_layers=1, pool_size=4); integration test runs a 2-batch loop end-to-end
**Target Platform**: Local workstation (CPU or GPU; no deployment)
**Project Type**: CLI tool (extension of existing `python -m sealed` module)
**Performance Goals**: No throughput targets; training speed is secondary to correctness
**Constraints**: Encoder weights are frozen during Stage 1; the pool transformer is the only trainable component
**Scale/Scope**: ~10,000 pools per dataset; replay buffer ≤ 1,000 episodes; batch size 32 (default)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Fast Automated Tests | ✅ Pass | Unit tests use d_model=8, n_layers=1, pool_size=4; run in milliseconds. Integration test uses 2 episodes, tiny .npz fixtures. |
| II. Simplicity First | ✅ Pass | PPO complexity is unavoidable (spec requirement). No critic head added beyond what Stage 1 needs (running-mean baseline only). No speculative Stage 2/3 abstractions. |
| III. Data Integrity | ✅ Pass | pools.txt validated at startup (empty = abort). Missing .npz = abort with card name. Checkpoint writes are atomic (temp-file + rename pattern from existing EmbeddingStore). |
| IV. Domain-Driven Design | ✅ Pass | Pool transformer and PPO logic live in `sealed/domain/`. Orchestration in `sealed/application/`. File I/O in `sealed/infrastructure/`. No PyTorch imports in application layer (injected via constructor). |
| V. Forge Interoperability | N/A | Training feature; no Forge integration, no Java stub needed. |
| VI. Documentation | ✅ Pass | quickstart.md covers train/sample commands, monitoring, resume, and rollback. research.md documents PPO design rationale. README update included as a deliverable task. |

**Post-Design Re-check**: All gates pass. No violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/012-sealed-stage1-training/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   └── cli-train-sample.md
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code

```text
src/sealed/
├── domain/
│   ├── card_encoder.py          (existing — unchanged)
│   ├── card_embedding_port.py   (NEW) CardEmbeddingPort Protocol — satisfied structurally by EmbeddingStore
│   ├── pool_transformer.py      (NEW) PoolTransformerModel, PoolTransformerConfig
│   ├── episode_runner.py        (NEW) EpisodeRunner — runs one episode, handles masking + termination
│   ├── replay_buffer.py         (NEW) ReplayBuffer, Episode dataclass
│   └── ppo_trainer.py           (NEW) PPOTrainer — computes PPO loss, updates weights, monitors KL
├── application/
│   ├── encode_cards.py          (existing — unchanged)
│   ├── generate_pools.py        (existing — unchanged)
│   ├── train_stage1.py          (NEW) TrainStage1UseCase — training loop orchestration
│   └── sample_stage1.py         (NEW) SampleStage1UseCase — generates N human-readable pick sequences
└── infrastructure/
    ├── embedding_store.py       (existing — unchanged)
    ├── pool_connector.py        (existing — unchanged)
    ├── pool_loader.py           (NEW) PoolLoader — reads pools.txt, assembles (96, 516) tensors from .npz
    ├── pool_model_store.py      (NEW) PoolModelStore — atomic checkpoint save/load
    └── cli.py                   (MODIFIED) add train + sample subcommands

tests/
├── unit/sealed/
│   ├── domain/
│   │   ├── test_pool_transformer.py    (NEW)
│   │   ├── test_episode_runner.py      (NEW)
│   │   ├── test_replay_buffer.py       (NEW)
│   │   └── test_ppo_trainer.py         (NEW)
│   ├── application/
│   │   ├── test_train_stage1.py        (NEW)
│   │   └── test_sample_stage1.py       (NEW)
│   └── infrastructure/
│       ├── test_pool_loader.py         (NEW)
│       ├── test_pool_model_store.py    (NEW)
│       ├── test_cli_sealed_train.py    (NEW)
│       └── test_cli_sealed_sample.py   (NEW)
└── integration/sealed/
    ├── test_encode_cards_integration.py  (existing — unchanged)
    └── test_train_stage1_integration.py  (NEW)
```

**Structure Decision**: Single-project layout extending the existing `src/sealed/` DDD structure. Follows the same `domain / application / infrastructure` layering as `price_predictor` and the existing `sealed` module.

## Deliverables

### Domain layer (`src/sealed/domain/`)

**`pool_transformer.py`**
- `PoolTransformerConfig` dataclass: `n_layers=8`, `n_heads=8`, `d_model=516`, `ff_dim=2048`, `n_slots=96`, `card_embed_dim=512`, `dropout=0.1`
- `PoolTransformerModel(nn.Module)`:
  - `nn.TransformerEncoder` with `batch_first=True`, no positional encoding
  - Output linear head: `nn.Linear(516, 96)` → per-slot logits
  - `forward(slot_features: Tensor[batch, 96, 516]) → logits: Tensor[batch, 96]`
  - No masking applied to logits — the model samples from the full 96-slot distribution in Stage 1

**`card_embedding_port.py`** *(NEW — domain protocol)*
- `CardEmbeddingPort` (`typing.Protocol`): `get_embedding(card_name: str) → np.ndarray`
  - Satisfied structurally by `EmbeddingStore` (infrastructure) — no cross-layer import required
  - Keeps `EpisodeRunner` (domain) free of any infrastructure dependency per constitution IV

**`episode_runner.py`**
- `EpisodeRunner`:
  - `run(pool_names: str, card_port: CardEmbeddingPort, model, rng_seed) → Episode`
    - `pool_names` is a semicolon-separated card name string (same format as pools.txt line)
  - Assembles the 96-slot base tensors from the card port + basic land embeddings
  - At each step: applies the step's shuffle seed to permute the non-basic-land slots into a shuffled input tensor; slot flags reflect current picked state
  - Model outputs logits over the 96 shuffled input positions (no masking)
  - Samples a shuffled input position; translates it to the corresponding **pool index** via the inverse permutation
  - Legality check: has this pool index been picked before? If yes → terminate
  - Records the **pool index** (not the shuffled input position) in `actions`, and the log-probability of the sampled input position in `log_probs`
  - Computes reward using `best_run` (passed in, not stored in runner)

**`replay_buffer.py`**
- `Episode` dataclass: `pool_names: str` *(semicolon-separated card names, same format as pools.txt line)*, `shuffle_seeds: np.ndarray[40, int32]`, `actions: np.ndarray[n, int32]`, `log_probs: np.ndarray[n, float32]`, `reward: float`
- `ReplayBuffer`:
  - `max_size: int` (default 1000)
  - `append(episode: Episode) → None` — FIFO eviction
  - `sample(n: int) → list[Episode]` — random sample without replacement (or all if `n > len`)
  - `__len__`, `to_list() → list[Episode]`, `from_list(episodes: list[Episode]) → ReplayBuffer`

**`ppo_trainer.py`**
- `PPOTrainer`:
  - `__init__(model, optimizer, clip_eps=0.2, kl_warn_threshold=1.5)`
  - `update(episodes: list[Episode], pool_loader, best_run) → TrainBatchResult`
  - Reconstructs episode states from stored seeds: for each step, applies the stored shuffle seed to recover the input-position → pool-index permutation, then looks up the shuffled input position of the stored pool-index action to compute the new log-prob under the current policy
  - Computes per-step importance ratios; PPO loss (clipped surrogate); backward + step
  - KL divergence monitoring: emits `print("[warn] KL divergence ...")` to stdout when exceeded
  - `reward_baseline`: EMA (decay 0.99), updated per episode processed
- `TrainBatchResult` dataclass: `mean_reward: float`, `episode_runs: list[int]`, `kl_warnings: int`

### Application layer (`src/sealed/application/`)

**`train_stage1.py`**
- `TrainStage1UseCase`:
  - `execute(pools_path, cards_path, model_path, batch_size, set_code) → None`
  - Startup validation: pools.txt non-empty, all card embeddings present; creates model-path dir; loads or initializes checkpoint
  - Main loop: collect `batch_size` episodes → add to replay buffer → PPO update → print summary → checkpoint
  - Consecutive-successes tracking; halts and reports when ≥ 100
  - Saves timestamped checkpoint every 1000 episodes

**`sample_stage1.py`**
- `SampleStage1UseCase`:
  - `execute(pools_path, cards_path, model_path, n_samples) → None`
  - Loads checkpoint; runs N episodes with `model.eval()` + `torch.no_grad()`; prints formatted pick sequence for each

### Infrastructure layer (`src/sealed/infrastructure/`)

**`pool_loader.py`**
- `PoolLoader`:
  - `load_pools(pools_path: Path) → list[str]` — reads pools.txt line by line; each element is a semicolon-separated string of card names (one pool per line); raises `ValueError` if empty
  - `assemble_pool_tensor(pool_names: str, cards_path: Path) → Tensor[96, 516]` — `pool_names` is a semicolon-separated card name string; loads .npz files, appends basic land embeddings (Plains/Island/Swamp/Mountain/Forest/Wastes), zero-pads to 96 slots, concatenates all four flag fields at their initial values (all zero); raises `FileNotFoundError` identifying missing card name
  - Basic land embeddings are loaded from cards-path by name (e.g., `Plains.npz`, `Island.npz`, etc.)
  - Note: PoolLoader produces the **initial** base tensor only (all flags at zero). EpisodeRunner owns per-step flag mutation.

**`pool_model_store.py`**
- `PoolModelStore`:
  - `save(path: Path, model, optimizer, training_state, replay_buffer) → None` — atomic (temp + rename)
  - `load(path: Path) → CheckpointData` — returns model state dicts + training state + replay buffer
  - `save_timestamped(base_path: Path, ...) → Path` — writes to `base_path.parent/checkpoints/{ISO8601}.pt`

**`cli.py` (modified)**
- Add `train` subparser: `--stage`, `--set`, `--pools-path`, `--cards-path`, `--model-path`, `--batch-size`
- Add `sample` subparser: `--set`, `--pools-path`, `--cards-path`, `--model-path`, `--n-samples`
- `run_train(args) → int` — delegates to `TrainStage1UseCase`
- `run_sample(args) → int` — delegates to `SampleStage1UseCase`
- Extend `main()` dispatch

### Tests

**Unit tests** (miniaturized parameters: `n_slots=4`, `d_model=8`, `n_layers=1`, `n_heads=2`, `card_embed_dim=8`):

| Test file | Key scenarios |
|-----------|--------------|
| `test_pool_transformer.py` | Forward pass shape, output logits cover all 96 slots with no masking, log-probs sum to 0 |
| `test_episode_runner.py` | Legal episode completes 40 picks, illegal pick terminates early, reward formula correct, basic land slot increments count correctly; same pool index at two different shuffled input positions is correctly identified as a duplicate (illegal), two different pool indices at the same shuffled input position across steps are correctly identified as legal |
| `test_replay_buffer.py` | FIFO eviction at max_size, append/sample, serialization round-trip |
| `test_ppo_trainer.py` | KL divergence warning fires above threshold, PPO loss gradient flows, reward baseline updates |
| `test_train_stage1.py` | Startup: empty pools file → ValueError, missing .npz → FileNotFoundError, model-path dir created automatically |
| `test_sample_stage1.py` | Output format: SUCCESS and ILLEGAL PICK cases, n_samples controls count |
| `test_pool_loader.py` | Empty pools.txt → ValueError, missing card .npz → FileNotFoundError with card name, correct 516-dim tensor shape |
| `test_pool_model_store.py` | Save + load round-trip preserves all fields, atomic write (temp file absent after save), timestamped checkpoint naming |
| `test_cli_sealed_train.py` | `--stage 1` dispatches to train use case, `--batch-size` passed through, unknown `--stage` exits with code 1 |
| `test_cli_sealed_sample.py` | `--n-samples` passed through, missing checkpoint exits with code 2 |

**Integration test** (`test_train_stage1_integration.py`):
- Create a tiny fixture: 4 fake card .npz files (8-dim, NOT 512-dim — use miniaturized config), 1-line pools.txt with 4 cards
- Run `TrainStage1UseCase.execute(batch_size=2, ...)` for 2 batches
- Assert: `latest.pt` checkpoint written, `best_run >= 1`, no exception raised
- Optionally assert completion fires when a fixture model is pre-configured to always pick legally (inject mock episode runner)
