# Protocol contract: `DraftWorkerMain` → supervisor (stdout sentinel)

Mirrors the existing forge-connector worker pattern (`PoolMain`,
`MatchWorkerMain`). The Java worker runs Forge's draft AI for all pod seats and
streams completed-draft transcripts; the Python supervisor completes each
record (deck build + score) and appends it to `drafts.jsonl`.

## Transport (FR-010, FR-011)

- The worker emits **one flushed line per completed draft** on stdout:

  ```
  <<DRAFT-EVENT-JSON>>{"draft_id":"…","boosters":[…],"seats":[{"agent":"forge-full"}…]}
  ```

  - Prefix sentinel: literal `<<DRAFT-EVENT-JSON>>`.
  - Suffix: compact, **newline-free** JSON transcript = `draft_id` + `boosters`
    (each `{set_code, picks}`) + per-seat `agent` ids. **No** `deck` /
    `deck_score` (the supervisor fills those).
  - `flush()` after each line.
- The worker's diagnostics and Forge's incidental logging go to **stderr**,
  piped to a log file; the supervisor ignores non-sentinel stdout.

## Supervisor processing (FR-005, FR-007, FR-010, FR-012)

1. Generate one `run_id` (UUID) at startup.
2. Read worker stdout line-by-line; keep only lines starting with the sentinel.
3. Defensively `json.loads` the suffix; **skip** anything that fails to parse
   (no crash). Forge stdout noise is silently dropped.
4. For each parsed transcript:
   - reconstruct each seat's full drafted pool from the boosters (FR-016),
   - build a 40-card deck per seat via `--build-method` (picker default / SA),
   - score the non-basic subset with the frozen scorer → `deck_score`
     (`null` + `deck=[]` on failed build),
   - assemble the complete record (`draft_id`, `run_id`, `timestamp`, `seats`
     with `agent`/`deck`/`deck_score`, `boosters`) and append one line to
     `drafts.jsonl`.

## Crash semantics (FR-011, SC-003)

- **Worker JVM crash**: supervisor restarts a fresh worker and continues toward
  `--n-drafts`; partial transcripts (no sentinel line emitted) are simply not
  produced.
- **Supervisor crash mid-record**: the in-flight draft is lost (acceptable at
  data-gen scale); already-appended records are unaffected.
- **Trailing partial line** in `drafts.jsonl`: tolerated by all readers.

## Agent degradation (FR-006)

- `forge-full`: Forge draft AI picks every card for the seat.
- `forge-r30` / `forge-r100`: 30% / 100% of that seat's picks replaced by a
  uniform-random legal pick from the seat's current pack. The sampled identifier
  is recorded verbatim in `seats[i].agent`.
