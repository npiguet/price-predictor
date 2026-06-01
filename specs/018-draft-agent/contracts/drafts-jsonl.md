# Data contract: `output/draft/drafts.jsonl`

One self-contained JSON record per line, append-only (FR-013). Card names are
Forge canonical names. Readers MUST tolerate a trailing partial final line
(JVM-crash-mid-write recovery).

## Record schema

```json
{
  "draft_id": "<uuid>",
  "run_id": "<uuid>",
  "timestamp": "<ISO 8601 UTC>",
  "seats": [
    {"agent": "forge-full", "deck": ["...40 names..."], "deck_score": 12.34}
  ],
  "boosters": [
    {"set_code": "BLB", "picks": ["...P names in pick-order..."]}
  ]
}
```

(One element shown per array; a real record has `pod_size` seats and
`pod_size × packs` boosters.)

## Field semantics

| Field | Type | Meaning |
|---|---|---|
| `draft_id` | string (UUID) | Groups this record; train/val split unit (FR-013, FR-035). |
| `run_id` | string (UUID) | One per `generate-draft-data` invocation, stamped on every record (FR-005). |
| `timestamp` | string (ISO 8601 UTC) | Draft completion time. |
| `seats[i].agent` | string | Per-seat agent id sampled from `--agent-mix` (FR-006/FR-014). |
| `seats[i].deck` | array<string> (len 40) \| `[]` | Built deck incl. basics; `[]` on failed build (FR-014). |
| `seats[i].deck_score` | number \| `null` | Scorer scalar over non-basics; `null` on failed build (FR-014). |
| `boosters[k].set_code` | string | Per-booster set (Chaos-draft-capable) (FR-015). |
| `boosters[k].picks` | array<string> (len `pack_size`) | Cards in pick order; fully drained; multiset = initial pack contents (FR-015). |

## Geometry conventions (self-contained reconstruction — FR-016)

Derived sizes: `pod_size = len(seats)`, `packs = len(boosters)/len(seats)`,
`pack_size P = len(boosters[0].picks)`.

For `boosters[k]`:
- `pack_number = floor(k / pod_size) + 1`
- `opening_seat = k mod pod_size`
- the pick at position `j` was made by seat
  `(opening_seat + j · dir_p) mod pod_size`, where `dir_p = +1` for packs 1 & 3
  (pass left) and `dir_p = −1` for pack 2 (pass right).

Inverse (loader, FR-031), for a target `(seat s, pack p, pick i)`:
- `s_open = (s − (i − 1) · dir_p) mod pod_size`
- `k = (p − 1) · pod_size + s_open`
- offset `j = i − 1`; the seat's legal actions are `boosters[k].picks[j:]`,
  the taken card is `boosters[k].picks[j]`.

## Reader rules

- Tolerate and skip a trailing partial line (FR-013).
- `--resume` counts complete records toward `--n-drafts` (FR-012).
- A record round-trips: any seat's POOL/PACK/PASSED/TAKEN at any pick is
  reconstructable from the record alone (SC-002).
