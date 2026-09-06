# Contract: provenance sidecar

**Feature**: `023-ability-effect-model`
Authority: [`../../2026-09-05-ability-effect-model.md`](../../2026-09-05-ability-effect-model.md) § Ability identity.

The sidecar is the join between runtime Forge trait objects and converted ability lines. It is the
single most load-bearing contract in the feature: every effect record names an ability through it, and
every cache row is aligned by it.

## Why converted-line ordinals cannot be the key

The converter's bracket numbers run one per-face counter across five mixed line kinds; keyword-derived
lines are emitted first in the file but appended last in Forge's runtime lists; and the converter
merges, deduplicates, and splits abilities relative to the runtime objects. An ordinal therefore names
different things on the two sides. The stable key is **printed provenance**.

## Key

```
ProvenanceKey = (script_file, face, trait_kind, index_within_kind)
```

- `script_file` includes its tree **and that tree's own layout**, which differ: `cardsfolder/` is
  letter-keyed and `tokenscripts/` is flat, so Ajani's Pridemate is
  `cardsfolder/a/ajanis_pridemate.txt` on one side and `tokenscripts/ajanis_pridemate.txt` on the
  other. The key is the path as written, produced identically by the Java writer and the Python
  reader — a mismatch is the fail-loudly case below.
- `index_within_kind` is the index within that kind's slice of the face's raw trait list, not a
  global ordinal.

## File

`<name>.provenance.json`, written beside the converted `<name>.txt` — the same pairing convention the
`.npz` embedding files already use. **Writing it never alters the converted text.**

```jsonc
{
  "card": "Ajani's Pridemate",
  "script_file": "cardsfolder/a/ajanis_pridemate.txt",
  "lines": [
    {
      "line_index": 3,                       // index into the converted file's rendered lines
      "line_kind": "triggered",
      "provenance": [                        // several keys where the line merged several traits
        { "face": 0, "trait_kind": "trigger", "index_within_kind": 0 }
      ],
      "sub_ability_links": [                 // index paths below the trait
        { "path": [0], "label": "DBPutCounter" }
      ],
      "script_api_type": "PutCounter",
      "script_param_keys": ["Defined", "CounterType", "CounterNum"],
      "script_text": "DB$ PutCounter | Defined$ Self | CounterType$ P1P1 | CounterNum$ 1",
      "role_spans": [
        { "start": 0, "end": 34, "role": "trigger-condition" },
        { "start": 35, "end": 71, "role": "effect" }
      ]
    }
  ],
  "dropped_keys": [                          // traits that map to no rendered line
    { "face": 0, "trait_kind": "static", "index_within_kind": 2 }
  ]
}
```

`dropped_keys` exists because the collector computes keys from runtime trait accessors, never from the
sidecar. A trait the converter deduplicated away is still live at runtime and still produces a key, so
without this list an expected dedup and a genuine corpus/sidecar mismatch would look identical at the
join — and the mismatch is the one condition that must fail loudly.

## Rules

| Rule | Consequence |
|---|---|
| A rendered line merged from several runtime traits carries **several** provenance keys | the join is many-to-one, never assumed one-to-one |
| A trait whose line was deduplicated away maps to **no** line, and its key is listed in `dropped_keys` | expected, not an error. A record naming it is kept, and the record simply has no line to join to — it still supervises through its state and payload |
| A record names a provenance key that is in neither `lines` nor `dropped_keys` | fail loudly — the sidecar does not describe the card the record was collected against, i.e. a reconversion between collection and training |
| An event attributed to a sub-ability link absent from `sub_ability_links` falls back to the **root line** | attribution never drops an event |
| `script_text` is the stage-four primary encoding surface | every command that encodes reads it here and needs no Forge-cardsfolder path of its own |
| `role_spans` are character ranges over the **converted prose** | roles are `cost` \| `effect` \| `trigger-condition` \| `target-spec` |
| Row `i` of the ability cache corresponds to `lines[i]` | rendered lines for a converted tree, script lines for the variant tree |

## Runtime-side key computation

Keys come from the trait accessors, verified present in Forge 2.0.15-SNAPSHOT
(`forge-game/src/main/java/forge/game/CardTraitBase.java`, and `SpellAbility` for the last):

| Accessor | Use |
|---|---|
| `getCardState()` | the printed face |
| `isIntrinsic()` | printed vs granted |
| `getKeyword()` | the originating keyword, where the trait came from one |
| `getOriginalHost()` | the card whose state hosts the trait. **For a granted trait this is the recipient, not the donor** — `CardTraitBase.getOriginalHost()` returns `getCardState().getCard()` — so it must not be used to key granted abilities; use the grantor accessors below |
| `isCopiedTrait()` | copy detection |
| `getGrantorStatic()` | **`SpellAbility` only**, and does not survive copies |

Fallback chain, in order:

1. **Granted abilities** resolve through the grantor accessors to the donor card's printed line.
2. **Copied abilities** resolve through the original-ability back-reference (set only for copies).
3. **Copy-spell effects** (Fork, Reverberate) carry only a copied flag, so they resolve through the
   stack object's source card.

## Trees

| Tree | Source | Sidecar written by |
|---|---|---|
| `cardsfolder` | `../forge/forge-gui/res/cardsfolder/` → `output/cardsfolder/` | `price_predictor convert` |
| `tokenscripts` | `../forge/forge-gui/res/tokenscripts/` → `output/tokenscripts/` | `price_predictor convert` |
| `variant-scripts` | perturbed scripts → `output/effects/variant-scripts/` | `effects collect-variants` (stage four) |

Token scripts stay out of `output/cardsfolder/` because converted token and card filenames collide and
the sealed pipeline treats that tree as its card corpus.

## Rejected alternative

**Normalized-text matching** between runtime description text and converted lines was considered as a
no-converter-change fallback and rejected: it silently fails exactly where the converter merged or
deduplicated lines, which is precisely the case the sidecar exists to record.
