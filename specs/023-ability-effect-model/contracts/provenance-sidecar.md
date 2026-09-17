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
| A trait whose text a rendered line carries — merged, deduplicated, a secondary of a pair, keyword-derived, or a Class level — is claimed by that line | the join is many-to-one; `dropped_keys` holds only traits with no text, such as the implicit permanent spell on every permanent |
| A record whose acting keys all sit in `dropped_keys` (or name an unconverted script) | `build-corpus` refuses it as `no-acting-text`; on the board, the surface builder emits no token for such a key. A runtime-only key is neither: it keeps the record and gets no token |
| A key names a script file with **no sidecar at all** under its tree | not an error: the converted corpus never held that card, and the ability contributes no text. Token entities do this — the collector files a token's provenance as `cardsfolder/<letter>/<sanitized token name>.txt` while tokens convert into `output/tokenscripts/` under Forge's own script filenames, so the path resolves to nothing. Readers raise `UnconvertedScript` and count it; `SidecarCache.line_for` returns None. Distinct from the row below, where the sidecar exists and disagrees |
| A record names a provenance key **inside** a declared `(face, trait_kind)` range that is in neither `lines` nor `dropped_keys` | fail loudly — the sidecar does not describe the card the record was collected against, i.e. a reconversion between collection and training. `dropped_keys` is declared-minus-claimed, so every index the converter parsed sits in one of the two lists and a gap between them can only come from a different trait list |
| A key whose face or trait kind the sidecar never declared, or whose index is past the highest it declared for that pair | runtime-only: a trait Forge attaches to a live card — level up, bestow, scavenge; a trigger granted by another card's static, keyed to the granting card's script; a disguise creature's face-down face. No line, the record is kept, and the surface builder emits no token |
| A record whose `ability` is empty and whose `ability_unresolved` is `engine_effect` | expected, not an error, and **not** a `dropped_keys` case: the acting card is engine-built (The Monarch, The Initiative, dungeons, emblems, speed) and has no script file in any tree, so there is no sidecar to drop from. A sentinel `script_file` would turn that honest answer into a hard join failure, inverting the fail-loudly rule above |
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
| `getTrigger()` | the trigger a wrapped or `Execute$`-SVar ability belongs to |
| `getReplacementEffect()` | the replacement effect an ability belongs to |
| `getRootAbility()` | the head of a sub-ability chain |
| `getOriginalMapParams()` | the structural fingerprint a `copy()` preserves, since `putParam` writes only `mapParams` |
| `Card.getEffectSourceAbility()` | the ability that created an effect card, which owns its printed line |

Fallback chain, in order. Each step is tried only after the ones above it miss, and the structural
match refuses to answer when it is ambiguous — **no key beats a wrong key**:

1. **Unwrap** a triggered ability's wrapper, then a sub-ability to its root.
2. **Trigger- and replacement-borne abilities** key to the trait that owns them (`getTrigger()`,
   `getReplacementEffect()`), not to the `Execute$` SVar ability, which is a member of no trait slice.
3. **Keyword-derived traits** key to their keyword's ordinal (see below).
4. **The trait where it stands**, located by identity in its own `CardState` slice.
5. **The trait where it stands**, located by *structure*: the unique member of the slice with the same
   class and the same `getOriginalMapParams()`. This is what reaches an alternative-cost or
   extra-keyword-cost copy, which carries no back-reference at all.
6. **Granted abilities** resolve through the grantor accessors to the donor card's printed line.
7. **Copied abilities** resolve through the original-ability back-reference.
8. **Effect cards** resolve through `Card.getEffectSourceAbility()` to the ability that created them.
9. Give up, and record *why* in the record's `ability_unresolved`.

Step 4 alone was the whole chain's identity test in stage one, and it fails for every activated ability
the stack hands back: `MagicStack.add` replaces a non-mana activated ability with a fresh copy before
pushing it, so identity finds nothing and the key is null. Mana abilities return before that copy,
which is why land mana abilities keyed and Fountain of Youth did not.

### Keyword ordinals

`index_within_kind` for `trait_kind = keyword` is the position of the keyword's printed text in the
trait's own `CardState.getIntrinsicKeywords()` **sorted by that text**. Not the position in
`getKeywords()`: `KeywordCollection` is a `MultimapBuilder.hashKeys()` over the `Keyword` enum, whose
`hashCode()` is the identity hash, so that order differs between the convert JVM and the collect JVM —
an ordinal neither side could reproduce. Restricting it to intrinsics also correctly refuses a *granted*
keyword, which names no printed line on the card that received it.

Both sides compute it from one shared function, `ProvenanceKey.keywordIndex`, the same discipline
`faceIndex`/`faceOrder` already enforce for faces. The two are coupled: the moment the runtime emits
`keyword@N`, the sidecars must have been rebuilt by the fixed converter, or every such record trips the
fail-loudly rule below.

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
