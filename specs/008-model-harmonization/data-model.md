# Data Model: 008 Model Harmonization

## Entities (unchanged)

No new domain entities are introduced. The existing `Card`, `PriceEstimate`, `TrainedModel`, `TransformerConfig`, and `EvaluationMetrics` entities remain as-is.

## Key Data Flows (changed)

### Converted Card Text (canonical input format)

The converted card text becomes the sole input representation for all operations. Format:

```text
name: <card name>
mana cost: {<mana symbols>}
types: <supertype> <type> <subtype>
power toughness: <P>/<T>          (creatures only)
loyalty: <N>                       (planeswalkers only)
spell[N]: <ability text>           (instants/sorceries)
triggered: <ability text>          (triggered abilities)
activated: <ability text>          (activated abilities)
keyword[N]: <keyword>              (keyword abilities)
replacement: <ability text>        (replacement effects)
planeswalker[N]: <ability text>    (planeswalker abilities)
```

### Model Artifact Layout (changed)

Both model types follow the same versioning convention: timestamped filenames + a `latest` copy.

```text
models/
├── sklearn/
│   ├── latest.joblib          (copy of most recent version)
│   └── <timestamp>.joblib     (versioned artifacts)
└── transformer/
    ├── latest.pt              (copy of most recent version)
    └── <timestamp>.pt         (versioned artifacts)
```

### Training Data Flow (changed)

**Before** (separate paths per model):
- sklearn: Forge scripts → `parse_forge_cards()` → Card → FeatureEngineering → train
- transformer: Forge scripts → Card → match to converted text files → tokenize → train

**After** (unified input):
- sklearn: Converted text files → `parse_converted_cards()` → Card → match prices → FeatureEngineering → train
- transformer: Converted text files → read raw text → match prices → tokenize → train

Both pipelines read from `./output/` and match cards to prices via MTGJSON data.

### Prediction Data Flow (changed)

**Before** (multiple entry points):
- `predict`: Manual card attributes → Card → sklearn predict
- `eval`: Forge script file → HTTP POST to server → server parses → dual predict
- Server: Forge script text body → parse → dual predict

**After** (unified):
- `predict sklearn`: Converted text (file or inline) → parse → Card → sklearn predict (local)
- `predict transformer`: Converted text (file or inline) → tokenize raw text → transformer predict (local)
- Server: Converted text body → parse for sklearn + tokenize for transformer → dual predict
