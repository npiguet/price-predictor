"""Shared loaders and statistics for the effect-embedding probe battery.

Measures nothing on its own. It builds the one table every other script in
this directory reads: one row per unique ability text the shipping model
encodes, with its 64-d ``e`` from the shipping cache, its ``e`` from the
``taxonomy`` baseline's cache, and a set of hand-parsed features of the script
and of the card that carries it.

Unique means unique on the **script surface**: the text is
``encoding_text(line, prose, "script")`` (the script, falling back to prose
where a line has none), which is what the encoder actually reads. Two cards
printing the same script share one row.

Run nothing here directly; ``build_texts.py`` materialises the table.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "effects" / "reports" / "embedding-probes-20260919"
CACHE = OUT / "cache"
ABILITIES = ROOT / "output" / "effects" / "abilities"
TREES = ("cardsfolder", "tokenscripts")
MANIFEST = ROOT / "output" / "effects" / "corpus" / "manifest.json"
TRAINING = ROOT / "output" / "effects" / "corpus" / "training"
TEXT_TABLE = CACHE / "texts.pkl"
PROFILE_TABLE = OUT / "effect_profiles.csv"
SEED = 42

# ── script parsing ──────────────────────────────────────────────────────

_PAIR_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\$\s?(.*)$")
_INT_RE = re.compile(r"^[+-]?\d+$")


def parse_script(text: str) -> dict[str, str]:
    """``Key$ Value | Key$ Value`` into a dict; a keyword line gives ``{}``."""
    pairs: dict[str, str] = {}
    if "$" not in text:
        return pairs
    for part in text.split(" | "):
        match = _PAIR_RE.match(part)
        if match:
            pairs.setdefault(match.group(1), match.group(2).strip())
    return pairs


def as_int(value: str | None) -> float:
    """An integer parameter, NaN for a variable (``X``, an SVar name)."""
    if value is None:
        return float("nan")
    value = value.strip()
    return float(int(value)) if _INT_RE.match(value) else float("nan")


_MANA_SYMBOL_RE = re.compile(r"^(?:[WUBRGCS]|[WUBRG2]/[WUBRGP]|[WUBRG]/P)$")


def cost_features(cost: str | None) -> dict[str, float]:
    """The cost of an activated ability, from its ``Cost$`` parameter."""
    out = {
        "cost_present": 0.0, "cost_tap": 0.0, "cost_untap_symbol": 0.0,
        "cost_mana": 0.0, "cost_x": 0.0, "cost_sacrifice": 0.0,
        "cost_life": 0.0, "cost_discard": 0.0, "cost_exile": 0.0,
        "cost_loyalty_plus": 0.0, "cost_loyalty_minus": 0.0,
        "cost_remove_counter": 0.0,
    }
    if not cost:
        return out
    out["cost_present"] = 1.0
    mana = 0
    for token in cost.split():
        if token == "T":
            out["cost_tap"] = 1.0
        elif token == "Q":
            out["cost_untap_symbol"] = 1.0
        elif token == "X":
            out["cost_x"] = 1.0
        elif token.isdigit():
            mana += int(token)
        elif _MANA_SYMBOL_RE.match(token):
            mana += 1
        elif token.startswith("Sac<"):
            out["cost_sacrifice"] = 1.0
        elif token.startswith("PayLife<"):
            out["cost_life"] = 1.0
        elif token.startswith("Discard<"):
            out["cost_discard"] = 1.0
        elif token.startswith("Exile"):
            out["cost_exile"] = 1.0
        elif token.startswith("AddCounter<") and "LOYALTY" in token:
            out["cost_loyalty_plus"] = 1.0
        elif token.startswith("SubCounter<") and "LOYALTY" in token:
            out["cost_loyalty_minus"] = 1.0
        elif token.startswith("SubCounter<"):
            out["cost_remove_counter"] = 1.0
    out["cost_mana"] = float(mana)
    return out


_PERMANENT_WORDS = {
    "Permanent", "Artifact", "Enchantment", "Land", "Planeswalker", "Battle",
}


def target_type(pairs: dict[str, str]) -> str:
    """What the ability targets: player / creature / permanent / any / card /
    spell / other, or ``none`` for an untargeted line."""
    valid = pairs.get("ValidTgts")
    if pairs.get("TargetType") == "Spell" or pairs.get("TgtZone") == "Stack":
        return "spell"
    if not valid:
        return "none"
    heads = {segment.split(".")[0].split("+")[0] for segment in valid.split(",")}
    if "Any" in heads:
        return "any"
    has_player = bool(heads & {"Player", "Opponent"})
    has_creature = "Creature" in heads
    has_permanent = bool(heads & _PERMANENT_WORDS)
    if has_player and (has_creature or has_permanent):
        return "any"
    if has_player:
        return "player"
    if has_creature and not has_permanent:
        return "creature"
    if has_creature or has_permanent:
        return "permanent"
    if heads & {"Card", "Instant", "Sorcery"}:
        return "card"
    return "other"


def affected_features(pairs: dict[str, str], api: str) -> dict[str, float]:
    """Who the script names as affected, read off its parameter values."""
    values = " ".join(
        v for k, v in pairs.items()
        if k not in ("SpellDescription", "TriggerDescription", "Description",
                     "StackDescription", "TgtPrompt")
    )
    defined = pairs.get("Defined", "")
    return {
        "aff_you": float(
            defined in ("You", "Self", "Remembered")
            or "YouCtrl" in values or bool(re.search(r"\bYou\b", values))
        ),
        "aff_opponent": float(
            "Opponent" in values or "OppCtrl" in values
        ),
        "aff_each": float(
            api.endswith("All") or api.endswith("EachPlayer")
            or defined == "Player" or "ValidCards" in pairs
            or defined.startswith("Player")
        ),
    }


def count_amounts(pairs: dict[str, str], api: str) -> dict[str, float]:
    """Numeric amounts; 0 where the line has no such amount, NaN for ``X``."""
    def amount(*keys: str) -> float:
        for key in keys:
            if key in pairs:
                return as_int(pairs[key])
        return 0.0

    out = {
        "amt_damage": amount("NumDmg") if "Damage" in api or "NumDmg" in pairs
        else 0.0,
        "amt_counters": amount("CounterNum") if "CounterType" in pairs else 0.0,
        "amt_power": amount("NumAtt", "AddPower"),
        "amt_toughness": amount("NumDef", "AddToughness"),
        "amt_cards": amount("NumCards") if api in ("Draw", "Discard", "Mill",
                                                     "Dig", "Surveil", "Scry")
        else 0.0,
        "amt_life": amount("LifeAmount"),
        "amt_tokens": amount("TokenAmount") if api == "Token" else 0.0,
    }
    if api == "Token" and "TokenAmount" not in pairs:
        out["amt_tokens"] = 1.0
    if api == "Draw" and "NumCards" not in pairs:
        out["amt_cards"] = 1.0
    if "CounterType" in pairs and "CounterNum" not in pairs:
        out["amt_counters"] = 1.0
    return out


# ── card facts from the converted text ──────────────────────────────────

_PIP_RE = re.compile(r"\{([^}]+)\}")


def card_facts(converted_txt: Path) -> dict[str, object]:
    """Type line, mana value and colours of a card's first face."""
    facts: dict[str, object] = {"type_line": "", "mana_cost": "", "is_token": 0}
    if not converted_txt.exists():
        return facts
    for line in converted_txt.read_text(encoding="utf-8").splitlines():
        if line.startswith("types: ") and not facts["type_line"]:
            facts["type_line"] = line[len("types: "):]
        elif line.startswith("mana cost: ") and not facts["mana_cost"]:
            facts["mana_cost"] = line[len("mana cost: "):]
        elif line == "ALTERNATE":
            break
    mana_value, colours = 0, set()
    for pip in _PIP_RE.findall(str(facts["mana_cost"])):
        if pip.isdigit():
            mana_value += int(pip)
        elif pip == "X":
            continue
        else:
            mana_value += 1
            colours.update(c for c in pip if c in "WUBRG")
    facts["cmc"] = mana_value
    for colour in "WUBRG":
        facts[f"colour_{colour}"] = int(colour in colours)
    facts["n_colours"] = len(colours)
    return facts


CARD_TYPES = ("creature", "instant", "sorcery", "land", "artifact",
              "enchantment", "planeswalker")


# ── statistics ──────────────────────────────────────────────────────────


def collapse_rare(values: pd.Series, min_count: int = 20,
                  other: str = "(rare)") -> pd.Series:
    """Merge categories with fewer than ``min_count`` rows into one level."""
    counts = values.value_counts()
    rare = counts[counts < min_count].index
    return values.where(~values.isin(rare), other)


def eta_squared(y: np.ndarray, groups: pd.Series) -> float:
    """One-way ANOVA R²: the share of ``y``'s variance between group means."""
    frame = pd.DataFrame({"y": y, "g": groups.to_numpy()})
    total = float(((frame.y - frame.y.mean()) ** 2).sum())
    means = frame.groupby("g").y.transform("mean")
    between = float(((means - frame.y.mean()) ** 2).sum())
    return between / total if total > 0 else float("nan")


def multivariate_eta_squared(E: np.ndarray, groups: pd.Series) -> float:
    """The share of the whole space's variance (trace) between group means."""
    centred = E - E.mean(axis=0)
    total = float((centred ** 2).sum())
    frame = pd.DataFrame(centred)
    means = frame.groupby(groups.to_numpy()).transform("mean").to_numpy()
    return float((means ** 2).sum()) / total


def group_means_residual(E: np.ndarray, groups: pd.Series) -> np.ndarray:
    """``E`` with each group's mean subtracted (the grand mean kept)."""
    frame = pd.DataFrame(E)
    means = frame.groupby(groups.to_numpy()).transform("mean").to_numpy()
    return E - means + E.mean(axis=0)


def weighted_corr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b) & (w > 0)
    a, b, w = a[ok], b[ok], w[ok]
    if a.size < 3:
        return float("nan")
    w = w / w.sum()
    am, bm = w @ a, w @ b
    cov = w @ ((a - am) * (b - bm))
    va, vb = w @ ((a - am) ** 2), w @ ((b - bm) ** 2)
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")


def pca(E: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(scores, components, variance share)`` of the centred matrix."""
    centred = E - E.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    share = s ** 2 / float((s ** 2).sum())
    scores = centred @ vt.T
    # Fix each component's sign so the reported direction is reproducible:
    # the positive end is the one holding the larger absolute tail.
    for k in range(vt.shape[0]):
        if abs(scores[:, k].min()) > abs(scores[:, k].max()):
            scores[:, k] *= -1
            vt[k] *= -1
    return scores, vt, share


def load_texts() -> pd.DataFrame:
    """The table ``build_texts.py`` wrote, with ``e_full`` / ``e_tax`` arrays."""
    return pd.read_pickle(TEXT_TABLE)


def matrix(table: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(table[column].to_numpy()).astype(np.float64)


def load_manifest_sets() -> tuple[set[str], dict[str, int]]:
    """``(held-out texts, rarity table)`` from the curated corpus's manifest."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return set(manifest["held_out_texts"]), dict(manifest["rarity"])


def top_keys(texts: pd.Series, min_share: float = 0.01) -> list[str]:
    """Parameter keys present in at least ``min_share`` of the texts."""
    counts: Counter[str] = Counter()
    for pairs in texts:
        counts.update(pairs.keys())
    floor = min_share * len(texts)
    skip = {"SpellDescription", "TriggerDescription", "Description"}
    return sorted(k for k, c in counts.items() if c >= floor and k not in skip)


def write_markdown(frame: pd.DataFrame, path: Path, title: str,
                   note: str = "", floatfmt: str = ".3f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = to_markdown(frame, floatfmt)
    text = f"# {title}\n\n{note}\n\n{body}\n" if note else f"# {title}\n\n{body}\n"
    path.write_text(text, encoding="utf-8")


# ── the candidate features, shared by every script ──────────────────────

CATEGORICAL = ("api", "line_kind", "mode", "target", "script_head",
               "keyword_name")

BINARY_SCRIPT = (
    "cost_present", "cost_tap", "cost_x", "cost_sacrifice", "cost_life",
    "cost_discard", "cost_exile", "cost_loyalty_plus", "cost_loyalty_minus",
    "cost_remove_counter", "aff_you", "aff_opponent", "aff_each",
    "trigger_etb", "trigger_dies", "dur_eot", "dur_permanent",
    "has_subability", "has_condition", "optional", "is_curse",
)
NUMERIC_SCRIPT = (
    "cost_mana", "amt_damage", "amt_counters", "amt_power", "amt_toughness",
    "amt_cards", "amt_life", "amt_tokens", "n_params", "log_len",
)
CARD = (
    "card_creature", "card_instant", "card_sorcery", "card_land",
    "card_artifact", "card_enchantment", "card_planeswalker", "card_token",
    "card_cmc", "card_colour_W", "card_colour_U", "card_colour_B",
    "card_colour_R", "card_colour_G", "card_n_colours",
)
CORPUS = ("held_out", "log_rarity", "log_carriers", "log_train_records")


def with_key_presence(table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add a ``key_<Name>`` 0/1 column for every common parameter key."""
    keys = top_keys(table.pairs)
    columns = {f"key_{k}": table.pairs.map(lambda p, k=k: float(k in p))
               for k in keys}
    return pd.concat([table, pd.DataFrame(columns, index=table.index)], axis=1), \
        [f"key_{k}" for k in keys]


def with_profiles(table: pd.DataFrame) -> pd.DataFrame:
    """Join the corpus effect profiles on the text, where they exist."""
    if not PROFILE_TABLE.exists():
        table = table.copy()
        table["log_train_records"] = 0.0
        return table
    profiles = pd.read_csv(PROFILE_TABLE)
    joined = table.merge(profiles, on="text", how="left")
    joined["n_records"] = joined["n_records"].fillna(0.0)
    joined["log_train_records"] = np.log1p(joined["n_records"])
    return joined


def prepared() -> tuple[pd.DataFrame, list[str]]:
    """The text table with key-presence columns and effect profiles joined."""
    table = load_texts()
    table, key_columns = with_key_presence(table)
    table = with_profiles(table)
    for column in CATEGORICAL:
        table[column] = table[column].astype(str)
    table["api_c"] = collapse_rare(table["api"])
    table["mode_c"] = collapse_rare(table["mode"])
    table["keyword_c"] = collapse_rare(table["keyword_name"])
    table["head_c"] = collapse_rare(table["script_head"])
    return table, key_columns


def design(table: pd.DataFrame, *, categorical=(), numeric=()) -> np.ndarray:
    """A dense design matrix: one-hot categoricals plus standardized numerics.

    NaN numerics (a variable amount such as ``X``) become 0 after
    standardizing, i.e. the column mean; standardized values clip at ±5.
    """
    parts = []
    for column in categorical:
        parts.append(pd.get_dummies(table[column], dtype=float).to_numpy())
    for column in numeric:
        values = table[column].to_numpy(dtype=float)
        mean, std = np.nanmean(values), np.nanstd(values)
        values = (values - mean) / (std if std > 0 else 1.0)
        # Clipped: a handful of scripts carry amounts like 99 or 1000, and one
        # such row in a test fold would dominate every out-of-fold score.
        parts.append(np.clip(np.nan_to_num(values), -5.0, 5.0)[:, None])
    return np.hstack(parts) if parts else np.zeros((len(table), 0))


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A pipe table without the optional ``tabulate`` dependency."""
    def cell(value: object) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else format(value, floatfmt)
        return str(value).replace("|", "/")

    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join(
        "---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---"
        for c in frame.columns) + "|"
    rows = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def winsorized(values: np.ndarray, tail: float = 0.001) -> np.ndarray:
    """Clip a numeric target at its ``tail`` quantiles, keeping NaN as NaN.

    A handful of scripts carry amounts like -9999 toughness or 200 tokens;
    one of them in a test fold would decide an R² on its own.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return np.where(np.isfinite(values), np.clip(values, low, high), np.nan)
