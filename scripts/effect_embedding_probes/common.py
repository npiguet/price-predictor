"""Shared loaders and statistics for the effect-embedding probe battery.

Measures nothing on its own. It builds the one table every other script in
this directory reads: one row per unique ability text the checkpoint encodes,
with its ``e`` from that checkpoint's cache, its ``e`` from the ``taxonomy``
baseline's cache where one exists, and a set of hand-parsed features of the
script and of the card that carries it.

Every script takes ``--checkpoint`` and ``--abilities-root`` (FR-072) and
reads everything else from them through :func:`resolve_paths`: the
vocabulary's encoding surface, the sidecar trees the checkpoint trained
against, the curated corpus's manifest, and the width of ``e`` from the cache
itself. Nothing is hardcoded to one run, so the battery runs unchanged on any
sweep arm. Output goes under
``<reports-dir>/embedding-probes-<checkpoint>-<date>/``, the reports dir
defaulting to ``output/effects/reports/``.

Unique means unique on the checkpoint's **encoding surface**: the text is
``encoding_text(line, prose, surface)``, which is what the encoder actually
reads. Two cards printing the same script share one row.

Run nothing here directly; ``build_texts.py`` materialises the table.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from _lib.markdown import md_table
from _lib.paths import REPO

from effects.domain import default_paths

REPORTS = REPO / default_paths.DEFAULT_REPORTS_DIR
DEFAULT_CHECKPOINT = REPO / default_paths.DEFAULT_CHECKPOINT
DEFAULT_ABILITIES = REPO / default_paths.DEFAULT_ABILITIES_ROOT
DEFAULT_CARDS_FOLDERS = tuple(REPO / folder for folder in default_paths.DEFAULT_CARDS_FOLDERS)
VARIANT_TREE = REPO / default_paths.DEFAULT_VARIANT_SCRIPTS
SEED = 42


def checkpoint_label(checkpoint: Path) -> str:
    """The checkpoint's name in a report directory.

    Its stem, except that every run's rolling checkpoint is ``latest.pt``, so
    that one is named by its run directory as well and two arms' reports
    never share a directory.
    """
    checkpoint = Path(checkpoint)
    if checkpoint.stem == "latest":
        return f"{checkpoint.parent.name}-{checkpoint.stem}"
    return checkpoint.stem


@dataclass(frozen=True)
class ProbePaths:
    """Everything one checkpoint's probe run reads and writes."""

    checkpoint: Path
    abilities_root: Path
    out: Path
    cards_folders: tuple[Path, ...]
    surface: str
    e_width: int
    manifest: Path | None

    @property
    def label(self) -> str:
        return checkpoint_label(self.checkpoint)

    @property
    def cache(self) -> Path:
        return self.out / "cache"

    @property
    def text_table(self) -> Path:
        return self.cache / "texts.pkl"

    @property
    def keymap(self) -> Path:
        return self.cache / "keymap.pkl"

    @property
    def profile_table(self) -> Path:
        return self.out / "effect_profiles.csv"

    @property
    def training(self) -> Path | None:
        """The curated corpus's training shards, beside its manifest."""
        return None if self.manifest is None else self.manifest.parent / "training"

    def tree_root(self, tree: str) -> Path | None:
        """The sidecar folder a source tree is read from, if this run has one."""
        from effects.infrastructure.sidecar_io import tree_of_folder

        for folder in self.cards_folders:
            if tree_of_folder(folder) == tree:
                return folder
        return None

    def reusable(self, relative: str) -> Path:
        """``relative`` under this run's output, or the newest earlier run's.

        ``build_texts.py`` and ``effect_profiles.py`` write the tables the
        other scripts read; a probe run the next day finds them under the
        previous day's directory for the same checkpoint rather than asking for
        a rebuild.
        """
        mine = self.out / relative
        if mine.exists():
            return mine
        earlier = sorted(
            (path for path in self.out.parent.glob(
                f"embedding-probes-{self.label}-*/{relative}"
            )),
            key=lambda path: path.parent.as_posix(),
        )
        return earlier[-1] if earlier else mine


def add_probe_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The two flags every script takes (FR-072), plus the sidecar trees."""
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="Effects checkpoint whose vocabulary and cache the probes read",
    )
    parser.add_argument(
        "--abilities-root", type=Path, default=DEFAULT_ABILITIES,
        help="The ability cache that checkpoint encoded",
    )
    parser.add_argument(
        "--cards-folder", action="append", type=Path, dest="cards_folders",
        default=None,
        help=(
            "Sidecar tree the cache rows align with (repeatable); defaults to "
            "the trees the checkpoint recorded, else output/effects/gen3/converted/cardsfolder and "
            "output/effects/gen3/converted/tokenscripts"
        ),
    )
    parser.add_argument(
        "--reports-dir", type=Path, default=None,
        help=(
            "Directory the run's embedding-probes-<checkpoint>-<date>/ folder "
            "goes in (default: output/effects/reports/)"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "The run's report directory itself, its run.json included, shared "
            "by every script of the battery given it (default: the dated "
            "folder under --reports-dir)"
        ),
    )
    return parser


def cache_e_width(abilities_root: Path) -> int:
    """The width of ``e``, read from the first shipping cache file found."""
    from effects.infrastructure.ability_cache_store import AbilityCacheStore

    for _, matrix in AbilityCacheStore(abilities_root).iter_matrices():
        if matrix.ndim == 2 and matrix.shape[1]:
            return int(matrix.shape[1])
    raise FileNotFoundError(
        f"no ability cache under {abilities_root}; run encode-abilities first"
    )


@dataclass(frozen=True)
class PathOptions:
    """What :func:`paths_for` may override: where the sidecar trees come from
    (``cards_folders`` first, then ``recorded_folders``, the trees the checkpoint
    trained against) and where the report directory goes."""

    recorded_folders: tuple = ()
    cards_folders: list | tuple | None = None
    today: datetime.date | None = None
    reports_dir: Path | None = None
    output: Path | None = None


def paths_for(
    checkpoint: Path,
    abilities_root: Path,
    provenance,
    options: PathOptions = PathOptions(),
) -> ProbePaths:
    """The run's paths from a checkpoint's recorded provenance.

    ``provenance`` is the checkpoint's ``SplitProvenance``: its vocabulary
    fixes the encoding surface and its corpus path the manifest. Sidecar trees
    come from ``options.cards_folders`` when given, else from what the
    checkpoint recorded it trained against, else the converted defaults.
    """
    from effects.domain.ability_encoder import surface_of

    checkpoint = Path(checkpoint)
    folders = tuple(Path(f) for f in (
        options.cards_folders or options.recorded_folders or DEFAULT_CARDS_FOLDERS
    ))
    corpus = getattr(provenance, "corpus_path", "") or ""
    manifest = Path(corpus) / "manifest.json" if corpus else None
    return ProbePaths(
        checkpoint=checkpoint,
        abilities_root=Path(abilities_root),
        out=report_dir(checkpoint, options.reports_dir, options.today, output=options.output),
        cards_folders=folders,
        surface=surface_of(Path(provenance.vocab_path)),
        e_width=cache_e_width(abilities_root),
        manifest=manifest,
    )


def report_dir(
    checkpoint: Path, reports_dir: Path | None = None, today: datetime.date | None = None,
    *, output: Path | None = None,
) -> Path:
    """The run's report directory: ``output`` (``--output``), else
    ``<reports-dir>/embedding-probes-<checkpoint>-<date>/``."""
    if output is not None:
        return Path(output)
    date = (today or datetime.date.today()).strftime("%Y%m%d")
    return (Path(reports_dir) if reports_dir else REPORTS) / (
        f"embedding-probes-{checkpoint_label(Path(checkpoint))}-{date}"
    )


def main_with_manifest(main: Callable[[], object], script: str) -> None:
    """Run a probe script's ``main`` inside its report directory's run manifest.

    The battery's scripts share one report directory, so its manifest lists
    each script run as one more command. The common flags are read here once
    more, ahead of ``main``, for the directory; ``--help`` runs bare.
    """
    from price_predictor.infrastructure.run_manifest import run_script

    if {"-h", "--help"} & set(sys.argv[1:]):
        main()
        return
    args, _ = add_probe_arguments(argparse.ArgumentParser(add_help=False)).parse_known_args()
    raise SystemExit(run_script(
        report_dir(args.checkpoint, args.reports_dir, output=args.output), main, script=script,
        inputs=[args.checkpoint, args.abilities_root, *(args.cards_folders or ())],
    ))


def resolve_paths(args: argparse.Namespace) -> ProbePaths:
    """Load ``--checkpoint`` and resolve the run's paths from it."""
    from effects.infrastructure.effect_model_store import load_effect_checkpoint

    checkpoint = Path(args.checkpoint)
    loaded = load_effect_checkpoint(checkpoint)
    paths = paths_for(checkpoint, args.abilities_root, loaded.provenance, PathOptions(
        recorded_folders=getattr(loaded, "cards_folders", ()),
        cards_folders=getattr(args, "cards_folders", None),
        reports_dir=getattr(args, "reports_dir", None),
        output=getattr(args, "output", None),
    ))
    if loaded.encoder_config.e_dim != paths.e_width:
        raise ValueError(
            f"{args.abilities_root} holds {paths.e_width}-wide e but "
            f"{checkpoint} encodes {loaded.encoder_config.e_dim}; point "
            "--abilities-root at the cache this checkpoint wrote"
        )
    paths.out.mkdir(parents=True, exist_ok=True)
    return paths

# ── script parsing ──────────────────────────────────────────────────────

_INT_RE = re.compile(r"^[+-]?\d+$")


def as_int(value: str | None) -> float:
    """An integer parameter, NaN for a variable (``X``, an SVar name)."""
    if value is None:
        return float("nan")
    value = value.strip()
    return float(int(value)) if _INT_RE.match(value) else float("nan")


_MANA_SYMBOL_RE = re.compile(r"^(?:[WUBRGCS]|[WUBRG2]/[WUBRGP]|[WUBRG]/P)$")


# Not ``ManaCost.parse``: it reads Forge's compact shards (``WU``, ``W2``), these tokens
# and the converted text's braces are slash-separated, and a ``2/W`` pip counts one here
# (a symbol paid) but two as a mana value.
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
    return _head_category({segment.split(".")[0].split("+")[0] for segment in valid.split(",")})


def _head_category(heads: set[str]) -> str:
    """The target category of a set of ``ValidTgts`` head words."""
    if "Any" in heads:
        return "any"
    has_player = bool(heads & {"Player", "Opponent"})
    has_creature = "Creature" in heads
    has_permanent = bool(heads & _PERMANENT_WORDS)
    if has_player:
        return "any" if has_creature or has_permanent else "player"
    if has_permanent:
        return "permanent"
    if has_creature:
        return "creature"
    if heads & {"Card", "Instant", "Sorcery"}:
        return "card"
    return "other"


_DESCRIPTION_KEYS = ("SpellDescription", "TriggerDescription", "Description",
                     "StackDescription", "TgtPrompt")


def affected_features(pairs: dict[str, str], api: str) -> dict[str, float]:
    """Who the script names as affected, read off its parameter values."""
    values = " ".join(v for k, v in pairs.items() if k not in _DESCRIPTION_KEYS)
    defined = pairs.get("Defined", "")
    return {
        "aff_you": float(
            defined in ("You", "Self", "Remembered")
            or "YouCtrl" in values or bool(re.search(r"\bYou\b", values))
        ),
        "aff_opponent": float("Opponent" in values or "OppCtrl" in values),
        "aff_each": float(_names_each(pairs, api, defined)),
    }


def _names_each(pairs: dict[str, str], api: str, defined: str) -> bool:
    """Whether the script's ``Defined`` or API names every player or card."""
    return (
        api.endswith(("All", "EachPlayer")) or defined == "Player"
        or "ValidCards" in pairs or defined.startswith("Player")
    )


def _amount(pairs: dict[str, str], *keys: str) -> float:
    """The first of ``keys`` the script gives, 0 when it gives none."""
    for key in keys:
        if key in pairs:
            return as_int(pairs[key])
    return 0.0


def _amount_if(applies: bool, pairs: dict[str, str], *keys: str) -> float:
    return _amount(pairs, *keys) if applies else 0.0


def _implied_one(pairs: dict[str, str], api: str) -> list[str]:
    """The amounts a script leaves unwritten that count as one."""
    implied = []
    if api == "Token" and "TokenAmount" not in pairs:
        implied.append("amt_tokens")
    if api == "Draw" and "NumCards" not in pairs:
        implied.append("amt_cards")
    if "CounterType" in pairs and "CounterNum" not in pairs:
        implied.append("amt_counters")
    return implied


def count_amounts(pairs: dict[str, str], api: str) -> dict[str, float]:
    """Numeric amounts; 0 where the line has no such amount, NaN for ``X``."""
    draws = api in ("Draw", "Discard", "Mill", "Dig", "Surveil", "Scry")
    out = {
        "amt_damage": _amount_if("Damage" in api or "NumDmg" in pairs, pairs, "NumDmg"),
        "amt_counters": _amount_if("CounterType" in pairs, pairs, "CounterNum"),
        "amt_power": _amount(pairs, "NumAtt", "AddPower"),
        "amt_toughness": _amount(pairs, "NumDef", "AddToughness"),
        "amt_cards": _amount_if(draws, pairs, "NumCards"),
        "amt_life": _amount(pairs, "LifeAmount"),
        "amt_tokens": _amount_if(api == "Token", pairs, "TokenAmount"),
    }
    out.update({name: 1.0 for name in _implied_one(pairs, api)})
    return out


# ── card facts from the converted text ──────────────────────────────────

_PIP_RE = re.compile(r"\{([^}]+)\}")


def card_facts(converted_txt: Path) -> dict[str, object]:
    """Type line, mana value and colours of a card's first face."""
    facts: dict[str, object] = {"type_line": "", "mana_cost": "", "is_token": 0}
    if not converted_txt.exists():
        return facts
    facts["type_line"], facts["mana_cost"] = _first_face(
        converted_txt.read_text(encoding="utf-8").splitlines())
    mana_value, colours = _mana_value_and_colours(str(facts["mana_cost"]))
    facts["cmc"] = mana_value
    for colour in "WUBRG":
        facts[f"colour_{colour}"] = int(colour in colours)
    facts["n_colours"] = len(colours)
    return facts


def _first_face(lines: list[str]) -> tuple[str, str]:
    """``(type line, mana cost)`` of the converted text's first face."""
    type_line = mana_cost = ""
    for line in lines:
        if line == "ALTERNATE":
            break
        if line.startswith("types: ") and not type_line:
            type_line = line[len("types: "):]
        elif line.startswith("mana cost: ") and not mana_cost:
            mana_cost = line[len("mana cost: "):]
    return type_line, mana_cost


def _mana_value_and_colours(mana_cost: str) -> tuple[int, set[str]]:
    """The mana value a cost spells and the colours its pips name."""
    mana_value, colours = 0, set()
    for pip in _PIP_RE.findall(mana_cost):
        if pip.isdigit():
            mana_value += int(pip)
        elif pip != "X":
            mana_value += 1
            colours.update(c for c in pip if c in "WUBRG")
    return mana_value, colours


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


def profile_weight(table: pd.DataFrame, column: str) -> np.ndarray:
    """``n / (n + 5)`` for an effect-profile statistic, ``n`` the records it is a share of.

    A share over a handful of records is mostly noise, so it counts for less.
    """
    if column.startswith(("p_cost", "mean_cost")):
        n = table["cost_n"]
    elif column.startswith("mean_cont"):
        n = table["cont_n"]
    elif column.startswith("p_trigger"):
        n = table["n_trigger"]
    elif column.startswith("kind_share"):
        n = table["n_records"]
    else:
        n = table["res_n"]
    n = n.fillna(0).to_numpy(float)
    return n / (n + 5.0)


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


def load_texts(paths: ProbePaths) -> pd.DataFrame:
    """The table ``build_texts.py`` wrote, with ``e_full`` / ``e_tax`` arrays."""
    return pd.read_pickle(paths.reusable("cache/texts.pkl"))


def has_taxonomy(table: pd.DataFrame) -> bool:
    """Whether every row carries a taxonomy-baseline ``e``.

    An arm need not have a taxonomy cache; ``build_texts.py`` then leaves
    ``e_tax`` empty, and every column that reads it is omitted (FR-072).
    """
    return bool(len(table)) and all(
        vector is not None and len(vector) for vector in table["e_tax"]
    )


def matrix(table: pd.DataFrame, column: str) -> np.ndarray:
    return np.stack(table[column].to_numpy()).astype(np.float64)


def load_manifest_sets(paths: ProbePaths) -> tuple[set[str], dict[str, int]]:
    """``(held-out texts, rarity table)`` from the curated corpus's manifest.

    Empty for a checkpoint that names no curated corpus: every text then reads
    as trained and unrated rather than the run failing.
    """
    from effects.infrastructure.corpus_store import CorpusStore

    if paths.manifest is None or not paths.manifest.exists():
        return set(), {}
    manifest = CorpusStore(paths.manifest.parent).load()
    return set(manifest.held_out_texts), dict(manifest.rarity)


def top_keys(texts: pd.Series, min_share: float = 0.01) -> list[str]:
    """Parameter keys present in at least ``min_share`` of the texts."""
    counts: Counter[str] = Counter()
    for pairs in texts:
        counts.update(pairs.keys())
    floor = min_share * len(texts)
    skip = {"SpellDescription", "TriggerDescription", "Description"}
    return sorted(k for k, c in counts.items() if c >= floor and k not in skip)


@dataclass(frozen=True)
class MarkdownStyle:
    """What a written table carries besides its title: a note under it and
    the float format of its cells."""

    note: str = ""
    floatfmt: str = ".3f"


def write_markdown(frame: pd.DataFrame, path: Path, title: str,
                   style: MarkdownStyle = MarkdownStyle()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = to_markdown(frame, style.floatfmt)
    note = style.note
    text = f"# {title}\n\n{note}\n\n{body}\n" if note else f"# {title}\n\n{body}\n"
    path.write_text(text, encoding="utf-8")


# ── the candidate features, shared by every script ──────────────────────

#: The categorical columns of the script-feature design matrix.
SCRIPT_CATS = ["api_c", "line_kind", "mode_c", "target", "head_c", "keyword_c"]

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


def with_profiles(table: pd.DataFrame, paths: ProbePaths) -> pd.DataFrame:
    """Join the corpus effect profiles on the text, where they exist."""
    profile_table = paths.reusable("effect_profiles.csv")
    if not profile_table.exists():
        table = table.copy()
        table["log_train_records"] = 0.0
        return table
    profiles = pd.read_csv(profile_table)
    joined = table.merge(profiles, on="text", how="left")
    joined["n_records"] = joined["n_records"].fillna(0.0)
    joined["log_train_records"] = np.log1p(joined["n_records"])
    return joined


def prepared(paths: ProbePaths) -> tuple[pd.DataFrame, list[str]]:
    """The text table with key-presence columns and effect profiles joined."""
    table = load_texts(paths)
    table, key_columns = with_key_presence(table)
    table = with_profiles(table, paths)
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
    """A pipe table of ``frame``: floats formatted, NaN blank, numeric columns right-aligned."""
    def cell(value: object) -> object:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else format(value, floatfmt)
        return value

    align = "".join("r" if pd.api.types.is_numeric_dtype(frame[c]) else "l"
                    for c in frame.columns)
    return md_table(list(frame.columns),
                    ([cell(v) for v in row] for row in frame.itertuples(index=False)),
                    align)


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
