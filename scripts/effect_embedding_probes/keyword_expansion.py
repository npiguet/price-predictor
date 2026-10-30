"""Does keyword expansion work, and can the encoder still read unknown keywords?

Two questions about one checkpoint's ability encoder (``--checkpoint``, on the
encoding surface and vocabulary it recorded):

1. **Does a keyword's encoding look like the encoding of its definition?**
   Training replaces a keyword token with its definition with probability
   ``--keyword-expand-p`` (0.25), so the two should land together.
2. **Can the encoder still read a keyword the vocabulary does not know?** An
   unknown keyword is meant to be expanded unconditionally.

What it measures, all on CPU with the checkpoint's own encoder and tokenizer:

- **Token sequences** the encoder receives for representative keyword lines,
  with expansion off, forced (``probability=1.0``), and with the keyword
  removed from the vocabulary (the unknown-keyword path), next to what
  ``tokenize`` produces under the script surface's rules.
- **Keyword vs definition**: for every keyword line in the converted corpus
  whose expansion fires, the cosine between the bare line (expansion off) and
  the forced expansion, and the rank of the keyword's own definition among all
  keywords' definitions (top-1, top-5, mean reciprocal rank, against chance),
  in both directions, broken down by corpus frequency and by kind. Also the
  rank of the definition among *every* unique ability text of the corpus.
- **Longhand twins**: hand-picked real card lines that spell a keyword's
  effect without naming it (ward's twins from ``ward_twins.py`` and more), and
  which keyword each one retrieves, bare and expanded.
- **Unknown keywords**: the definitions whose token the vocabulary lacks, and a
  simulation of a new keyword — each known keyword deleted from the
  vocabulary, encoded with its definitions available (forced expansion, as a
  new keyword would be) and without (a bare ``[UNK]``) — and whether that lands
  near the real keyword's encoding and near its longhand twins.
- **Cache vs model input**: how far forced expansion moves every corpus text
  containing a keyword word, and how far the cache sits from each encoding.

Run from the repo root (a few minutes on CPU; the corpus encoding is cached in
the output directory and reused)::

    python scripts/effect_embedding_probes/keyword_expansion.py \
        --checkpoint PATH --abilities-root DIR

Writes CSVs and ``summary.txt`` to ``keyword-expansion/`` under the
checkpoint's ``embedding-probes-<checkpoint>-<date>/`` report directory.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_embedding_probes.common import (  # noqa: E402
    ROOT,
    ProbePaths,
    add_probe_arguments,
    load_manifest_sets,
    resolve_paths,
)
from effects.domain.ability_encoder import (  # noqa: E402
    AbilityEncoder,
    collate_lines,
    prepare_line,
)
from effects.domain.ability_tokenizer import (  # noqa: E402
    HOST_BODIED_KEYWORDS,
    AbilityTokenizer,
    Surface,
    keyword_token,
    tokenizer_surface,
)
from effects.infrastructure.ability_cache_store import AbilityCacheStore  # noqa: E402
from effects.infrastructure.effect_model_store import (  # noqa: E402
    inference_files,
    load_effect_checkpoint,
)
from effects.infrastructure.sidecar_io import iter_sidecars, tree_of_folder  # noqa: E402
from effects.infrastructure.tokenizer_loading import load_keyword_definitions  # noqa: E402
from price_predictor.infrastructure.tokenizer_store import load_vocabulary  # noqa: E402

BATCH = 256

#: Hand classification of the corpus's keywords by what they do. A keyword
#: carrying a parameter on its line (``Ward:2``) is also counted as
#: parameterized, independently of this table.
KIND: dict[str, str] = {}
for _kind, _names in {
    "evasion": (
        "Flying", "Menace", "Shadow", "Fear", "Intimidate", "Skulk",
        "Horsemanship", "Landwalk", "Reach", "Defender",
    ),
    "combat damage": (
        "First Strike", "Double Strike", "Deathtouch", "Lifelink", "Trample",
        "Wither", "Infect", "Toxic", "Banding", "Flanking", "Bushido",
        "Rampage", "Vigilance", "Indestructible",
    ),
    "triggered": (
        "Prowess", "Afflict", "Exalted", "Ward", "Afterlife", "Persist",
        "Undying", "Evolve", "Extort", "Mentor", "Myriad", "Battle cry",
        "Annihilator", "Renown", "Soulshift", "Exploit", "Fabricate",
        "Training", "Melee", "Dethrone", "Ingest", "Enlist", "Mobilize",
        "Backup", "Cascade", "Storm", "Ravenous", "Riot", "Firebending",
        "Modular", "Bloodthirst", "Graft", "Decayed", "Unleash",
        "Hideaway", "Squad", "Offspring", "For Mirrodin", "Living Weapon",
        "Increment", "Casualty", "Provoke", "Tribute", "Sunburst",
        "Amplify", "Devour", "Fading", "Vanishing", "Cumulative upkeep",
        "Echo", "Gift",
    ),
    "cost-modifying": (
        "Affinity", "Convoke", "Delve", "Improvise", "Kicker", "Multikicker",
        "Buyback", "Entwine", "Replicate", "Assist", "Emerge", "Surge",
        "Spectacle", "Prowl", "Bargain", "Strive", "Escalate", "Casualty",
        "Offering", "Conspire", "Splice", "Squad",
    ),
    "alt cast / activated": (
        "Cycling", "Flashback", "Equip", "Crew", "Morph", "Megamorph",
        "Disguise", "Madness", "Unearth", "Foretell", "Ninjutsu", "Bestow",
        "Warp", "Evoke", "Escape", "Plot", "Disturb", "Suspend", "Dash",
        "Blitz", "Encore", "Embalm", "Eternalize", "Mutate", "Overload",
        "Miracle", "Reconfigure", "Scavenge", "Transmute", "Dredge",
        "Outlast", "Retrace", "Rebound", "Jump-start", "Aftermath",
        "Harmonize", "Freerunning", "Mayhem", "Sneak", "Saddle", "Station",
        "Level up", "Fortify", "Reinforce", "Craft", "Web-slinging",
        "Specialize", "TypeCycling", "Recover", "Prototype", "Awaken",
        "Flash",
    ),
}.items():
    for _name in _names:
        KIND.setdefault(_name, _kind)

#: Real card lines that spell a keyword's effect without naming it, as
#: ``(keyword, sidecar path stem, description)``; the line is the first of
#: that sidecar matching ``TWIN_MARKERS``. Ward's first four come from
#: ``effects/domain/ward_twins.py``; the rest were found by searching the
#: converted corpus's script text for the reminder wording.
TWINS: tuple[tuple[str, str, str], ...] = (
    ("Ward", "cardsfolder/f/frost_titan", "counter ... unless its controller pays {2}"),
    ("Ward", "cardsfolder/d/diffusion_sliver", "Sliver becomes target ... pays {2}"),
    ("Ward", "cardsfolder/u/unsettled_mariner", "you or a permanent ... pays {1}"),
    ("Ward", "cardsfolder/b/boreal_elemental", "spells that target it cost {2} more"),
    ("Ward", "cardsfolder/a/amulet_of_safekeeping", "you become target ... pays {1}"),
    ("Lifelink", "cardsfolder/d/doubtless_one", "deals damage, you gain that much life"),
    ("Lifelink", "cardsfolder/h/horned_cheetah", "deals damage, you gain that much life"),
    ("Deathtouch", "cardsfolder/l/lowland_basilisk", "damage to a creature, destroy it"),
    ("Exalted", "cardsfolder/s/strategic_intervention", "attacks alone, +1/+1"),
    ("Exalted", "cardsfolder/e/eiganjo_exemplar", "Samurai/Warrior attacks alone, +1/+1"),
    ("Afflict", "cardsfolder/v/vedalken_ghoul", "becomes blocked, defender loses 4"),
    ("Bushido", "cardsfolder/j/jukai_trainee", "blocks or becomes blocked, +1/+1"),
    ("Flying", "cardsfolder/c/canopy_cover", "can't be blocked except by flying/reach"),
    ("Hexproof", "cardsfolder/c/canopy_cover", "can't be targeted by opponents"),
    ("Toxic", "cardsfolder/p/pit_scorpion", "damage to a player, poison counter"),
    ("Haste", "cardsfolder/i/instill_energy", "can attack as though it had haste"),
    ("Prowess", "cardsfolder/b/black_bolt_inhuman_king", "noncreature spell, +2/+2"),
)

#: What to search each twin's sidecar lines for, to pick the right line.
TWIN_MARKERS: dict[str, str] = {
    "Ward": "BecomesTarget|RaiseCost",
    "Lifelink": "DamageDealtOnce",
    "Deathtouch": "DamageDone",
    "Exalted": "Alone$ True",
    "Afflict": "AttackerBlocked",
    "Bushido": "Mode$ Blocks",
    "Flying": "CantBlockBy",
    "Hexproof": "CantTarget",
    "Toxic": "TrigPoison",
    "Haste": "CanAttackIfHaste",
    "Prowess": "SpellCast",
}

#: Keyword lines shown token by token in ``token_sequences.txt``.
SHOWCASE = (
    "Flying", "First Strike", "Double Strike", "Ward:2", "Toxic:1",
    "Afflict:2", "Cycling:2", "Prowess", "Protection from red",
    "Split second", "Start your engines", "Living Weapon", "Firebending:1",
    "Equip:2", "Enchant:Creature", "Cumulative upkeep:1",
)


# ── loading ────────────────────────────────────────────────────────────


@dataclass
class Readers:
    """The checkpoint's encoder and the tokenizers every section reads it through.

    ``tok`` is the checkpoint's own surface, so every sequence is what that
    encoder read; ``script_tok`` shows the script surface's rules beside it.
    """

    encoder: AbilityEncoder
    vocab: dict
    definitions: dict
    tok: AbilityTokenizer
    script_tok: AbilityTokenizer

    def show(self, tokens) -> str:
        unk = self.tok.unk_id
        return " ".join(t.text if t.token_id != unk else f"[UNK:{t.text}]" for t in tokens)

    def without(self, token: str, definitions: dict | None = None) -> AbilityTokenizer:
        """A tokenizer whose vocabulary lacks ``token``, as a new keyword's would."""
        reduced = {k: v for k, v in self.vocab.items() if k != token}
        return AbilityTokenizer(reduced, self.definitions if definitions is None else definitions)

    def encode(self, token_lists) -> np.ndarray:
        """``e`` for each token list, batched by length."""
        order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
        out = np.zeros((len(token_lists), self.encoder.config.e_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(order), BATCH):
                chunk = order[start:start + BATCH]
                lines = [prepare_line(self.tok, token_lists[i]) for i in chunk]
                vectors, _ = self.encoder(**collate_lines(lines, self.tok.pad_id))
                out[chunk] = vectors.numpy()
        return out


def load_readers(paths: ProbePaths):
    """The checkpoint's encoder at the size it recorded, its tokenizers, and its provenance."""
    checkpoint = load_effect_checkpoint(paths.checkpoint)
    provenance = checkpoint.provenance
    # The recorded paths are repository-relative and this script may run from
    # anywhere, so they are anchored at the repository root.
    files = inference_files(checkpoint, vocab=ROOT / provenance.vocab_path,
                            keywords=ROOT / provenance.keyword_definitions_path)
    encoder = AbilityEncoder(checkpoint.encoder_config)
    encoder.load_state_dict(checkpoint.encoder_state)
    encoder.eval()
    vocab = load_vocabulary(files.vocab)
    definitions = load_keyword_definitions(files.keywords)
    tok = AbilityTokenizer(vocab, definitions, surface=tokenizer_surface(
        paths.surface, checkpoint.training_settings))
    script_tok = AbilityTokenizer(vocab, definitions, surface=Surface.SCRIPT)
    return Readers(encoder, vocab, definitions, tok, script_tok), provenance


def corpus_lines(trees) -> list[dict]:
    """Every sidecar line with script text, with its card and tree path."""
    rows = []
    for tree in trees:
        for path, sidecar in iter_sidecars(tree):
            # Named by the source tree the keys spell, not the folder's own
            # name: a kept-aside ``gen1-cardsfolder`` still holds ``cardsfolder``.
            rel = (Path(tree_of_folder(tree)) / path.relative_to(tree)).as_posix()
            stem = rel.removesuffix(".provenance.json")
            for index, line in enumerate(sidecar.lines):
                if line.script_text:
                    rows.append({"source": stem, "card": sidecar.card, "index": index,
                                 "api": line.script_api_type, "text": line.script_text})
    return rows


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


# ── statistics ─────────────────────────────────────────────────────────


def ranks(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Rank (1 = nearest) of ``keys[i]`` among all keys for ``query[i]``."""
    sims = unit(query) @ unit(keys).T
    own = np.diag(sims)[:, None]
    return (sims > own).sum(axis=1) + 1


def retrieval(rank_array: np.ndarray, n: int) -> dict:
    harmonic = sum(1 / k for k in range(1, n + 1))
    return {
        "n": len(rank_array),
        "top1": float(np.mean(rank_array == 1)),
        "top5": float(np.mean(rank_array <= 5)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "median_rank": float(np.median(rank_array)),
        "chance_top1": 1 / n,
        "chance_top5": min(5, n) / n,
        "chance_mrr": harmonic / n,
        "chance_median_rank": (n + 1) / 2,
    }


def freq_bin(carriers: int) -> str:
    if carriers >= 300:
        return "a: >=300 cards"
    if carriers >= 50:
        return "b: 50-299 cards"
    if carriers >= 10:
        return "c: 10-49 cards"
    return "d: <10 cards"


def table_text(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda v: f"{v:.3f}")


# ── the sections ───────────────────────────────────────────────────────


@dataclass
class Corpus:
    """Every unique script text of the converted corpus, encoded bare and expanded."""

    lines: list[dict]
    texts: list[str]
    plain: np.ndarray
    forced: np.ndarray

    def __post_init__(self) -> None:
        self.row_of = {t: i for i, t in enumerate(self.texts)}
        self.by_source: dict[str, list[dict]] = defaultdict(list)
        for r in self.lines:
            self.by_source[r["source"]].append(r)

    def twin_text(self, keyword: str, source: str) -> str | None:
        """The first line of ``source`` carrying the keyword's twin marker."""
        marker = TWIN_MARKERS[keyword].split("|")
        return next((r["text"] for r in self.by_source.get(source, [])
                     if any(m in r["text"] for m in marker)), None)


def definitions_table(r: Readers, carriers: Counter) -> pd.DataFrame:
    """Each definition against the vocabulary: its token, and what its expansion reads."""
    rows = []
    for name, definition in r.definitions.items():
        token = keyword_token(name)
        template = definition.reminder_template or ""
        parts = r.tok.tokenize(r.tok._instantiate(template))  # noqa: SLF001
        rows.append({
            "keyword": name,
            "token": token,
            "multi_word": "_" in token,
            "in_vocab": token in r.vocab,
            "host_bodied": name.lower() in HOST_BODIED_KEYWORDS,
            "has_script": bool(definition.generated_script),
            "placeholder": "%d" in template or "$d" in template or "%s" in template,
            "leftover_percent_tokens": sum(1 for p in parts if p.text == "%"),
            "unk_in_expansion": sum(1 for p in parts if p.token_id == r.tok.unk_id),
            "corpus_carriers": carriers.get(name, 0),
            "expansion": r.show(parts),
        })
    return pd.DataFrame(rows).sort_values("keyword")


def keyword_line_table(r: Readers, keyword_lines: list[dict], carriers: Counter,
                       rarity: dict) -> pd.DataFrame:
    """Per keyword, its most common line and whether expansion fires on it."""
    variants: dict[str, Counter] = defaultdict(Counter)
    for row in keyword_lines:
        variants[row["text"].split(":")[0]][row["text"]] += 1
    rows = []
    for name, counter in variants.items():
        text, _ = counter.most_common(1)[0]
        tokens = r.tok.tokenize(text)
        fires = [t.text for t in tokens if r.tok.expandable(t)]
        rows.append({
            "keyword": name, "line": text, "carriers": carriers[name],
            "in_definitions": name in r.definitions,
            "tokens": r.show(tokens), "expands_on": " ".join(fires),
            "fires": bool(fires),
            "acting_games": sum(rarity.get(t, 0) for t in counter),
        })
    return pd.DataFrame(rows).sort_values("carriers", ascending=False)


def token_sequences(r: Readers) -> str:
    """The showcase lines token by token, expansion off, forced, and unknown."""
    seq: list[str] = []
    for text in SHOWCASE:
        tokens = r.tok.tokenize(text)
        forced = r.tok.expand_keywords(tokens, probability=1.0)
        seq.append(f"== {text!r}")
        seq.append(f"  tokenize (expansion off, validation/gates): {r.show(tokens)}")
        seq.append(f"  forced expansion (p=1, the shipping cache): {r.show(forced)}")
        seq.append(f"  tokenize, script surface:                    "
                   f"{r.show(r.script_tok.tokenize(text))}")
        for t in tokens:
            if not r.tok.expandable(t):
                continue
            unknown = r.without(t.text)
            seq.append(f"  '{t.text}' deleted from vocab, definitions loaded: "
                       f"{r.show(unknown.expand_keywords(unknown.tokenize(text)))}")
            if "_" not in t.text:
                bare = r.without(t.text, {})
                seq.append(f"  '{t.text}' deleted from vocab, no definitions:     "
                           f"{r.show(bare.expand_keywords(bare.tokenize(text)))}")
    return "\n".join(seq)


def encode_corpus(r: Readers, lines: list[dict], out: Path) -> Corpus:
    """The corpus encoded with expansion off and forced, cached beside the report."""
    texts = sorted({row["text"] for row in lines})
    cache_file = out / "corpus_encodings.npz"
    if cache_file.exists():
        stored = np.load(cache_file, allow_pickle=True)
        if list(stored["texts"]) == texts:
            return Corpus(lines, texts, unit(stored["plain"]), unit(stored["forced"]))
        cache_file.unlink()
    plain_tokens = [r.tok.tokenize(t) for t in texts]
    forced_tokens = [r.tok.expand_keywords(list(t), probability=1.0) for t in plain_tokens]
    plain = r.encode(plain_tokens)
    changed = [i for i, (a, b) in enumerate(zip(plain_tokens, forced_tokens))
               if len(a) != len(b) or any(x is not y for x, y in zip(a, b))]
    forced = plain.copy()
    forced[changed] = r.encode([forced_tokens[i] for i in changed])
    np.savez(cache_file, texts=np.array(texts, dtype=object), plain=plain, forced=forced)
    return Corpus(lines, texts, unit(plain), unit(forced))


def expansion_effect(paths: ProbePaths, corpus: Corpus, keyword_lines: list[dict]) -> list[str]:
    """Whether the shipping cache is the forced encoding, and how far forcing moves a text."""
    summary = []
    check = []
    cache = AbilityCacheStore(paths.abilities_root)
    for row in keyword_lines[:2000:50]:
        # A sidecar sits beside its converted ``<name>.txt``, so its stem
        # plus ``.txt`` is the script file the cache is keyed by.
        matrix = (cache.read(row["source"] + ".txt")
                  if row["source"].startswith("cardsfolder/") else None)
        if matrix is not None:
            e = unit(matrix[row["index"]][None])[0]
            i = corpus.row_of[row["text"]]
            check.append((float(e @ corpus.forced[i]), float(e @ corpus.plain[i])))
    if check:
        c = np.array(check)
        summary.append(
            f"Shipping cache vs this script's encodings, {len(c)} keyword "
            f"rows: mean cos to forced expansion {c[:, 0].mean():.4f}, to "
            f"expansion-off {c[:, 1].mean():.4f}")
    plain_u, forced_u = corpus.plain, corpus.forced
    moved = np.einsum("ij,ij->i", plain_u, forced_u)
    affected = moved < 0.999999
    summary.append(
        f"Corpus unique script texts: {len(corpus.texts)}; changed by forced "
        f"expansion: {int(affected.sum())} "
        f"({affected.mean():.1%}); cos(plain, forced) over those: median "
        f"{np.median(moved[affected]):.3f}, 10th pct "
        f"{np.percentile(moved[affected], 10):.3f}, min {moved.min():.3f}")
    # Nearest-neighbour agreement between the two encodings of a text.
    sample = np.random.default_rng(0).choice(
        np.flatnonzero(affected), size=min(500, int(affected.sum())), replace=False)
    nn_plain = np.argsort(-(plain_u[sample] @ plain_u.T), axis=1)[:, 1:11]
    nn_forced = np.argsort(-(forced_u[sample] @ forced_u.T), axis=1)[:, 1:11]
    overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(nn_plain, nn_forced)])
    summary += [f"  10-NN overlap between the plain and forced spaces for 500 "
                f"affected texts: {overlap:.2f}", ""]
    return summary


def keyword_vs_definition(corpus: Corpus, klines: pd.DataFrame) -> tuple:
    """Question 1 per keyword whose expansion fires: the bare line against its definition.

    Also the bare lines' and the expansions' encodings, ``A`` and ``B``, row by row.
    """
    measured = klines[klines.fires].copy()
    A = corpus.plain[[corpus.row_of[t] for t in measured.line]]
    B = corpus.forced[[corpus.row_of[t] for t in measured.line]]
    measured["cos_keyword_definition"] = np.einsum("ij,ij->i", A, B)
    measured["rank_def_among_defs"] = ranks(A, B)
    measured["rank_kw_among_kws"] = ranks(B, A)
    # Rank of the definition among every corpus text (plus itself), by
    # similarity to the bare keyword.
    own = measured["cos_keyword_definition"].to_numpy()[:, None]
    measured["def_rank_in_corpus"] = (A @ corpus.plain.T > own).sum(axis=1) + 1
    # And the bare keyword's rank among every corpus text for its definition.
    own_b = np.einsum("ij,ij->i", B, A)[:, None]
    measured["kw_rank_in_corpus_from_def"] = (B @ corpus.plain.T > own_b).sum(axis=1) + 1
    # The null: how close are *other* keywords' definitions?
    sims = A @ B.T
    np.fill_diagonal(sims, np.nan)
    measured["cos_other_defs_median"] = np.nanmedian(sims, axis=1)
    measured["kind"] = [KIND.get(n, "other") for n in measured.keyword]
    measured["parameterized"] = measured.line.str.contains(":")
    measured["freq"] = [freq_bin(c) for c in measured.carriers]
    return measured, A, B


def definition_retrieval(measured: pd.DataFrame) -> pd.DataFrame:
    """Question 1's retrieval scores overall and by frequency, kind and parameter."""
    n = len(measured)

    def block(frame, label):
        a = retrieval(frame.rank_def_among_defs.to_numpy(), n)
        b = retrieval(frame.rank_kw_among_kws.to_numpy(), n)
        return {
            "group": label, "n": a["n"],
            "cos_kw_def_median": frame.cos_keyword_definition.median(),
            "cos_other_median": frame.cos_other_defs_median.median(),
            "kw->def top1": a["top1"], "kw->def top5": a["top5"],
            "kw->def MRR": a["mrr"], "def->kw top1": b["top1"],
            "def->kw MRR": b["mrr"],
            "median def rank in corpus": frame.def_rank_in_corpus.median(),
            "chance top1": a["chance_top1"], "chance top5": a["chance_top5"],
            "chance MRR": a["chance_mrr"],
        }

    table = [block(measured, "all")]
    for key in ("freq", "kind", "parameterized"):
        table += [block(frame, f"{key}={value}") for value, frame in measured.groupby(key)]
    return pd.DataFrame(table)


def longhand_twins(corpus: Corpus, names: list[str], A: np.ndarray,
                   B: np.ndarray) -> pd.DataFrame:
    """Which keyword each longhand twin retrieves, bare and expanded."""
    kw_index = {k: i for i, k in enumerate(names)}
    rows = []
    for keyword, source, note in TWINS:
        text = corpus.twin_text(keyword, source)
        if text is None or keyword not in kw_index:
            rows.append({"keyword": keyword, "card": source, "note": note, "resolved": False})
            continue
        twin = corpus.plain[corpus.row_of[text]]
        k = kw_index[keyword]
        to_bare, to_def = A @ twin, B @ twin
        rows.append({
            "keyword": keyword, "card": source, "note": note, "resolved": True,
            "cos_to_bare": float(to_bare[k]),
            "cos_to_definition": float(to_def[k]),
            "rank_of_keyword_bare": int((to_bare > to_bare[k]).sum() + 1),
            "rank_of_keyword_def": int((to_def > to_def[k]).sum() + 1),
            "nearest_bare_keyword": names[int(np.argmax(to_bare))],
            "nearest_definition": names[int(np.argmax(to_def))],
            "twin_text": text[:160],
        })
    return pd.DataFrame(rows)


def unknown_keywords(r: Readers, corpus: Corpus, measured: pd.DataFrame, A: np.ndarray,
                     B: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Question 2's simulation: each measured keyword's token deleted from the vocabulary,
    encoded with its definitions available and without."""
    rows, new_tokens, bare_tokens = [], [], []
    for name, text in zip(measured.keyword, measured.line):
        fired = next(t.text for t in r.tok.tokenize(text) if r.tok.expandable(t))
        unknown, bare = r.without(fired), r.without(fired, {})
        u_tokens = unknown.expand_keywords(unknown.tokenize(text))
        b_tokens = bare.expand_keywords(bare.tokenize(text))
        rows.append({
            "keyword": name, "line": text, "token": fired,
            "unknown_expands": any(t.expanded_from for t in u_tokens),
            "unknown_tokens": r.show(u_tokens)[:200],
            "bare_tokens": r.show(b_tokens),
        })
        new_tokens.append(u_tokens)
        bare_tokens.append(b_tokens)
    U = unit(r.encode(new_tokens))
    U0 = unit(r.encode(bare_tokens))
    sim = pd.DataFrame(rows)
    sim["cos_new_to_real"] = np.einsum("ij,ij->i", U, A)
    sim["cos_new_to_forced"] = np.einsum("ij,ij->i", U, B)
    sim["rank_real_among_kws"] = ranks(U, A)
    sim["real_rank_in_corpus"] = (
        (U @ corpus.plain.T) > sim.cos_new_to_real.to_numpy()[:, None]).sum(axis=1) + 1
    sim["cos_unk_to_real"] = np.einsum("ij,ij->i", U0, A)
    sim["rank_real_among_kws_unk"] = ranks(U0, A)
    sim["kind"] = measured.kind.to_numpy()
    sim["freq"] = measured.freq.to_numpy()
    return sim, U


def new_keyword_retrieval(sim: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    groups = [("all", sim), *[(f"freq={k}", f) for k, f in sim.groupby("freq")],
              *[(f"kind={k}", f) for k, f in sim.groupby("kind")]]
    for label, frame in groups:
        a = retrieval(frame.rank_real_among_kws.to_numpy(), n)
        b = retrieval(frame.rank_real_among_kws_unk.to_numpy(), n)
        rows.append({
            "group": label, "n": len(frame),
            "cos new->real median": frame.cos_new_to_real.median(),
            "new top1": a["top1"], "new top5": a["top5"], "new MRR": a["mrr"],
            "median real rank in corpus": frame.real_rank_in_corpus.median(),
            "[UNK] top1": b["top1"], "[UNK] MRR": b["mrr"],
            "chance top1": a["chance_top1"], "chance MRR": a["chance_mrr"],
        })
    return pd.DataFrame(rows)


def twins_for_new_keyword(corpus: Corpus, twins: pd.DataFrame, names: list[str],
                          U: np.ndarray, A: np.ndarray) -> pd.DataFrame:
    kw_index = {k: i for i, k in enumerate(names)}
    rows = []
    for _, row in twins[twins.resolved].iterrows():
        k = kw_index[row.keyword]
        twin = corpus.plain[corpus.row_of[corpus.twin_text(row.keyword, row.card)]]
        rows.append({
            "keyword": row.keyword, "card": row.card,
            "cos_new_to_twin": float(U[k] @ twin),
            "cos_real_to_twin": float(A[k] @ twin),
            "twin_rank_among_corpus_for_new": int(
                ((U[k] @ corpus.plain.T) > U[k] @ twin).sum() + 1),
            "twin_rank_among_corpus_for_real": int(
                ((A[k] @ corpus.plain.T) > A[k] @ twin).sum() + 1),
        })
    return pd.DataFrame(rows)


def vocabulary_gaps(defs: pd.DataFrame, klines: pd.DataFrame) -> list[str]:
    """Question 2's census: definitions without a token, lines that never expand."""
    missing = defs[~defs.in_vocab]
    multi = defs[defs.multi_word]
    no_fire = klines[klines.in_definitions & ~klines.fires]
    return [
        f"Q2 — {len(missing)} of {len(defs)} definitions have no vocabulary "
        f"token: {', '.join(missing.keyword)}",
        f"  multi-word definitions: {len(multi)}, of which in the vocabulary "
        f"as a merged token: {int(multi.in_vocab.sum())}",
        f"  corpus keywords with a definition whose line never expands: "
        f"{len(no_fire)} ({int(no_fire.carriers.sum())} card lines): "
        + ", ".join(f"{k} [{t}]" for k, t in zip(no_fire.keyword, no_fire.tokens)),
        f"  definitions whose expansion leaves a '%' token (a %d placeholder "
        f"the tokenizer does not strip): "
        f"{int((defs.leftover_percent_tokens > 0).sum())}; with [UNK] "
        f"tokens: {int((defs.unk_in_expansion > 0).sum())}",
    ]


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = add_probe_arguments(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    report(resolve_paths(parser.parse_args()))


def report(paths: ProbePaths) -> None:
    out = paths.out / "keyword-expansion"
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    r, provenance = load_readers(paths)
    summary: list[str] = [
        f"checkpoint {paths.checkpoint}",
        f"vocab {provenance.vocab_path}, keywords "
        f"{provenance.keyword_definitions_path}, withheld keyword "
        f"{provenance.withheld_keyword!r}",
        "",
    ]
    lines = corpus_lines(paths.cards_folders)
    keyword_lines = [row for row in lines if row["api"] == "Keyword"]
    carriers = Counter(row["text"].split(":")[0] for row in keyword_lines)
    _, rarity = load_manifest_sets(paths)

    defs = definitions_table(r, carriers)
    defs.to_csv(out / "definitions_vs_vocab.csv", index=False)
    klines = keyword_line_table(r, keyword_lines, carriers, rarity)
    klines.to_csv(out / "keyword_lines.csv", index=False)
    (out / "token_sequences.txt").write_text(token_sequences(r), encoding="utf-8")

    corpus = encode_corpus(r, lines, out)
    summary += expansion_effect(paths, corpus, keyword_lines)

    measured, A, B = keyword_vs_definition(corpus, klines)
    measured.to_csv(out / "q1_keyword_vs_definition.csv", index=False)
    names, n = list(measured.keyword), len(measured)
    q1 = definition_retrieval(measured)
    q1.to_csv(out / "q1_retrieval.csv", index=False)
    summary += [
        f"Q1 — {n} corpus keywords whose expansion fires (of {len(klines)} keyword names "
        f"in the corpus)",
        f"  cos(bare, expanded): median {measured.cos_keyword_definition.median():.3f}; "
        f"median cos to the other keywords' definitions "
        f"{measured.cos_other_defs_median.median():.3f}",
        table_text(q1),
        f"  corpus size for the in-corpus rank: {len(corpus.texts)} texts", ""]

    twins = longhand_twins(corpus, names, A, B)
    twins.to_csv(out / "q1_longhand_twins.csv", index=False)
    summary += [f"Longhand twins (rank of the right keyword among {n}):",
                table_text(twins.drop(columns=["twin_text"])), ""]

    summary += vocabulary_gaps(defs, klines)
    sim, U = unknown_keywords(r, corpus, measured, A, B)
    sim.to_csv(out / "q2_simulated_new_keyword.csv", index=False)
    q2 = new_keyword_retrieval(sim, n)
    q2.to_csv(out / "q2_retrieval.csv", index=False)
    summary += ["  Simulated new keyword (token deleted from the vocabulary):", table_text(q2)]
    twin_new = twins_for_new_keyword(corpus, twins, names, U, A)
    twin_new.to_csv(out / "q2_twins_for_new_keyword.csv", index=False)
    summary += ["  Longhand twins seen from the simulated new keyword:", table_text(twin_new)]

    summary += ["", "  Nearest corpus lines to a simulated new keyword:"]
    for name in ("Flying", "Deathtouch", "Lifelink", "Prowess", "Ward",
                 "Exalted", "Afflict", "Toxic", "Cycling", "Firebending"):
        if name in names:
            order = np.argsort(-(U[names.index(name)] @ corpus.plain.T))[:5]
            summary.append(f"   {name}: " + " || ".join(corpus.texts[i][:70] for i in order))
    absent = klines[klines.keyword.isin(set(defs[~defs.in_vocab].keyword))]
    summary += ["", "  Corpus keywords whose definition token the vocabulary lacks: "
                + (", ".join(f"{k} -> {t}" for k, t in zip(absent.keyword, absent.tokens))
                   or "none")]
    (out / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
