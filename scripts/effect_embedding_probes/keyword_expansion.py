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
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ProbePaths,
    add_probe_arguments,
    load_manifest_sets,
    resolve_paths,
)

from effects.application.extract_keyword_definitions import (  # noqa: E402
    load_keyword_definitions,
)
from effects.domain.ability_encoder import (  # noqa: E402
    AbilityEncoder,
    collate_lines,
    prepare_line,
)
from effects.domain.ability_tokenizer import (  # noqa: E402
    HOST_BODIED_KEYWORDS,
    AbilityTokenizer,
    tokenizer_surface,
)
from effects.infrastructure.effect_model_store import (  # noqa: E402
    EffectModelStore,
)
from effects.infrastructure.sidecar_io import tree_of_folder  # noqa: E402
from price_predictor.infrastructure.tokenizer_store import (  # noqa: E402
    load_vocabulary,
)

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


def load_encoder(checkpoint_path: Path):
    """The checkpoint's encoder, at the size it recorded, its vocabulary, and
    the training settings that name the tokenization rules it read."""
    checkpoint = EffectModelStore(checkpoint_path.parent).load(checkpoint_path)
    provenance = checkpoint.provenance
    vocab_path = ROOT / provenance.vocab_path
    keyword_path = ROOT / provenance.keyword_definitions_path
    provenance.verify_hashes(vocab_path=vocab_path, keyword_path=keyword_path)
    encoder = AbilityEncoder(checkpoint.encoder_config)
    encoder.load_state_dict(checkpoint.encoder_state)
    encoder.eval()
    return (
        encoder, load_vocabulary(vocab_path), provenance, keyword_path,
        checkpoint.training_settings,
    )


def corpus_lines(trees) -> list[dict]:
    """Every sidecar line with script text, with its card and tree path."""
    rows = []
    for tree in trees:
        for path in sorted(tree.rglob("*.provenance.json")):
            sidecar = json.loads(path.read_text(encoding="utf-8"))
            # Named by the source tree the keys spell, not the folder's own
            # name: a kept-aside ``gen1-cardsfolder`` still holds ``cardsfolder``.
            rel = (Path(tree_of_folder(tree)) / path.relative_to(tree)).as_posix()
            stem = rel.removesuffix(".provenance.json")
            for index, line in enumerate(sidecar["lines"]):
                text = line.get("script_text")
                if not text:
                    continue
                rows.append({
                    "source": stem, "card": sidecar["card"], "index": index,
                    "api": line.get("script_api_type"), "text": text,
                })
    return rows


def encode(encoder, tokenizer, token_lists) -> np.ndarray:
    """Unit-normalized ``e`` for each token list, batched by length."""
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    out = np.zeros((len(token_lists), encoder.config.e_dim), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(order), BATCH):
            chunk = order[start:start + BATCH]
            lines = [prepare_line(tokenizer, token_lists[i]) for i in chunk]
            batch = collate_lines(lines, tokenizer.pad_id)
            vectors, _ = encoder(**batch)
            out[chunk] = vectors.numpy()
    return out


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def show(tokens) -> str:
    return " ".join(
        t.text if t.token_id != UNK_ID else f"[UNK:{t.text}]" for t in tokens
    )


UNK_ID = -1


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


# ── main ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = add_probe_arguments(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
    )
    report(resolve_paths(parser.parse_args()))


def report(paths: ProbePaths) -> None:  # noqa: C901, PLR0912, PLR0915 — one linear report
    global UNK_ID
    OUT = paths.out / "keyword-expansion"  # noqa: N806 — the report directory
    CACHE_ROOT = paths.abilities_root / "cardsfolder"  # noqa: N806
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    encoder, vocab, provenance, keyword_path, settings = load_encoder(paths.checkpoint)
    definitions = load_keyword_definitions(keyword_path)
    # The checkpoint's own surface, so every sequence below is what that
    # encoder read; ``script_tok`` shows the script surface's rules beside it.
    tok = AbilityTokenizer(
        vocab, definitions,
        surface=tokenizer_surface(paths.surface, settings),
    )
    script_tok = AbilityTokenizer(vocab, definitions, surface="script")
    UNK_ID = tok.unk_id
    summary: list[str] = [
        f"checkpoint {paths.checkpoint}",
        f"vocab {provenance.vocab_path}, keywords "
        f"{provenance.keyword_definitions_path}, withheld keyword "
        f"{provenance.withheld_keyword!r}",
        "",
    ]

    lines = corpus_lines(paths.cards_folders)
    keyword_lines = [r for r in lines if r["api"] == "Keyword"]
    carriers = Counter(r["text"].split(":")[0] for r in keyword_lines)
    variants: dict[str, Counter] = defaultdict(Counter)
    for r in keyword_lines:
        variants[r["text"].split(":")[0]][r["text"]] += 1
    _, rarity = load_manifest_sets(paths)

    # ── 1. the definition table against the vocabulary ────────────────
    def_rows = []
    for name, definition in definitions.items():
        token = tok._keyword_token(name)  # noqa: SLF001 — the tokenizer's own key
        template = definition.reminder_template or ""
        expanded = tok._instantiate(template, None)  # noqa: SLF001
        parts = tok.tokenize(expanded)
        def_rows.append({
            "keyword": name,
            "token": token,
            "multi_word": "_" in token,
            "in_vocab": token in vocab,
            "host_bodied": name.lower() in HOST_BODIED_KEYWORDS,
            "has_script": bool(definition.generated_script),
            "placeholder": "%d" in template or "$d" in template
            or "%s" in template,
            "leftover_percent_tokens": sum(1 for p in parts if p.text == "%"),
            "unk_in_expansion": sum(1 for p in parts if p.token_id == UNK_ID),
            "corpus_carriers": carriers.get(name, 0),
            "expansion": show(parts),
        })
    defs = pd.DataFrame(def_rows).sort_values("keyword")
    defs.to_csv(OUT / "definitions_vs_vocab.csv", index=False)

    # ── 2. per keyword line: does expansion fire? ──────────────────────
    line_rows = []
    for name, counter in variants.items():
        text, _count = counter.most_common(1)[0]
        tokens = tok.tokenize(text)
        fires = [t.text for t in tokens if tok.expandable(t)]
        line_rows.append({
            "keyword": name, "line": text, "carriers": carriers[name],
            "in_definitions": name in definitions,
            "tokens": show(tokens), "expands_on": " ".join(fires),
            "fires": bool(fires),
            "acting_games": sum(rarity.get(t, 0) for t in counter),
        })
    klines = pd.DataFrame(line_rows).sort_values("carriers", ascending=False)
    klines.to_csv(OUT / "keyword_lines.csv", index=False)

    # ── 3. token sequences for the showcase lines ──────────────────────
    seq: list[str] = []
    for text in SHOWCASE:
        tokens = tok.tokenize(text)
        forced = tok.expand_keywords(tokens, probability=1.0)
        seq.append(f"== {text!r}")
        seq.append(f"  tokenize (expansion off, validation/gates): {show(tokens)}")
        seq.append(f"  forced expansion (p=1, the shipping cache): {show(forced)}")
        seq.append(
            f"  tokenize, script surface:                    "
            f"{show(script_tok.tokenize(text))}"
        )
        for t in tokens:
            if tok.expandable(t) and "_" not in t.text:
                reduced = {k: v for k, v in vocab.items() if k != t.text}
                unknown = AbilityTokenizer(reduced, definitions)
                bare = AbilityTokenizer(reduced, {})
                seq.append(
                    f"  '{t.text}' deleted from vocab, definitions loaded: "
                    f"{show(unknown.expand_keywords(unknown.tokenize(text)))}"
                )
                seq.append(
                    f"  '{t.text}' deleted from vocab, no definitions:     "
                    f"{show(bare.expand_keywords(bare.tokenize(text)))}"
                )
            elif tok.expandable(t):
                reduced = {k: v for k, v in vocab.items() if k != t.text}
                unknown = AbilityTokenizer(reduced, definitions)
                seq.append(
                    f"  '{t.text}' deleted from vocab, definitions loaded: "
                    f"{show(unknown.expand_keywords(unknown.tokenize(text)))}"
                )
    (OUT / "token_sequences.txt").write_text("\n".join(seq), encoding="utf-8")

    # ── 4. encode the whole corpus with expansion off (and forced) ─────
    texts = sorted({r["text"] for r in lines})
    cache_file = OUT / "corpus_encodings.npz"
    if cache_file.exists():
        stored = np.load(cache_file, allow_pickle=True)
        if list(stored["texts"]) == texts:
            plain, forced_all = stored["plain"], stored["forced"]
        else:
            cache_file.unlink()
    if not cache_file.exists():
        plain_tokens = [tok.tokenize(t) for t in texts]
        forced_tokens = [
            tok.expand_keywords(list(t), probability=1.0) for t in plain_tokens
        ]
        plain = encode(encoder, tok, plain_tokens)
        changed = [i for i, (a, b) in enumerate(zip(plain_tokens, forced_tokens))
                   if len(a) != len(b) or any(x is not y for x, y in zip(a, b))]
        forced_all = plain.copy()
        forced_all[changed] = encode(
            encoder, tok, [forced_tokens[i] for i in changed],
        )
        np.savez(cache_file, texts=np.array(texts, dtype=object),
                 plain=plain, forced=forced_all)
    row_of = {t: i for i, t in enumerate(texts)}
    plain_u, forced_u = unit(plain), unit(forced_all)

    # Sanity check: the shipping cache equals the forced encoding.
    check = []
    for r in keyword_lines[:2000:50]:
        npz = CACHE_ROOT / (r["source"].removeprefix("cardsfolder/") + ".npz")
        if r["source"].startswith("cardsfolder/") and npz.exists():
            e = np.load(npz)["e"][r["index"]]
            i = row_of[r["text"]]
            check.append((
                float(unit(e[None])[0] @ forced_u[i]),
                float(unit(e[None])[0] @ plain_u[i]),
            ))
    if check:
        c = np.array(check)
        summary.append(
            f"Shipping cache vs this script's encodings, {len(c)} keyword "
            f"rows: mean cos to forced expansion {c[:, 0].mean():.4f}, to "
            f"expansion-off {c[:, 1].mean():.4f}"
        )

    # How much forced expansion moves each corpus text (cache vs model input).
    moved = np.einsum("ij,ij->i", plain_u, forced_u)
    affected = moved < 0.999999
    summary.append(
        f"Corpus unique script texts: {len(texts)}; changed by forced "
        f"expansion: {int(affected.sum())} "
        f"({affected.mean():.1%}); cos(plain, forced) over those: median "
        f"{np.median(moved[affected]):.3f}, 10th pct "
        f"{np.percentile(moved[affected], 10):.3f}, min {moved.min():.3f}"
    )
    # Nearest-neighbour agreement between the two encodings of a text.
    sample = np.random.default_rng(0).choice(
        np.flatnonzero(affected), size=min(500, int(affected.sum())),
        replace=False,
    )
    nn_plain = np.argsort(-(plain_u[sample] @ plain_u.T), axis=1)[:, 1:11]
    nn_forced = np.argsort(-(forced_u[sample] @ forced_u.T), axis=1)[:, 1:11]
    overlap = np.mean([
        len(set(a) & set(b)) / 10 for a, b in zip(nn_plain, nn_forced)
    ])
    summary.append(
        f"  10-NN overlap between the plain and forced spaces for 500 "
        f"affected texts: {overlap:.2f}"
    )
    summary.append("")

    # ── 5. question 1: keyword vs its definition ───────────────────────
    measured = klines[klines.fires].copy()
    names = list(measured.keyword)
    A = plain_u[[row_of[t] for t in measured.line]]
    B = forced_u[[row_of[t] for t in measured.line]]
    measured["cos_keyword_definition"] = np.einsum("ij,ij->i", A, B)
    measured["rank_def_among_defs"] = ranks(A, B)
    measured["rank_kw_among_kws"] = ranks(B, A)
    # Rank of the definition among every corpus text (plus itself), by
    # similarity to the bare keyword.
    corpus_sims = A @ plain_u.T
    own = measured["cos_keyword_definition"].to_numpy()[:, None]
    measured["def_rank_in_corpus"] = (corpus_sims > own).sum(axis=1) + 1
    # And the bare keyword's rank among every corpus text for its definition.
    corpus_sims_b = B @ plain_u.T
    own_b = np.einsum("ij,ij->i", B, A)[:, None]
    measured["kw_rank_in_corpus_from_def"] = (corpus_sims_b > own_b).sum(axis=1) + 1
    # The null: how close are *other* keywords' definitions?
    sims = A @ B.T
    np.fill_diagonal(sims, np.nan)
    measured["cos_other_defs_median"] = np.nanmedian(sims, axis=1)
    measured["kind"] = [KIND.get(n, "other") for n in names]
    measured["parameterized"] = measured.line.str.contains(":")
    measured["freq"] = [freq_bin(c) for c in measured.carriers]
    measured.to_csv(OUT / "q1_keyword_vs_definition.csv", index=False)

    n = len(measured)
    summary.append(f"Q1 — {n} corpus keywords whose expansion fires "
                   f"(of {len(klines)} keyword names in the corpus)")
    summary.append(
        f"  cos(bare, expanded): median "
        f"{measured.cos_keyword_definition.median():.3f}; median cos to the "
        f"other keywords' definitions "
        f"{measured.cos_other_defs_median.median():.3f}"
    )

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
        for value, frame in measured.groupby(key):
            table.append(block(frame, f"{key}={value}"))
    q1 = pd.DataFrame(table)
    q1.to_csv(OUT / "q1_retrieval.csv", index=False)
    summary.append(q1.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    summary.append(f"  corpus size for the in-corpus rank: {len(texts)} texts")
    summary.append("")

    # ── 6. longhand twins ───────────────────────────────────────────────
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in lines:
        by_source[r["source"]].append(r)
    twin_rows = []
    kw_index = {k: i for i, k in enumerate(names)}
    for keyword, source, note in TWINS:
        marker = TWIN_MARKERS[keyword].split("|")
        candidates = [r for r in by_source.get(source, [])
                      if any(m in r["text"] for m in marker)]
        if not candidates or keyword not in kw_index:
            twin_rows.append({"keyword": keyword, "card": source,
                              "note": note, "resolved": False})
            continue
        twin = plain_u[row_of[candidates[0]["text"]]]
        k = kw_index[keyword]
        to_bare = A @ twin
        to_def = B @ twin
        twin_rows.append({
            "keyword": keyword, "card": source, "note": note, "resolved": True,
            "cos_to_bare": float(to_bare[k]),
            "cos_to_definition": float(to_def[k]),
            "rank_of_keyword_bare": int((to_bare > to_bare[k]).sum() + 1),
            "rank_of_keyword_def": int((to_def > to_def[k]).sum() + 1),
            "nearest_bare_keyword": names[int(np.argmax(to_bare))],
            "nearest_definition": names[int(np.argmax(to_def))],
            "twin_text": candidates[0]["text"][:160],
        })
    twins = pd.DataFrame(twin_rows)
    twins.to_csv(OUT / "q1_longhand_twins.csv", index=False)
    summary.append(f"Longhand twins (rank of the right keyword among {n}):")
    summary.append(twins.drop(columns=["twin_text"]).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    summary.append("")

    # ── 7. question 2: unknown keywords ─────────────────────────────────
    missing = defs[~defs.in_vocab]
    summary.append(
        f"Q2 — {len(missing)} of {len(defs)} definitions have no vocabulary "
        f"token: {', '.join(missing.keyword)}"
    )
    multi = defs[defs.multi_word]
    summary.append(
        f"  multi-word definitions: {len(multi)}, of which in the vocabulary "
        f"as a merged token: {int(multi.in_vocab.sum())}"
    )
    no_fire = klines[klines.in_definitions & ~klines.fires]
    summary.append(
        f"  corpus keywords with a definition whose line never expands: "
        f"{len(no_fire)} ({int(no_fire.carriers.sum())} card lines): "
        + ", ".join(f"{k} [{t}]" for k, t in zip(no_fire.keyword, no_fire.tokens))
    )
    summary.append(
        f"  definitions whose expansion leaves a '%' token (a %d placeholder "
        f"the tokenizer does not strip): "
        f"{int((defs.leftover_percent_tokens > 0).sum())}; with [UNK] "
        f"tokens: {int((defs.unk_in_expansion > 0).sum())}"
    )

    # Simulate a new keyword: delete each measured keyword's token.
    sim_rows = []
    new_vectors, bare_vectors = [], []
    for name, text in zip(names, measured.line):
        fired = next(t.text for t in tok.tokenize(text) if tok.expandable(t))
        reduced = {k: v for k, v in vocab.items() if k != fired}
        unknown = AbilityTokenizer(reduced, definitions)
        bare = AbilityTokenizer(reduced, {})
        u_tokens = unknown.expand_keywords(unknown.tokenize(text))
        b_tokens = bare.expand_keywords(bare.tokenize(text))
        sim_rows.append({
            "keyword": name, "line": text, "token": fired,
            "unknown_expands": any(t.expanded_from for t in u_tokens),
            "unknown_tokens": show(u_tokens)[:200],
            "bare_tokens": show(b_tokens),
        })
        new_vectors.append(u_tokens)
        bare_vectors.append(b_tokens)
    U = unit(encode(encoder, tok, new_vectors))
    U0 = unit(encode(encoder, tok, bare_vectors))
    sim = pd.DataFrame(sim_rows)
    sim["cos_new_to_real"] = np.einsum("ij,ij->i", U, A)
    sim["cos_new_to_forced"] = np.einsum("ij,ij->i", U, B)
    sim["rank_real_among_kws"] = ranks(U, A)
    sim["real_rank_in_corpus"] = (
        (U @ plain_u.T) > sim.cos_new_to_real.to_numpy()[:, None]
    ).sum(axis=1) + 1
    sim["cos_unk_to_real"] = np.einsum("ij,ij->i", U0, A)
    sim["rank_real_among_kws_unk"] = ranks(U0, A)
    sim["kind"] = measured.kind.to_numpy()
    sim["freq"] = measured.freq.to_numpy()
    sim.to_csv(OUT / "q2_simulated_new_keyword.csv", index=False)

    q2 = []
    for label, frame in [("all", sim), *[
        (f"freq={k}", f) for k, f in sim.groupby("freq")
    ], *[(f"kind={k}", f) for k, f in sim.groupby("kind")]]:
        a = retrieval(frame.rank_real_among_kws.to_numpy(), n)
        b = retrieval(frame.rank_real_among_kws_unk.to_numpy(), n)
        q2.append({
            "group": label, "n": len(frame),
            "cos new->real median": frame.cos_new_to_real.median(),
            "new top1": a["top1"], "new top5": a["top5"], "new MRR": a["mrr"],
            "median real rank in corpus": frame.real_rank_in_corpus.median(),
            "[UNK] top1": b["top1"], "[UNK] MRR": b["mrr"],
            "chance top1": a["chance_top1"], "chance MRR": a["chance_mrr"],
        })
    q2t = pd.DataFrame(q2)
    q2t.to_csv(OUT / "q2_retrieval.csv", index=False)
    summary.append("  Simulated new keyword (token deleted from the vocabulary):")
    summary.append(q2t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Twins for the simulated new keyword.
    rows = []
    for _, row in twins[twins.resolved].iterrows():
        k = kw_index[row.keyword]
        twin_text = next(r["text"] for r in by_source[row.card]
                         if any(m in r["text"]
                                for m in TWIN_MARKERS[row.keyword].split("|")))
        twin = plain_u[row_of[twin_text]]
        rows.append({
            "keyword": row.keyword, "card": row.card,
            "cos_new_to_twin": float(U[k] @ twin),
            "cos_real_to_twin": float(A[k] @ twin),
            "twin_rank_among_corpus_for_new": int(
                ((U[k] @ plain_u.T) > U[k] @ twin).sum() + 1),
            "twin_rank_among_corpus_for_real": int(
                ((A[k] @ plain_u.T) > A[k] @ twin).sum() + 1),
        })
    twin_new = pd.DataFrame(rows)
    twin_new.to_csv(OUT / "q2_twins_for_new_keyword.csv", index=False)
    summary.append("  Longhand twins seen from the simulated new keyword:")
    summary.append(twin_new.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Nearest corpus lines for a few simulated new keywords.
    summary.append("")
    summary.append("  Nearest corpus lines to a simulated new keyword:")
    for name in ("Flying", "Deathtouch", "Lifelink", "Prowess", "Ward",
                 "Exalted", "Afflict", "Toxic", "Cycling", "Firebending"):
        if name not in kw_index:
            continue
        k = kw_index[name]
        order = np.argsort(-(U[k] @ plain_u.T))[:5]
        summary.append(f"   {name}: " + " || ".join(
            texts[i][:70] for i in order))

    # Real keywords absent from the vocabulary but present in the corpus.
    absent = klines[klines.keyword.isin(set(missing.keyword))]
    summary.append("")
    summary.append(
        "  Corpus keywords whose definition token the vocabulary lacks: "
        + (", ".join(f"{k} -> {t}" for k, t in zip(absent.keyword, absent.tokens))
           or "none")
    )
    (OUT / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
