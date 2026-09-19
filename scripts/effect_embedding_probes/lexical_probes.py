"""How much of the embedding a bag of words predicts, and from which part.

The script surface carries two kinds of words: the script's own keys and
values (``SP$ DealDamage | NumDmg$ 3 | ValidTgts$ Any``) and the English the
script embeds in its description parameters (``SpellDescription$``,
``TriggerDescription$``, ``Description$``, ...), which is the card's rules
text. A trigger line's script names the executed ability only by an SVar name
(``Execute$ TrigDraw``), so for triggers most of the "what happens" lives in
the description.

For four word views of each unique text —

- ``whole text``: every word;
- ``script only``: the text with description values removed;
- ``description only``: only the description values;
- ``API type + description``: the API type as one extra word, plus the
  description —

this fits a ridge regression from word presence (binary, words in at least 5
texts) to the text's scores on the top nine principal components of the
shipping embedding (which hold 99.9% of its variance) and reports the
out-of-fold R² per component and pooled over the nine, with folds grouped by
carrying card. A high R² from a view means the embedding is close to a linear
function of which words that view contains.

Run: ``python scripts/effect_embedding_probes/lexical_probes.py`` (CPU, a few
minutes). Writes ``lexical_probes.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import OUT, load_texts, matrix, pca, write_markdown  # noqa: E402

DESCRIPTION_KEYS = ("SpellDescription", "TriggerDescription", "Description",
                    "StackDescription", "PrecostDesc", "CostDesc", "TgtPrompt",
                    "ChangeTypeDesc", "ValidTgtsDesc")
_PART_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\$\s?(.*)$")
TOKEN = r"(?u)\{[^}]+\}|[A-Za-z_]+|\d+"
N_PCS = 9


def split_text(text: str) -> tuple[str, str]:
    """``(script words, description words)`` of one encoding text."""
    if "$" not in text:
        return text, ""
    script, description = [], []
    for part in text.split(" | "):
        match = _PART_RE.match(part)
        if match and match.group(1) in DESCRIPTION_KEYS:
            script.append(match.group(1))
            description.append(match.group(2))
        else:
            script.append(part)
    return " ".join(script), " ".join(description)


def cv_r2(X, Y, groups) -> np.ndarray:
    pred = np.zeros_like(Y)
    for train, test in GroupKFold(5).split(X, Y, groups):
        model = Ridge(alpha=3.0, solver="sparse_cg").fit(X[train], Y[train])
        pred[test] = model.predict(X[test])
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return np.append(1 - ss_res / ss_tot, 1 - ss_res.sum() / ss_tot.sum())


def main() -> None:
    table = load_texts()
    scores, _, share = pca(matrix(table, "e_full"))
    Y = scores[:, :N_PCS]
    groups = table["group"].to_numpy()
    parts = table.text.map(split_text)
    views = {
        "whole text": table.text,
        "script only": parts.map(lambda p: p[0]),
        "description only": parts.map(lambda p: p[1]),
        "API type + description": "apitype_" + table.api.str.replace(
            r"\W", "_", regex=True) + " " + parts.map(lambda p: p[1]),
    }
    rows = []
    for name, docs in views.items():
        vectorizer = CountVectorizer(token_pattern=TOKEN, lowercase=True,
                                     binary=True, min_df=5)
        X = vectorizer.fit_transform(docs.fillna("")).astype(np.float64)
        r2 = cv_r2(X, Y, groups)
        row = {"word view": name, "vocabulary": X.shape[1]}
        for k in range(N_PCS):
            row[f"PC{k + 1}"] = r2[k]
        row["pooled (99.9% of variance)"] = r2[-1]
        rows.append(row)
        print(row, flush=True)
    frame = pd.DataFrame(rows)
    has_description = (parts.map(lambda p: p[1]) != "").mean()
    write_markdown(frame, OUT / "lexical_probes.md",
                   "Bag-of-words ridge probes onto the top nine components",
                   f"Out-of-fold R², GroupKFold(5) by carrying card, ridge "
                   f"alpha 3 on binary word presence (min 5 texts). "
                   f"{has_description:.0%} of texts carry a description "
                   f"parameter. Component shares: "
                   + ", ".join(f"{s:.1%}" for s in share[:N_PCS]) + ".")


if __name__ == "__main__":
    main()
