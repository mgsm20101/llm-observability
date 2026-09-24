"""An in-process dense store over a small local corpus.

Replaces Qdrant. Qdrant was a fourth service to stand up — after Ollama,
Langfuse and a corpus loader — for a project whose subject is observability,
and a service that has to be running before the project can be run at all is a
service that stops it being run. Twelve documents do not need a vector
database; they need a matrix and a dot product.

What this costs: nothing here demonstrates ANN indexing, sharding, filtered
search or persistence. Those are real Qdrant features and this file does not
stand in for them. What it buys is that `python -m src.demo` works on a clean
checkout with no containers.

The encoder is `intfloat/multilingual-e5-base`, which needs its `query:` and
`passage:` prefixes — e5 is trained with them and dropping them measurably
degrades retrieval, so they are applied here rather than left to callers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class StoredDoc:
    id: str
    title: str
    text: str


def load_corpus(path: Path) -> list[StoredDoc]:
    if not path.exists():
        raise FileNotFoundError(
            f"no corpus at {path}. It ships with the repository; if it is "
            f"missing, the checkout is incomplete."
        )
    docs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(StoredDoc(id=row["id"], title=row["title"], text=row["text"]))
    if not docs:
        raise ValueError(f"{path} is empty")
    return docs


@lru_cache(maxsize=2)
def _encoder(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class DenseStore:
    """Embeddings held in memory, cosine similarity by dot product.

    Vectors are L2-normalised once at build time, which makes cosine
    similarity a plain matrix multiply and keeps the query path free of
    per-call normalisation.
    """

    def __init__(self, docs: list[StoredDoc], model_name: str):
        self.docs = docs
        self.model_name = model_name
        model = _encoder(model_name)
        passages = [f"passage: {d.title}\n{d.text}" for d in docs]
        self._matrix = model.encode(
            passages, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int) -> list[tuple[StoredDoc, float]]:
        model = _encoder(self.model_name)
        vector = model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self._matrix @ vector
        order = np.argsort(-scores)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in order]
