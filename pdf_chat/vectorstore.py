"""Step 4 of RAG: keep all the chunk vectors and find the closest ones to a query.

This is a "vector database" with the mystery removed. A hosted vector DB (Chroma,
FAISS, Pinecone, Qdrant) does clever things to stay fast over millions of vectors,
but at its core it does precisely what this class does: store a big matrix of
vectors, and when asked, compute the similarity of a query vector to every stored
vector and hand back the closest few. For the sizes in this project, the plain
numpy version below is not just clear -- it is genuinely fast enough.

The similarity measure is COSINE SIMILARITY: the cosine of the angle between two
vectors, which ignores their length and looks only at direction. Because our
embedder already normalises every vector to length 1, the cosine is just the dot
product -- so searching is one matrix-times-vector multiply. A score of 1.0 means
"same direction" (as similar as it gets); 0.0 means "unrelated".
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    """An in-memory store of chunk vectors with brute-force cosine search."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.zeros((0, dim), dtype=np.float64)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their matching vectors (row i of `vectors` is `chunks[i]`)."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Got a different number of chunks and vectors.")
        if vectors.shape[0] == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vectors have width {vectors.shape[1]}, store expects {self.dim}.")
        self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        """Return the `k` chunks most similar to `query_vector`, best first.

        Each result is a (chunk, score) pair where score is the cosine similarity.
        With normalised vectors the whole search is `self._vectors @ query_vector`.
        """
        if len(self) == 0:
            return []
        scores = self._vectors @ query_vector  # one dot product per stored chunk
        k = min(k, len(self))
        # argpartition grabs the top-k cheaply; then we sort just those k by score.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._chunks[i], float(scores[i])) for i in top_idx]

    # --- saving and loading, so you can index once and query many times --------
    def save(self, folder: str | Path) -> None:
        """Persist the store to a folder: vectors as .npy, chunks as .jsonl."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self._vectors)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._chunks:
                fh.write(json.dumps(asdict(chunk)) + "\n")
        (folder / "meta.json").write_text(json.dumps({"dim": self.dim}), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "VectorStore":
        """Rebuild a store previously written by `save`."""
        folder = Path(folder)
        dim = json.loads((folder / "meta.json").read_text(encoding="utf-8"))["dim"]
        store = cls(dim=dim)
        vectors = np.load(folder / "vectors.npy")
        chunks: list[Chunk] = []
        with (folder / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                chunks.append(Chunk(**json.loads(line)))
        store.add(chunks, vectors)
        return store
