"""Step 3 of RAG: turn a piece of text into a vector (a fixed-length list of numbers).

This is the idea the whole field rests on: if you can place every passage as a
point in the same space, then "find text about X" becomes "find the nearest
points to X" -- geometry, which computers are very fast at.

We ship TWO embedders behind one tiny interface (an object with `.embed(text)`
returning a numpy vector, and `.embed_many(list)`):

  HashingEmbedder    The default. Pure numpy, no downloads, no key, instant. It
                     is a classic *lexical* vectoriser: it counts words (with a
                     TF-IDF weighting and stop-words removed) and hashes them
                     into a fixed number of buckets. Two passages that share
                     distinctive words land near each other. This is the honest
                     pre-neural baseline -- real information retrieval worked this
                     way for decades. Its blind spot: it matches WORDS, not
                     MEANING, so "biggest" will not match "largest". That blind
                     spot is exactly what the neural embedder below fixes, and
                     feeling the difference is one of the lessons of this project.

  SentenceTransformerEmbedder  The real thing, optional. If you `pip install
                     sentence-transformers`, this loads a small neural model
                     (all-MiniLM) that maps *meaning* to nearby vectors, so
                     "biggest" and "largest" land together. It downloads a model
                     the first time, so it is off by default.

Both produce L2-normalised vectors (length 1), which makes cosine similarity --
the standard way to compare embeddings -- a plain dot product later on.
"""

from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

# A compact English stop-word list. These words appear in almost every sentence,
# so they carry no signal about topic -- worse, they make unrelated texts look
# similar. Dropping them is what lets an off-topic question score ~0 (and so trip
# the "I don't know" guardrail) instead of matching on shared "the"s and "is"es.
STOP_WORDS: frozenset[str] = frozenset(
    """a an the and or but if then else of to in on at for from by with without
    is are was were be been being am do does did have has had having this that
    these those it its he she they them his her their we you your i me my our us
    as so than then there here what which who whom whose how why when where while
    about into over under again further once not no nor only own same too very
    can could will would shall should may might must just also about above below
    up down out off again more most some such only own""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case, split into word/number tokens, drop stop-words and 1-char tokens."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if tok not in STOP_WORDS and len(tok) > 1
    ]


class Embedder(Protocol):
    """The contract every embedder honours: text in, unit vector out."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...
    def embed_many(self, texts: list[str]) -> np.ndarray: ...


def _bucket(token: str, dim: int) -> int:
    """Map a word to one of `dim` buckets with a STABLE hash.

    We use zlib.crc32, not Python's built-in hash(): hash() of a string is
    randomly salted per process for security, so it would give different vectors
    every run and break reproducibility. crc32 is the same on every machine, every
    run -- which is what we want for a store we save to disk.
    """
    return zlib.crc32(token.encode("utf-8")) % dim


class HashingEmbedder:
    """Offline TF-IDF-style lexical embedder. The project's default.

    "TF-IDF" = term frequency x inverse document frequency. Term frequency: a word
    that appears a lot in a passage describes that passage. Inverse document
    frequency: a word that appears in *every* passage (like "planet" in a book of
    planets) tells you little, so we weigh it down. The IDF weights are learned by
    calling `.fit(corpus)`; if you skip fit, every word is weighted equally (still
    works, just blunter).
    """

    def __init__(self, dim: int = 2048) -> None:
        # dim = how many "buckets" words hash into. Bigger = fewer accidental
        # collisions (two different words landing in the same bucket, which can
        # make unrelated texts look similar). 2048 is plenty for these documents.
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float64)  # until fit() is called

    def fit(self, corpus: list[str]) -> "HashingEmbedder":
        """Learn inverse-document-frequency weights from a list of passages."""
        n_docs = max(len(corpus), 1)
        doc_freq = np.zeros(self.dim, dtype=np.float64)
        for doc in corpus:
            seen = {_bucket(tok, self.dim) for tok in tokenize(doc)}
            for bucket in seen:
                doc_freq[bucket] += 1.0
        # +1s are "smoothing": they stop us dividing by zero and tame extremes.
        self.idf = np.log((1.0 + n_docs) / (1.0 + doc_freq)) + 1.0
        return self

    def embed(self, text: str) -> np.ndarray:
        """Text -> a unit vector: bucketed word counts, IDF-weighted, then normalised."""
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in tokenize(text):
            vec[_bucket(tok, self.dim)] += 1.0
        vec *= self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into a 2-D array, one unit vector per row."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder:
    """Real neural embeddings via sentence-transformers. Optional, downloads a model.

    Same interface as HashingEmbedder, so the rest of the program cannot tell them
    apart. Use it to feel the jump from lexical to semantic matching. `.fit()` is a
    no-op here (neural models need no corpus statistics), which keeps the pipeline
    code identical whichever embedder it is handed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def fit(self, corpus: list[str]) -> "SentenceTransformerEmbedder":
        return self  # nothing to learn; here so it is a drop-in for HashingEmbedder

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return self._model.encode(texts, normalize_embeddings=True)


def make_embedder(kind: str = "hashing", **kwargs) -> Embedder:
    """Factory: 'hashing' (offline default) or 'sentence-transformers' (neural)."""
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    if kind in {"sentence-transformers", "st", "neural"}:
        return SentenceTransformerEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder {kind!r}. Use 'hashing' or 'sentence-transformers'.")
